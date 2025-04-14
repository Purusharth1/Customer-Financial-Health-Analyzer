"""Transaction Categorization Using LLM.

This module categorizes transactions into predefined categories using a
Language Model (LLM).
Key functionalities include:
- Mapping transaction descriptions to categories like food, utilities, entertainment.
- Handling ambiguous or unclear transaction descriptions.
- Enhancing categorization with context-aware rules and fallback mechanisms.
- Storing categorized data for downstream analysis.
"""

import logging
import sys
from pathlib import Path

import mlflow
import pandas as pd

from utils import (
    ensure_no_active_run,
    get_llm_config,
    sanitize_metric_name,
    setup_mlflow,
)

sys.path.append(str(Path(__file__).parent.parent))
from llm_setup.config import LLMConfig
from llm_setup.ollama_manager import query_llm, setup_ollama

# Custom logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def apply_rules(row: pd.Series) -> str:
    """Apply rule-based categorization to a single transaction."""
    narration = str(row["Narration"]).upper()
    withdrawal = row["Withdrawal (INR)"] if not pd.isna(row["Withdrawal (INR)"]) else 0
    deposit = row["Deposit (INR)"] if not pd.isna(row["Deposit (INR)"]) else 0

    # Income rules
    if deposit > 0 and any(
        kw in narration
        for kw in ["SALARY", "CREDIT", "REFUND", "INTEREST", "CASH DEP",
                   "NEFTCR", "FT-CR", "IMPS", "RTGS CR", "UPI"]
    ):
        return "Income"

    # Expense rules
    if withdrawal > 0:
        if any(kw in narration for kw in ["POS", "PURCHASE", "DEBIT", "PAYTM",
                                          "PAYU", "RELIANCE", "NWD"]):
            return "Expense (retail)"
        if any(kw in narration for kw in ["BILLPAY", "ELECTRICITY", "WATER",
                                          "GAS", "AIRTEL", "CITRUSAIRTEL"]):
            return "Expense (utilities)"
        if any(kw in narration for kw in ["LOAN", "EMI", "MORTGAGE",
                                          "CREDIT CARD", "FUNDTRANSFERTO"]):
            return "Expense (loan)"
        if any(kw in narration for kw in ["ATM", "ATW", "CASH", "CHQ PAID",
                            "CHEQUE", "NEFT", "IMPS", "UPI", "RTGS DR"]):
            return "Expense (other)"

    # Default
    return ""


def apply_llm_fallback(transactions_df: pd.DataFrame,
                       llm_config: LLMConfig,
                       valid_categories: list[str]) -> None:
    """Use LLM for ambiguous cases."""
    llm_needed = transactions_df[transactions_df["category"] == ""].index
    if llm_needed.any():
        logger.info("Using LLM for %d transactions", len(llm_needed))
        for idx in llm_needed:
            row = transactions_df.loc[idx]
            desc = row["Narration"]
            withdrawal = (row["Withdrawal (INR)"]
                        if not pd.isna(row["Withdrawal (INR)"]) else 0)
            deposit = row["Deposit (INR)"] if not pd.isna(row["Deposit (INR)"]) else 0
            amount = deposit - withdrawal

            prompt = (
                f"""Classify this HDFC bank transaction into
                EXACTLY one of these categories: """
                f"""{', '.join(valid_categories)}. Return ONLY the
                category name as a single word or phrase, """
                f"""with no explanations, newlines, or additional text.
                For example: 'Income' or 'Expense (retail)'. """
                f"""Description: '{desc}'. Amount: {amount:.2f} INR
                (positive=deposit, negative=withdrawal)."""
            )

            try:
                category = query_llm(prompt, llm_config).strip()
                # Extract valid category if response is verbose
                for valid_cat in valid_categories:
                    if valid_cat.lower() in category.lower():
                        category = valid_cat
                        break
                else:
                    category = None
                if category in valid_categories:
                    transactions_df.loc[idx, "category"] = category
                else:
                    logger.warning(
                        "Invalid LLM category '%s' for index %d, using fallback",
                        category,
                        idx,
                    )
                    transactions_df.loc[idx, "category"] = "Expense (other)"
            except Exception:
                logger.exception("LLM failed for index %d", idx)
                transactions_df.loc[idx, "category"] = "Expense (other)"


def categorize_transactions(timeline_csv: str, output_csv: str) -> pd.DataFrame | None:
    """Categorize transactions using rules, with LLM as fallback."""
    setup_mlflow()
    llm_config = get_llm_config()
    logger.info("Starting transaction categorization")

    if not setup_ollama(llm_config):
        logger.warning("Ollama setup failed, proceeding with rule-based categorization")

    ensure_no_active_run()
    with mlflow.start_run(run_name="Transaction_Categorization", nested=True):
        mlflow.log_param("input_csv", timeline_csv)
        mlflow.log_param("llm_model", llm_config.model_name)

        try:
            transactions_df = pd.read_csv(timeline_csv)
            logger.info("Loaded CSV with %d rows", len(transactions_df))
        except FileNotFoundError:
            logger.exception("Input CSV not found: %s", timeline_csv)
            mlflow.log_param("error", "Input CSV not found")
            return None

        if transactions_df.empty:
            logger.warning("No transactions to categorize")
            mlflow.log_param("warning", "No transactions to categorize")
            return transactions_df

        valid_categories = [
            "Income",
            "Expense (utilities)",
            "Expense (retail)",
            "Expense (loan)",
            "Expense (other)",
        ]

        # Rule-based categorization
        start_time = pd.Timestamp.now()
        transactions_df["category"] = transactions_df.apply(
            lambda row: apply_rules(row, valid_categories), axis=1,
        )
        rule_time = (pd.Timestamp.now() - start_time).total_seconds()
        logger.info(
            "Rules categorized %d/%d transactions in %.2f seconds",
            len(transactions_df[transactions_df["category"] != ""]),
            len(transactions_df),
            rule_time,
        )

        # LLM fallback for ambiguous cases
        apply_llm_fallback(transactions_df, llm_config, valid_categories)

        # Default for uncategorized transactions
        transactions_df.loc[
            transactions_df["category"] == "", "category",
        ] = "Expense (other)"

        # Count categories
        category_counts = transactions_df["category"].value_counts().to_dict()

        # Log metrics
        for category, count in category_counts.items():
            try:
                sanitized_metric = sanitize_metric_name(f"category_{category}")
                mlflow.log_metric(sanitized_metric, count)
            except Exception:
                logger.exception("Failed to log metric for '%s'", category)

        try:
            transactions_df.to_csv(output_csv, index=False)
            mlflow.log_artifact(output_csv)
            mlflow.log_metric("transactions_categorized", len(transactions_df))
            mlflow.log_metric("rule_processing_time_s", rule_time)
            logger.info("Categorized %d transactions", len(transactions_df))
        except Exception:
            logger.exception("Error saving output")

        return transactions_df


if __name__ == "__main__":
    timeline_csv = "data/output/timeline.csv"
    output_csv = "data/output/categorized.csv"
    transactions_df = categorize_transactions(timeline_csv, output_csv)
