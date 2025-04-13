from typing import Optional
import logging
from pathlib import Path
import sys
import pandas as pd
import mlflow

from utils import setup_mlflow, get_llm_config, ensure_no_active_run, sanitize_metric_name

# Add project root to sys.path for module imports
project_root = Path(__file__).parent.parent
from llm_setup.ollama_manager import query_llm, setup_ollama
from llm_setup.config import LLMConfig
sys.path.insert(0, str(project_root))

def categorize_transactions(timeline_csv: str, output_csv: str) -> Optional[pd.DataFrame]:
    """Categorize transactions using rules, with LLM as fallback."""
    setup_mlflow()
    llm_config = get_llm_config()
    logging.info("Starting transaction categorization")

    if not setup_ollama(llm_config):
        logging.warning("Ollama setup failed, proceeding with rule-based categorization")

    ensure_no_active_run()
    with mlflow.start_run(run_name="Transaction_Categorization", nested=True):
        mlflow.log_param("input_csv", timeline_csv)
        mlflow.log_param("llm_model", llm_config.model_name)

        try:
            df = pd.read_csv(timeline_csv)
            logging.info(f"Loaded CSV with {len(df)} rows")
        except FileNotFoundError:
            logging.error("Input CSV not found: %s", timeline_csv)
            mlflow.log_param("error", "Input CSV not found")
            return None

        if df.empty:
            logging.warning("No transactions to categorize")
            mlflow.log_param("warning", "No transactions to categorize")
            return df

        valid_categories = [
            "Income",
            "Expense (utilities)",
            "Expense (retail)",
            "Expense (loan)",
            "Expense (other)"
        ]
        category_counts = {cat: 0 for cat in valid_categories}

        # Rule-based categorization
        def apply_rules(row: pd.Series) -> str:
            narration = str(row["Narration"]).upper()
            withdrawal = row["Withdrawal (INR)"] if not pd.isna(row["Withdrawal (INR)"]) else 0
            deposit = row["Deposit (INR)"] if not pd.isna(row["Deposit (INR)"]) else 0
            amount = deposit - withdrawal

            # Income rules
            if deposit > 0:
                if any(kw in narration for kw in [
                    "SALARY", "CREDIT", "REFUND", "INTEREST", "CASH DEP",
                    "NEFTCR", "FT-CR", "IMPS", "RTGS CR", "UPI"
                ]):
                    return "Income"

            # Expense rules
            if withdrawal > 0:
                if any(kw in narration for kw in [
                    "POS", "PURCHASE", "DEBIT", "PAYTM", "PAYU", "RELIANCE", "NWD"
                ]):
                    return "Expense (retail)"
                if any(kw in narration for kw in [
                    "BILLPAY", "ELECTRICITY", "WATER", "GAS", "AIRTEL", "CITRUSAIRTEL"
                ]):
                    return "Expense (utilities)"
                if any(kw in narration for kw in [
                    "LOAN", "EMI", "MORTGAGE", "CREDIT CARD", "FUNDTRANSFERTO"
                ]):
                    return "Expense (loan)"
                if any(kw in narration for kw in [
                    "ATM", "ATW", "CASH", "CHQ PAID", "CHEQUE", "NEFT", "IMPS", "UPI", "RTGS DR"
                ]):
                    return "Expense (other)"

            # Default
            return ""

        # Apply rules vectorized
        start_time = pd.Timestamp.now()
        df["category"] = df.apply(apply_rules, axis=1)

        rule_time = (pd.Timestamp.now() - start_time).total_seconds()
        logging.info(f"Rules categorized {len(df[df['category'] != ''])}/{len(df)} transactions in {rule_time:.2f}s")

        # LLM fallback for ambiguous cases
        llm_needed = df[df["category"] == ""].index
        if llm_needed.any():
            logging.info(f"Using LLM for {len(llm_needed)} transactions")
            for idx in llm_needed:
                row = df.loc[idx]
                desc = row["Narration"]
                withdrawal = row["Withdrawal (INR)"] if not pd.isna(row["Withdrawal (INR)"]) else 0
                deposit = row["Deposit (INR)"] if not pd.isna(row["Deposit (INR)"]) else 0
                amount = deposit - withdrawal

                prompt = (
                    f"Classify this HDFC bank transaction into EXACTLY one of these categories: "
                    f"{', '.join(valid_categories)}. Return ONLY the category name as a single word or phrase, "
                    f"with no explanations, newlines, or additional text. For example: 'Income' or 'Expense (retail)'. "
                    f"Description: '{desc}'. Amount: {amount:.2f} INR (positive=deposit, negative=withdrawal)."
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
                        df.at[idx, "category"] = category
                    else:
                        logging.warning("Invalid LLM category '%s' for index %d, using fallback", category, idx)
                        df.at[idx, "category"] = "Expense (other)"
                except Exception as e:
                    logging.error("LLM failed for index %d: %s", idx, str(e))
                    df.at[idx, "category"] = "Expense (other)"
        else:
            df.loc[df["category"] == "", "category"] = "Expense (other)"

        # Count categories
        for category in df["category"]:
            category_counts[category] += 1

        # Log metrics
        for category, count in category_counts.items():
            try:
                sanitized_metric = sanitize_metric_name(f"category_{category}")
                mlflow.log_metric(sanitized_metric, count)
            except Exception as e:
                logging.error("Failed to log metric for '%s': %s", category, str(e))

        try:
            df.to_csv(output_csv, index=False)
            mlflow.log_artifact(output_csv)
            mlflow.log_metric("transactions_categorized", len(df))
            mlflow.log_metric("rule_processing_time_s", rule_time)
            logging.info("Categorized %d transactions", len(df))
        except Exception as e:
            logging.error("Error saving output: %s", str(e))

        return df


if __name__ == "__main__":
    timeline_csv = "data/output/timeline.csv"
    output_csv = "data/output/categorized.csv"
    df = categorize_transactions(timeline_csv, output_csv)
    if df is not None:
        print(df.head())