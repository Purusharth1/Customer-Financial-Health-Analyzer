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
import time
from pathlib import Path

import mlflow
import pandas as pd
import yaml

sys.path.append(str(Path(__file__).parent.parent))
from llm_setup.config import LLMConfig
from llm_setup.ollama_manager import query_llm, setup_ollama
from src.models import CategorizedTransaction, CategorizerInput, CategorizerOutput
from src.utils import (
    ensure_no_active_run,
    get_llm_config,
    sanitize_metric_name,
    setup_mlflow,
)

# Custom logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Constants
DESCRIPTION_TRUNCATE_LENGTH = 30
BATCH_SIZE = 15
EXAMPLE_COUNT_PER_CATEGORY = 1
KEYWORD_LIMIT = 3
INCOME_AMOUNT_BASE = 5000
EXPENSE_AMOUNT_BASE = 2000
DELAY_SECONDS = 0.2

def load_config() -> dict:
    """Load configuration from config file."""
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    try:
        with config_path.open() as file:
            return yaml.safe_load(file)
    except (FileNotFoundError, yaml.YAMLError):
        logger.exception("Error loading config")
        # Return a minimal default config
        return {
            "transaction_categories": {
                "income": ["Income (Other)"],
                "essential_expenses": ["Expense (Other)"],
            },
        }

def get_all_categories(config: dict) -> list[str]:
    """Get a flattened list of all transaction categories from config."""
    categories = []
    cat_groups = config.get("transaction_categories", {})
    for group in cat_groups.values():
        categories.extend(group)
    return categories

def get_category_keywords(config: dict) -> dict[str, list[str]]:
    """Get category keywords mapping from config."""
    return config.get("category_keywords", {})

def apply_rules(row: pd.Series, config: dict) -> str:
    """Apply rule-based categorization to a single transaction."""
    narration = str(row["Narration"]).upper()
    withdrawal = row["Withdrawal (INR)"] if not pd.isna(row["Withdrawal (INR)"]) else 0
    deposit = row["Deposit (INR)"] if not pd.isna(row["Deposit (INR)"]) else 0

    # Get category keywords from config
    category_keywords = get_category_keywords(config)

    # Check each category's keywords
    for category, keywords in category_keywords.items():
        is_income = category.startswith("Income") and deposit > 0
        is_expense = (
            (category.startswith("Expense") or category == "Savings/Investment")
            and withdrawal > 0
        )
        if (is_income or is_expense) and any(kw in narration for kw in keywords):
            return category

    # Default categorization based on transaction type
    if deposit > 0:
        return "Income (Other)"
    if withdrawal > 0:
        return "Expense (Other)"
    return ""

def get_example_transactions(config: dict) -> dict[str, list[tuple[str, float]]]:
    """Create example transactions for each category based on keywords."""
    examples: dict[str, list[tuple[str, float]]] = {}
    category_keywords = get_category_keywords(config)

    for category, keywords in category_keywords.items():
        examples[category] = []
        for i, keyword in enumerate(keywords[:KEYWORD_LIMIT]):
            if category.startswith("Income"):
                amount = INCOME_AMOUNT_BASE * (i + 1)
                desc = f"{keyword} TRANSACTION FROM XYZ"
            else:
                amount = -EXPENSE_AMOUNT_BASE * (i + 1)
                desc = f"PAYMENT FOR {keyword} SERVICE"
            examples[category].append((desc, amount))
    return examples

def _build_llm_prompt(
    desc: str, amount: float, valid_categories: list[str], context_examples: list[str],
) -> str:
    """Build LLM prompt with context examples."""
    return f"""As a financial analyst, categorize the following bank transaction
    into EXACTLY ONE of these categories:
{', '.join(valid_categories)}

Here are some examples of how to categorize similar transactions:
{chr(10).join(context_examples)}

Now, please categorize this transaction:
Description: '{desc}'
Amount: {amount:.2f} INR (positive=deposit, negative=withdrawal)

Respond ONLY with the exact category name from the provided list. No explanations or
additional text."""

def apply_llm_fallback(transactions_df: pd.DataFrame,
                       llm_config: LLMConfig, config: dict) -> None:
    """Use LLM for transaction categorization with few-shot learning examples."""
    valid_categories = get_all_categories(config)
    example_transactions = get_example_transactions(config)
    context_examples = [
        f"Description: '{desc}', Amount: {amount} INR → Category: {category}"
        for category, examples in example_transactions.items()
        for desc, amount in examples[:EXAMPLE_COUNT_PER_CATEGORY]
    ]

    llm_needed = transactions_df[transactions_df["category"] == ""].index
    if not llm_needed.empty:
        logger.info("Using LLM for %d transactions", len(llm_needed))
        for i in range(0, len(llm_needed), BATCH_SIZE):
            batch_indices = llm_needed[i:i+BATCH_SIZE]
            logger.info("Processing batch %d to %d", i, i+len(batch_indices)-1)

            for idx in batch_indices:
                row = transactions_df.loc[idx]
                desc = row["Narration"]
                withdrawal = (row["Withdrawal (INR)"]
                        if not pd.isna(row["Withdrawal (INR)"]) else 0)
                deposit = (row["Deposit (INR)"]
                        if not pd.isna(row["Deposit (INR)"]) else 0)
                amount = deposit - withdrawal

                prompt = _build_llm_prompt(desc, amount,
                                        valid_categories, context_examples)
                try:
                    llm_response = query_llm(prompt, llm_config).strip()
                    logger.debug("LLM raw response: '%s'", llm_response)

                    matched_category = next(
                        (cat for cat in valid_categories
                         if cat.lower() in llm_response.lower()), None,
                    )
                    if matched_category:
                        transactions_df.loc[idx, "category"] = matched_category
                        logger.info(
                            "Transaction '%s' (%.2f INR) categorized as '%s'",
                            desc[:DESCRIPTION_TRUNCATE_LENGTH] + (
                                "..." if len(desc) > DESCRIPTION_TRUNCATE_LENGTH else ""
                            ),
                            amount,
                            matched_category,
                        )
                    else:
                        default_category = ("Income (Other)"
                                        if amount > 0 else "Expense (Other)")
                        logger.warning(
                            "Invalid LLM category '%s' for '%s', using '%s'",
                            llm_response,
                            desc[:DESCRIPTION_TRUNCATE_LENGTH] + (
                                "..." if len(desc) > DESCRIPTION_TRUNCATE_LENGTH else ""
                            ),
                            default_category,
                        )
                        transactions_df.loc[idx, "category"] = default_category

                    time.sleep(DELAY_SECONDS)
                except Exception:
                    logger.exception(
                        "LLM failed for transaction '%s'",
                        desc[:DESCRIPTION_TRUNCATE_LENGTH] + (
                            "..." if len(desc) > DESCRIPTION_TRUNCATE_LENGTH else ""
                        ),
                    )
                    default_category = ("Income (Other)"
                                if amount > 0 else "Expense (Other)")
                    transactions_df.loc[idx, "category"] = default_category

def categorize_transactions(input_model: CategorizerInput) -> CategorizerOutput:
    """Categorize transactions using rules, with LLM as fallback."""
    setup_mlflow()
    llm_config = get_llm_config()
    config = load_config()
    logger.info("Starting transaction categorization")

    timeline_csv = input_model.timeline_csv
    output_csv = input_model.output_csv

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
            return CategorizerOutput(transactions=[])

        if transactions_df.empty:
            logger.warning("No transactions to categorize")
            mlflow.log_param("warning", "No transactions to categorize")
            return CategorizerOutput(transactions=[])

        valid_categories = get_all_categories(config)
        mlflow.log_param("categories_count", len(valid_categories))
        mlflow.log_param("categories", ", ".join(valid_categories))

        start_time = pd.Timestamp.now()
        transactions_df["category"] = transactions_df.apply(
            lambda row: apply_rules(row, config), axis=1,
        )
        rule_time = (pd.Timestamp.now() - start_time).total_seconds()
        logger.info(
            "Rules categorized %d/%d transactions in %.2f seconds",
            len(transactions_df[transactions_df["category"] != ""]),
            len(transactions_df),
            rule_time,
        )

        apply_llm_fallback(transactions_df, llm_config, config)
        transactions_df.loc[
            transactions_df["category"] == "", "category",
        ] = "Expense (Other)"

        category_counts = transactions_df["category"].value_counts().to_dict()
        for category, count in category_counts.items():
            try:
                sanitized_metric = sanitize_metric_name(f"category_{category}")
                mlflow.log_metric(sanitized_metric, count)
            except Exception:
                logger.exception("Failed to log metric for '%s'", category)

        transactions = [
            CategorizedTransaction(**row.to_dict())
            for _, row in transactions_df.iterrows()
        ]

        try:
            transactions_df.to_csv(output_csv, index=False)
            mlflow.log_artifact(output_csv)
            mlflow.log_metric("transactions_categorized", len(transactions_df))
            mlflow.log_metric("rule_processing_time_s", rule_time)
            logger.info("Categorized %d transactions", len(transactions_df))
        except Exception:
            logger.exception("Error saving output")

        return CategorizerOutput(transactions=transactions)

if __name__ == "__main__":
    input_model = CategorizerInput(
        timeline_csv="data/output/timeline.csv",
        output_csv="data/output/categorized.csv",
    )
    result = categorize_transactions(input_model)
