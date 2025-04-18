"""Timeline Construction for Financial Transactions.

This module builds a chronological timeline from a transactions CSV,
standardizing column names, parsing dates, and converting data into
CategorizedTransaction objects for use in the financial pipeline.
"""
import logging
import sys
from pathlib import Path

import mlflow
import pandas as pd
from dateutil.parser import parse

sys.path.append(str(Path(__file__).parent.parent))
from src.models import CategorizedTransaction, TimelineInput, TimelineOutput
from src.utils import setup_mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _read_transactions(input_csv: Path) -> pd.DataFrame | None:
    """Read and validate transactions CSV."""
    start_time = pd.Timestamp.now()
    try:
        transactions_df = pd.read_csv(input_csv)
        logger.info(
            "Read CSV: %.3f s",
            (pd.Timestamp.now() - start_time).total_seconds(),
        )
        # Handle the CSV content in a separate function
        return _process_transactions_df(transactions_df, input_csv)
    except FileNotFoundError:
        logger.exception("Input CSV not found: %s", input_csv)
        mlflow.log_param("error", f"Input CSV not found: {input_csv}")
        return None

def _process_transactions_df(transactions_df: pd.DataFrame,
                input_csv: Path) -> pd.DataFrame | None:
    """Process the transactions DataFrame."""
    if transactions_df.empty:
        logger.warning("Empty CSV: %s", input_csv)
        mlflow.log_param("warning", "Empty CSV")
        return None

    logger.info("Read CSV with %d rows", len(transactions_df))
    return transactions_df


def _standardize_columns(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names using alias mapping."""
    column_mapping = {
        "Reference_Number": "Reference Number",
        "Value_Date": "Value Date",
        "Withdrawal_INR": "Withdrawal (INR)",
        "Deposit_INR": "Deposit (INR)",
        "Closing_Balance_INR": "Closing Balance (INR)",
    }

    for orig_col, alias_col in column_mapping.items():
        if orig_col in transactions_df.columns and alias_col in transactions_df.columns:
            if not transactions_df[orig_col].isna().all():
                transactions_df[alias_col] = transactions_df[orig_col]
            transactions_df = transactions_df.drop(columns=[orig_col])
        elif orig_col in transactions_df.columns:
            transactions_df = transactions_df.rename(columns={orig_col: alias_col})

    return transactions_df

def _validate_columns(transactions_df: pd.DataFrame) -> pd.DataFrame | None:
    """Validate presence of expected columns."""
    expected_columns = [
        "Date",
        "Narration",
        "Reference Number",
        "Value Date",
        "Withdrawal (INR)",
        "Deposit (INR)",
        "Closing Balance (INR)",
        "Source_File",
    ]
    missing_columns = [col for col in expected_columns
                       if col not in transactions_df.columns]
    if missing_columns:
        logger.error("Missing columns in CSV: %s", missing_columns)
        mlflow.log_param("error", f"Missing columns: {missing_columns}")
        return None
    return transactions_df

def _clean_data(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """Clean string and numeric columns."""
    string_columns = ["Reference Number", "Value Date", "Source_File"]
    for col in string_columns:
        if col in transactions_df.columns:
            transactions_df[col] = transactions_df[col].fillna("")

    numeric_columns = ["Withdrawal (INR)", "Deposit (INR)", "Closing Balance (INR)"]
    for col in numeric_columns:
        if col in transactions_df.columns:
            transactions_df[col] = transactions_df[col].fillna(0.0)

    return transactions_df

def _parse_dates(transactions_df: pd.DataFrame) -> pd.DataFrame | None:
    """Parse dates and drop invalid entries."""
    def parse_date(date_str: str) -> str | None:
        try:
            return parse(date_str, dayfirst=True).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            return None

    transactions_df["parsed_date"] = transactions_df["Date"].apply(parse_date)
    invalid_dates = transactions_df["parsed_date"].isna().sum()
    mlflow.log_metric("invalid_dates_dropped", invalid_dates)

    transactions_df = transactions_df.dropna(subset=["parsed_date"])
    if transactions_df.empty:
        logger.warning("No valid dates after parsing")
        mlflow.log_param("warning", "No valid dates after parsing")
        return None
    return transactions_df

def _convert_to_transactions(transactions_df: pd.DataFrame,
                            ) -> list[CategorizedTransaction]:
    """Convert DataFrame rows to CategorizedTransaction objects."""
    categorized_transactions: list[CategorizedTransaction] = []
    for _, row in transactions_df.iterrows():
        try:
            transaction_dict = {
                "Date": row.get("Date", ""),
                "Narration": row.get("Narration", ""),
                "Reference_Number": row.get("Reference Number", ""),
                "Value_Date": row.get("Value Date", ""),
                "Withdrawal_INR": float(row.get("Withdrawal (INR)", 0.0)),
                "Deposit_INR": float(row.get("Deposit (INR)", 0.0)),
                "Closing_Balance_INR": float(row.get("Closing Balance (INR)", 0.0)),
                "Source_File": row.get("Source_File", ""),
                "parsed_date": row.get("parsed_date", ""),
                "category": row.get("category", "Uncategorized"),
            }
            categorized_transaction = CategorizedTransaction(**transaction_dict)
            categorized_transactions.append(categorized_transaction)
        except ValueError as e:
            logger.warning("Failed to validate row: %s, error: %s", row.to_dict(), e)
            continue
    return categorized_transactions

def build_timeline(input_model: TimelineInput) -> TimelineOutput:
    """Build a chronological timeline from transactions CSV."""
    setup_mlflow()

    with mlflow.start_run(run_name="Timeline_Construction"):
        mlflow.log_param("input_csv", str(input_model.transactions_csv))

        transactions_df = _read_transactions(input_model.transactions_csv)
        if transactions_df is None:
            return TimelineOutput(transactions=[])

        transactions_df = _standardize_columns(transactions_df)
        transactions_df = _validate_columns(transactions_df)
        if transactions_df is None:
            return TimelineOutput(transactions=[])

        transactions_df = _clean_data(transactions_df)
        transactions_df = _parse_dates(transactions_df)
        if transactions_df is None:
            return TimelineOutput(transactions=[])

        transactions_df = transactions_df.sort_values("parsed_date")
        transactions_df["category"] = "Uncategorized"

        output_columns = [
            "Date",
            "Narration",
            "Reference Number",
            "Value Date",
            "Withdrawal (INR)",
            "Deposit (INR)",
            "Closing Balance (INR)",
            "Source_File",
            "parsed_date",
            "category",
        ]
        output_columns = [col for col in output_columns
                          if col in transactions_df.columns]
        df_output = transactions_df[output_columns]
        df_output.to_csv(input_model.output_csv, index=False)
        mlflow.log_artifact(str(input_model.output_csv))

        categorized_transactions = _convert_to_transactions(transactions_df)
        mlflow.log_metric("transactions_timed", len(categorized_transactions))
        return TimelineOutput(transactions=categorized_transactions)

if __name__ == "__main__":
    input_model = TimelineInput(
        transactions_csv=Path("data/output/all_transactions.csv"),
        output_csv=Path("data/output/timeline.csv"),
    )
    output = build_timeline(input_model)
    if output.transactions:
        logger.info(
            "Timeline transactions: %s",
            pd.DataFrame([{
                "Date": t.Date,
                "Narration": t.Narration,
                "parsed_date": t.parsed_date,
                "category": t.category,
            } for t in output.transactions]).head().to_dict("records"),
        )
