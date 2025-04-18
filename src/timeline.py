"""Timeline Construction for Financial Transactions.

This module builds a chronological timeline from a transactions CSV,
standardizing column names, parsing dates, and converting data into
CategorizedTransaction objects for use in the financial pipeline.
"""

import sys
from pathlib import Path

import mlflow
import pandas as pd
from dateutil.parser import parse

sys.path.append(str(Path(__file__).parent.parent))
from src.models import CategorizedTransaction, TimelineInput, TimelineOutput
from src.utils import ensure_no_active_run, setup_mlflow


def _read_transactions(input_csv: Path) -> pd.DataFrame | None:
    """Read and validate transactions CSV."""
    start_time = pd.Timestamp.now()
    try:
        transactions_df = pd.read_csv(input_csv)
        elapsed = (pd.Timestamp.now() - start_time).total_seconds()
        mlflow.log_metric("read_csv_duration_seconds", elapsed)
        # Handle the CSV content in a separate function
        return _process_transactions_df(transactions_df, input_csv)
    except FileNotFoundError:
        mlflow.log_param("read_csv_error", f"Input CSV not found: {input_csv}")
        return None


def _process_transactions_df(
    transactions_df: pd.DataFrame, input_csv: Path,
) -> pd.DataFrame | None:
    """Process the transactions DataFrame."""
    if transactions_df.empty:
        mlflow.log_param("process_csv_status", "Empty CSV")
        return None

    mlflow.log_metric("csv_rows_read", len(transactions_df))
    return transactions_df


def _standardize_columns(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names using alias mapping."""
    start_time = pd.Timestamp.now()
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

    elapsed = (pd.Timestamp.now() - start_time).total_seconds()
    mlflow.log_metric("standardize_columns_duration_seconds", elapsed)
    mlflow.log_metric("columns_standardized", len(column_mapping))
    return transactions_df


def _validate_columns(transactions_df: pd.DataFrame) -> pd.DataFrame | None:
    """Validate presence of expected columns."""
    start_time = pd.Timestamp.now()
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
    missing_columns = [
        col for col in expected_columns if col not in transactions_df.columns
    ]
    if missing_columns:
        mlflow.log_param(
            "validate_columns_error", f"Missing columns: {missing_columns}",
        )
        return None
    elapsed = (pd.Timestamp.now() - start_time).total_seconds()
    mlflow.log_metric("validate_columns_duration_seconds", elapsed)
    return transactions_df


def _clean_data(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """Clean string and numeric columns."""
    start_time = pd.Timestamp.now()
    string_columns = ["Reference Number", "Value Date", "Source_File"]
    for col in string_columns:
        if col in transactions_df.columns:
            transactions_df[col] = transactions_df[col].fillna("")

    numeric_columns = ["Withdrawal (INR)", "Deposit (INR)", "Closing Balance (INR)"]
    for col in numeric_columns:
        if col in transactions_df.columns:
            transactions_df[col] = transactions_df[col].fillna(0.0)

    elapsed = (pd.Timestamp.now() - start_time).total_seconds()
    mlflow.log_metric("clean_data_duration_seconds", elapsed)
    mlflow.log_metric("string_columns_cleaned", len(string_columns))
    mlflow.log_metric("numeric_columns_cleaned", len(numeric_columns))
    return transactions_df


def _parse_dates(transactions_df: pd.DataFrame) -> pd.DataFrame | None:
    """Parse dates and drop invalid entries."""
    start_time = pd.Timestamp.now()

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
        mlflow.log_param("parse_dates_status", "No valid dates after parsing")
        return None
    elapsed = (pd.Timestamp.now() - start_time).total_seconds()
    mlflow.log_metric("parse_dates_duration_seconds", elapsed)
    return transactions_df


def _convert_to_transactions(
    transactions_df: pd.DataFrame,
) -> list[CategorizedTransaction]:
    """Convert DataFrame rows to CategorizedTransaction objects."""
    start_time = pd.Timestamp.now()
    categorized_transactions: list[CategorizedTransaction] = []
    invalid_rows = 0
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
        except ValueError:
            invalid_rows += 1
            continue
    mlflow.log_metric("invalid_rows_dropped", invalid_rows)
    mlflow.log_metric("transactions_converted", len(categorized_transactions))
    elapsed = (pd.Timestamp.now() - start_time).total_seconds()
    mlflow.log_metric("convert_to_transactions_duration_seconds", elapsed)
    return categorized_transactions


def build_timeline(input_model: TimelineInput) -> TimelineOutput:
    """Build a chronological timeline from transactions CSV."""
    setup_mlflow()
    ensure_no_active_run()

    with mlflow.start_run(run_name="Timeline_Construction"):
        start_time = pd.Timestamp.now()
        mlflow.log_param("input_csv", str(input_model.transactions_csv))
        mlflow.log_param("output_csv", str(input_model.output_csv))

        transactions_df = _read_transactions(input_model.transactions_csv)
        if transactions_df is None:
            mlflow.log_metric("transactions_processed", 0)
            return TimelineOutput(transactions=[])

        transactions_df = _standardize_columns(transactions_df)
        transactions_df = _validate_columns(transactions_df)
        if transactions_df is None:
            mlflow.log_metric("transactions_processed", 0)
            return TimelineOutput(transactions=[])

        transactions_df = _clean_data(transactions_df)
        transactions_df = _parse_dates(transactions_df)
        if transactions_df is None:
            mlflow.log_metric("transactions_processed", 0)
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
        output_columns = [
            col for col in output_columns if col in transactions_df.columns
        ]
        df_output = transactions_df[output_columns]
        df_output.to_csv(input_model.output_csv, index=False)
        mlflow.log_artifact(str(input_model.output_csv))

        categorized_transactions = _convert_to_transactions(transactions_df)
        mlflow.log_metric("transactions_timed", len(categorized_transactions))
        elapsed = (pd.Timestamp.now() - start_time).total_seconds()
        mlflow.log_metric("build_timeline_duration_seconds", elapsed)
        return TimelineOutput(transactions=categorized_transactions)


if __name__ == "__main__":
    input_model = TimelineInput(
        transactions_csv=Path("data/output/all_transactions.csv"),
        output_csv=Path("data/output/timeline.csv"),
    )
    output = build_timeline(input_model)
