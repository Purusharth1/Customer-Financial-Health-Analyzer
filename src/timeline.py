"""Date Recognition and Timeline Construction.

This module is responsible for identifying and organizing transactions into a chronological timeline.
Key functionalities include:
- Parsing and standardizing date formats from extracted data.
- Sorting transactions by date to create a coherent timeline.
- Detecting and resolving inconsistencies in date sequences.
- Generating time-based metadata (e.g., month, weekday) for analysis.
"""
import mlflow
import pandas as pd
from dateutil.parser import parse

from src.utils import setup_mlflow


def build_timeline(transactions_csv: str, output_csv: str) -> pd.DataFrame | None:
    """Build a chronological timeline from transactions CSV."""
    setup_mlflow()
    with mlflow.start_run(run_name="Timeline_Construction"):
        mlflow.log_param("input_csv", transactions_csv)

        try:
            df = pd.read_csv(transactions_csv)
        except FileNotFoundError:
            mlflow.log_param("error", "Input CSV not found")
            return None

        if df.empty:
            mlflow.log_param("warning", "No transactions in CSV")
            return df

        def parse_date(date_str: str) -> str | None:
            try:
                return parse(date_str, dayfirst=True).strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                return None

        df["parsed_date"] = df["Date"].apply(parse_date)
        invalid_dates = df["parsed_date"].isna().sum()
        mlflow.log_metric("invalid_dates_dropped", invalid_dates)

        df = df.dropna(subset=["parsed_date"])
        df = df.sort_values("parsed_date")

        df.to_csv(output_csv, index=False)
        mlflow.log_artifact(output_csv)
        mlflow.log_metric("transactions_timed", len(df))

        return df


if __name__ == "__main__":
    transactions_csv = "data/output/all_transactions.csv"
    output_csv = "data/output/timeline.csv"
    df = build_timeline(transactions_csv, output_csv)
    if df is not None:
        print(df.head())
