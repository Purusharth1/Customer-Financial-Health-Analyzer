"""Visualization of Financial Data.

This module generates data for spending trends, expense breakdown, and account
overview, saving results as JSON for use in the financial pipeline's visualization
components.
"""
import json
import logging
import sys
from pathlib import Path

import mlflow
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from src.models import (
    AccountOverview,
    ExpenseBreakdown,
    SpendingTrends,
    VisualizerInput,
    VisualizerOutput,
)
from src.utils import setup_mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _create_default_output() -> VisualizerOutput:
    """Create default VisualizerOutput for error cases."""
    return VisualizerOutput(
        spending_trends=SpendingTrends(labels=["No Data"], expenses=[0], budget=[0]),
        expense_breakdown=ExpenseBreakdown(categories=["No Data"], percentages=[100]),
        account_overview=AccountOverview(
            total_balance=0.0,
            monthly_income=0.0,
            monthly_expense=0.0,
            balance_percentage=0.0,
            income_percentage=0.0,
            expense_percentage=0.0,
        ),
    )

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

def _validate_columns(transactions_df: pd.DataFrame) -> pd.DataFrame | None:
    """Validate required columns in DataFrame."""
    required = ["parsed_date", "Narration",
                "Withdrawal (INR)", "Deposit (INR)", "category"]
    if not all(col in transactions_df for col in required):
        missing = [col for col in required if col not in transactions_df]
        logger.error("Missing columns: %s", missing)
        return None
    return transactions_df

def _save_results(results: VisualizerOutput,
                  output_file: Path,
                  charts_file: Path | None = None) -> None:
    """Save visualization results to JSON file(s)."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as f:
        json.dump(results.dict(), f)
    logger.info("Default visualization data saved to %s", output_file)

    if (charts_file and charts_file.parent.exists()
        and charts_file.parent != output_file.parent):
        with charts_file.open("w") as f:
            json.dump(results.dict(), f)
        logger.info("Visualization data also saved to %s", charts_file)

def _compute_spending_trends(transactions_df: pd.DataFrame) -> SpendingTrends:
    """Compute spending trends data."""
    transactions_df["parsed_date"] = pd.to_datetime(
        transactions_df["parsed_date"], errors="coerce",
    )
    transactions_df["month"] = transactions_df["parsed_date"].dt.to_period("M")

    monthly_expenses = (
        transactions_df[transactions_df["Withdrawal (INR)"] > 0]
        .groupby("month")["Withdrawal (INR)"]
        .sum()
    )
    monthly_income = (
        transactions_df[transactions_df["Deposit (INR)"] > 0]
        .groupby("month")["Deposit (INR)"]
        .sum()
    )
    labels = sorted(set(monthly_expenses.index).union(monthly_income.index))
    labels = [str(m) for m in labels]
    expenses = [float(monthly_expenses.get(m, 0)) for m in labels]
    budget = [sum(expenses) / len(labels) * 1.2 if labels else 0 for _ in labels]
    return SpendingTrends(
        labels=labels or ["No Data"],
        expenses=expenses or [0],
        budget=budget or [0],
    )

def _compute_expense_breakdown(transactions_df: pd.DataFrame) -> ExpenseBreakdown:
    """Compute expense breakdown data."""
    expense_cats = (
        transactions_df[transactions_df["Withdrawal (INR)"] > 0]
        .groupby("category")["Withdrawal (INR)"]
        .sum()
    )
    total_expense = expense_cats.sum()
    categories = expense_cats.index.tolist() or ["No Expenses"]
    percentages = [
        float(amt / total_expense * 100)
        if total_expense else 100 for amt in expense_cats
    ] or [100]
    return ExpenseBreakdown(categories=categories, percentages=percentages)

def _compute_account_overview(transactions_df: pd.DataFrame) -> AccountOverview:
    """Compute account overview data."""
    total_income = transactions_df["Deposit (INR)"].sum()
    total_expense = transactions_df["Withdrawal (INR)"].sum()
    total_balance = total_income - total_expense
    latest_month = transactions_df["month"].max()
    prev_month = latest_month - 1
    latest_income = transactions_df[transactions_df["month"] == latest_month][
        "Deposit (INR)"
    ].sum()
    latest_expense = transactions_df[transactions_df["month"] == latest_month][
        "Withdrawal (INR)"
    ].sum()
    prev_income = transactions_df[transactions_df["month"] == prev_month][
        "Deposit (INR)"
    ].sum()
    prev_expense = transactions_df[transactions_df["month"] == prev_month][
        "Withdrawal (INR)"
    ].sum()
    prev_balance = prev_income - prev_expense
    return AccountOverview(
        total_balance=float(total_balance),
        monthly_income=float(latest_income),
        monthly_expense=float(latest_expense),
        balance_percentage=float(
            ((total_balance - prev_balance)/ prev_balance * 100) if prev_balance else 0,
        ),
        income_percentage=float(
            ((latest_income - prev_income) / prev_income * 100) if prev_income else 0,
        ),
        expense_percentage=float(
            ((latest_expense - prev_expense)/prev_expense * 100) if prev_expense else 0,
        ),
    )

def generate_visualizations(input_model: VisualizerInput) -> VisualizerOutput:
    """Generate data for spending trends, expense breakdown, and account overview."""
    setup_mlflow()
    logger.info("Generating visualization data: %s", input_model.input_csv)

    results = _create_default_output()
    output_file = input_model.output_dir / "visualization_data.json"
    charts_file = input_model.output_dir / "charts" / "visualization_data.json"

    with mlflow.start_run(run_name="Visualization"):
        mlflow.log_param("input_csv", str(input_model.input_csv))
        start_time = pd.Timestamp.now()

        transactions_df = _read_transactions(input_model.input_csv)
        if transactions_df is None:
            _save_results(results, output_file, charts_file)
            return results

        mlflow.log_metric("transactions_visualized", len(transactions_df))
        transactions_df = _validate_columns(transactions_df)
        if transactions_df is None:
            _save_results(results, output_file, charts_file)
            return results

        results.spending_trends = _compute_spending_trends(transactions_df)
        results.expense_breakdown = _compute_expense_breakdown(transactions_df)
        results.account_overview = _compute_account_overview(transactions_df)

        mlflow.log_metrics({
            "total_balance": results.account_overview.total_balance,
            "monthly_income": results.account_overview.monthly_income,
            "monthly_expense": results.account_overview.monthly_expense,
            "expense_categories_count": len(results.expense_breakdown.categories),
            "months_analyzed": len(results.spending_trends.labels),
        })

        _save_results(results, output_file, charts_file)
        logger.info(
            "Visualization data generated: %.3f s",
            (pd.Timestamp.now() - start_time).total_seconds(),
        )
        return results

if __name__ == "__main__":
    input_model = VisualizerInput(
        input_csv=Path("data/output/categorized.csv"),
        output_dir=Path("data/output/charts"),
    )
    results = generate_visualizations(input_model)
    logger.info("Visualization results: %s", results.dict())
