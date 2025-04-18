"""Visualization of Financial Data."""
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

def generate_visualizations(input_model: VisualizerInput) -> VisualizerOutput:
    """Generate data for spending trends, expense breakdown, and account overview."""
    setup_mlflow()
    logger.info(f"Generating visualization data: {input_model.input_csv}")

    input_csv = input_model.input_csv
    output_dir = input_model.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results = VisualizerOutput(
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

    with mlflow.start_run(run_name="Visualization"):
        mlflow.log_param("input_csv", str(input_csv))
        start_time = pd.Timestamp.now()
        try:
            df = pd.read_csv(input_csv)
            logger.info(f"Read CSV: {(pd.Timestamp.now() - start_time).total_seconds():.3f}s")
        except FileNotFoundError:
            logger.exception("Input CSV not found: %s", input_csv)
            output_file = output_dir / "visualization_data.json"
            with open(output_file, "w") as f:
                json.dump(results.dict(), f)  # Changed to results.dict()
            logger.info(f"Default visualization data saved to {output_file}")
            return results
        if df.empty:
            logger.warning(f"Empty CSV: {input_csv}")
            output_file = output_dir / "visualization_data.json"
            with open(output_file, "w") as f:
                json.dump(results.dict(), f)  # Changed to results.dict()
            logger.info(f"Default visualization data saved to {output_file}")
            return results
        mlflow.log_metric("transactions_visualized", len(df))

        required = ["parsed_date", "Narration", "Withdrawal (INR)", "Deposit (INR)", "category"]
        if not all(col in df for col in required):
            missing = [col for col in required if col not in df]
            logger.error("Missing columns: %s", missing)
            output_file = output_dir / "visualization_data.json"
            with open(output_file, "w") as f:
                json.dump(results.dict(), f)  # Changed to results.dict()
            logger.info(f"Default visualization data saved to {output_file}")
            return results

        df["parsed_date"] = pd.to_datetime(df["parsed_date"], errors="coerce")
        df["month"] = df["parsed_date"].dt.to_period("M")

        monthly_expenses = df[df["Withdrawal (INR)"] > 0].groupby("month")["Withdrawal (INR)"].sum()
        monthly_income = df[df["Deposit (INR)"] > 0].groupby("month")["Deposit (INR)"].sum()
        labels = sorted(set(monthly_expenses.index).union(monthly_income.index))
        labels = [str(m) for m in labels]
        expenses = [float(monthly_expenses.get(m, 0)) for m in labels]
        budget = [sum(expenses) / len(labels) * 1.2 if labels else 0 for _ in labels]
        results.spending_trends = SpendingTrends(
            labels=labels or ["No Data"],
            expenses=expenses or [0],
            budget=budget or [0],
        )

        expense_cats = df[df["Withdrawal (INR)"] > 0].groupby("category")["Withdrawal (INR)"].sum()
        total_expense = expense_cats.sum()
        categories = expense_cats.index.tolist() or ["No Expenses"]
        percentages = [float(amt / total_expense * 100) if total_expense else 100 for amt in expense_cats] or [100]
        results.expense_breakdown = ExpenseBreakdown(
            categories=categories,
            percentages=percentages,
        )

        total_income = df["Deposit (INR)"].sum()
        total_expense = df["Withdrawal (INR)"].sum()
        total_balance = total_income - total_expense
        latest_month = df["month"].max()
        prev_month = latest_month - 1
        latest_income = df[df["month"] == latest_month]["Deposit (INR)"].sum()
        latest_expense = df[df["month"] == latest_month]["Withdrawal (INR)"].sum()
        prev_income = df[df["month"] == prev_month]["Deposit (INR)"].sum()
        prev_expense = df[df["month"] == prev_month]["Withdrawal (INR)"].sum()
        prev_balance = prev_income - prev_expense
        results.account_overview = AccountOverview(
            total_balance=float(total_balance),
            monthly_income=float(latest_income),
            monthly_expense=float(latest_expense),
            balance_percentage=float(((total_balance - prev_balance) / prev_balance * 100) if prev_balance else 0),
            income_percentage=float(((latest_income - prev_income) / prev_income * 100) if prev_income else 0),
            expense_percentage=float(((latest_expense - prev_expense) / prev_expense * 100) if prev_expense else 0),
        )

        mlflow.log_metrics({
            "total_balance": total_balance,
            "monthly_income": latest_income,
            "monthly_expense": latest_expense,
            "expense_categories_count": len(categories),
            "months_analyzed": len(labels),
        })

        output_file = output_dir / "visualization_data.json"
        with open(output_file, "w") as f:
            json.dump(results.dict(), f)  # Changed to results.dict()
        logger.info(f"Visualization data saved to {output_file}")

        charts_dir = output_dir / "charts"
        if charts_dir != output_dir and charts_dir.exists():
            charts_file = charts_dir / "visualization_data.json"
            with open(charts_file, "w") as f:
                json.dump(results.dict(), f)  # Changed to results.dict()
            logger.info(f"Visualization data also saved to {charts_file}")

        logger.info(f"Visualization data generated: {(pd.Timestamp.now() - start_time).total_seconds():.3f}s")
        return results

if __name__ == "__main__":
    input_model = VisualizerInput(
        input_csv=Path("data/output/categorized.csv"),
        output_dir=Path("data/output"),
    )
    results = generate_visualizations(input_model)
    print(results.dict())
