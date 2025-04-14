"""Visualization of Financial Data.

This module generates visualizations such as pie charts, bar charts, and line charts to represent
financial data. Key functionalities include:
- Creating pie charts for category-wise spending distribution.
- Plotting bar charts for monthly or weekly spending trends.
- Generating line charts to visualize spending over time.
- Saving or exporting visualizations for reports.
"""
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from utils import setup_mlflow

logging.basicConfig(level=logging.INFO)

def generate_visualizations(input_csv: str, output_dir: str) -> dict | None:
    """Generate pie, bar, and line charts for transactions.
    
    Args:
        input_csv: Path to categorized transactions CSV.
        output_dir: Directory to save charts.
    
    Returns:
        Dictionary with chart file paths.

    """
    setup_mlflow()
    logging.info("Generating visualizations")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = {}

    with mlflow.start_run(run_name="Visualization"):
        mlflow.log_param("input_csv", input_csv)
        df = pd.read_csv(input_csv)
        df["parsed_date"] = pd.to_datetime(df["parsed_date"])
        mlflow.log_metric("transactions_visualized", len(df))

        # Pie Chart: Expense Breakdown
        expenses = df[df["category"].str.contains("Expense")].groupby("category")["Withdrawal (INR)"].sum()
        if not expenses.empty:
            plt.figure(figsize=(8, 6))
            expenses.plot.pie(autopct="%1.1f%%", startangle=90)
            plt.title("Expense Breakdown")
            pie_file = Path(output_dir) / "expense_pie.png"
            plt.savefig(pie_file, bbox_inches="tight")
            plt.close()
            results["pie_chart"] = str(pie_file)
            mlflow.log_artifact(pie_file)

        # Bar Chart: Monthly Spending
        df["month"] = df["parsed_date"].dt.to_period("M")
        monthly = df[df["category"].str.contains("Expense")].groupby("month")["Withdrawal (INR)"].sum()
        if not monthly.empty:
            plt.figure(figsize=(10, 6))
            monthly.plot.bar()
            plt.title("Monthly Spending")
            plt.xlabel("Month")
            plt.ylabel("Amount (INR)")
            bar_file = Path(output_dir) / "monthly_bar.png"
            plt.savefig(bar_file, bbox_inches="tight")
            plt.close()
            results["bar_chart"] = str(bar_file)
            mlflow.log_artifact(bar_file)

        # Line Chart: Spending Trends
        daily = df[df["category"].str.contains("Expense")].groupby("parsed_date")["Withdrawal (INR)"].sum()
        if not daily.empty:
            plt.figure(figsize=(10, 6))
            daily.plot.line()
            plt.title("Spending Trends Over Time")
            plt.xlabel("Date")
            plt.ylabel("Amount (INR)")
            line_file = Path(output_dir) / "trends_line.png"
            plt.savefig(line_file, bbox_inches="tight")
            plt.close()
            results["line_chart"] = str(line_file)
            mlflow.log_artifact(line_file)

        logging.info("Visualizations generated")
        return results

if __name__ == "__main__":
    input_csv = "data/output/categorized.csv"
    output_dir = "data/output/charts"
    results = generate_visualizations(input_csv, output_dir)
    print(results)
