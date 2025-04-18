"""Financial Summaries and Narrative Generation.

This module generates concise summaries and narratives based on analyzed financial data.
Key functionalities include:
- Creating textual summaries of spending habits and trends.
- Highlighting key insights and actionable recommendations.
- Formatting narratives for user-friendly presentation.
- Supporting integration with conversational interfaces.
"""
import logging
import sys
from pathlib import Path

import mlflow
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from src.utils import setup_mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_stories(input_csv: str, output_file: str) -> list[str]:
    """Generate financial summaries and narratives.
    
    Args:
        input_csv: Path to categorized transactions CSV.
        output_file: Path to save story text.
    
    Returns:
        List of story strings.

    """
    setup_mlflow()
    logger.info("Generating financial stories")

    with mlflow.start_run(run_name="Storytelling"):
        mlflow.log_param("input_csv", input_csv)
        start_time = pd.Timestamp.now()
        try:
            df = pd.read_csv(input_csv)
            logger.info(f"Read CSV: {(pd.Timestamp.now() - start_time).total_seconds():.3f}s")
        except FileNotFoundError:
            logger.exception("Input CSV not found: %s", input_csv)
            return []
        mlflow.log_metric("transactions_storied", len(df))

        t = pd.Timestamp.now()
        df["parsed_date"] = pd.to_datetime(df["parsed_date"], errors="coerce")
        df["month"] = df["parsed_date"].dt.to_period("M")

        # Aggregate by month
        monthly = df.groupby("month").agg({
            "Withdrawal (INR)": "sum",
            "Deposit (INR)": "sum",
            "category": lambda x: x.value_counts().idxmax(),  # Most frequent category
        }).reset_index()
        monthly["net"] = monthly["Deposit (INR)"] - monthly["Withdrawal (INR)"]
        logger.info(f"Aggregate: {(pd.Timestamp.now() - t).total_seconds():.3f}s")

        # Generate stories
        t = pd.Timestamp.now()
        stories = []
        for idx, row in monthly.iterrows():
            month = str(row["month"])
            withdrawals = row["Withdrawal (INR)"]
            deposits = row["Deposit (INR)"]
            top_category = row["category"]
            net = row["net"]

            # Rule-based narrative
            summary = f"In {month}, you spent ₹{withdrawals:.2f} and received ₹{deposits:.2f}"
            if net > 0:
                insight = f", saving ₹{net:.2f} with most spending on {top_category}."
            elif net < 0:
                insight = f", overspending by ₹{-net:.2f}, mainly on {top_category}."
            else:
                insight = f", balancing spending and income, primarily on {top_category}."

            stories.append(f"{month}: {summary}{insight}")

        logger.info(f"Stories: {(pd.Timestamp.now() - t).total_seconds():.3f}s")

        # Save stories
        t = pd.Timestamp.now()
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            f.write("\n".join(stories))
        mlflow.log_artifact(output_file)
        logger.info(f"Save: {(pd.Timestamp.now() - t).total_seconds():.3f}s")

        logger.info(f"Total: {(pd.Timestamp.now() - start_time).total_seconds():.3f}s")
        return stories

if __name__ == "__main__":
    input_csv = "data/output/categorized.csv"
    output_file = "data/output/stories.txt"
    stories = generate_stories(input_csv, output_file)
    print(stories)
