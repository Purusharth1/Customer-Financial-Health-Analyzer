"""Financial Summaries and Narrative Generation.

This module generates concise summaries and narratives based on analyzed financial data using an LLM.
Key functionalities include:
- Creating engaging textual summaries of spending habits and trends.
- Highlighting key insights and actionable recommendations.
- Formatting narratives for user-friendly presentation.
- Supporting integration with conversational interfaces.
"""

import logging
import sys
from pathlib import Path

import mlflow
import ollama
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from src.utils import get_llm_config, setup_mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_stories(input_csv: str, output_file: str) -> list[str]:
    """Generate financial summaries and narratives using an LLM.

    Args:
    ----
        input_csv: Path to categorized transactions CSV.
        output_file: Path to save story text.

    Returns:
    -------
        List of story strings.

    """
    setup_mlflow()
    logger.info("Generating financial stories")

    with mlflow.start_run(run_name="Storytelling"):
        mlflow.log_param("input_csv", input_csv)
        start_time = pd.Timestamp.now()
        try:
            df = pd.read_csv(input_csv)
            logger.info(
                f"Read CSV: {(pd.Timestamp.now() - start_time).total_seconds():.3f}s",
            )
        except FileNotFoundError:
            logger.exception("Input CSV not found: %s", input_csv)
            mlflow.log_param("error", f"Input CSV not found: {input_csv}")
            return []
        mlflow.log_metric("transactions_storied", len(df))

        t = pd.Timestamp.now()
        df["parsed_date"] = pd.to_datetime(df["parsed_date"], errors="coerce")
        df["month"] = df["parsed_date"].dt.to_period("M")

        # Aggregate by month
        monthly = (
            df.groupby("month")
            .agg(
                {
                    "Withdrawal (INR)": "sum",
                    "Deposit (INR)": "sum",
                    "category": lambda x: x.value_counts().idxmax(),  # Most frequent category
                },
            )
            .reset_index()
        )
        monthly["net"] = monthly["Deposit (INR)"] - monthly["Withdrawal (INR)"]
        logger.info(f"Aggregate: {(pd.Timestamp.now() - t).total_seconds():.3f}s")

        # Initialize LLM
        llm_config = get_llm_config()
        try:
            client = ollama.Client(host=llm_config.api_endpoint)
            mlflow.log_param("llm_model", llm_config.model_name)
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            mlflow.log_param("llm_error", str(e))
            client = None

        # Generate stories
        t = pd.Timestamp.now()
        stories = []
        for idx, row in monthly.iterrows():
            month = str(row["month"])
            withdrawals = row["Withdrawal (INR)"]
            deposits = row["Deposit (INR)"]
            top_category = row["category"]
            net = row["net"]

            # Get sample transactions for this month
            month_df = df[df["month"] == row["month"]][
                ["Narration", "Withdrawal (INR)", "Deposit (INR)", "category"]
            ]
            sample_transactions = month_df.head(3).to_dict(orient="records")
            sample_text = (
                "\n".join(
                    f"- {t['Narration']}: ₹{t['Withdrawal (INR)'] or t['Deposit (INR)']} ({t['category']})"
                    for t in sample_transactions
                )
                if sample_transactions
                else "No specific transactions available."
            )

            # LLM prompt
            prompt = f"""
You are a financial advisor crafting an engaging, concise, and actionable story about a user's financial activity for {month}. Based on the following data, create a narrative (100-150 words) that summarizes their spending and income, highlights key trends, and offers one actionable recommendation. Make the tone friendly, professional, and motivating.

- Total Spending: ₹{withdrawals:.2f}
- Total Income: ₹{deposits:.2f}
- Net Balance: ₹{net:.2f} ({'savings' if net > 0 else 'overspending'})
- Most Frequent Category: {top_category}
- Sample Transactions:
{sample_text}

Example Output:
"In January 2016, your financial story shows a busy month! You spent ₹99,012, mainly on shopping and miscellaneous expenses, while earning ₹98,014. This led to a slight overspend of ₹997. Frequent POS transactions suggest regular retail purchases. To balance your budget, consider setting a monthly shopping limit to boost savings."
"""
            try:
                if client:
                    response = client.generate(
                        model=llm_config.model_name,
                        prompt=prompt,
                    )
                    story = response.get("response", "").strip()
                    if not story:
                        raise ValueError("Empty LLM response")
                else:
                    # Fallback to rule-based narrative if LLM fails
                    summary = f"In {month}, you spent ₹{withdrawals:.2f} and received ₹{deposits:.2f}"
                    insight = (
                        f", saving ₹{net:.2f} with most spending on {top_category}."
                        if net > 0
                        else f", overspending by ₹{-net:.2f}, mainly on {top_category}."
                        if net < 0
                        else f", balancing spending and income, primarily on {top_category}."
                    )
                    story = f"{month}: {summary}{insight}"
            except Exception as e:
                logger.warning(f"LLM failed for {month}: {e}")
                mlflow.log_param(f"llm_error_{month}", str(e))
                # Fallback to rule-based narrative
                summary = f"In {month}, you spent ₹{withdrawals:.2f} and received ₹{deposits:.2f}"
                insight = (
                    f", saving ₹{net:.2f} with most spending on {top_category}."
                    if net > 0
                    else f", overspending by ₹{-net:.2f}, mainly on {top_category}."
                    if net < 0
                    else f", balancing spending and income, primarily on {top_category}."
                )
                story = f"{month}: {summary}{insight}"

            stories.append(story)
            mlflow.log_metric(f"story_length_{month}", len(story))

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
