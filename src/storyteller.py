"""Financial Summaries and Narrative Generation.

This module generates a cohesive financial narrative based on analyzed financial data using an LLM.
Key functionalities include:
- Creating an engaging, comprehensive story of spending habits and trends.
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
from src.models import StorytellerInput, StorytellerOutput
from src.utils import get_llm_config, setup_mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_stories(input_model: StorytellerInput) -> StorytellerOutput:
    """Generate a single financial narrative using an LLM.

    Args:
        input_model: StorytellerInput with path to categorized transactions CSV and output file.

    Returns:
        StorytellerOutput with a single story string.
    """
    setup_mlflow()
    logger.info("Generating financial story")

    input_csv = input_model.input_csv
    output_file = input_model.output_file

    with mlflow.start_run(run_name="Storytelling"):
        mlflow.log_param("input_csv", str(input_csv))
        start_time = pd.Timestamp.now()
        try:
            df = pd.read_csv(input_csv)
            logger.info(f"Read CSV: {(pd.Timestamp.now() - start_time).total_seconds():.3f}s")
        except FileNotFoundError:
            logger.exception("Input CSV not found: %s", input_csv)
            mlflow.log_param("error", f"Input CSV not found: {input_csv}")
            return StorytellerOutput(stories=[])

        if df.empty:
            logger.warning(f"Empty CSV: {input_csv}")
            mlflow.log_param("warning", "Empty CSV")
            return StorytellerOutput(stories=[])

        mlflow.log_metric("transactions_storied", len(df))

        t = pd.Timestamp.now()
        df["parsed_date"] = pd.to_datetime(df["parsed_date"], errors="coerce")
        df["month"] = df["parsed_date"].dt.to_period("M")

        # Aggregate financial data
        total_withdrawals = df["Withdrawal (INR)"].sum()
        total_deposits = df["Deposit (INR)"].sum()
        net_balance = total_deposits - total_withdrawals
        monthly_agg = df.groupby("month").agg({
            "Withdrawal (INR)": "sum",
            "Deposit (INR)": "sum",
            "category": lambda x: x.value_counts().idxmax(),
        }).reset_index()
        monthly_agg["net"] = monthly_agg["Deposit (INR)"] - monthly_agg["Withdrawal (INR)"]
        top_category = df["category"].value_counts().idxmax()
        overspending_months = len(monthly_agg[monthly_agg["net"] < 0])
        saving_months = len(monthly_agg[monthly_agg["net"] > 0])
        
        # Get sample transactions (up to 5 across the period)
        sample_transactions = df[["Narration", "Withdrawal (INR)", "Deposit (INR)", "category", "month"]].head(5).to_dict(orient="records")
        sample_text = "\n".join(
            f"- {t['month']}: {t['Narration']}: ₹{t['Withdrawal (INR)'] or t['Deposit (INR)']} ({t['category']})"
            for t in sample_transactions
        ) if sample_transactions else "No specific transactions available."

        logger.info(f"Aggregate: {(pd.Timestamp.now() - t).total_seconds():.3f}s")

        # Initialize LLM
        llm_config = get_llm_config()
        try:
            client = ollama.Client(host=llm_config.api_endpoint)
            mlflow.log_param("llm_model", llm_config.model_name)
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            mlflow.log_param("llm_error", str(e))
            return StorytellerOutput(stories=[])

        # LLM prompt for a single comprehensive story
        t = pd.Timestamp.now()
        prompt = f"""
You are a financial advisor crafting an engaging, cohesive, and actionable financial story about a user's financial activity from {str(monthly_agg['month'].min())} to {str(monthly_agg['month'].max())}. Based on the following data, create a narrative (300-500 words) that summarizes their spending and income trends, highlights key patterns (e.g., overspending vs. saving months, dominant categories), and offers two actionable recommendations. Make the tone friendly, professional, and motivating.

- Total Spending: ₹{total_withdrawals:.2f}
- Total Income: ₹{total_deposits:.2f}
- Net Balance: ₹{net_balance:.2f} ({'savings' if net_balance > 0 else 'overspending'})
- Most Frequent Category: {top_category}
- Number of Overspending Months: {overspending_months}
- Number of Saving Months: {saving_months}
- Sample Transactions:
{sample_text}

Example Output:
"From January to December 2016, your financial journey was a mix of highs and lows! You spent ₹1,200,000 and earned ₹1,250,000, netting a solid ₹50,000 in savings. Shopping dominated your expenses, with frequent retail purchases. You overspent in 6 months, particularly in September due to large POS transactions, but saved significantly in April and October. To stay on track, consider setting a monthly budget for discretionary spending and automate savings to build a stronger financial cushion."
"""
        try:
            response = client.generate(
                model=llm_config.model_name,
                prompt=prompt,
            )
            story = response.get("response", "").strip()
            if not story:
                raise ValueError("Empty LLM response")
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            mlflow.log_param("llm_error", str(e))
            return StorytellerOutput(stories=[])

        logger.info(f"Story generation: {(pd.Timestamp.now() - t).total_seconds():.3f}s")

        # Save story
        t = pd.Timestamp.now()
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            f.write(story)
        mlflow.log_artifact(output_file)
        logger.info(f"Save: {(pd.Timestamp.now() - t).total_seconds():.3f}s")

        logger.info(f"Total: {(pd.Timestamp.now() - start_time).total_seconds():.3f}s")
        return StorytellerOutput(stories=[story])

if __name__ == "__main__":
    input_model = StorytellerInput(
        input_csv=Path("data/output/categorized.csv"),
        output_file=Path("data/output/stories.txt")
    )
    output = generate_stories(input_model)
    print(output.stories)