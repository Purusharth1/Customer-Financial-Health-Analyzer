"""Financial Summaries and Narrative Generation.

This module generates a cohesive financial narrative
based on analyzed financial data using an LLM.
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
import yaml

sys.path.append(str(Path(__file__).parent.parent))
from src.models import StorytellerInput, StorytellerOutput
from src.utils import get_llm_config, setup_mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _raise_empty_llm_response_error() -> None:
    """Raise ValueError for empty LLM response."""
    msg = "Empty LLM response"
    raise ValueError(msg)

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

def _aggregate_financial_data(transactions_df: pd.DataFrame) -> dict[str, any]:
    """Aggregate financial data for storytelling."""
    start_time = pd.Timestamp.now()
    transactions_df["parsed_date"] = pd.to_datetime(
        transactions_df["parsed_date"], errors="coerce",
    )
    transactions_df["month"] = transactions_df["parsed_date"].dt.to_period("M")

    total_withdrawals = transactions_df["Withdrawal (INR)"].sum()
    total_deposits = transactions_df["Deposit (INR)"].sum()
    monthly_agg = transactions_df.groupby("month").agg({
        "Withdrawal (INR)": "sum",
        "Deposit (INR)": "sum",
        "category": lambda x: x.value_counts().idxmax(),
    }).reset_index()
    monthly_agg["net"] = (
        monthly_agg["Deposit (INR)"] - monthly_agg["Withdrawal (INR)"]
    )
    sample_transactions = (
        transactions_df[
            ["Narration", "Withdrawal (INR)", "Deposit (INR)", "category", "month"]
        ]
        .head(5)
        .to_dict(orient="records")
    )
    sample_text = (
        "\n".join(
            "- {}: {}: ₹{:.2f} ({})".format(
                t["month"],
                t["Narration"],
                t["Withdrawal (INR)"] or t["Deposit (INR)"],
                t["category"],
            )
            for t in sample_transactions
        )
        if sample_transactions
        else "No specific transactions available."
    )

    logger.info(
        "Aggregate: %.3f s",
        (pd.Timestamp.now() - start_time).total_seconds(),
    )
    return {
        "total_withdrawals": total_withdrawals,
        "total_deposits": total_deposits,
        "net_balance": total_deposits - total_withdrawals,
        "monthly_agg": monthly_agg,
        "top_category": transactions_df["category"].value_counts().idxmax(),
        "overspending_months": len(monthly_agg[monthly_agg["net"] < 0]),
        "saving_months": len(monthly_agg[monthly_agg["net"] > 0]),
        "sample_text": sample_text,
    }

def _load_config() -> dict[str, any]:
    """Load configuration from config.yaml."""
    config_path = Path("config/config.yaml")
    try:
        with config_path.open("r") as f:
            return yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError):
        logger.exception("Failed to load config")
        raise

def _generate_llm_story(data: dict[str, any], llm_config: dict) -> str | None:
    """Generate financial story using LLM."""
    generation_time = pd.Timestamp.now()
    try:
        client = ollama.Client(host=llm_config.api_endpoint)
        mlflow.log_param("llm_model", llm_config.model_name)
    except (ConnectionError, ValueError) as e:
        logger.exception("Failed to initialize LLM")
        mlflow.log_param("llm_error", str(e))
        return None

    config = _load_config()
    prompt_template = config["storyteller"]["prompt"]
    prompt = prompt_template.format(
        start_month=data["monthly_agg"]["month"].min(),
        end_month=data["monthly_agg"]["month"].max(),
        total_withdrawals=data["total_withdrawals"],
        total_deposits=data["total_deposits"],
        net_balance=data["net_balance"],
        balance_status="savings" if data["net_balance"] > 0 else "overspending",
        top_category=data["top_category"],
        overspending_months=data["overspending_months"],
        saving_months=data["saving_months"],
        sample_text=data["sample_text"],
    )

    try:
        response = client.generate(model=llm_config.model_name, prompt=prompt)
        story = response.get("response", "").strip()
        if not story:
            _raise_empty_llm_response_error()
        else:
            logger.info(
                "Story generation: %.3f s",
                (pd.Timestamp.now() - generation_time).total_seconds(),
            )
            return story
    except (ConnectionError, ValueError) as e:
        logger.exception("LLM generation failed")
        mlflow.log_param("llm_error", str(e))
        return None

def generate_stories(input_model: StorytellerInput) -> StorytellerOutput:
    """Generate a single financial narrative using an LLM."""
    setup_mlflow()
    logger.info("Generating financial story")

    with mlflow.start_run(run_name="Storytelling"):
        mlflow.log_param("input_csv", str(input_model.input_csv))
        start_time = pd.Timestamp.now()

        transactions_df = _read_transactions(input_model.input_csv)
        if transactions_df is None:
            return StorytellerOutput(stories=[])

        mlflow.log_metric("transactions_storied", len(transactions_df))
        data = _aggregate_financial_data(transactions_df)
        llm_config = get_llm_config()
        story = _generate_llm_story(data, llm_config)
        if story is None:
            return StorytellerOutput(stories=[])

        save_time = pd.Timestamp.now()
        output_file = input_model.output_file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w") as f:
            f.write(story)
        mlflow.log_artifact(str(output_file))
        logger.info(
            "Save: %.3f s",
            (pd.Timestamp.now() - save_time).total_seconds(),
        )

        logger.info(
            "Total: %.3f s",
            (pd.Timestamp.now() - start_time).total_seconds(),
        )
        return StorytellerOutput(stories=[story])

if __name__ == "__main__":
    input_model = StorytellerInput(
        input_csv=Path("data/output/categorized.csv"),
        output_file=Path("data/output/stories.txt"),
    )
    output = generate_stories(input_model)
    logger.info("Generated stories: %s", output.stories)
