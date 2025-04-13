from typing import List
import logging
from pathlib import Path
import pandas as pd
import mlflow
from utils import setup_mlflow, get_llm_config
import sys
sys.path.append(str(Path(__file__).parent.parent))
from llm_setup.ollama_manager import query_llm
from llm_setup.config import LLMConfig

logging.basicConfig(level=logging.INFO)

def generate_stories(input_csv: str, output_file: str) -> List[str]:
    """
    Generate financial summaries and narratives.
    
    Args:
        input_csv: Path to categorized transactions CSV.
        output_file: Path to save story text.
    
    Returns:
        List of story strings.
    """
    setup_mlflow()
    llm_config = get_llm_config()
    logging.info("Generating financial stories")
    
    with mlflow.start_run(run_name="Storytelling"):
        mlflow.log_param("input_csv", input_csv)
        df = pd.read_csv(input_csv)
        df["parsed_date"] = pd.to_datetime(df["parsed_date"])
        mlflow.log_metric("transactions_storied", len(df))
        
        # Group by month
        df["month"] = df["parsed_date"].dt.to_period("M")
        monthly = df.groupby("month").agg({
            "Withdrawal (INR)": "sum",
            "Deposit (INR)": "sum",
            "category": lambda x: x.value_counts().to_dict()
        }).reset_index()
        
        stories = []
        for _, row in monthly.iterrows():
            month = str(row["month"])
            withdrawals = row["Withdrawal (INR)"]
            deposits = row["Deposit (INR)"]
            categories = row["category"]
            
            prompt = (
                f"Summarize this month's financial activity for {month}: "
                f"Total spent: {withdrawals:.2f} INR, Total received: {deposits:.2f} INR, "
                f"Category breakdown: {categories}. "
                "Return a concise summary (1-2 sentences) with no extra formatting."
            )
            try:
                story = query_llm(prompt, llm_config).strip()
                stories.append(f"{month}: {story}")
            except Exception as e:
                logging.error(f"LLM storytelling failed for {month}: {e}")
                stories.append(f"{month}: Spent {withdrawals:.2f} INR, received {deposits:.2f} INR.")
        
        # Save stories
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            f.write("\n".join(stories))
        mlflow.log_artifact(output_file)
        
        logging.info("Stories generated")
        return stories

if __name__ == "__main__":
    input_csv = "data/output/categorized.csv"
    output_file = "data/output/stories.txt"
    stories = generate_stories(input_csv, output_file)
    print(stories)