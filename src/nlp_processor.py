import logging
import sys
from pathlib import Path

import mlflow
import pandas as pd

from utils import get_llm_config, setup_mlflow

sys.path.append(str(Path(__file__).parent.parent))
from llm_setup.ollama_manager import query_llm

logging.basicConfig(level=logging.INFO)

def process_nlp_queries(input_csv: str, query: str, output_file: str) -> str | None:
    """Process NLP queries (search, memory, conversational).
    
    Args:
        input_csv: Path to categorized transactions CSV.
        query: User query (e.g., "Restaurant spending last month").
        output_file: Path to save query response.
    
    Returns:
        Query response or None if failed.

    """
    setup_mlflow()
    llm_config = get_llm_config()
    logging.info(f"Processing NLP query: {query}")

    with mlflow.start_run(run_name="NLP_Query"):
        mlflow.log_param("input_csv", input_csv)
        mlflow.log_param("query", query)

        df = pd.read_csv(input_csv)
        df["parsed_date"] = pd.to_datetime(df["parsed_date"])

        # Simple keyword search
        if "search" in query.lower():
            keywords = query.lower().split()[1:]  # e.g., "search restaurant" -> ["restaurant"]
            matches = df[df["Narration"].str.lower().str.contains("|".join(keywords), na=False)]
            if not matches.empty:
                response = matches[["parsed_date", "Narration", "Withdrawal (INR)", "Deposit (INR)", "category"]].to_string()
            else:
                response = "No matching transactions found."

        # LLM-based conversational query
        else:
            last_month = df["parsed_date"].max() - pd.offsets.MonthBegin(1)
            recent_df = df[df["parsed_date"] >= last_month]
            data_summary = recent_df[["Narration", "Withdrawal (INR)", "Deposit (INR)", "category"]].to_dict()

            prompt = (
                f"Answer this financial query based on recent transactions: '{query}'. "
                f"Data: {data_summary}. "
                "Return a concise answer (1-2 sentences) with no extra formatting."
            )
            try:
                response = query_llm(prompt, llm_config).strip()
            except Exception as e:
                logging.exception(f"LLM query failed: {e}")
                response = "Unable to process query."

        # Save response
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            f.write(response)
        mlflow.log_artifact(output_file)

        logging.info("Query processed")
        return response

if __name__ == "__main__":
    input_csv = "data/output/categorized.csv"
    query = "Restaurant spending last month"
    output_file = "data/output/nlp_response.txt"
    response = process_nlp_queries(input_csv, query, output_file)
    print(response)
