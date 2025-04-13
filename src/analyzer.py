from typing import List, Dict, Optional
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import mlflow
from utils import setup_mlflow, get_llm_config, sanitize_metric_name
import sys
sys.path.append(str(Path(__file__).parent.parent))
from llm_setup.ollama_manager import query_llm
from llm_setup.config import LLMConfig

logging.basicConfig(level=logging.INFO)

def analyze_transactions(input_csv: str, output_dir: str) -> Dict:
    """
    Analyze transactions for patterns, fees, recurring payments, and anomalies.
    
    Args:
        input_csv: Path to categorized transactions CSV.
        output_dir: Directory to save analysis outputs.
    
    Returns:
        Dictionary with analysis results.
    """
    setup_mlflow()
    llm_config = get_llm_config()
    logging.info("Starting transaction analysis")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = {
        "patterns": [],
        "fees": [],
        "recurring": [],
        "anomalies": []
    }
    
    with mlflow.start_run(run_name="Transaction_Analysis"):
        mlflow.log_param("input_csv", input_csv)
        df = pd.read_csv(input_csv)
        mlflow.log_metric("transactions_analyzed", len(df))
        
        # Spending Patterns
        patterns = detect_patterns(df, llm_config)
        results["patterns"] = patterns
        patterns_file = Path(output_dir) / "patterns.txt"
        with open(patterns_file, "w") as f:
            f.write("\n".join(patterns))
        mlflow.log_artifact(patterns_file)
        
        # Fees and Interest
        fees = detect_fees(df)
        results["fees"] = fees
        fees_file = Path(output_dir) / "fees.csv"
        pd.DataFrame(fees).to_csv(fees_file, index=False)
        mlflow.log_artifact(fees_file)
        
        # Recurring Payments
        recurring = detect_recurring(df)
        results["recurring"] = recurring
        recurring_file = Path(output_dir) / "recurring.csv"
        pd.DataFrame(recurring).to_csv(recurring_file, index=False)
        mlflow.log_artifact(recurring_file)
        
        # Anomalies
        anomalies = detect_anomalies(df)
        results["anomalies"] = anomalies
        anomalies_file = Path(output_dir) / "anomalies.csv"
        pd.DataFrame(anomalies).to_csv(anomalies_file, index=False)
        mlflow.log_artifact(anomalies_file)
        
        # Log counts
        for key, items in results.items():
            mlflow.log_metric(sanitize_metric_name(f"{key}_count"), len(items))
        
        logging.info("Analysis complete")
        return results

def detect_patterns(df: pd.DataFrame, llm_config: LLMConfig) -> List[str]:
    """Use LLM to identify spending patterns."""
    categories = df.groupby("category")["Withdrawal (INR)"].sum().to_dict()
    prompt = (
        f"Analyze this spending data: {categories}. "
        "Identify patterns, e.g., high spending on weekends or specific categories. "
        "Return a list of short insights, one per line, no explanations."
    )
    try:
        response = query_llm(prompt, llm_config).strip()
        return response.split("\n")
    except Exception as e:
        logging.error(f"LLM pattern detection failed: {e}")
        return ["No patterns detected"]

def detect_fees(df: pd.DataFrame) -> List[Dict]:
    """Identify transactions with fees or interest."""
    keywords = ["FEE", "CHARGE", "INTEREST", "PENALTY"]
    fees = []
    for _, row in df.iterrows():
        if any(kw in str(row["Narration"]).upper() for kw in keywords):
            fees.append({
                "date": row["parsed_date"],
                "narration": row["Narration"],
                "amount": row["Withdrawal (INR)"]
            })
    return fees

def detect_recurring(df: pd.DataFrame) -> List[Dict]:
    """Detect recurring payments based on periodicity."""
    df["parsed_date"] = pd.to_datetime(df["parsed_date"])
    grouped = df.groupby(["Narration", "Withdrawal (INR)"]).agg(
        dates=("parsed_date", list),
        count=("parsed_date", "count")
    )
    recurring = []
    for (narration, amount), row in grouped.iterrows():
        if row["count"] > 1:
            dates = pd.to_datetime(row["dates"])
            deltas = np.diff(dates).astype("timedelta64[D]").astype(int)
            if len(deltas) > 0 and np.std(deltas) < 5:  # Consistent intervals
                recurring.append({
                    "narration": narration,
                    "amount": amount,
                    "frequency": "monthly" if np.mean(deltas) > 25 else "weekly"
                })
    return recurring

def detect_anomalies(df: pd.DataFrame) -> List[Dict]:
    """Flag unusual transactions based on amount."""
    threshold = df["Withdrawal (INR)"].mean() + 3 * df["Withdrawal (INR)"].std()
    anomalies = []
    for _, row in df.iterrows():
        if row["Withdrawal (INR)"] > threshold:
            anomalies.append({
                "date": row["parsed_date"],
                "narration": row["Narration"],
                "amount": row["Withdrawal (INR)"]
            })
    return anomalies

if __name__ == "__main__":
    input_csv = "data/output/categorized.csv"
    output_dir = "data/output/analysis"
    results = analyze_transactions(input_csv, output_dir)
    print(results)