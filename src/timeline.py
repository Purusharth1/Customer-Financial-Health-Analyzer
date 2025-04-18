import sys
import logging
from pathlib import Path
from typing import List
import mlflow
import pandas as pd
from dateutil.parser import parse

sys.path.append(str(Path(__file__).parent.parent))
from src.models import TimelineInput, TimelineOutput, Transaction, CategorizedTransaction
from src.utils import setup_mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_timeline(input_model: TimelineInput) -> TimelineOutput:
    """Build a chronological timeline from transactions CSV."""
    setup_mlflow()
    
    with mlflow.start_run(run_name="Timeline_Construction"):
        mlflow.log_param("input_csv", str(input_model.transactions_csv))
        
        try:
            df = pd.read_csv(input_model.transactions_csv)
            logger.info(f"Loaded CSV with columns: {list(df.columns)}")
        except FileNotFoundError:
            logger.error("Input CSV not found: %s", input_model.transactions_csv)
            mlflow.log_param("error", "Input CSV not found")
            return TimelineOutput(transactions=[])
            
        if df.empty:
            logger.warning("No transactions in CSV")
            mlflow.log_param("warning", "No transactions in CSV")
            return TimelineOutput(transactions=[])
        
        # Check if we have duplicate representations of the same columns
        # This handles both original and alias columns being present
        column_mapping = {
            "Reference_Number": "Reference Number", 
            "Value_Date": "Value Date",
            "Withdrawal_INR": "Withdrawal (INR)", 
            "Deposit_INR": "Deposit (INR)",
            "Closing_Balance_INR": "Closing Balance (INR)"
        }
        
        # Keep only one version of each column
        for orig_col, alias_col in column_mapping.items():
            if orig_col in df.columns and alias_col in df.columns:
                # Prioritize the non-empty values from either column
                if not df[orig_col].isna().all():
                    df[alias_col] = df[orig_col]
                df = df.drop(columns=[orig_col])
            elif orig_col in df.columns:
                # Rename to the alias format expected by the model
                df = df.rename(columns={orig_col: alias_col})
                
        # Define expected columns based on Transaction model aliases
        expected_columns = [
            "Date",
            "Narration",
            "Reference Number",
            "Value Date",
            "Withdrawal (INR)",
            "Deposit (INR)",
            "Closing Balance (INR)",
            "Source_File"
        ]
        
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            logger.error("Missing columns in CSV: %s", missing_columns)
            mlflow.log_param("error", f"Missing columns: {missing_columns}")
            return TimelineOutput(transactions=[])
            
        # Replace NaN with empty strings for optional string fields
        string_columns = ["Reference Number", "Value Date", "Source_File"]
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].fillna("")
                
        # Replace NaN with 0.0 for numeric fields
        numeric_columns = ["Withdrawal (INR)", "Deposit (INR)", "Closing Balance (INR)"]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)
                
        def parse_date(date_str: str) -> str | None:
            try:
                return parse(date_str, dayfirst=True).strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                return None
                
        df["parsed_date"] = df["Date"].apply(parse_date)
        invalid_dates = df["parsed_date"].isna().sum()
        mlflow.log_metric("invalid_dates_dropped", invalid_dates)
        
        df = df.dropna(subset=["parsed_date"])
        if df.empty:
            logger.warning("No valid dates after parsing")
            mlflow.log_param("warning", "No valid dates after parsing")
            return TimelineOutput(transactions=[])
            
        df = df.sort_values("parsed_date")
        
        # Add only the category field to the DataFrame
        df["category"] = "Uncategorized"
        
        # Save processed data - use only the required columns to avoid duplicates
        output_columns = [
            "Date", 
            "Narration", 
            "Reference Number", 
            "Value Date", 
            "Withdrawal (INR)", 
            "Deposit (INR)", 
            "Closing Balance (INR)", 
            "Source_File",
            "parsed_date",
            "category"
        ]
        
        # Make sure we only keep columns that actually exist
        output_columns = [col for col in output_columns if col in df.columns]
        df_output = df[output_columns]
        df_output.to_csv(input_model.output_csv, index=False)
        mlflow.log_artifact(str(input_model.output_csv))
        
        # Convert DataFrame to CategorizedTransaction objects
        categorized_transactions: List[CategorizedTransaction] = []
        for _, row in df.iterrows():
            try:
                # Create a clean dictionary with the correct field names for CategorizedTransaction
                transaction_dict = {
                    "Date": row.get("Date", ""),
                    "Narration": row.get("Narration", ""),
                    "Reference_Number": row.get("Reference Number", ""),
                    "Value_Date": row.get("Value Date", ""),
                    "Withdrawal_INR": float(row.get("Withdrawal (INR)", 0.0)),
                    "Deposit_INR": float(row.get("Deposit (INR)", 0.0)),
                    "Closing_Balance_INR": float(row.get("Closing Balance (INR)", 0.0)),
                    "Source_File": row.get("Source_File", ""),
                    "parsed_date": row.get("parsed_date", ""),
                    "category": row.get("category", "Uncategorized")
                }
                
                # Create CategorizedTransaction
                categorized_transaction = CategorizedTransaction(**transaction_dict)
                categorized_transactions.append(categorized_transaction)
            except Exception as e:
                logger.warning(f"Failed to validate row: {row.to_dict()}, error: {e}")
                continue
        
        mlflow.log_metric("transactions_timed", len(categorized_transactions))
        return TimelineOutput(transactions=categorized_transactions)

if __name__ == "__main__":
    input_model = TimelineInput(
        transactions_csv=Path("data/output/all_transactions.csv"),
        output_csv=Path("data/output/timeline.csv")
    )
    output = build_timeline(input_model)
    if output.transactions:
        print(pd.DataFrame([{
            "Date": t.Date,
            "Narration": t.Narration,
            "parsed_date": t.parsed_date,
            "category": t.category
        } for t in output.transactions]).head())