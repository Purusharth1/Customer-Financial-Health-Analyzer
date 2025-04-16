"""
Natural Language Processing for Financial Health Analyzer.

Handles:
- Search across transactions.
- Financial memory for purchases/events.
- Conversational queries and summaries with visualizations.
"""
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union

import mlflow
import pandas as pd
import sys
sys.path.append(str(Path(__file__).parent.parent))
from llm_setup.ollama_manager import query_llm
from src.utils import get_llm_config, setup_mlflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 30

class FinancialMemory:
    """Manages query history and context."""

    def __init__(self, persist_path: str = "data/state/financial_memory.json"):
        """
        Initialize memory.

        Args:
            persist_path: Path to save/load memory.
        """
        self.queries = []
        self.context = {}
        self.persist_path = persist_path
        if Path(persist_path).exists():
            self._load()

    def add_query(self, query: str, result: str) -> None:
        """Add query and result."""
        self.queries.append({"query": query, "result": result, "timestamp": datetime.now().isoformat()})
        self.queries = self.queries[-10:]  # Keep last 10
        self._save()

    def add_context(self, key: str, value: str) -> None:
        """Add context (e.g., laptop_purchase)."""
        self.context[key] = value
        self._save()

    def get_context(self) -> str:
        """Format recent queries and context."""
        context = "Recent queries:\n"
        for q in self.queries[-3:]:
            context += f"- Q: {q['query']}\n  A: {q['result']}\n"
        if self.context:
            context += "Known info:\n"
            for k, v in self.context.items():
                context += f"- {k}: {v}\n"
        return context

    def _save(self) -> None:
        """Save memory to disk."""
        try:
            state = {"queries": self.queries, "context": self.context}
            Path(self.persist_path).parent.mkdir(parents=True, exist_ok=True)
            with Path(self.persist_path).open("w") as f:
                json.dump(state, f)
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")

    def _load(self) -> None:
        """Load memory from disk."""
        try:
            with Path(self.persist_path).open() as f:
                state = json.load(f)
            self.queries = state.get("queries", [])
            self.context = state.get("context", {})
        except Exception as e:
            logger.error(f"Failed to load memory: {e}")

class QueryProcessor:
    """Processes search, memory, and conversational queries."""

    def __init__(self, df: pd.DataFrame, llm_config: dict):
        """
        Initialize with transaction data.

        Args:
            df: DataFrame with transactions.
            llm_config: LLM configuration.
        """
        self.df = df.copy() if not df.empty else df
        self.llm_config = llm_config
        self.memory = FinancialMemory()
        self.df["parsed_date"] = pd.to_datetime(self.df["parsed_date"], errors="coerce")
        if df.empty:
            logger.warning("Transaction DataFrame is empty")

    def _filter_by_time(self, query: str) -> pd.DataFrame:
        """Filter transactions by time period."""
        query = query.lower()
        now = datetime.now()
        try:
            if "2016" in query:
                logger.info("Filtering for year: 2016")
                start = datetime(2016, 1, 1)
                end = datetime(2016, 12, 31)
            elif "last month" in query:
                start = (now.replace(day=1) - pd.offsets.MonthBegin(1)).replace(day=1)
                end = now.replace(day=1) - pd.offsets.Day(1)
            elif "this month" in query:
                start = now.replace(day=1)
                end = now
            elif "last year" in query:
                start = datetime(now.year - 1, 1, 1)
                end = datetime(now.year - 1, 12, 31)
            else:
                start = now - pd.offsets.MonthBegin(3)
                end = now
            filtered = self.df[(self.df["parsed_date"] >= start) & (self.df["parsed_date"] <= end)]
            logger.info(f"Filtered {len(filtered)} transactions for {start} to {end}")
            return filtered
        except Exception as e:
            logger.error(f"Time filter error: {e}")
            return self.df

    def search(self, query: str) -> pd.DataFrame:
        """Search transactions by keywords."""
        keywords = [word for word in query.lower().split() if len(word) > 2]
        if not keywords:
            logger.debug("No valid keywords")
            return pd.DataFrame()
        mask = pd.Series(False, index=self.df.index)
        for col in ["Narration", "category"]:
            for kw in keywords:
                mask |= self.df[col].str.lower().str.contains(re.escape(kw), na=False)
        matches = self.df[mask]
        logger.info(f"Found {len(matches)} matches for query: {query}")
        return matches

    def process_query(self, query: str) -> Dict[str, Optional[Union[str, Dict]]]:
        """Handle all query types."""
        query_lower = query.lower()

        # Search query
        if any(term in query_lower for term in ["search", "find", "show me"]):
            matches = self.search(query)
            if matches.empty:
                response = "No matching transactions found."
                logger.debug(f"No matches for search query: {query}")
            else:
                transactions = matches[["parsed_date", "Narration", "Withdrawal (INR)", "category"]].head(5).to_dict("records")
                response = f"Found {len(matches)} transaction{'s' if len(matches) > 1 else ''}: {json.dumps(transactions, default=str)}"
            self.memory.add_query(query, response)
            return {"text_response": response}

        # Financial memory query
        if any(term in query_lower for term in ["how much did i", "when did i", "did i buy"]):
            matches = self.search(query)
            context = self.memory.get_context()
            if "how much" in query_lower:
                item = query_lower.split("on")[-1].strip() if "on" in query_lower else query_lower.split()[-1]
                if not matches.empty:
                    total = matches["Withdrawal (INR)"].sum()
                    response = f"You spent ₹{total:.2f} on {item}."
                    self.memory.add_context(f"{item}_purchase", response)
                else:
                    response = f"No record of {item} purchase."
                    logger.debug(f"No purchase found for {item}")
            elif "when did i" in query_lower:
                place = query_lower.split("to")[-1].strip() if "to" in query_lower else query_lower.split()[-1]
                if not matches.empty:
                    date = matches["parsed_date"].min().strftime("%Y-%m-%d")
                    response = f"You went to {place} around {date}."
                    self.memory.add_context(f"{place}_visit", response)
                else:
                    response = f"No record of visiting {place}."
                    logger.debug(f"No visit found for {place}")
            else:
                prompt = f"Answer based on context:\n{context}\nQuery: {query}\nKeep it short."
                response = query_llm(prompt, self.llm_config, timeout=TIMEOUT_SECONDS).strip()
            self.memory.add_query(query, response)
            return {"text_response": response}

        # Conversational/summary query
        time_df = self._filter_by_time(query)
        if time_df.empty:
            response = "No transactions found for this period."
            logger.warning(f"Empty time-filtered data for query: {query}")
            self.memory.add_query(query, response)
            return {"text_response": response}

        # Extract category (e.g., "Expense (Debt Payment)")
        category = None
        possible_categories = time_df["category"].unique()
        for cat in possible_categories:
            if cat.lower() in query_lower:
                category = cat
                break
        if not category and "expense" in query_lower:
            # Try to match full category after "expense"
            expense_part = query_lower.split("expense")[-1].strip().split("in")[0].strip()
            for cat in possible_categories:
                if expense_part in cat.lower():
                    category = cat
                    break
        if not category:
            # Fallback: extract last meaningful word
            words = query_lower.split()
            for word in reversed(words):
                if word in time_df["Narration"].str.lower().values:
                    category = word
                    break
        if not category:
            response = "Couldn’t identify a category."
            logger.debug(f"No category matched in query: {query}")
            self.memory.add_query(query, response)
            return {"text_response": response}

        # Match category or narration (escape regex chars)
        matches = time_df[
            time_df["category"].str.lower().str.contains(re.escape(category.lower()), na=False) |
            time_df["Narration"].str.lower().str.contains(re.escape(category.lower()), na=False)
        ]
        if matches.empty:
            response = f"No {category} transactions found."
            logger.info(f"No matches for category '{category}'")
            self.memory.add_query(query, response)
            return {"text_response": response}

        total = matches["Withdrawal (INR)"].sum()
        count = len(matches)
        period = query_lower.split("in")[-1].strip() if "in" in query_lower else "the period"
        if "summary" in query_lower:
            top_narrations = matches["Narration"].value_counts().head(2).index.tolist()
            response = f"In {period}, you spent ₹{total:.2f} on {category} across {count} transaction{'s' if count > 1 else ''}. Common transactions: {', '.join(top_narrations)}."
        else:
            response = f"You spent ₹{total:.2f} on {category} in {period} across {count} transaction{'s' if count > 1 else ''}."

        viz_data = None
        if count > 1:
            # Convert dates to strings for JSON serialization
            bar_data = matches.groupby(matches["parsed_date"].dt.strftime("%Y-%m-%d"))["Withdrawal (INR)"].sum().reset_index()
            viz_data = {
                "type": "bar",
                "data": bar_data[["parsed_date", "Withdrawal (INR)"]].values.tolist(),
                "columns": ["Date", "Amount (INR)"],
                "title": f"{category} Spending in {period}"
            }
            logger.info(f"Generated visualization for {category}")

        self.memory.add_query(query, response)
        return {"text_response": response, "visualization": viz_data}

def process_nlp_queries(
    input_csv: str,
    query: str,
    output_file: str,
    visualization_file: Optional[str] = None
) -> str:
    """
    Process NLP queries.

    Args:
        input_csv: Path to transactions CSV.
        query: User query.
        output_file: Path to save response.
        visualization_file: Optional path for visualization.

    Returns:
        Response string.
    """
    setup_mlflow()
    llm_config = get_llm_config()
    logger.info(f"Processing query: {query}")

    with mlflow.start_run(run_name="NLP_Query"):
        mlflow.log_param("input_csv", input_csv)
        mlflow.log_param("query", query)

        try:
            df = pd.read_csv(input_csv)
            if df.empty:
                raise ValueError("Empty CSV")
        except Exception as e:
            error_msg = f"Failed to load CSV: {e}"
            logger.error(error_msg)
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with Path(output_file).open("w") as f:
                f.write(error_msg)
            return error_msg

        try:
            processor = QueryProcessor(df, llm_config)
            result = processor.process_query(query)

            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with Path(output_file).open("w") as f:
                f.write(result["text_response"])
            mlflow.log_artifact(output_file)

            if visualization_file and result.get("visualization"):
                Path(visualization_file).parent.mkdir(parents=True, exist_ok=True)
                with Path(visualization_file).open("w") as f:
                    json.dump(result["visualization"], f)
                mlflow.log_artifact(visualization_file)
                logger.info(f"Saved visualization to {visualization_file}")

            return result["text_response"]

        except Exception as e:
            error_msg = f"Error processing query: {e}"
            logger.error(error_msg)
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with Path(output_file).open("w") as f:
                f.write(error_msg)
            return error_msg

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="give me a summary for Expense (Other) in 2016")
    args = parser.parse_args()
    response = process_nlp_queries(
        "data/output/categorized.csv",
        args.query,
        "data/output/nlp_response.txt",
        "data/output/visualization_data.json"
    )
    print("\nQuery Response:")
    print(response)