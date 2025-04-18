"""Natural Language Processing for Financial Health Analyzer.

Handles:
- Search across transactions.
- Financial memory for purchases/events.
- Conversational queries and summaries with visualizations.
"""
import json
import logging
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from llm_setup.ollama_manager import query_llm
from src.models import (
    FinancialMemoryState,
    NlpProcessorInput,
    NlpProcessorOutput,
    QueryRecord,
    VisualizationData,
)
from src.utils import get_llm_config, setup_mlflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
TIMEOUT_SECONDS = 30
MIN_KEYWORD_LENGTH = 3
RECENT_QUERIES_LIMIT = 10
RECENT_QUERIES_CONTEXT = 3
DEFAULT_TIMEFRAME_MONTHS = 3
EMPTY_CSV_MESSAGE = "Empty CSV"
TRANSACTION_LIMIT = 5
BATCH_TOP_NARRATIONS = 2

class FinancialMemory:
    """Manages financial memory state with query history and context."""

    def __init__(self, persist_path: str = "data/state/financial_memory.json") -> None:
        """Initialize FinancialMemory with a persistent storage path."""
        self.queries: list[QueryRecord] = []
        self.context: dict[str, str] = {}
        self.persist_path = persist_path
        if Path(persist_path).exists():
            self._load()

    def add_query(self, query: str, result: str) -> None:
        """Add a query and its result to memory, keeping the most recent ones."""
        self.queries.append(
            QueryRecord(
                query=query,
                result=result,
                timestamp=datetime.now(UTC).isoformat(),
            ),
        )
        self.queries = self.queries[-RECENT_QUERIES_LIMIT:]
        self._save()

    def add_context(self, key: str, value: str) -> None:
        """Add or update context information in memory."""
        self.context[key] = value
        self._save()

    def get_context(self) -> str:
        """Retrieve recent queries and context as a formatted string."""
        context = "Recent queries:\n"
        for q in self.queries[-RECENT_QUERIES_CONTEXT:]:
            context += f"- Q: {q.query}\n  A: {q.result}\n"
        if self.context:
            context += "Known info:\n"
            for k, v in self.context.items():
                context += f"- {k}: {v}\n"
        return context

    def _save(self) -> None:
        """Save memory state to a JSON file."""
        try:
            state = FinancialMemoryState(queries=self.queries, context=self.context)
            Path(self.persist_path).parent.mkdir(parents=True, exist_ok=True)
            with Path(self.persist_path).open("w") as f:
                json.dump(state.dict(), f)
        except (OSError, json.JSONEncodeError):
            logger.exception("Failed to save memory")

    def _load(self) -> None:
        """Load memory state from a JSON file."""
        try:
            with Path(self.persist_path).open() as f:
                state_data = json.load(f)
            state = FinancialMemoryState(**state_data)
            self.queries = state.queries
            self.context = state.context
        except (OSError, json.JSONDecodeError, ValueError):
            logger.exception("Failed to load memory")

class QueryProcessor:
    """Processes NLP queries on financial transaction data."""

    def __init__(self, transactions_df: pd.DataFrame, llm_config: dict) -> None:
        """Initialize QueryProcessor with transaction data and LLM config."""
        self.df = (
            transactions_df.copy()
            if not transactions_df.empty
            else transactions_df
        )
        self.llm_config = llm_config
        self.memory = FinancialMemory()
        self.df["parsed_date"] = pd.to_datetime(self.df["parsed_date"], errors="coerce")
        if transactions_df.empty:
            logger.warning("Transaction DataFrame is empty")

    def _extract_time_range(self, query: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Extract start and end timestamps from query or use default timeframe."""
        query_lower = query.lower()
        # Patterns for common time expressions
        year_pattern = r"\b(20\d{2})\b"
        month_year_pattern = r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(20\d{2})\b"
        date_pattern = r"\b(\d{4}-\d{2}-\d{2})\b"

        # Try extracting specific date
        date_match = re.search(date_pattern, query_lower)
        if date_match:
            date = pd.to_datetime(date_match.group(1))
            start = date.replace(hour=0, minute=0, second=0)
            end = date.replace(hour=23, minute=59, second=59)
            return start, end

        # Try extracting month and year
        month_year_match = re.search(month_year_pattern, query_lower)
        if month_year_match:
            month_str, year = month_year_match.groups()
            month = [
                "january", "february", "march", "april", "may", "june",
                "july", "august", "september", "october", "november", "december",
            ].index(month_str.lower()) + 1
            start = pd.Timestamp(year=int(year), month=month, day=1)
            end = (start + pd.offsets.MonthEnd(0)).replace(hour=23, minute=59, second=59)
            return start, end

        # Try extracting year
        year_match = re.search(year_pattern, query_lower)
        if year_match:
            year = int(year_match.group(1))
            start = pd.Timestamp(year=year, month=1, day=1)
            end = pd.Timestamp(year=year, month=12, day=31, hour=23, minute=59, second=59)
            return start, end

        # Default to last DEFAULT_TIMEFRAME_MONTHS months
        end = pd.Timestamp.now()
        start = end - pd.offsets.MonthBegin(DEFAULT_TIMEFRAME_MONTHS)
        logger.info("No time range specified in query, using default: %s to %s", start, end)
        return start, end

    def _filter_by_time(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Filter DataFrame by time range."""
        logger.info("Filtering by time: start=%s, end=%s", start, end)
        # Ensure start and end are tz-naive
        start_naive = start.tz_localize(None) if start.tzinfo else start
        end_naive = end.tz_localize(None) if end.tzinfo else end
        # Verify parsed_date is tz-naive
        if self.df["parsed_date"].dtype == "datetime64[ns]":
            logger.debug("parsed_date is tz-naive")
        else:
            logger.warning("Unexpected parsed_date dtype: %s", self.df["parsed_date"].dtype)
        # Filter DataFrame
        mask = (self.df["parsed_date"] >= start_naive) & (self.df["parsed_date"] <= end_naive)
        return self.df[mask]

    def search(self, query: str) -> pd.DataFrame:
        """Search transactions by keywords in narration or category."""
        keywords = [
            word
            for word in query.lower().split()
            if len(word) > MIN_KEYWORD_LENGTH
        ]
        if not keywords:
            logger.debug("No valid keywords")
            return pd.DataFrame()
        mask = pd.Series(data=False, index=self.df.index)
        for col in ["Narration", "category"]:
            for kw in keywords:
                mask |= self.df[col].str.lower().str.contains(
                    re.escape(kw), na=False,
                )
        matches = self.df[mask]
        logger.info("Found %d matches for query: %s", len(matches), query)
        return matches

    def _process_search_query(self, query: str) -> NlpProcessorOutput:
        """Handle search-related queries (e.g., 'search', 'find', 'show me')."""
        matches = self.search(query)
        if matches.empty:
            response = "No matching transactions found."
            logger.debug("No matches for search query: %s", query)
        else:
            transactions = (
                matches[
                    ["parsed_date", "Narration", "Withdrawal (INR)", "category"]
                ]
                .head(TRANSACTION_LIMIT)
                .to_dict("records")
            )
            response = (
                f"Found {len(matches)} transaction{'s' if len(matches) > 1 else ''}: "
                f"{json.dumps(transactions, default=str)}"
            )
        self.memory.add_query(query, response)
        return NlpProcessorOutput(text_response=response)

    def _process_purchase_query(
        self, query: str, query_lower: str,
    ) -> NlpProcessorOutput:
        """Handle purchase-related queries (e.g., 'how much did I spend')."""
        matches = self.search(query)
        item = (
            query_lower.split("on")[-1].strip()
            if "on" in query_lower
            else query_lower.split()[-1]
        )
        if not matches.empty:
            total = matches["Withdrawal (INR)"].sum()
            response = f"You spent ₹{total:.2f} on {item}."
            self.memory.add_context(f"{item}_purchase", response)
        else:
            response = f"No record of {item} purchase."
            logger.debug("No purchase found for %s", item)
        self.memory.add_query(query, response)
        return NlpProcessorOutput(text_response=response)

    def _process_visit_query(self, query: str, query_lower: str) -> NlpProcessorOutput:
        """Handle visit-related queries (e.g., 'when did I go to')."""
        matches = self.search(query)
        place = (
            query_lower.split("to")[-1].strip()
            if "to" in query_lower
            else query_lower.split()[-1]
        )
        if not matches.empty:
            date = matches["parsed_date"].min().strftime("%Y-%m-%d")
            response = f"You went to {place} around {date}."
            self.memory.add_context(f"{place}_visit", response)
        else:
            response = f"No record of visiting {place}."
            logger.debug("No visit found for %s", place)
        self.memory.add_query(query, response)
        return NlpProcessorOutput(text_response=response)

    def _process_fallback_query(self, query: str) -> NlpProcessorOutput:
        """Handle generic queries using LLM with context."""
        context = self.memory.get_context()
        prompt = f"Answer based on context:\n{context}\nQuery: {query}\nKeep it short."
        response = query_llm(
            prompt, self.llm_config, timeout=TIMEOUT_SECONDS,
        ).strip()
        self.memory.add_query(query, response)
        return NlpProcessorOutput(text_response=response)

    def _extract_category(self, query_lower: str, time_df: pd.DataFrame) -> str | None:
        """Extract a category from the query based on transaction data."""
        possible_categories = time_df["category"].unique()
        for cat in possible_categories:
            if cat.lower() in query_lower:
                return cat
        if "expense" in query_lower:
            expense_part = query_lower.split("expense")[-1].strip()
            expense_part = expense_part.split("in")[0].strip()
            for cat in possible_categories:
                if expense_part in cat.lower():
                    return cat
        for word in reversed(query_lower.split()):
            if word in time_df["Narration"].str.lower().to_numpy():
                return word
        return None

    def _generate_visualization(
        self, matches: pd.DataFrame, category: str, period: str,
    ) -> VisualizationData | None:
        """Generate bar chart data for transactions if applicable."""
        if len(matches) <= 1:
            return None
        bar_data = (
            matches.groupby(matches["parsed_date"].dt.strftime("%Y-%m-%d"))[
                "Withdrawal (INR)"
            ]
            .sum()
            .reset_index()
        )
        viz_data = VisualizationData(
            type="bar",
            data=bar_data[["parsed_date", "Withdrawal (INR)"]].to_numpy().tolist(),
            columns=["Date", "Amount (INR)"],
            title=f"{category} Spending in {period}",
        )
        logger.info("Generated visualization for %s", category)
        return viz_data

    def _process_category_query(
        self, query_lower: str, time_df: pd.DataFrame,
    ) -> NlpProcessorOutput:
        """Handle queries involving transaction categories."""
        category = self._extract_category(query_lower, time_df)
        if not category:
            response = "Could not identify a category."
            logger.debug("No category matched in query: %s", query_lower)
            self.memory.add_query(query_lower, response)
            return NlpProcessorOutput(text_response=response)

        matches = time_df[
            time_df["category"]
            .str.lower()
            .str.contains(re.escape(category.lower()), na=False)
            | time_df["Narration"]
            .str.lower()
            .str.contains(re.escape(category.lower()), na=False)
        ]
        if matches.empty:
            response = f"No {category} transactions found."
            logger.info("No matches for category '%s'", category)
            self.memory.add_query(query_lower, response)
            return NlpProcessorOutput(text_response=response)

        total = matches["Withdrawal (INR)"].sum()
        count = len(matches)
        period = (
            query_lower.split("in")[-1].strip()
            if "in" in query_lower
            else "the period"
        )
        if "summary" in query_lower:
            top_narrations = (
                matches["Narration"]
                .value_counts()
                .head(BATCH_TOP_NARRATIONS)
                .index.tolist()
            )
            response = (
                f"In {period}, you spent ₹{total:.2f} on {category} across "
                f"{count} transaction{'s' if count > 1 else ''}. "
                f"Common transactions: {', '.join(top_narrations)}."
            )
        else:
            response = (
                f"You spent ₹{total:.2f} on {category} in {period} across "
                f"{count} transaction{'s' if count > 1 else ''}."
            )

        viz_data = self._generate_visualization(matches, category, period)
        self.memory.add_query(query_lower, response)
        return NlpProcessorOutput(text_response=response, visualization=viz_data)

    def process_query(self, query: str) -> NlpProcessorOutput:
        """Process a financial query with optional visualization."""
        query_lower = query.lower()
        if any(term in query_lower for term in ["search", "find", "show me"]):
            return self._process_search_query(query)

        if any(
            term in query_lower
            for term in ["how much did i", "when did i", "did i buy"]
        ):
            if "how much" in query_lower:
                return self._process_purchase_query(query, query_lower)
            if "when did i" in query_lower:
                return self._process_visit_query(query, query_lower)
            return self._process_fallback_query(query)

        # Extract time range from query
        start, end = self._extract_time_range(query)
        time_df = self._filter_by_time(start, end)
        if time_df.empty:
            response = "No transactions found for this period."
            logger.warning("Empty time-filtered data for query: %s", query)
            self.memory.add_query(query, response)
            return NlpProcessorOutput(text_response=response)

        return self._process_category_query(query_lower, time_df)

def _raise_empty_csv_error() -> None:
    """Raise ValueError for empty CSV with a predefined message."""
    error_msg = EMPTY_CSV_MESSAGE
    raise ValueError(error_msg)

def process_nlp_queries(input_model: NlpProcessorInput) -> NlpProcessorOutput:
    """Process NLP queries and save results to output files."""
    setup_mlflow()
    llm_config = get_llm_config()
    logger.info("Processing query: %s", input_model.query)

    with mlflow.start_run(run_name="NLP_Query"):
        mlflow.log_param("input_csv", str(input_model.input_csv))
        mlflow.log_param("query", input_model.query)

        try:
            transactions_df = pd.read_csv(input_model.input_csv)
            if transactions_df.empty:
                _raise_empty_csv_error()
        except (FileNotFoundError, pd.errors.EmptyDataError, ValueError) as e:
            error_msg = f"Failed to load CSV: {e}"
            logger.exception(error_msg)
            input_model.output_file.parent.mkdir(parents=True, exist_ok=True)
            with input_model.output_file.open("w") as f:
                f.write(error_msg)
            return NlpProcessorOutput(text_response=error_msg)

        processor = QueryProcessor(transactions_df, llm_config)
        result = processor.process_query(input_model.query)

        input_model.output_file.parent.mkdir(parents=True, exist_ok=True)
        with input_model.output_file.open("w") as f:
            f.write(result.text_response)
        mlflow.log_artifact(str(input_model.output_file))

        if input_model.visualization_file and result.visualization:
            input_model.visualization_file.parent.mkdir(parents=True, exist_ok=True)
            with input_model.visualization_file.open("w") as f:
                json.dump(result.visualization.dict(), f)
            mlflow.log_artifact(str(input_model.visualization_file))
            logger.info(
                "Saved visualization to %s",
                input_model.visualization_file,
            )

        return result

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        default="give me a summary for Expense (Other) in 2016",
    )
    args = parser.parse_args()
    response = process_nlp_queries(
        NlpProcessorInput(
            input_csv=Path("data/output/categorized.csv"),
            query=args.query,
            output_file=Path("data/output/nlp_response.txt"),
            visualization_file=Path("data/output/visualization_data.json"),
        ),
    )
    logger.info("Query Response: %s", response)
