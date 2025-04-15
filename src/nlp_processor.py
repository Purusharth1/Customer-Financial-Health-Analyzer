"""
Natural Language Processing for Financial Health Analyzer.

This module provides advanced NLP capabilities for financial data analysis including:
- Sophisticated search across bank statements and transactions
- Financial memory management for contextual queries (e.g., "How much did I spend on my laptop?")
- Conversational financial analysis (e.g., "Show me restaurant spending last month")
- Integration with LLMs for natural language understanding and generation
"""
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Union
import re
import mlflow
import pandas as pd
from dateutil.relativedelta import relativedelta
from fuzzywuzzy import process
from utils import get_llm_config, setup_mlflow
import sys
sys.path.append(str(Path(__file__).parent.parent))
from llm_setup.ollama_manager import query_llm


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 60

class FinancialMemory:
    """Manages context and memory for financial queries."""

    def __init__(self, persist_path: Optional[str] = None):
        """
        Initialize financial memory.

        Args:
            persist_path: Path to save/load memory state, if provided.
        """
        self.recent_queries: list = []
        self.context: dict = {}
        self.persist_path = persist_path

        if persist_path and Path(persist_path).exists():
            self.load_state()

    def add_query(self, query: str, result: str) -> None:
        """
        Add query and result to memory.

        Args:
            query: User query.
            result: Response to the query.
        """
        self.recent_queries.append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "result": result,
        })
        self.recent_queries = self.recent_queries[-10:]
        if self.persist_path:
            self.save_state()

    def update_context(self, key: str, value: any) -> None:
        """
        Update context with new information.

        Args:
            key: Context key (e.g., "laptop_purchase").
            value: Context value.
        """
        self.context[key] = value
        if self.persist_path:
            self.save_state()

    def get_recent_context(self) -> str:
        """Get formatted recent context for LLM prompt."""
        context_str = "Recent queries:\n"
        for q in self.recent_queries[-3:]:
            context_str += f"- Q: {q['query']}\n  A: {q['result']}\n"
        if self.context:
            context_str += "\nKnown information:\n"
            for k, v in self.context.items():
                context_str += f"- {k}: {v}\n"
        return context_str

    def save_state(self) -> None:
        """Save memory state to disk."""
        state = {
            "recent_queries": self.recent_queries,
            "context": self.context,
        }
        try:
            Path(self.persist_path).parent.mkdir(parents=True, exist_ok=True)
            with Path(self.persist_path).open("w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save memory state: {e}")

    def load_state(self) -> None:
        """Load memory state from disk."""
        try:
            with Path(self.persist_path).open() as f:
                state = json.load(f)
            if not isinstance(state, dict):
                raise ValueError("Invalid memory state format")
            self.recent_queries = state.get("recent_queries", [])
            self.context = state.get("context", {})
            self.recent_queries = [
                q for q in self.recent_queries
                if isinstance(q, dict) and "query" in q and "result" in q
            ]
        except Exception as e:
            logger.error(f"Failed to load memory state: {e}")

class QueryProcessor:
    """Process and analyze financial queries."""

    def __init__(self, transaction_df: pd.DataFrame, llm_config: dict, analysis_dir: str = "data/output/analysis"):
        """
        Initialize query processor.

        Args:
            transaction_df: DataFrame with transaction data.
            llm_config: Configuration for LLM.
            analysis_dir: Directory with analysis outputs (e.g., anomalies.csv).
        """
        if transaction_df.empty:
            raise ValueError("Transaction DataFrame is empty")
        self.df = transaction_df.copy()
        self.llm_config = llm_config
        self.memory = FinancialMemory(persist_path="data/state/financial_memory.json")
        self.analysis_dir = analysis_dir

        if "parsed_date" not in self.df.columns:
            raise KeyError("Missing 'parsed_date' column")
        self.df["parsed_date"] = pd.to_datetime(self.df["parsed_date"], errors="coerce")

        # Load analysis data
        self.anomalies = self._load_analysis_file("anomalies.csv")
        self.fees = self._load_analysis_file("fees.csv")

    def _load_analysis_file(self, filename: str) -> pd.DataFrame:
        """Load analysis file if it exists."""
        file_path = Path(self.analysis_dir) / filename
        try:
            if file_path.exists():
                return pd.read_csv(file_path)
        except Exception as e:
            logger.warning(f"Failed to load {filename}: {e}")
        return pd.DataFrame()

    def _get_time_filtered_data(self, time_period: str) -> pd.DataFrame:
        """
        Filter data based on time period, fall back to all data if empty.

        Args:
            time_period: Time period (e.g., "last month").

        Returns:
            Filtered DataFrame.
        """
        now = datetime.now()
        time_period = time_period.lower()

        if "last month" in time_period:
            start_date = (now.replace(day=1) - timedelta(days=1)).replace(day=1)
            end_date = now.replace(day=1) - timedelta(days=1)
        elif "this month" in time_period:
            start_date = now.replace(day=1)
            end_date = now
        elif "last year" in time_period:
            start_date = datetime(now.year - 1, 1, 1)
            end_date = datetime(now.year - 1, 12, 31)
        elif "this year" in time_period:
            start_date = datetime(now.year, 1, 1)
            end_date = now
        elif "last week" in time_period:
            start_date = now - timedelta(days=now.weekday() + 7)
            end_date = start_date + timedelta(days=6)
        else:
            start_date = now - relativedelta(months=3)
            end_date = now

        filtered = self.df[
            (self.df["parsed_date"] >= start_date) &
            (self.df["parsed_date"] <= end_date)
        ]
        if filtered.empty:
            logger.warning(f"No data for {time_period}, falling back to all data")
            filtered = self.df
        return filtered

    def search_transactions(self, query: str) -> pd.DataFrame:
        """
        Search transactions using keywords, fuzzy matching, and synonyms.

        Args:
            query: Search query.

        Returns:
            DataFrame with matching transactions.
        """
        # Synonyms for common categories
        synonyms = {
            "restaurant": ["cafe", "dining", "food", "eatery", "bistro"],
            "travel": ["flight", "hotel", "train", "taxi"],
            "shopping": ["retail", "store", "online"],
        }

        # Extract keywords
        stop_words = {"the", "and", "for", "in", "on", "at", "to", "with", "by"}
        query_lower = query.lower()
        keywords = [
            word for word in query_lower.split()
            if word not in stop_words and len(word) > 2
        ]
        if not keywords:
            logger.debug("No valid keywords in query")
            return pd.DataFrame()

        # Add synonyms
        all_keywords = keywords.copy()
        for keyword in keywords:
            for key, syn_list in synonyms.items():
                if keyword in key or any(syn in keyword for syn in syn_list):
                    all_keywords.extend(syn_list)
        all_keywords = list(set(all_keywords))

        # Time filter
        time_filters = ["last month", "this month", "last year", "this year", "last week"]
        time_filter = next((tf for tf in time_filters if tf in query_lower), None)
        df = self._get_time_filtered_data(time_filter) if time_filter else self.df

        # Search columns
        search_cols = ["Narration", "category"]
        if "Description" in df.columns:
            search_cols.append("Description")

        # Fuzzy and exact matching
        matches = []
        for col in search_cols:
            unique_vals = df[col].dropna().str.lower().unique()
            for keyword in all_keywords:
                # Fuzzy match
                fuzzy_matches = process.extract(keyword, unique_vals, limit=10)
                matched_vals = [m[0] for m in fuzzy_matches if m[1] > 70]  # Lowered threshold
                # Exact match
                exact_mask = df[col].str.lower().str.contains(keyword, na=False)
                combined_mask = df[col].str.lower().isin(matched_vals) | exact_mask
                if combined_mask.any():
                    matches.append(df[combined_mask])
                else:
                    logger.debug(f"No matches for '{keyword}' in {col}")

        if not matches:
            logger.info(f"No transactions found for query: {query}")
            return pd.DataFrame()

        result = pd.concat(matches).drop_duplicates()
        return result

    def process_financial_memory_query(self, query: str) -> str:
        """
        Process queries related to financial memory.

        Args:
            query: User query.

        Returns:
            Response string.
        """
        patterns = [
            (r"(buy|bought|spend|spent|cost|pay|paid)\s+(.+?)\s+(for|on)\s+(.+)", "purchase"),
            (r"when did I (go|visit|travel to|visit)\s+(.+)", "visit"),
        ]

        for pattern, query_type in patterns:
            match = re.search(pattern, query.lower(), re.IGNORECASE)
            if match:
                if query_type == "purchase":
                    item = match.group(4).strip()
                    prompt = (
                        f"Find the purchase of '{item}' in the transaction data. "
                        "Return when it happened and how much was spent in 1-2 sentences. "
                        f"If not found, say 'No record of {item} purchase.'"
                    )
                else:
                    place = match.group(2).strip()
                    prompt = (
                        f"Find transactions related to '{place}' visit. "
                        "Return the most recent visit date in 1 sentence. "
                        f"If not found, say 'No record of visiting {place}.'"
                    )

                matches = self.search_transactions(match.group(0))
                if matches.empty:
                    response = f"No record of {item if query_type == 'purchase' else 'visiting ' + place}."
                else:
                    data_summary = matches[[
                        "parsed_date", "Narration", "Withdrawal (INR)", "Deposit (INR)", "category"
                    ]].head(5).to_dict("records")
                    full_prompt = f"{prompt}\nData: {json.dumps(data_summary, default=str)}"
                    response = query_llm(full_prompt, self.llm_config, timeout=TIMEOUT_SECONDS).strip()

                self.memory.add_query(query, response)
                if "no record" not in response.lower():
                    key = f"{item}_purchase" if query_type == "purchase" else f"{place}_visit"
                    self.memory.update_context(key, response)
                return response

        context = self.memory.get_recent_context()
        prompt = (
            f"Answer the query based on context:\n"
            f"CONTEXT:\n{context}\n"
            f"QUERY: {query}\n"
            "Keep it concise (1-2 sentences). "
            "If no relevant context, say 'No information available.'"
        )
        response = query_llm(prompt, self.llm_config).strip()
        self.memory.add_query(query, response)
        return response

    def process_conversational_query(self, query: str) -> Dict[str, Optional[Union[str, Dict]]]:
        """
        Process conversational financial queries.

        Args:
            query: User query.

        Returns:
            Dict with text response and optional visualization data.
        """
        time_periods = ["last month", "this month", "last year", "this year", "last week"]
        time_period = next((tp for tp in time_periods if tp in query.lower()), "last month")
        filtered_df = self._get_time_filtered_data(time_period)

        # Category spending
        category_patterns = [
            r"(spend|spent|spending)\s+on\s+(.+)",
            r"(.+)\s+(spending|expenses)",
            r"how much\s+(?:did I|have I\s+|)(spend|spent)\s+on\s+(.+)",
        ]

        for pattern in category_patterns:
            match = re.search(pattern, query.lower(), re.IGNORECASE)
            if match:
                category = match.groups()[-1].strip()
                # Fuzzy match category with synonyms
                synonyms = {
                    "restaurant": ["cafe", "dining", "food", "eatery", "bistro"],
                }
                search_terms = [category]
                for key, syn_list in synonyms.items():
                    if category in key or any(category in syn for syn in syn_list):
                        search_terms.extend(syn_list)

                cat_matches = pd.DataFrame()
                if "category" in filtered_df.columns:
                    categories = filtered_df["category"].dropna().str.lower().unique()
                    for term in search_terms:
                        best_match, score = process.extractOne(term, categories)
                        if score > 70:
                            cat_matches = pd.concat([
                                cat_matches,
                                filtered_df[filtered_df["category"].str.lower() == best_match]
                            ])
                        # Search Narration as fallback
                        narration_mask = filtered_df["Narration"].str.lower().str.contains(term, na=False)
                        cat_matches = pd.concat([
                            cat_matches,
                            filtered_df[narration_mask]
                        ])

                cat_matches = cat_matches.drop_duplicates()
                if not cat_matches.empty:
                    total_spent = cat_matches["Withdrawal (INR)"].sum()
                    count = len(cat_matches)
                    viz_data = None
                    if count > 1:
                        bar_data = (
                            cat_matches.groupby(cat_matches["parsed_date"].dt.date)["Withdrawal (INR)"]
                            .sum()
                            .reset_index()
                        )
                        viz_data = {
                            "type": "bar",
                            "data": bar_data[["parsed_date", "Withdrawal (INR)"]].values.tolist(),
                            "columns": ["Date", "Amount (INR)"],
                            "title": f"{category} Spending {time_period}"
                        }
                        if len(bar_data) > 3:
                            viz_data["alternative"] = {
                                "type": "line",
                                "data": bar_data[["parsed_date", "Withdrawal (INR)"]].values.tolist(),
                                "columns": ["Date", "Amount (INR)"],
                                "title": f"{category} Spending Trend {time_period}"
                            }
                    response = (
                        f"You spent ₹{total_spent:.2f} on {category} {time_period} "
                        f"across {count} transaction{'s' if count > 1 else ''}."
                    )
                    logger.info(f"Found {count} matches for '{category}'")
                    return {
                        "text_response": response,
                        "visualization": viz_data
                    }
                logger.info(f"No matches for category '{category}' in {time_period}")

        # General conversational query with analysis context
        data_summary = {
            "time_period": time_period,
            "total_transactions": len(filtered_df),
            "total_spent": filtered_df["Withdrawal (INR)"].sum(),
            "total_received": filtered_df["Deposit (INR)"].sum(),
            "top_categories": (
                filtered_df.groupby("category")["Withdrawal (INR)"]
                .sum()
                .nlargest(5)
                .to_dict()
                if "category" in filtered_df.columns
                else {}
            ),
            "anomalies": (
                self.anomalies[["parsed_date", "Narration", "amount", "type", "category"]]
                .head(5)
                .to_dict("records")
                if not self.anomalies.empty
                else []
            ),
        }
        context = self.memory.get_recent_context()
        prompt = (
            f"Answer this query based on the transaction summary and context:\n"
            f"QUERY: {query}\n"
            f"SUMMARY:\n{json.dumps(data_summary, indent=2, default=str)}\n"
            f"CONTEXT:\n{context}\n"
            "Keep it concise (2-3 sentences). "
            "If no relevant data, suggest checking transactions or provide a general insight."
        )
        response = query_llm(prompt, self.llm_config, timeout=TIMEOUT_SECONDS).strip()
        self.memory.add_query(query, response)
        logger.debug(f"General query response: {response}")
        return {"text_response": response}

def process_nlp_queries(
    input_csv: str,
    query: str,
    output_file: str,
    visualization_file: Optional[str] = None
) -> str:
    """
    Process NLP queries for financial data analysis.

    Args:
        input_csv: Path to categorized transactions CSV.
        query: User query (e.g., "Restaurant spending last month").
        output_file: Path to save text response.
        visualization_file: Optional path to save visualization data.

    Returns:
        Response string (for workflow compatibility).
    """
    setup_mlflow()
    llm_config = get_llm_config()
    logger.info(f"Processing NLP query: {query}")

    with mlflow.start_run(run_name="Financial_NLP_Query"):
        mlflow.log_param("input_csv", input_csv)
        mlflow.log_param("query", query)

        try:
            df = pd.read_csv(input_csv)
            if df.empty:
                raise ValueError("Input CSV is empty")
        except Exception as e:
            error_msg = f"Failed to load CSV: {e}"
            logger.error(error_msg)
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with Path(output_file).open("w") as f:
                f.write(error_msg)
            return error_msg

        try:
            processor = QueryProcessor(df, llm_config)

            classification_prompt = (
                f"Classify this query into one type: 'search', 'memory', 'conversational', 'analyzer', 'storyteller'. "
                f"Query: {query}\n"
                "Return only the type."
            )
            query_type = query_llm(classification_prompt, llm_config).strip().lower()
            logger.debug(f"Query classified as: {query_type}")

            query_lower = query.lower()
            if any(term in query_lower for term in ["search", "find", "look for", "show me"]):
                query_type = "search"
            elif any(term in query_lower for term in ["when did i", "how much did i", "did i buy", "what did i"]):
                query_type = "memory"
            elif "story" in query_lower or "summary" in query_lower:
                query_type = "storyteller"
            elif any(term in query_lower for term in ["recurring", "subscription", "unusual", "suspicious"]):
                query_type = "analyzer"

            if query_type == "search":
                matches = processor.search_transactions(query)
                if matches.empty:
                    response = {
                        "text_response": "No matching transactions found.",
                        "matches_count": 0,
                    }
                else:
                    display_cols = ["parsed_date", "Narration", "Withdrawal (INR)", "Deposit (INR)", "category"]
                    display_cols = [col for col in display_cols if col in matches]
                    response = {
                        "text_response": f"Found {len(matches)} matching transaction{'s' if len(matches) > 1 else ''}.",
                        "matches_count": len(matches),
                        "matches": matches[display_cols].to_dict("records"),
                    }
            elif query_type == "memory":
                response = {
                    "text_response": processor.process_financial_memory_query(query)
                }
            elif query_type == "storyteller":
                response = {
                    "text_response": "This query will be handled by the storyteller module."
                }
            elif query_type == "analyzer":
                response = {
                    "text_response": "This query will be handled by the analyzer module."
                }
            else:
                response = processor.process_conversational_query(query)

            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with Path(output_file).open("w") as f:
                f.write(response["text_response"])
            mlflow.log_artifact(output_file)

            if visualization_file and response.get("visualization"):
                try:
                    Path(visualization_file).parent.mkdir(parents=True, exist_ok=True)
                    with Path(visualization_file).open("w") as f:
                        json.dump(response["visualization"], f, default=str)
                    mlflow.log_artifact(visualization_file)
                except Exception as e:
                    logger.warning(f"Failed to save visualization: {e}")

            logger.info("Query processed successfully")
            return response["text_response"]

        except ValueError as ve:
            error_msg = f"Invalid input: {ve}"
            logger.error(error_msg)
            with Path(output_file).open("w") as f:
                f.write(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Error processing query: {e}"
            logger.exception(error_msg)
            with Path(output_file).open("w") as f:
                f.write(error_msg)
            return error_msg

if __name__ == "__main__":
    input_csv = "data/output/categorized.csv"
    query = "How much did I spend on Expenses (loan) last month?"
    output_file = "data/output/nlp_response.txt"
    visualization_file = "data/output/visualization_data.json"
    response = process_nlp_queries(input_csv, query, output_file, visualization_file)
    print("\nQuery Response:")
    print(response)