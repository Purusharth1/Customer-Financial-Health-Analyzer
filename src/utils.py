"""Shared Utilities for Logging and Helper Functions.

This module provides reusable utility functions and
logging configurations for the application.
Key functionalities include:
- Configuring and managing custom loggers for debugging and monitoring.
- Providing helper functions for common operations (e.g., date formatting).
- Ensuring consistency across the codebase with shared utilities.
"""

import logging
import sys
from pathlib import Path

import mlflow
import yaml

sys.path.append(str(Path(__file__).parent.parent))
from llm_setup.config import LLMConfig

logger = logging.getLogger(__name__)
def setup_mlflow() -> None:
    """Configure MLflow tracking."""
    tracking_uri = Path("logs/mlruns").resolve()
    Path(tracking_uri).mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{tracking_uri}")
    mlflow.set_experiment("Financial_Analyzer")


def ensure_no_active_run() -> None:
    """End any active MLflow run to prevent conflicts."""
    if mlflow.active_run():
        logger.info("Ending active MLflow run: %s", mlflow.active_run().info.run_id)
        mlflow.end_run()


def sanitize_metric_name(name: str) -> str:
    """Sanitize metric name for MLflow compatibility."""
    # Replace invalid characters with underscores
    invalid_chars = r"[^a-zA-Z0-9_\-\.:/ ]"
    import re

    return re.sub(invalid_chars, "_", name)


def load_config() -> dict:
    """Load configuration from config.yaml."""
    try:
        with Path("config/config.yaml").open() as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning("config.yaml not found, returning empty config")
        return {}


def setup_logging() -> None:
    """Configure logging based on logging.yaml."""
    try:
        with Path("config/logging.yaml").open() as f:
            config = yaml.safe_load(f)
            logging.config.dictConfig(config)
    except FileNotFoundError:
        logging.basicConfig(level=logging.INFO)


def get_llm_config() -> LLMConfig:
    """Load LLM configuration."""
    config = load_config()
    llm_settings = config.get("llm", {})
    return LLMConfig(
        model_name=llm_settings.get("model_name", "gemma:2b"),
        api_endpoint=llm_settings.get("api_endpoint", "http://localhost:11434"),
    )
