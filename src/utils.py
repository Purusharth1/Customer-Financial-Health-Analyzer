import logging
from typing import Dict
import os

import mlflow
import yaml
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from llm_setup.config import LLMConfig

def setup_mlflow() -> None:
    """Configure MLflow tracking."""
    tracking_uri = os.path.abspath("logs/mlruns")
    os.makedirs(tracking_uri, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{tracking_uri}")
    mlflow.set_experiment("Financial_Analyzer")


def ensure_no_active_run() -> None:
    """End any active MLflow run to prevent conflicts."""
    if mlflow.active_run():
        logging.info("Ending active MLflow run: %s", mlflow.active_run().info.run_id)
        mlflow.end_run()


def sanitize_metric_name(name: str) -> str:
    """Sanitize metric name for MLflow compatibility."""
    # Replace invalid characters with underscores
    invalid_chars = r"[^a-zA-Z0-9_\-\.:/ ]"
    import re
    return re.sub(invalid_chars, "_", name)


def load_config() -> Dict:
    """Load configuration from config.yaml."""
    try:
        with open("config/config.yaml", "r") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logging.warning("config.yaml not found, returning empty config")
        return {}


def setup_logging() -> None:
    """Configure logging based on logging.yaml."""
    try:
        with open("config/logging.yaml", "r") as f:
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
        api_endpoint=llm_settings.get("api_endpoint", "http://localhost:11434")
    )