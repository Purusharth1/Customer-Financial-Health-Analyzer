import logging
import subprocess
from typing import Optional

import mlflow
import ollama
from llm_setup.config import LLMConfig


def setup_ollama(config: LLMConfig) -> bool:
    """Set up Ollama service and pull specified LLM model."""
    logging.info("Setting up Ollama with model: %s", config.model_name)
    mlflow.log_param("llm_model", config.model_name)

    try:
        # Check if Ollama is running
        ollama.list()  # Raises exception if service is down
        logging.info("Ollama service is already running")
    except Exception as e:
        logging.warning("Ollama service not running, attempting to start: %s", e)
        try:
            subprocess.run(["ollama", "serve"], check=False, capture_output=True, text=True)
            logging.info("Started Ollama service")
        except FileNotFoundError:
            logging.error("Ollama not installed. Please install from https://ollama.ai")
            return False
        except subprocess.CalledProcessError as e:
            logging.error("Failed to start Ollama service: %s", e.stderr)
            return False

    # Check and pull model
    try:
        response = ollama.list()
        models = response.get("models", [])
        model_names = [model.get("model", "") for model in models]
        if config.model_name not in model_names:
            logging.info("Pulling model: %s", config.model_name)
            subprocess.run(["ollama", "pull", config.model_name], check=True, capture_output=True, text=True)
            mlflow.log_param("model_pulled", config.model_name)
        else:
            logging.info("Model %s already available", config.model_name)
    except subprocess.CalledProcessError as e:
        logging.error("Failed to pull model %s: %s", config.model_name, e.stderr)
        return False
    except Exception as e:
        logging.error("Error checking models: %s", e)
        return False

    return True


def query_llm(prompt: str, config: LLMConfig) -> Optional[str]:
    """Query the LLM with a given prompt."""
    try:
        response = ollama.chat(
            model=config.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"].strip()
    except Exception as e:
        logging.error("LLM query failed: %s", e)
        mlflow.log_param("llm_error", str(e))
        return None