"""Ollama LLM Service Manager.

This module provides utilities for setting up and querying the Ollama LLM service,
ensuring the specified model is available and handling service lifecycle for integration
with the financial pipeline.
"""
import logging
import subprocess

import mlflow
import ollama

from llm_setup.config import LLMConfig

logger = logging.getLogger(__name__)

# Define allowed model names to prevent untrusted input
ALLOWED_MODELS = {"llama3.1", "qwen2.5:7b", "llama3.2:3b", "gemma:2b"}

OLLAMA_EXECUTABLE = "/usr/local/bin/ollama"

def setup_ollama(config: LLMConfig) -> bool:
    """Set up Ollama service and pull specified LLM model."""
    logger.info("Setting up Ollama with model: %s", config.model_name)
    mlflow.log_param("llm_model", config.model_name)

    # Validate model name
    if config.model_name not in ALLOWED_MODELS:
        logger.error("Invalid model name: %s. Allowed models: %s",
                     config.model_name, ALLOWED_MODELS)
        return False

    try:
        # Check if Ollama is running
        ollama.list()  # Raises exception if service is down
        logger.info("Ollama service is already running")
    except (ConnectionError, RuntimeError) as e:
        logger.warning("Ollama service not running, attempting to start: %s", e)
        try:
            # Use trusted, hardcoded command for starting Ollama service
            subprocess.run(
                [OLLAMA_EXECUTABLE, "serve"],
                check=False,
                capture_output=True,
                text=True,
            )
            logger.info("Started Ollama service")
        except FileNotFoundError:
            logger.exception("Ollama not installed. Please install from https://ollama.ai")
            return False
        except subprocess.CalledProcessError as e:
            logger.exception("Failed to start Ollama service: %s", e.stderr)
            return False

    # Check and pull model
    try:
        response = ollama.list()
        models = response.get("models", [])
        model_names = [model.get("model", "") for model in models]
        if config.model_name not in model_names:
            logger.info("Pulling model: %s", config.model_name)
            # Use trusted command with validated model name
            subprocess.run(
                [OLLAMA_EXECUTABLE, "pull", config.model_name],
                check=True,
                capture_output=True,
                text=True,
            )
            mlflow.log_param("model_pulled", config.model_name)
        else:
            logger.info("Model %s already available", config.model_name)
    except subprocess.CalledProcessError as e:
        logger.exception("Failed to pull model %s: %s", config.model_name, e.stderr)
        return False
    except (ConnectionError, RuntimeError):
        logger.exception("Error checking models")
        return False

    return True

def query_llm(prompt: str, config: LLMConfig) -> str | None:
    """Query the LLM with a given prompt."""
    if config.model_name not in ALLOWED_MODELS:
        logger.error("Invalid model name: %s. Allowed models: %s",
                     config.model_name, ALLOWED_MODELS)
        return None

    try:
        response = ollama.chat(
            model=config.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"].strip()
    except (ConnectionError, RuntimeError, KeyError):
        logger.exception("LLM query failed")
        mlflow.log_param("llm_error", "Query failed")
        return None
