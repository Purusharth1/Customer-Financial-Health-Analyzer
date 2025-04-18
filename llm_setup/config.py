"""Configuration for LLM Setup.

This module defines the LLMConfig dataclass for configuring the Ollama LLM service,
including model name and other parameters used in the financial pipeline.
"""
from dataclasses import dataclass


@dataclass
class LLMConfig:
    """Configuration for LLM setup."""

    model_name: str = "llama3.2:3b"
    api_endpoint: str = "http://localhost:11434"
