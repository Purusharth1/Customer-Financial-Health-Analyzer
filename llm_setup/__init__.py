"""Initialization for the LLM Setup Package.

This package provides utilities for configuring and managing the Ollama LLM service,
including setup and query functionalities for integration with the financial pipeline.
"""
from .config import LLMConfig
from .ollama_manager import query_llm, setup_ollama

__all__ = ["LLMConfig", "query_llm", "setup_ollama"]
