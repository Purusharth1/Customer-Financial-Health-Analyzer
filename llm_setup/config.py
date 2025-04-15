from dataclasses import dataclass


@dataclass
class LLMConfig:
    """Configuration for LLM setup."""

    model_name: str = "llama3.2:3b"
    api_endpoint: str = "http://localhost:11434"
