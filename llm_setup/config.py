from dataclasses import dataclass


@dataclass
class LLMConfig:
    """Configuration for LLM setup."""
    model_name: str = "phi3:3.8b"
    api_endpoint: str = "http://localhost:11434"