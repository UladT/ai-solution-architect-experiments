"""
Configuration management for Task 3 - Agentic AI.
Loads all settings from .env file.
"""

import os
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()


@dataclass
class Config:
    """Central configuration for Task 3."""

    # Azure OpenAI
    azure_openai_api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    azure_openai_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    azure_openai_api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    model_name: str = os.getenv("MODEL_NAME", "gpt-4")
    max_tokens: int = int(os.getenv("MAX_TOKENS", "2000"))
    temperature: float = float(os.getenv("TEMPERATURE", "0"))

    # GNews.io (free tier – get key at https://gnews.io)
    gnews_api_key: str = os.getenv("GNEWS_API_KEY", "")

    # App settings
    save_results: bool = os.getenv("SAVE_RESULTS", "true").lower() == "true"
    results_dir: str = "results"

    def validate(self) -> None:
        """Validate required configuration."""
        if not self.azure_openai_api_key:
            raise ValueError(
                "AZURE_OPENAI_API_KEY is required.\n"
                "Add it to your .env file."
            )
        if not self.azure_openai_endpoint:
            raise ValueError(
                "AZURE_OPENAI_ENDPOINT is required.\n"
                "Format: https://your-resource.openai.azure.com/"
            )
        if not self.gnews_api_key:
            raise ValueError(
                "GNEWS_API_KEY is required.\n"
                "Get a free key at https://gnews.io (100 req/day free tier).\n"
                "Add GNEWS_API_KEY=<your-key> to your .env file."
            )

    def print_config(self) -> None:
        """Print current configuration (safely)."""
        print("\n⚙️  CONFIGURATION:")
        print(f"  Model:        {self.model_name}")
        print(f"  Endpoint:     {self.azure_openai_endpoint}")
        print(f"  Max Tokens:   {self.max_tokens}")
        print(f"  Temperature:  {self.temperature}")
        print(f"  Save Results: {self.save_results}")
        print(f"  OAI Key:      ...{self.azure_openai_api_key[-4:]}")
        print(f"  GNews Key:    ...{self.gnews_api_key[-4:]}")
        print()


# Global config instance
config = Config()
