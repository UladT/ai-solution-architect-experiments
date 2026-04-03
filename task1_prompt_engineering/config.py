"""
Configuration management.
Loads all settings from .env file.
"""

import os
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()


@dataclass
class Config:
    """Central configuration for Task 1."""

    # API Settings (Azure OpenAI)
    azure_openai_api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    azure_openai_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    azure_openai_api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    model_name: str = os.getenv("MODEL_NAME", "gpt-4")

    # LLM Parameters
    max_tokens: int = int(os.getenv("MAX_TOKENS", "2000"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.3"))

    # App Settings
    save_results: bool = os.getenv("SAVE_RESULTS", "true").lower() == "true"
    verbose: bool = os.getenv("VERBOSE", "true").lower() == "true"

    # Paths
    results_dir: str = "results"
    metrics_dir: str = "metrics"

    def validate(self) -> None:
        """Validate required configuration."""
        if not self.azure_openai_api_key:
            raise ValueError(
                "AZURE_OPENAI_API_KEY is required.\n"
                "1. Copy .env.example to .env\n"
                "2. Add your Azure OpenAI API key\n"
                "3. Get key at: https://portal.azure.com/"
            )
        if not self.azure_openai_endpoint:
            raise ValueError(
                "AZURE_OPENAI_ENDPOINT is required.\n"
                "Format: https://your-resource-name.openai.azure.com/"
            )
        if not self.azure_openai_api_version:
            raise ValueError(
                "AZURE_OPENAI_API_VERSION is required.\n"
                "Example: 2024-02-15-preview"
            )

    def print_config(self) -> None:
        """Print current configuration (safely)."""
        print("\n⚙️  CONFIGURATION (Azure OpenAI):")
        print(f"  Model:        {self.model_name}")
        print(f"  Endpoint:     {self.azure_openai_endpoint}")
        print(f"  API Version:  {self.azure_openai_api_version}")
        print(f"  Max Tokens:   {self.max_tokens}")
        print(f"  Temperature:  {self.temperature}")
        print(f"  Save Results: {self.save_results}")
        print(f"  Verbose:      {self.verbose}")
        print(f"  API Key:      ...{self.azure_openai_api_key[-4:]}")
        print()


# Global config instance
config = Config()