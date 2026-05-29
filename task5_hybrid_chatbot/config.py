"""Configuration for Task 5 hybrid chatbot."""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """Runtime configuration for Task 5."""

    azure_openai_api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    azure_openai_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    azure_openai_api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    model_name: str = os.getenv("MODEL_NAME", "gpt-4")
    max_tokens: int = int(os.getenv("MAX_TOKENS", "1800"))
    temperature: float = float(os.getenv("TEMPERATURE", "0"))

    disasters_csv_dir: str = os.getenv(
        "DISASTERS_CSV_DIR",
        "/Users/Uladzimir_Tulinau/Library/CloudStorage/OneDrive-EPAM/SAS AI course/final_task/DISASTERS",
    )

    results_dir: str = "results"
    verbose: bool = os.getenv("VERBOSE", "true").lower() == "true"

    def validate(self) -> None:
        """Validate required settings."""
        if not self.azure_openai_api_key:
            raise ValueError("AZURE_OPENAI_API_KEY is required in .env")
        if not self.azure_openai_endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT is required in .env")
        if not os.path.isdir(self.disasters_csv_dir):
            raise ValueError(
                "DISASTERS_CSV_DIR directory does not exist: "
                f"{self.disasters_csv_dir}"
            )


config = Config()
