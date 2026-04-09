"""
Configuration management for Task 2 RAG Document Analysis.
Extends the same Azure OpenAI setup used in Task 1.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """Central configuration for Task 2."""

    # ── Azure OpenAI ──────────────────────────────────────────────────────────
    azure_openai_api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    azure_openai_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    azure_openai_api_version: str = os.getenv(
        "AZURE_OPENAI_API_VERSION", "2024-02-15-preview"
    )
    model_name: str = os.getenv("MODEL_NAME", "gpt-4")
    # Embedding model deployed in the same Azure OpenAI resource
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")

    # ── LLM Parameters ────────────────────────────────────────────────────────
    max_tokens: int = int(os.getenv("MAX_TOKENS", "1000"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.1"))  # Low = factual

    # ── RAG / FAISS Settings ──────────────────────────────────────────────────
    chunk_size: int = 800         # Characters per chunk
    chunk_overlap: int = 100      # Overlap between chunks
    retrieval_k: int = 5          # Documents to retrieve per query

    # ── Paths ─────────────────────────────────────────────────────────────────
    base_dir: str = os.path.dirname(os.path.abspath(__file__))
    results_dir: str = os.path.join(base_dir, "results")
    charts_dir: str = os.path.join(base_dir, "results", "charts")
    faiss_index_path: str = os.path.join(base_dir, "results", "faiss_index")
    doc_index_path: str = os.path.join(base_dir, "results", "doc_index.json")

    # ── App Settings ──────────────────────────────────────────────────────────
    save_results: bool = True
    verbose: bool = True

    def validate(self) -> None:
        """Raise ValueError for any missing required settings."""
        missing = []
        if not self.azure_openai_api_key:
            missing.append("AZURE_OPENAI_API_KEY")
        if not self.azure_openai_endpoint:
            missing.append("AZURE_OPENAI_ENDPOINT")
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}\n"
                "Copy .env.example to .env and fill in your Azure OpenAI credentials."
            )

    def print_config(self) -> None:
        """Print current configuration (no secrets)."""
        print("\n⚙️  CONFIGURATION (Azure OpenAI + FAISS RAG):")
        print(f"  Chat model:      {self.model_name}")
        print(f"  Embedding model: {self.embedding_model}")
        print(f"  Endpoint:        {self.azure_openai_endpoint}")
        print(f"  Chunk size:      {self.chunk_size} chars")
        print(f"  Retrieval K:     {self.retrieval_k}")
        print(f"  Results dir:     {self.results_dir}")


config = Config()
