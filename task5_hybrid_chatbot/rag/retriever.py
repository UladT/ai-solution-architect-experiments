"""Lightweight RAG retriever reusing Task 2 document corpus patterns."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import List

from task2_rag_document_analysis.document_loader import DocumentLoader


@dataclass
class RetrievedChunk:
    """Single retrieved chunk."""

    doc_id: str
    category: str
    score: float
    text: str


class ResumeRAGRetriever:
    """Lexical RAG retriever over Task 2 sample resume corpus."""

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 80) -> None:
        loader = DocumentLoader()
        documents = loader.load_sample_data()
        chunks = loader.chunk_documents(documents, chunk_size, chunk_overlap)

        self._chunks: List[dict] = chunks
        self._chunk_vectors = [self._to_counter(c["page_content"]) for c in chunks]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[A-Za-z0-9_+.#-]+", text.lower())

    def _to_counter(self, text: str) -> Counter:
        return Counter(self._tokenize(text))

    @staticmethod
    def _cosine_sim(a: Counter, b: Counter) -> float:
        if not a or not b:
            return 0.0

        dot = sum(a[token] * b[token] for token in (a.keys() & b.keys()))
        if dot == 0:
            return 0.0

        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def search(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        """Retrieve top-k chunks for a query."""
        q_vec = self._to_counter(query)
        scored: list[RetrievedChunk] = []

        for chunk, vec in zip(self._chunks, self._chunk_vectors):
            score = self._cosine_sim(q_vec, vec)
            if score <= 0:
                continue
            meta = chunk.get("metadata", {})
            scored.append(
                RetrievedChunk(
                    doc_id=meta.get("doc_id", "unknown"),
                    category=meta.get("category", "Unknown"),
                    score=score,
                    text=chunk["page_content"],
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[: max(1, min(top_k, 10))]

    def search_as_text(self, query: str, top_k: int = 3) -> str:
        """Return retrieval result formatted for tool output."""
        hits = self.search(query=query, top_k=top_k)
        if not hits:
            return "No relevant RAG documents found for this question."

        lines = [f"RAG results ({len(hits)}):"]
        for i, hit in enumerate(hits, start=1):
            preview = hit.text[:280].replace("\n", " ")
            lines.append(
                f"{i}. category={hit.category}; score={hit.score:.3f}; "
                f"doc_id={hit.doc_id}; snippet={preview}"
            )
        return "\n".join(lines)
