"""
RAG Engine - Task 2 Resume Intelligence Platform

LangChain + FAISS Retrieval-Augmented Generation engine.

AC-2 Scalability:
  - FAISS supports adding vectors incrementally (no full rebuild).
  - Works with 100 or 100,000 documents — just call add_documents().

Ninja Challenges:
  1) Corpus update w/o Vector DB rebuild:
       engine.add_documents(new_chunks)   ← appends to existing FAISS index
  2) Access control aware RAG:
       engine.query(query, role="hr_viewer")  ← post-filters by allowed categories
"""

import os
import math
import re
import hashlib
from typing import List, Dict, Optional, Tuple

from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import PromptTemplate

from config import config


# ─────────────────────────────────────────────────────────────────────────────
# Role → Allowed resume categories (None = unrestricted)
# AC-5: Access control aware RAG
# ─────────────────────────────────────────────────────────────────────────────
ROLE_PERMISSIONS: Dict[str, Optional[List[str]]] = {
    "hr_admin": None,  # All categories, full PII visible
    "hr_viewer": [     # Job-category search only; PII will be masked downstream
        "Data Science", "Python Developer", "Java Developer",
        "DevOps Engineer", "Network Security Engineer",
        "Web Designing", "Business Analyst",
    ],
    "candidate": None,  # Own documents only (filtered by user_id in metadata)
}

RAG_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a resume intelligence assistant. Use ONLY the provided resume
excerpts to answer the question. If the information is not in the context, say
"I don't have enough information in the retrieved documents."

Resume excerpts:
{context}

Question: {question}

Answer (be specific, cite the candidate name and category when relevant):""",
)


class LocalHashEmbeddings(Embeddings):
    """
    Deterministic local embedding fallback used when remote embedding APIs are blocked.
    """

    def __init__(self, dim: int = 1536) -> None:
        self.dim = dim

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text)

    def _embed_text(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        tokens = re.findall(r"[A-Za-z0-9_+.#-]+", text.lower())
        if not tokens:
            return vec

        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            idx = int(digest, 16) % self.dim
            vec[idx] += 1.0

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


class RAGEngine:
    """
    LangChain + FAISS RAG engine for resume intelligence.

    Usage:
        engine = RAGEngine()
        engine.build_index(chunks)                    # first run
        engine.add_documents(new_chunks)              # incremental update
        answer, sources = engine.query("Python ML")
    """

    def __init__(self) -> None:
        # Default to local embeddings for reliability in restricted environments.
        # Set USE_AZURE_EMBEDDINGS=true to force Azure embedding usage.
        use_azure_embeddings = os.getenv("USE_AZURE_EMBEDDINGS", "false").lower() == "true"
        self._embeddings: Embeddings = LocalHashEmbeddings()
        self._embedding_mode = "local-hash"
        if use_azure_embeddings:
            try:
                self._embeddings = AzureOpenAIEmbeddings(
                    azure_deployment=config.embedding_model,
                    azure_endpoint=config.azure_openai_endpoint,
                    api_key=config.azure_openai_api_key,
                    api_version=config.azure_openai_api_version,
                )
                self._embedding_mode = "azure"
            except Exception as exc:  # noqa: BLE001
                if config.verbose:
                    print(
                        "  ! Could not initialize Azure embeddings; "
                        f"using local-hash instead: {exc}"
                    )
        self._llm = AzureChatOpenAI(
            azure_deployment=config.model_name,
            azure_endpoint=config.azure_openai_endpoint,
            api_key=config.azure_openai_api_key,
            api_version=config.azure_openai_api_version,
            temperature=config.temperature,
        )
        self._vectorstore: Optional[FAISS] = None

    # ── Index management ──────────────────────────────────────────────────────

    def _dicts_to_documents(self, chunks: List[Dict]) -> List[Document]:
        """Convert loader chunk dicts to LangChain Document objects."""
        return [
            Document(
                page_content=c["page_content"],
                metadata=c["metadata"],
            )
            for c in chunks
        ]

    def build_index(self, chunks: List[Dict]) -> None:
        """
        Build a new FAISS index from scratch and persist it to disk.
        Called once on first run.
        """
        if not chunks:
            raise ValueError("Cannot build index from empty chunk list.")

        docs = self._dicts_to_documents(chunks)
        try:
            self._vectorstore = FAISS.from_documents(docs, self._embeddings)
        except Exception as exc:  # noqa: BLE001
            # Common in restricted tenants: embeddings endpoint returns 403.
            if config.verbose:
                print(
                    "  ! Azure embeddings unavailable; "
                    "falling back to local hash embeddings."
                )
                print(f"    Reason: {exc}")
            self._embeddings = LocalHashEmbeddings()
            self._embedding_mode = "local-hash"
            self._vectorstore = FAISS.from_documents(docs, self._embeddings)
        self._save_index()
        if config.verbose:
            print(
                f"  ✓ Built FAISS index: {len(docs)} chunks "
                f"(embeddings={self._embedding_mode})"
            )

    def add_documents(self, new_chunks: List[Dict]) -> int:
        """
        NINJA: Incremental corpus update — append new docs to existing index
        without a full rebuild.  O(new_docs) rather than O(total_docs).

        Returns:
            Number of chunks added.
        """
        if not new_chunks:
            return 0

        self._ensure_index()
        new_docs = self._dicts_to_documents(new_chunks)
        self._vectorstore.add_documents(new_docs)  # type: ignore[union-attr]
        self._save_index()
        return len(new_docs)

    def load_index(self) -> bool:
        """Load a previously saved FAISS index from disk. Returns True if found."""
        index_path = config.faiss_index_path
        if os.path.exists(index_path):
            self._vectorstore = FAISS.load_local(
                index_path,
                self._embeddings,
                allow_dangerous_deserialization=True,
            )
            if config.verbose:
                print(f"  ✓ Loaded existing FAISS index from {index_path}")
            return True
        return False

    def _save_index(self) -> None:
        os.makedirs(config.results_dir, exist_ok=True)
        if self._vectorstore:
            self._vectorstore.save_local(config.faiss_index_path)

    def _ensure_index(self) -> None:
        if self._vectorstore is None:
            raise RuntimeError(
                "Index not initialized. Call build_index() or load_index() first."
            )

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        k: int = 0,
        role: str = "hr_admin",
        user_id: Optional[str] = None,
    ) -> List[Document]:
        """
        Retrieve top-k relevant document chunks.

        NINJA (access control): Post-filters results based on role permissions.
          - hr_admin : sees all results
          - hr_viewer: results limited to allowed categories
          - candidate : results limited to own doc_id

        Args:
            query:   Natural-language search query.
            k:       Number of results (0 = use config.retrieval_k).
            role:    Requester role for access control.
            user_id: For 'candidate' role — only return their own docs.
        """
        self._ensure_index()
        k = k or config.retrieval_k

        # Fetch extra candidates to absorb filtering losses
        fetch_k = k * 4 if role != "hr_admin" else k
        raw_docs: List[Document] = self._vectorstore.similarity_search(  # type: ignore
            query, k=fetch_k
        )

        filtered = self._apply_access_control(raw_docs, role, user_id)
        return filtered[:k]

    @staticmethod
    def _apply_access_control(
        docs: List[Document],
        role: str,
        user_id: Optional[str],
    ) -> List[Document]:
        """Filter documents based on role permissions (AC-5)."""
        allowed_categories = ROLE_PERMISSIONS.get(role)

        result = []
        for doc in docs:
            category = doc.metadata.get("category", "")
            doc_id = doc.metadata.get("doc_id", "")

            if role == "candidate":
                # Candidates see only their own documents
                if user_id and doc_id == user_id:
                    result.append(doc)
            elif allowed_categories is not None:
                if category in allowed_categories:
                    result.append(doc)
            else:
                result.append(doc)

        return result

    # ── Generation ────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        k: int = 0,
        role: str = "hr_admin",
        user_id: Optional[str] = None,
    ) -> Tuple[str, List[Document]]:
        """
        Full RAG: retrieve relevant chunks then generate a grounded answer.

        Returns:
            (answer_text, source_documents)
        """
        sources = self.retrieve(question, k=k, role=role, user_id=user_id)
        if not sources:
            return "No relevant documents found for the given query and access role.", []

        context = "\n\n---\n\n".join(
            f"[{doc.metadata.get('category', 'Unknown')}] {doc.page_content}"
            for doc in sources
        )

        prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
        try:
            response = self._llm.invoke(prompt)
            answer = response.content if hasattr(response, "content") else str(response)
        except Exception as exc:  # noqa: BLE001
            if config.verbose:
                print(f"  ! LLM answer generation failed, using fallback: {exc}")
            answer = self._fallback_answer(question, sources)

        return answer, sources

    @staticmethod
    def _fallback_answer(question: str, sources: List[Document]) -> str:
        """Simple extractive fallback answer when chat API is unavailable."""
        snippets = []
        for doc in sources[:3]:
            category = doc.metadata.get("category", "Unknown")
            snippet = " ".join(doc.page_content.split())[:220]
            snippets.append(f"[{category}] {snippet}")

        return (
            "Generated using extractive fallback because chat generation is unavailable. "
            f"Question: {question}\n"
            "Top evidence:\n- "
            + "\n- ".join(snippets)
        )

    def similarity_scores(
        self, query: str, k: int = 0
    ) -> List[Tuple[Document, float]]:
        """Return documents with their cosine similarity scores (for evaluation)."""
        self._ensure_index()
        k = k or config.retrieval_k
        return self._vectorstore.similarity_search_with_score(query, k=k)  # type: ignore
