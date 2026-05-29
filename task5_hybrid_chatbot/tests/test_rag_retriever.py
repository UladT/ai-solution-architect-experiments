"""Unit tests for hybrid RAG retriever."""

import unittest

from task5_hybrid_chatbot.rag.retriever import ResumeRAGRetriever


class ResumeRAGRetrieverTests(unittest.TestCase):
    """Test retrieval behavior using Task 2 sample corpus."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.retriever = ResumeRAGRetriever()

    def test_search_returns_hits_for_python_query(self) -> None:
        hits = self.retriever.search("Python Django FastAPI backend", top_k=3)
        self.assertGreaterEqual(len(hits), 1)
        top_categories = {hit.category for hit in hits}
        self.assertTrue(
            "Python Developer" in top_categories or "Data Science" in top_categories
        )

    def test_search_as_text_has_expected_format(self) -> None:
        text = self.retriever.search_as_text("Machine learning TensorFlow", top_k=2)
        self.assertIn("RAG results", text)
        self.assertIn("score=", text)


if __name__ == "__main__":
    unittest.main()
