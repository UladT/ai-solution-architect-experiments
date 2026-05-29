"""Unit tests for Task 5 evaluator metrics and AC-4 improvement flow."""

import unittest

from task5_hybrid_chatbot.evaluation.evaluator import Evaluator, EvaluationResult


class FakeOrchestrator:
    def __init__(self) -> None:
        self.last_tool_calls = []
        self.last_improvement_report = {}

    async def query_with_improvement(self, question: str, verbose: bool = False) -> str:
        if "flood" in question.lower():
            self.last_tool_calls = [{"tool": "disaster__get_disaster_summary", "args": {}}]
            self.last_improvement_report = {"attempted": True, "improved": True}
            return "India flood events summary with deaths and affected people"

        self.last_tool_calls = [{"tool": "rag__search_documents", "args": {}}]
        self.last_improvement_report = {"attempted": False, "improved": False}
        return "Python candidate profiles with FastAPI and Django"


class EvaluatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_aggregate_metrics_computation(self) -> None:
        results = [
            EvaluationResult(
                test_id="1",
                question="q1",
                category="rag",
                expected_tool_prefix="rag__",
                actual_tool_prefix="rag__",
                tool_selection_correct=True,
                keyword_score=1.0,
                completion_ok=True,
                second_attempt=False,
                improved=False,
                final_answer="ok",
            ),
            EvaluationResult(
                test_id="2",
                question="q2",
                category="disaster",
                expected_tool_prefix="disaster__",
                actual_tool_prefix="disaster__",
                tool_selection_correct=True,
                keyword_score=0.66,
                completion_ok=True,
                second_attempt=True,
                improved=True,
                final_answer="ok",
            ),
        ]

        metrics = Evaluator.aggregate_metrics(results)
        self.assertEqual(metrics["m1_tool_selection_accuracy"], 1.0)
        self.assertEqual(metrics["m3_task_completion_rate"], 1.0)
        self.assertEqual(metrics["m4_improvement_success_rate"], 1.0)
        self.assertEqual(metrics["improvement_attempts"], 1)

    async def test_run_test_case_uses_improvement_fields(self) -> None:
        orchestrator = FakeOrchestrator()
        evaluator = Evaluator(orchestrator)

        case = {
            "id": "d1",
            "question": "Top flood disasters in India",
            "expected_tool_prefix": "disaster__",
            "expected_keywords": ["india", "flood", "deaths"],
            "category": "disaster",
        }

        result = await evaluator.run_test_case(case)
        self.assertTrue(result.tool_selection_correct)
        self.assertTrue(result.second_attempt)
        self.assertTrue(result.improved)


if __name__ == "__main__":
    unittest.main()
