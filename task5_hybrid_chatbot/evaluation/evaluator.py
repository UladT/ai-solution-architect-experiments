"""Quantitative evaluator for Task 5 (AC-3/AC-4)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvaluationResult:
    test_id: str
    question: str
    category: str
    expected_tool_prefix: str
    actual_tool_prefix: str
    tool_selection_correct: bool
    keyword_score: float
    completion_ok: bool
    second_attempt: bool
    improved: bool
    final_answer: str


class Evaluator:
    """Computes quantitative metrics for Task 5 outputs."""

    def __init__(self, orchestrator) -> None:
        self.orchestrator = orchestrator

    @staticmethod
    def _tool_prefix_from_calls(tool_calls: list[dict]) -> str:
        if not tool_calls:
            return "none"
        tool_name = str(tool_calls[0].get("tool", ""))
        return tool_name.split("__")[0] + "__" if "__" in tool_name else tool_name

    @staticmethod
    def _keyword_score(answer: str, expected_keywords: list[str]) -> float:
        if not expected_keywords:
            return 1.0
        text = (answer or "").lower()
        hits = sum(1 for k in expected_keywords if k.lower() in text)
        return hits / len(expected_keywords)

    @staticmethod
    def _completion_ok(answer: str) -> bool:
        text = (answer or "").strip()
        if len(text) < 24:
            return False
        bad_markers = ["i could not complete", "unknown tool", "no data returned"]
        lowered = text.lower()
        return not any(marker in lowered for marker in bad_markers)

    async def run_test_case(self, test_case: dict) -> EvaluationResult:
        answer = await self.orchestrator.query_with_improvement(
            test_case["question"], verbose=False
        )

        tool_calls = getattr(self.orchestrator, "last_tool_calls", [])
        improvement = getattr(self.orchestrator, "last_improvement_report", {})
        actual_tool_prefix = self._tool_prefix_from_calls(tool_calls)
        expected_tool_prefix = test_case["expected_tool_prefix"]
        tool_selection_correct = actual_tool_prefix == expected_tool_prefix

        keyword_score = self._keyword_score(answer, test_case["expected_keywords"])
        completion_ok = self._completion_ok(answer)

        return EvaluationResult(
            test_id=test_case["id"],
            question=test_case["question"],
            category=test_case["category"],
            expected_tool_prefix=expected_tool_prefix,
            actual_tool_prefix=actual_tool_prefix,
            tool_selection_correct=tool_selection_correct,
            keyword_score=keyword_score,
            completion_ok=completion_ok,
            second_attempt=bool(improvement.get("attempted", False)),
            improved=bool(improvement.get("improved", False)),
            final_answer=answer,
        )

    async def run_all(self, test_cases: list[dict]) -> list[EvaluationResult]:
        results = []
        for case in test_cases:
            results.append(await self.run_test_case(case))
        return results

    @staticmethod
    def aggregate_metrics(results: list[EvaluationResult]) -> dict:
        if not results:
            return {
                "m1_tool_selection_accuracy": 0.0,
                "m2_avg_keyword_score": 0.0,
                "m3_task_completion_rate": 0.0,
                "m4_improvement_success_rate": 0.0,
                "improvement_attempts": 0,
            }

        total = len(results)
        m1 = sum(1 for r in results if r.tool_selection_correct) / total
        m2 = sum(r.keyword_score for r in results) / total
        m3 = sum(1 for r in results if r.completion_ok) / total

        attempts = sum(1 for r in results if r.second_attempt)
        improved = sum(1 for r in results if r.improved)
        m4 = (improved / attempts) if attempts else 0.0

        return {
            "m1_tool_selection_accuracy": round(m1, 4),
            "m2_avg_keyword_score": round(m2, 4),
            "m3_task_completion_rate": round(m3, 4),
            "m4_improvement_success_rate": round(m4, 4),
            "improvement_attempts": attempts,
        }
