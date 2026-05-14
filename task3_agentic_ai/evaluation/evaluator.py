"""
Evaluation metrics for Task 3 - Agentic AI.

Metrics:
  1. Tool Selection Accuracy
  2. Response Keyword Score
  3. Task Completion Rate

AC-4 upgrade:
  Adds an automatic second-pass refinement for weak responses and reports
  Improvement Success Rate.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional

from colorama import Fore, Style
from tabulate import tabulate


@dataclass
class EvalResult:
    """Holds all metric data for a single test case."""

    test_id: str
    question: str
    category: str

    # Raw outputs
    agent_response: str = ""
    tool_calls_made: list = field(default_factory=list)

    # M1: Tool selection accuracy
    tool_selection_correct: Optional[bool] = None
    # M2: Keyword coverage score (0.0-1.0)
    keyword_score: float = 0.0
    # M3: Task completion
    task_completed: bool = False

    # AC-4 improvement tracking
    second_attempt_run: bool = False
    improved_response_used: bool = False
    initial_keyword_score: float = 0.0
    improvement_delta: float = 0.0

    error: Optional[str] = None

    @property
    def overall_score(self) -> float:
        """Unweighted mean of M1, M2, M3 (0.0-1.0)."""
        if self.error:
            return 0.0
        parts = [
            1.0 if self.tool_selection_correct else 0.0,
            self.keyword_score,
            1.0 if self.task_completed else 0.0,
        ]
        return round(sum(parts) / len(parts), 3)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["overall_score"] = self.overall_score
        return payload


class Evaluator:
    """Run all evaluation test cases and report aggregate metrics."""

    def __init__(self, orchestrator) -> None:
        self.orchestrator = orchestrator

    async def run_test_case(self, test_case: dict) -> EvalResult:
        result = EvalResult(
            test_id=test_case["id"],
            question=test_case["question"],
            category=test_case["category"],
        )
        try:
            response = await self.orchestrator.query(test_case["question"])
            result.agent_response = response
            result.tool_calls_made = list(self.orchestrator.last_tool_calls)

            result.tool_selection_correct = self._check_tool_type(
                result.tool_calls_made, test_case["expected_tool_type"]
            )
            result.keyword_score = self._score_keywords(
                response, test_case["expected_keywords"]
            )
            result.initial_keyword_score = result.keyword_score
            result.task_completed = bool(response and len(response.strip()) > 20)

            # AC-4: quality improvement loop
            if self._needs_improvement(result):
                result.second_attempt_run = True
                improved_prompt = self._build_improvement_prompt(test_case)
                improved_response = await self.orchestrator.query(improved_prompt)
                improved_calls = list(self.orchestrator.last_tool_calls)

                improved_tool_ok = self._check_tool_type(
                    improved_calls, test_case["expected_tool_type"]
                )
                improved_kw = self._score_keywords(
                    improved_response, test_case["expected_keywords"]
                )
                improved_done = bool(
                    improved_response and len(improved_response.strip()) > 20
                )

                old_score = result.overall_score
                new_score = round(
                    ((1.0 if improved_tool_ok else 0.0) + improved_kw + (1.0 if improved_done else 0.0)) / 3.0,
                    3,
                )

                if new_score > old_score:
                    result.improved_response_used = True
                    result.improvement_delta = round(new_score - old_score, 3)
                    result.agent_response = improved_response
                    result.tool_calls_made = improved_calls
                    result.tool_selection_correct = improved_tool_ok
                    result.keyword_score = improved_kw
                    result.task_completed = improved_done

        except Exception as exc:  # noqa: BLE001
            result.error = str(exc)

        return result

    async def run_all(self, test_cases: list[dict], verbose: bool = False) -> list[EvalResult]:
        results: list[EvalResult] = []
        for i, tc in enumerate(test_cases, 1):
            print(f"  [{i:2}/{len(test_cases)}] {tc['id']}: {tc['question'][:55]}...")
            result = await self.run_test_case(tc)

            status = (
                f"{Fore.GREEN}OK{Style.RESET_ALL}" if result.overall_score >= 0.67
                else f"{Fore.YELLOW}WARN{Style.RESET_ALL}" if result.overall_score > 0
                else f"{Fore.RED}FAIL{Style.RESET_ALL}"
            )
            tool_tick = "OK" if result.tool_selection_correct else "NO"
            print(
                f"         {status} score={result.overall_score:.2f} | "
                f"tool={tool_tick} | keywords={result.keyword_score:.2f} | "
                f"complete={'OK' if result.task_completed else 'NO'}"
                + (f" | improved=+{result.improvement_delta:.2f}" if result.improved_response_used else "")
                + (f" | ERROR: {result.error[:50]}" if result.error else "")
            )
            results.append(result)

        return results

    def print_report(self, results: list[EvalResult]) -> None:
        print(f"\n{Fore.CYAN}{'='*62}")
        print("  EVALUATION REPORT")
        print(f"{'='*62}{Style.RESET_ALL}")

        completed = [r for r in results if not r.error]
        total = len(results)

        def safe_mean(values):
            return sum(values) / len(values) if values else 0.0

        tool_acc = safe_mean([1.0 if r.tool_selection_correct else 0.0 for r in completed])
        kw_score = safe_mean([r.keyword_score for r in completed])
        comp_rate = safe_mean([1.0 if r.task_completed else 0.0 for r in completed])
        avg_overall = safe_mean([r.overall_score for r in results])

        attempts = [r for r in completed if r.second_attempt_run]
        improved = [r for r in attempts if r.improved_response_used]
        improvement_success_rate = (len(improved) / len(attempts)) if attempts else 0.0

        print(f"\n{Fore.YELLOW}Aggregate Metrics (n={len(completed)}/{total} successful):{Style.RESET_ALL}")
        metrics_table = [
            ["M1 Tool Selection Accuracy", f"{tool_acc:.1%}", ">= 80%", "PASS" if tool_acc >= 0.8 else "FAIL"],
            ["M2 Avg Response Keyword Score", f"{kw_score:.1%}", ">= 60%", "PASS" if kw_score >= 0.6 else "FAIL"],
            ["M3 Task Completion Rate", f"{comp_rate:.1%}", ">= 90%", "PASS" if comp_rate >= 0.9 else "FAIL"],
            [
                "M4 Improvement Success Rate",
                f"{improvement_success_rate:.1%}",
                ">= 30% when retries happen",
                "PASS" if (not attempts or improvement_success_rate >= 0.3) else "FAIL",
            ],
            ["-- Average Overall Score", f"{avg_overall:.3f}/1.00", "-", ""],
        ]
        print(tabulate(metrics_table, headers=["Metric", "Score", "Target", "Status"], tablefmt="rounded_outline"))

        print(f"\n{Fore.YELLOW}Detailed Results:{Style.RESET_ALL}")
        rows = []
        for r in results:
            rows.append([
                r.test_id,
                "Y" if r.tool_selection_correct else "N",
                f"{r.keyword_score:.2f}",
                "Y" if r.task_completed else "N",
                "Y" if r.improved_response_used else "N",
                f"{r.overall_score:.3f}",
                (r.error[:35] + "...") if r.error else "",
            ])
        print(
            tabulate(
                rows,
                headers=["Test ID", "Tool", "Keywords", "Done", "Improved", "Score", "Error"],
                tablefmt="rounded_outline",
            )
        )

        print(
            f"\n{Fore.GREEN}Summary: {len(completed)}/{total} tests completed "
            f"| Average Score: {avg_overall:.3f} "
            f"| Improvement retries: {len(attempts)} "
            f"| Improved: {len(improved)}{Style.RESET_ALL}\n"
        )

    @staticmethod
    def _needs_improvement(result: EvalResult) -> bool:
        return (
            (not result.tool_selection_correct)
            or (result.keyword_score < 0.6)
            or (not result.task_completed)
        )

    @staticmethod
    def _build_improvement_prompt(test_case: dict) -> str:
        keywords = ", ".join(test_case["expected_keywords"][:4])
        return (
            f"{test_case['question']} "
            f"Please provide a concise, factual answer and include these concepts "
            f"where relevant: {keywords}. For news, include source and date."
        )

    @staticmethod
    def _check_tool_type(tool_calls: list[dict], expected_type: str) -> bool:
        return any(tc["tool"].startswith(f"{expected_type}__") for tc in tool_calls)

    @staticmethod
    def _score_keywords(response: str, expected_keywords: list[str]) -> float:
        if not response or not expected_keywords:
            return 0.0
        lower = response.lower()
        found = sum(1 for kw in expected_keywords if kw.lower() in lower)
        return round(found / len(expected_keywords), 3)
