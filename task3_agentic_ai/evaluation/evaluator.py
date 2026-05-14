"""
Evaluation metrics for Task 3 – Agentic AI.

Three quantitative metrics are defined, evaluated, and reported:

  1. Tool Selection Accuracy   – Did the agent call a tool of the right type
                                 (weather vs. news)?  Binary per test case.
  2. Response Keyword Score    – Fraction of expected topic keywords that appear
                                 in the final answer.  0.0–1.0 per test case.
  3. Task Completion Rate      – Did the agent produce a meaningful final answer
                                 (non-empty, > 20 characters)?  Binary.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Optional

from colorama import Fore, Style
from tabulate import tabulate


# ──────────────────────────────────────────────────────────────────────────── #
# Data classes                                                                  #
# ──────────────────────────────────────────────────────────────────────────── #

@dataclass
class EvalResult:
    """Holds all metric data for a single test case."""

    test_id: str
    question: str
    category: str

    # Raw outputs
    agent_response: str = ""
    tool_calls_made: list = field(default_factory=list)

    # ── Metrics ────────────────────────────────────────────────────────── #
    # M1: Tool Selection Accuracy
    tool_selection_correct: Optional[bool] = None

    # M2: Response Keyword Score (0.0–1.0)
    keyword_score: float = 0.0

    # M3: Task Completion (bool)
    task_completed: bool = False

    error: Optional[str] = None

    # ── Derived score ──────────────────────────────────────────────────── #
    @property
    def overall_score(self) -> float:
        """Unweighted mean of the three metric scores (0.0–1.0)."""
        if self.error:
            return 0.0
        parts = [
            1.0 if self.tool_selection_correct else 0.0,
            self.keyword_score,
            1.0 if self.task_completed else 0.0,
        ]
        return round(sum(parts) / len(parts), 3)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["overall_score"] = self.overall_score
        return d


# ──────────────────────────────────────────────────────────────────────────── #
# Evaluator                                                                     #
# ──────────────────────────────────────────────────────────────────────────── #

class Evaluator:
    """
    Runs the evaluation dataset against the AgentOrchestrator and produces
    a structured report with per-test and aggregate metrics.
    """

    def __init__(self, orchestrator) -> None:
        self.orchestrator = orchestrator

    # ── Public API ─────────────────────────────────────────────────────── #

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

            # M1
            result.tool_selection_correct = self._check_tool_type(
                result.tool_calls_made, test_case["expected_tool_type"]
            )
            # M2
            result.keyword_score = self._score_keywords(
                response, test_case["expected_keywords"]
            )
            # M3
            result.task_completed = bool(response and len(response.strip()) > 20)

        except Exception as exc:  # noqa: BLE001
            result.error = str(exc)

        return result

    async def run_all(
        self,
        test_cases: list[dict],
        verbose: bool = False,
    ) -> list[EvalResult]:
        """Run every test case and return a list of EvalResult objects."""
        results: list[EvalResult] = []
        for i, tc in enumerate(test_cases, 1):
            print(
                f"  [{i:2}/{len(test_cases)}] {tc['id']}: "
                f"{tc['question'][:55]}…"
            )
            result = await self.run_test_case(tc)

            status = (
                f"{Fore.GREEN}✅{Style.RESET_ALL}" if result.overall_score >= 0.67
                else f"{Fore.YELLOW}⚠️{Style.RESET_ALL}" if result.overall_score > 0
                else f"{Fore.RED}❌{Style.RESET_ALL}"
            )
            tool_tick = (
                f"{Fore.GREEN}✓{Style.RESET_ALL}"
                if result.tool_selection_correct
                else f"{Fore.RED}✗{Style.RESET_ALL}"
            )
            print(
                f"         {status} score={result.overall_score:.2f} | "
                f"tool={tool_tick} | "
                f"keywords={result.keyword_score:.2f} | "
                f"complete={'✓' if result.task_completed else '✗'}"
                + (f" | ERROR: {result.error[:50]}" if result.error else "")
            )
            results.append(result)

        return results

    def print_report(self, results: list[EvalResult]) -> None:
        """Print a formatted evaluation report with aggregate metrics."""
        print(f"\n{Fore.CYAN}{'='*62}")
        print("  EVALUATION REPORT")
        print(f"{'='*62}{Style.RESET_ALL}")

        completed = [r for r in results if not r.error]
        total = len(results)

        # ── Aggregate metrics ─────────────────────────────────────────── #
        def safe_mean(values):
            return sum(values) / len(values) if values else 0.0

        tool_acc = safe_mean(
            [1.0 if r.tool_selection_correct else 0.0 for r in completed]
        )
        kw_score = safe_mean([r.keyword_score for r in completed])
        comp_rate = safe_mean([1.0 if r.task_completed else 0.0 for r in completed])
        avg_overall = safe_mean([r.overall_score for r in results])

        print(f"\n{Fore.YELLOW}Aggregate Metrics (n={len(completed)}/{total} successful):{Style.RESET_ALL}")
        metrics_table = [
            ["M1 Tool Selection Accuracy",
             f"{tool_acc:.1%}", "≥ 80%",
             f"{Fore.GREEN}PASS{Style.RESET_ALL}" if tool_acc >= 0.8
             else f"{Fore.RED}FAIL{Style.RESET_ALL}"],
            ["M2 Avg Response Keyword Score",
             f"{kw_score:.1%}", "≥ 60%",
             f"{Fore.GREEN}PASS{Style.RESET_ALL}" if kw_score >= 0.6
             else f"{Fore.RED}FAIL{Style.RESET_ALL}"],
            ["M3 Task Completion Rate",
             f"{comp_rate:.1%}", "≥ 90%",
             f"{Fore.GREEN}PASS{Style.RESET_ALL}" if comp_rate >= 0.9
             else f"{Fore.RED}FAIL{Style.RESET_ALL}"],
            ["── Average Overall Score",
             f"{avg_overall:.3f}/1.00", "—", ""],
        ]
        print(tabulate(
            metrics_table,
            headers=["Metric", "Score", "Target", "Status"],
            tablefmt="rounded_outline",
        ))

        # ── Category breakdown ────────────────────────────────────────── #
        print(f"\n{Fore.YELLOW}By Category:{Style.RESET_ALL}")
        for cat in ("weather", "news"):
            cat_r = [r for r in results if r.category == cat]
            if cat_r:
                cat_tool = safe_mean(
                    [1.0 if r.tool_selection_correct else 0.0 for r in cat_r]
                )
                cat_kw = safe_mean([r.keyword_score for r in cat_r])
                cat_avg = safe_mean([r.overall_score for r in cat_r])
                print(
                    f"  {cat.capitalize():8} | avg={cat_avg:.3f} | "
                    f"tool={cat_tool:.1%} | keywords={cat_kw:.1%}"
                )

        # ── Per-test table ────────────────────────────────────────────── #
        print(f"\n{Fore.YELLOW}Detailed Results:{Style.RESET_ALL}")
        rows = []
        for r in results:
            rows.append([
                r.test_id,
                "✓" if r.tool_selection_correct else "✗",
                f"{r.keyword_score:.2f}",
                "✓" if r.task_completed else "✗",
                f"{r.overall_score:.3f}",
                (r.error[:35] + "…") if r.error else "",
            ])
        print(tabulate(
            rows,
            headers=["Test ID", "Tool✓", "Keywords", "Done", "Score", "Error"],
            tablefmt="rounded_outline",
        ))

        print(
            f"\n{Fore.GREEN}Summary: {len(completed)}/{total} tests completed "
            f"| Average Score: {avg_overall:.3f}{Style.RESET_ALL}\n"
        )

    # ── Internal helpers ───────────────────────────────────────────────── #

    @staticmethod
    def _check_tool_type(tool_calls: list[dict], expected_type: str) -> bool:
        """Return True if any tool call belongs to the expected server type."""
        return any(
            tc["tool"].startswith(f"{expected_type}__")
            for tc in tool_calls
        )

    @staticmethod
    def _score_keywords(response: str, expected_keywords: list[str]) -> float:
        """Return the fraction of expected keywords found in the response."""
        if not response or not expected_keywords:
            return 0.0
        lower = response.lower()
        found = sum(1 for kw in expected_keywords if kw.lower() in lower)
        return round(found / len(expected_keywords), 3)
