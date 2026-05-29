"""Task 5 entrypoint: Hybrid chatbot (RAG + Agent/MCP + disasters)."""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime

from colorama import Fore, Style, init

# Allow running directly from task folder.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")),
)

from config import config
from agent.orchestrator import HybridChatOrchestrator
from evaluation.dataset import TEST_CASES
from evaluation.evaluator import Evaluator

init(autoreset=True)

BANNER = """
╔═════════════════════════════════════════════════════════════════════╗
║ TASK 5: HYBRID CHATBOT (RAG + MCP AGENT + DISASTERS)              ║
╠═════════════════════════════════════════════════════════════════════╣
║ Sources:                                                           ║
║  1) RAG resume corpus (from Task 2)                               ║
║  2) Weather + News MCP tools (from Task 3)                        ║
║  3) Disaster MCP tool (new, pandas over disaster CSV files)       ║
╚═════════════════════════════════════════════════════════════════════╝
"""


async def run_interactive(orchestrator: HybridChatOrchestrator) -> None:
    print(f"{Fore.CYAN}Interactive mode. Type 'quit' to exit.{Style.RESET_ALL}")
    print(
        f"{Fore.YELLOW}Try: 'Top flood disasters in India since 2000 by deaths'\n"
        f"Try: 'Find Python developers with FastAPI experience'\n"
        f"Try: 'Weather in Lisbon today and latest AI headlines'{Style.RESET_ALL}"
    )

    while True:
        try:
            question = input(f"\n{Fore.GREEN}You: {Style.RESET_ALL}").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if question.lower() in {"quit", "exit", "q", ""}:
            break

        answer = await orchestrator.query(question, verbose=True)
        print(f"\n{Fore.CYAN}Assistant:{Style.RESET_ALL}\n{answer}")


async def run_single_question(orchestrator: HybridChatOrchestrator, question: str) -> None:
    print(f"{Fore.YELLOW}Q: {question}{Style.RESET_ALL}")
    answer = await orchestrator.query(question, verbose=True)
    print(f"\n{Fore.CYAN}A:{Style.RESET_ALL}\n{answer}")


async def run_evaluation(orchestrator: HybridChatOrchestrator) -> None:
    """Run AC-3 metrics and AC-4 improvement evaluation over fixed dataset."""
    print(f"{Fore.CYAN}Running evaluation for {len(TEST_CASES)} test cases...{Style.RESET_ALL}")
    evaluator = Evaluator(orchestrator)
    results = await evaluator.run_all(TEST_CASES)
    metrics = evaluator.aggregate_metrics(results)

    print(f"\n{Fore.YELLOW}Aggregate Metrics:{Style.RESET_ALL}")
    print(
        f"  M1 Tool Selection Accuracy:    "
        f"{metrics['m1_tool_selection_accuracy'] * 100:.1f}%"
    )
    print(
        f"  M2 Avg Keyword Score:          "
        f"{metrics['m2_avg_keyword_score'] * 100:.1f}%"
    )
    print(
        f"  M3 Task Completion Rate:       "
        f"{metrics['m3_task_completion_rate'] * 100:.1f}%"
    )
    print(
        f"  M4 Improvement Success Rate:   "
        f"{metrics['m4_improvement_success_rate'] * 100:.1f}% "
        f"(attempts={metrics['improvement_attempts']})"
    )

    os.makedirs(config.results_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(config.results_dir, f"task5_eval_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(
            {
                "metrics": metrics,
                "results": [r.__dict__ for r in results],
            },
            fp,
            indent=2,
        )
    print(f"\n{Fore.GREEN}Saved evaluation report: {out_path}{Style.RESET_ALL}")


async def amain() -> None:
    parser = argparse.ArgumentParser(description="Task 5 hybrid chatbot")
    parser.add_argument("--question", type=str, help="Ask one question and exit")
    parser.add_argument(
        "--mode",
        choices=["interactive", "evaluate"],
        default="interactive",
        help="Run mode",
    )
    args = parser.parse_args()

    print(BANNER)

    try:
        config.validate()
    except ValueError as exc:
        print(f"{Fore.RED}Configuration error: {exc}{Style.RESET_ALL}")
        raise SystemExit(1) from exc

    orchestrator = HybridChatOrchestrator(config)

    if args.question:
        await run_single_question(orchestrator, args.question)
        return

    if args.mode == "evaluate":
        await run_evaluation(orchestrator)
        return

    await run_interactive(orchestrator)


if __name__ == "__main__":
    asyncio.run(amain())
