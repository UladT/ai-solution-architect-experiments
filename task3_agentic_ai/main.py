"""
Task 3: Agentic AI – Weather & News Assistant
==============================================
Demonstrates:
  • MCP server orchestration (Open-Meteo + GNews.io)
  • ReAct-style agent loop with Azure OpenAI tool calling
  • Three quantitative evaluation metrics over a 12-question dataset

Usage:
  python main.py                    # demo questions + full evaluation
  python main.py --mode interactive # chat loop
  python main.py --mode evaluate    # evaluation only
  python main.py --question "..."   # single question
"""

import asyncio
import json
import os
import sys
import argparse
from datetime import datetime

from colorama import Fore, Style, init

# Allow running from the task directory directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from agent.orchestrator import AgentOrchestrator
from evaluation.dataset import TEST_CASES
from evaluation.evaluator import Evaluator

init(autoreset=True)

DEMO_QUESTIONS = [
    "What is the current weather in Amsterdam?",
    "What are the latest news about artificial intelligence?",
]


# ──────────────────────────────────────────────────────────────────────────── #
# Banner                                                                        #
# ──────────────────────────────────────────────────────────────────────────── #

def print_banner() -> None:
    print(f"\n{Fore.CYAN}{'='*62}")
    print("  TASK 3: AGENTIC AI – WEATHER & NEWS ASSISTANT")
    print("  AI Solution Architect Program – EPAM")
    print(f"{'='*62}{Style.RESET_ALL}")
    print(f"\n{Fore.YELLOW}Architecture:")
    print("  Agent Orchestrator  → Azure OpenAI (gpt-4) tool calling")
    print("  Weather MCP Server  → Open-Meteo API  (no API key)")
    print("  News MCP Server     → GNews.io API    (free-tier key)")
    print(
        f"  Evaluation          → 3 quantitative metrics, "
        f"{len(TEST_CASES)} test cases"
    )
    print(f"{Style.RESET_ALL}")


# ──────────────────────────────────────────────────────────────────────────── #
# Modes                                                                         #
# ──────────────────────────────────────────────────────────────────────────── #

async def run_demo(orchestrator: AgentOrchestrator) -> None:
    """Run a small set of demo questions to verify end-to-end flow."""
    print(f"{Fore.CYAN}── Demo Questions {'─'*43}{Style.RESET_ALL}")

    for question in DEMO_QUESTIONS:
        print(f"\n{Fore.YELLOW}Q: {question}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Agent is thinking…{Style.RESET_ALL}")
        try:
            response = await orchestrator.query(question, verbose=True)
            preview = response[:400] + ("…" if len(response) > 400 else "")
            print(f"\n{Fore.CYAN}A:{Style.RESET_ALL} {preview}\n")
        except Exception as exc:
            print(f"{Fore.RED}  Error: {exc}{Style.RESET_ALL}\n")


async def run_interactive(orchestrator: AgentOrchestrator) -> None:
    """Interactive chat loop – user types questions, agent answers."""
    print(f"{Fore.CYAN}── Interactive Mode (type 'quit' to exit) {'─'*18}{Style.RESET_ALL}")
    print(f"\n{Fore.YELLOW}Example questions:")
    for q in DEMO_QUESTIONS:
        print(f"  • {q}")
    print(f"{Style.RESET_ALL}")

    while True:
        try:
            question = input(f"{Fore.GREEN}You: {Style.RESET_ALL}").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if question.lower() in ("quit", "exit", "q", ""):
            break

        print(f"\n{Fore.YELLOW}Agent is thinking…{Style.RESET_ALL}")
        try:
            response = await orchestrator.query(question, verbose=True)
            print(f"\n{Fore.CYAN}Assistant:{Style.RESET_ALL}\n{response}\n")
        except Exception as exc:
            print(f"{Fore.RED}  Error: {exc}{Style.RESET_ALL}\n")


async def run_evaluation(orchestrator: AgentOrchestrator) -> None:
    """Run all evaluation test cases and print the metrics report."""
    print(f"{Fore.CYAN}── Evaluation ({len(TEST_CASES)} test cases) {'─'*36}{Style.RESET_ALL}\n")

    evaluator = Evaluator(orchestrator)
    results = await evaluator.run_all(TEST_CASES)
    evaluator.print_report(results)

    if config.save_results:
        os.makedirs(config.results_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(config.results_dir, f"evaluation_{ts}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump([r.to_dict() for r in results], fh, indent=2, ensure_ascii=False)
        print(f"  💾 Results saved: {path}")


# ──────────────────────────────────────────────────────────────────────────── #
# Entry point                                                                   #
# ──────────────────────────────────────────────────────────────────────────── #

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agentic AI – Weather & News Assistant"
    )
    parser.add_argument(
        "--mode",
        choices=["interactive", "evaluate", "both"],
        default="both",
        help="Run mode (default: both = demo + evaluate)",
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Answer a single question and exit.",
    )
    args = parser.parse_args()

    print_banner()

    try:
        config.validate()
        config.print_config()
    except ValueError as exc:
        print(f"{Fore.RED}❌ Configuration Error: {exc}{Style.RESET_ALL}")
        sys.exit(1)

    orchestrator = AgentOrchestrator(config)

    if args.question:
        print(f"{Fore.YELLOW}Q: {args.question}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Agent is thinking…{Style.RESET_ALL}")
        response = await orchestrator.query(args.question, verbose=True)
        print(f"\n{Fore.CYAN}A:{Style.RESET_ALL}\n{response}\n")

    elif args.mode == "interactive":
        await run_interactive(orchestrator)

    elif args.mode == "evaluate":
        await run_evaluation(orchestrator)

    else:  # "both"
        await run_demo(orchestrator)
        print(f"\n{Fore.YELLOW}{'─'*62}{Style.RESET_ALL}\n")
        await run_evaluation(orchestrator)


if __name__ == "__main__":
    asyncio.run(main())
