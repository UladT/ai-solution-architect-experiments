"""Hybrid orchestrator that combines Task 2 RAG and Task 3 MCP agent patterns."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AzureOpenAI

try:
    from task5_hybrid_chatbot.rag.retriever import ResumeRAGRetriever
    from task5_hybrid_chatbot.security_guard import SecurityGuard
except ModuleNotFoundError:
    from rag.retriever import ResumeRAGRetriever
    from security_guard import SecurityGuard


TASK3_SERVERS_DIR = Path(__file__).resolve().parents[2] / "task3_agentic_ai" / "mcp_servers"
TASK5_SERVERS_DIR = Path(__file__).resolve().parents[1] / "mcp_servers"

SYSTEM_PROMPT = (
    "You are a hybrid chatbot that can answer from 3 sources:\n"
    "1) RAG resume knowledge base (for hiring, resume, skills questions)\n"
    "2) MCP weather/news tools\n"
    "3) MCP natural disaster tools (historical disaster analytics from CSV)\n"
    "Guidelines:\n"
    "- For weather questions, call weather tools.\n"
    "- For latest headlines/news questions, call news tools.\n"
    "- For natural disaster questions (flood, earthquake, deaths, country disasters), call disaster tools.\n"
    "- For resume/skills/candidate questions, call rag__search_documents before answering.\n"
    "- Never fabricate facts. If tool data is missing, explain that clearly."
)

MAX_ITERATIONS = 10


class HybridChatOrchestrator:
    """OpenAI tool-calling loop over MCP and local RAG tools."""

    def __init__(self, config) -> None:
        self.config = config
        self.client = AzureOpenAI(
            api_key=config.azure_openai_api_key,
            api_version=config.azure_openai_api_version,
            azure_endpoint=config.azure_openai_endpoint,
        )
        self.rag = ResumeRAGRetriever()
        self.security_guard = SecurityGuard()
        self.last_tool_calls: list[dict] = []
        self.last_security_report: dict = {}
        self.last_improvement_report: dict = {
            "attempted": False,
            "improved": False,
            "initial_score": 0.0,
            "final_score": 0.0,
        }

    async def query(self, question: str, verbose: bool = False) -> str:
        """Answer one user question with security checks and improvement loop."""
        return await self.query_with_improvement(question=question, verbose=verbose)

    async def query_with_improvement(self, question: str, verbose: bool = False) -> str:
        """Answer a question and optionally retry when quality is weak (AC-4)."""
        self.last_tool_calls = []
        self.last_improvement_report = {
            "attempted": False,
            "improved": False,
            "initial_score": 0.0,
            "final_score": 0.0,
        }

        input_decision = self.security_guard.validate_input(question)
        self.last_security_report = {
            "input_safe": input_decision.is_safe,
            "blocked": input_decision.blocked,
            "threats_found": input_decision.threats_found,
            "warnings": input_decision.warnings,
            "output_issues": [],
            "tool_arg_issues": [],
        }

        if input_decision.blocked:
            return (
                "I cannot process that request because it appears unsafe or "
                "prompt-injection related. Please ask a normal question about "
                "resumes, weather, news, or natural disasters."
            )

        safe_question = input_decision.sanitized_input

        answer = await self._run_query_once(safe_question, verbose=verbose)
        initial_score = self._quality_score(answer)
        self.last_improvement_report["initial_score"] = initial_score

        if self._needs_improvement(answer):
            self.last_improvement_report["attempted"] = True
            refined_question = (
                f"{safe_question}\n\n"
                "Please provide a complete factual answer with concrete details "
                "from the most relevant tools and avoid vague wording."
            )

            improved_answer = await self._run_query_once(refined_question, verbose=False)
            final_score = self._quality_score(improved_answer)
            self.last_improvement_report["final_score"] = final_score

            if final_score > initial_score:
                self.last_improvement_report["improved"] = True
                answer = improved_answer
            else:
                self.last_improvement_report["improved"] = False
                self.last_improvement_report["final_score"] = initial_score
        else:
            self.last_improvement_report["final_score"] = initial_score

        safe_answer, output_issues = self.security_guard.validate_output(answer)
        self.last_security_report["output_issues"] = output_issues
        return safe_answer

    async def _run_query_once(self, question: str, verbose: bool = False) -> str:
        """Execute one end-to-end tool-calling cycle without AC-4 retries."""
        self.last_tool_calls = []

        weather_params = StdioServerParameters(
            command=sys.executable,
            args=[str(TASK3_SERVERS_DIR / "weather_server.py")],
        )
        news_params = StdioServerParameters(
            command=sys.executable,
            args=[str(TASK3_SERVERS_DIR / "news_server.py")],
        )
        disaster_params = StdioServerParameters(
            command=sys.executable,
            args=[str(TASK5_SERVERS_DIR / "disaster_server.py")],
        )

        async with stdio_client(weather_params) as (wr, ww):
            async with ClientSession(wr, ww) as weather_session:
                await weather_session.initialize()

                async with stdio_client(news_params) as (nr, nw):
                    async with ClientSession(nr, nw) as news_session:
                        await news_session.initialize()

                        async with stdio_client(disaster_params) as (dr, dw):
                            async with ClientSession(dr, dw) as disaster_session:
                                await disaster_session.initialize()
                                return await self._agent_loop(
                                    question=question,
                                    weather_session=weather_session,
                                    news_session=news_session,
                                    disaster_session=disaster_session,
                                    verbose=verbose,
                                )

    @staticmethod
    def _quality_score(answer: str) -> float:
        text = (answer or "").strip()
        if not text:
            return 0.0

        score = min(len(text) / 240.0, 1.0)
        lowered = text.lower()
        penalties = [
            "i could not complete",
            "unknown tool",
            "no data returned",
            "no relevant",
        ]
        if any(marker in lowered for marker in penalties):
            score *= 0.35
        return round(max(0.0, min(score, 1.0)), 4)

    def _needs_improvement(self, answer: str) -> bool:
        score = self._quality_score(answer)
        return score < 0.45

    async def _agent_loop(
        self,
        question: str,
        weather_session: ClientSession,
        news_session: ClientSession,
        disaster_session: ClientSession,
        verbose: bool,
    ) -> str:
        openai_tools, tool_routes = await self._build_tool_registry(
            weather_session=weather_session,
            news_session=news_session,
            disaster_session=disaster_session,
        )

        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        for iteration in range(MAX_ITERATIONS):
            if verbose:
                print(f"  [Iteration {iteration + 1}/{MAX_ITERATIONS}]")

            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                tools=openai_tools,
                tool_choice="auto",
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )

            choice = response.choices[0]

            if choice.finish_reason == "stop":
                return choice.message.content or ""

            if choice.finish_reason != "tool_calls":
                continue

            tool_calls = choice.message.tool_calls or []
            messages.append(
                {
                    "role": "assistant",
                    "content": choice.message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )

            for tc in tool_calls:
                tool_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                self.last_tool_calls.append({"tool": tool_name, "args": args})
                if verbose:
                    print(f"  -> {tool_name}({args})")

                tool_result = await self._call_tool(tool_name, args, tool_routes)

                if verbose:
                    print(f"  <- {tool_result[:140].replace(chr(10), ' ')}")

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result,
                    }
                )

        return "I could not complete the answer within the iteration limit."

    async def _build_tool_registry(
        self,
        weather_session: ClientSession,
        news_session: ClientSession,
        disaster_session: ClientSession,
    ) -> tuple[list[dict], dict]:
        openai_tools: list[dict] = []
        tool_routes: dict[str, tuple[ClientSession, str]] = {}

        weather_tools = await weather_session.list_tools()
        for t in weather_tools.tools:
            namespaced = f"weather__{t.name}"
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": namespaced,
                        "description": t.description,
                        "parameters": t.inputSchema,
                    },
                }
            )
            tool_routes[namespaced] = (weather_session, t.name)

        news_tools = await news_session.list_tools()
        for t in news_tools.tools:
            namespaced = f"news__{t.name}"
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": namespaced,
                        "description": t.description,
                        "parameters": t.inputSchema,
                    },
                }
            )
            tool_routes[namespaced] = (news_session, t.name)

        disaster_tools = await disaster_session.list_tools()
        for t in disaster_tools.tools:
            namespaced = f"disaster__{t.name}"
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": namespaced,
                        "description": t.description,
                        "parameters": t.inputSchema,
                    },
                }
            )
            tool_routes[namespaced] = (disaster_session, t.name)

        # Local RAG tool (from Task 2 style retrieval)
        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": "rag__search_documents",
                    "description": "Search resume RAG knowledge base for relevant candidate information.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
                        },
                        "required": ["query"],
                    },
                },
            }
        )

        return openai_tools, tool_routes

    async def _call_tool(self, tool_name: str, args: dict, tool_routes: dict) -> str:
        safe, issues = self.security_guard.validate_tool_args(tool_name, args)
        if not safe:
            self.last_security_report.setdefault("tool_arg_issues", []).extend(issues)
            return (
                "Tool arguments were rejected by security validation: "
                + ", ".join(issues)
            )

        if tool_name == "rag__search_documents":
            query = str(args.get("query", "")).strip()
            top_k = int(args.get("top_k", 3))
            return self.rag.search_as_text(query=query, top_k=top_k)

        if tool_name not in tool_routes:
            return f"Unknown tool: {tool_name}"

        session, actual_name = tool_routes[tool_name]
        result = await session.call_tool(actual_name, args)

        if result.content:
            return result.content[0].text
        return "No data returned from tool."
