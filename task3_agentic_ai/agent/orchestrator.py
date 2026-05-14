"""
Agent Orchestrator
ReAct-style agent that uses MCP servers for weather and news tools.
Spawns each MCP server as a subprocess (stdio transport) and routes
Azure OpenAI tool-calls to the appropriate server.
"""

import json
import sys
from pathlib import Path

from openai import AzureOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVERS_DIR = Path(__file__).parent.parent / "mcp_servers"

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to real-time weather data "
    "and the latest news.\n"
    "- For weather questions, always use the weather tools and include "
    "temperature (°C), conditions, and other relevant details.\n"
    "- For news questions, use the news tools and summarise key headlines "
    "with sources and dates.\n"
    "- Always call a tool before answering; never fabricate weather or news data.\n"
    "- Be concise and factual."
)

MAX_ITERATIONS = 10


class AgentOrchestrator:
    """
    Orchestrates an Azure OpenAI agent with weather and news MCP tools.

    After each call to `query()`, the list of tool calls made is available
    at `self.last_tool_calls` for evaluation purposes.
    """

    def __init__(self, config) -> None:
        self.config = config
        self.client = AzureOpenAI(
            api_key=config.azure_openai_api_key,
            api_version=config.azure_openai_api_version,
            azure_endpoint=config.azure_openai_endpoint,
        )
        self.last_tool_calls: list[dict] = []

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    async def query(self, question: str, verbose: bool = False) -> str:
        """
        Answer a question by orchestrating MCP tool calls.

        Args:
            question: User question about weather or news.
            verbose:  Print tool-call details to stdout.

        Returns:
            Final natural-language answer from the LLM.
        """
        self.last_tool_calls = []

        weather_params = StdioServerParameters(
            command=sys.executable,
            args=[str(SERVERS_DIR / "weather_server.py")],
        )
        news_params = StdioServerParameters(
            command=sys.executable,
            args=[str(SERVERS_DIR / "news_server.py")],
        )

        async with stdio_client(weather_params) as (wr, ww):
            async with ClientSession(wr, ww) as weather_session:
                await weather_session.initialize()

                async with stdio_client(news_params) as (nr, nw):
                    async with ClientSession(nr, nw) as news_session:
                        await news_session.initialize()

                        return await self._agent_loop(
                            question, weather_session, news_session, verbose
                        )

    # ------------------------------------------------------------------ #
    # Internal agent loop                                                  #
    # ------------------------------------------------------------------ #

    async def _agent_loop(
        self,
        question: str,
        weather_session: ClientSession,
        news_session: ClientSession,
        verbose: bool,
    ) -> str:
        # Discover tools from both MCP servers
        openai_tools, tool_routes = await self._build_tool_registry(
            weather_session, news_session
        )

        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        for iteration in range(MAX_ITERATIONS):
            if verbose:
                print(f"  [Agent iteration {iteration + 1}/{MAX_ITERATIONS}]")

            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                tools=openai_tools,
                tool_choice="auto",
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )

            choice = response.choices[0]

            # Final answer – no more tool calls
            if choice.finish_reason == "stop":
                return choice.message.content or ""

            # Tool calls requested
            if choice.finish_reason == "tool_calls":
                tool_calls = choice.message.tool_calls or []

                # Append assistant message (must include tool_calls field)
                messages.append({
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
                })

                for tc in tool_calls:
                    tool_name = tc.function.name
                    args = json.loads(tc.function.arguments)

                    if verbose:
                        print(f"  → {tool_name}({args})")

                    self.last_tool_calls.append({"tool": tool_name, "args": args})

                    # Route to the correct MCP server
                    tool_result = await self._call_tool(
                        tool_name, args, tool_routes
                    )

                    if verbose:
                        preview = tool_result[:120].replace("\n", " ")
                        print(f"  ← {preview}…")

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result,
                    })

        return "Could not produce an answer within the iteration limit."

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    async def _build_tool_registry(
        weather_session: ClientSession,
        news_session: ClientSession,
    ) -> tuple[list[dict], dict]:
        """
        Gather tools from both MCP servers and build:
        - openai_tools: list of tool dicts in OpenAI function-calling format
        - tool_routes:  mapping from namespaced tool name to (session, raw_name)
        """
        openai_tools: list[dict] = []
        tool_routes: dict[str, tuple[ClientSession, str]] = {}

        weather_tools = await weather_session.list_tools()
        for t in weather_tools.tools:
            namespaced = f"weather__{t.name}"
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": namespaced,
                    "description": t.description,
                    "parameters": t.inputSchema,
                },
            })
            tool_routes[namespaced] = (weather_session, t.name)

        news_tools = await news_session.list_tools()
        for t in news_tools.tools:
            namespaced = f"news__{t.name}"
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": namespaced,
                    "description": t.description,
                    "parameters": t.inputSchema,
                },
            })
            tool_routes[namespaced] = (news_session, t.name)

        return openai_tools, tool_routes

    @staticmethod
    async def _call_tool(
        tool_name: str,
        args: dict,
        tool_routes: dict,
    ) -> str:
        if tool_name not in tool_routes:
            return f"Unknown tool: {tool_name}"

        session, actual_name = tool_routes[tool_name]
        result = await session.call_tool(actual_name, args)

        if result.content:
            return result.content[0].text
        return "No data returned from tool."
