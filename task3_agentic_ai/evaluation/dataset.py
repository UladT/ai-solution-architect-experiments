"""
Evaluation dataset for Task 3 – Agentic AI.

12 test cases (6 weather, 6 news) with:
- expected_tool_type  : "weather" or "news"
- expected_tool_name  : specific MCP tool expected to be called
- expected_keywords   : keywords that should appear in the answer
- category            : grouping label
"""

TEST_CASES: list[dict] = [
    # ── Weather ───────────────────────────────────────────────────────── #
    {
        "id": "weather_001",
        "question": "What is the current weather in London?",
        "expected_tool_type": "weather",
        "expected_tool_name": "get_current_weather",
        "expected_keywords": ["london", "temperature", "°"],
        "category": "weather",
    },
    {
        "id": "weather_002",
        "question": "How is the weather in Tokyo today?",
        "expected_tool_type": "weather",
        "expected_tool_name": "get_current_weather",
        "expected_keywords": ["tokyo", "temperature"],
        "category": "weather",
    },
    {
        "id": "weather_003",
        "question": "Give me a 3-day weather forecast for Paris.",
        "expected_tool_type": "weather",
        "expected_tool_name": "get_weather_forecast",
        "expected_keywords": ["paris", "forecast"],
        "category": "weather",
    },
    {
        "id": "weather_004",
        "question": "Will it rain in Berlin tomorrow?",
        "expected_tool_type": "weather",
        "expected_tool_name": "get_weather_forecast",
        "expected_keywords": ["berlin", "rain", "precipitation"],
        "category": "weather",
    },
    {
        "id": "weather_005",
        "question": "What's the temperature in Sydney right now?",
        "expected_tool_type": "weather",
        "expected_tool_name": "get_current_weather",
        "expected_keywords": ["sydney", "temperature"],
        "category": "weather",
    },
    {
        "id": "weather_006",
        "question": "Show me a 5-day weather forecast for New York.",
        "expected_tool_type": "weather",
        "expected_tool_name": "get_weather_forecast",
        "expected_keywords": ["new york", "forecast"],
        "category": "weather",
    },

    # ── News ──────────────────────────────────────────────────────────── #
    {
        "id": "news_001",
        "question": "What are the latest news about artificial intelligence?",
        "expected_tool_type": "news",
        "expected_tool_name": "search_news",
        "expected_keywords": ["ai", "artificial intelligence", "technology", "model"],
        "category": "news",
    },
    {
        "id": "news_002",
        "question": "What is happening in the world of technology today?",
        "expected_tool_type": "news",
        "expected_tool_name": "get_top_headlines",
        "expected_keywords": ["technology", "tech"],
        "category": "news",
    },
    {
        "id": "news_003",
        "question": "Give me the latest business headlines.",
        "expected_tool_type": "news",
        "expected_tool_name": "get_top_headlines",
        "expected_keywords": ["business", "market", "company", "economy"],
        "category": "news",
    },
    {
        "id": "news_004",
        "question": "What are recent news about climate change?",
        "expected_tool_type": "news",
        "expected_tool_name": "search_news",
        "expected_keywords": ["climate", "environment", "carbon", "emissions"],
        "category": "news",
    },
    {
        "id": "news_005",
        "question": "What's the latest news in science?",
        "expected_tool_type": "news",
        "expected_tool_name": "get_top_headlines",
        "expected_keywords": ["science", "research", "study", "discovery"],
        "category": "news",
    },
    {
        "id": "news_006",
        "question": "Tell me about recent developments in space exploration.",
        "expected_tool_type": "news",
        "expected_tool_name": "search_news",
        "expected_keywords": ["space", "nasa", "rocket", "mission", "satellite"],
        "category": "news",
    },
]
