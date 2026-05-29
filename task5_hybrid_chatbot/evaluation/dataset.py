"""Evaluation dataset for Task 5 hybrid chatbot."""

TEST_CASES = [
    {
        "id": "rag_001",
        "question": "Find Python developers with FastAPI and Django experience",
        "expected_tool_prefix": "rag__",
        "expected_keywords": ["python", "fastapi", "django"],
        "category": "rag",
    },
    {
        "id": "rag_002",
        "question": "Show machine learning candidates with TensorFlow",
        "expected_tool_prefix": "rag__",
        "expected_keywords": ["machine", "learning", "tensorflow"],
        "category": "rag",
    },
    {
        "id": "weather_001",
        "question": "What is the weather in London today?",
        "expected_tool_prefix": "weather__",
        "expected_keywords": ["london", "temperature", "weather"],
        "category": "weather",
    },
    {
        "id": "news_001",
        "question": "What are the latest news about artificial intelligence?",
        "expected_tool_prefix": "news__",
        "expected_keywords": ["news", "artificial", "intelligence"],
        "category": "news",
    },
    {
        "id": "disaster_001",
        "question": "Top flood disasters in India since 2000 by total deaths",
        "expected_tool_prefix": "disaster__",
        "expected_keywords": ["india", "flood", "deaths"],
        "category": "disaster",
    },
    {
        "id": "disaster_002",
        "question": "Summarize earthquake disasters in Japan between 1990 and 2021",
        "expected_tool_prefix": "disaster__",
        "expected_keywords": ["japan", "earthquake", "events"],
        "category": "disaster",
    },
]
