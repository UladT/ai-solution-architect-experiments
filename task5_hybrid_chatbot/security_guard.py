"""Security guard for Task 5 hybrid chatbot (AC-5)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class SecurityDecision:
    is_safe: bool
    blocked: bool
    sanitized_input: str
    threats_found: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class SecurityGuard:
    """Input/output and tool-argument safety validation."""

    MAX_QUESTION_LEN = 600

    BLOCK_PATTERNS = [
        (r"ignore\s+previous\s+instructions", "prompt_injection"),
        (r"reveal\s+system\s+prompt", "system_prompt_exfiltration"),
        (r"developer\s+message", "prompt_exfiltration"),
        (r"tool\s*:\s*", "tool_channel_injection"),
        (r"rm\s+-rf|sudo\s+|drop\s+table|union\s+select", "unsafe_command_payload"),
        (r"169\.254\.169\.254", "metadata_service_probe"),
    ]

    SECRET_PATTERNS = [
        (r"sk-[A-Za-z0-9_-]{20,}", "openai_api_key"),
        (r"AKIA[0-9A-Z]{16}", "aws_access_key"),
        (r"AIza[0-9A-Za-z_-]{35}", "google_api_key"),
    ]

    SAFE_TEXT = re.compile(r"^[A-Za-z0-9 .,'()/_-]{1,120}$")

    def validate_input(self, question: str) -> SecurityDecision:
        text = (question or "").strip()
        threats: list[str] = []
        warnings: list[str] = []

        if not text:
            return SecurityDecision(
                is_safe=False,
                blocked=True,
                sanitized_input="",
                threats_found=["empty_input"],
            )

        if len(text) > self.MAX_QUESTION_LEN:
            text = text[: self.MAX_QUESTION_LEN]
            warnings.append("question_truncated")

        for pattern, label in self.BLOCK_PATTERNS:
            if re.search(pattern, text, flags=re.IGNORECASE):
                threats.append(label)

        blocked = len(threats) > 0
        return SecurityDecision(
            is_safe=not blocked,
            blocked=blocked,
            sanitized_input=text,
            threats_found=threats,
            warnings=warnings,
        )

    def validate_output(self, output: str) -> tuple[str, list[str]]:
        text = output or ""
        issues: list[str] = []

        for pattern, label in self.SECRET_PATTERNS:
            if re.search(pattern, text):
                text = re.sub(pattern, "[REDACTED_SECRET]", text)
                issues.append(label)

        return text, issues

    def validate_tool_args(self, tool_name: str, args: dict) -> tuple[bool, list[str]]:
        """Validate tool arguments before invocation."""
        issues: list[str] = []

        def check_text_field(field: str, max_len: int = 120) -> None:
            value = args.get(field)
            if value is None:
                return
            value = str(value).strip()
            if len(value) > max_len:
                issues.append(f"{field}_too_long")
            elif not self.SAFE_TEXT.match(value):
                issues.append(f"{field}_invalid_chars")

        if tool_name.startswith("rag__"):
            check_text_field("query", max_len=240)
        elif tool_name.startswith("weather__"):
            check_text_field("location", max_len=100)
        elif tool_name.startswith("news__"):
            check_text_field("query", max_len=120)
            category = args.get("category")
            if category is not None:
                allowed = {
                    "general", "world", "nation", "business", "technology",
                    "entertainment", "sports", "science", "health",
                }
                if str(category).lower() not in allowed:
                    issues.append("invalid_news_category")
        elif tool_name.startswith("disaster__"):
            check_text_field("country", max_len=80)
            check_text_field("disaster_type", max_len=80)
            check_text_field("metric", max_len=80)
            for year_field in ["start_year", "end_year"]:
                if year_field in args and args[year_field] is not None:
                    try:
                        year = int(args[year_field])
                    except (TypeError, ValueError):
                        issues.append(f"{year_field}_not_int")
                        continue
                    if year < 1900 or year > 2100:
                        issues.append(f"{year_field}_out_of_range")

        return len(issues) == 0, issues
