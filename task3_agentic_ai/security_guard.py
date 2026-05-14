"""
Security guard for Task 3.
Provides basic input/output validation against prompt-injection patterns,
unsafe command payloads, and accidental secret leakage.
"""

import re
from dataclasses import dataclass, field


@dataclass
class SecurityDecision:
    is_safe: bool
    sanitized_question: str
    blocked: bool = False
    threats_found: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class SecurityGuard:
    """Input/output safety checks for weather/news assistant."""

    MAX_QUESTION_LEN = 500

    BLOCK_PATTERNS = [
        (r"ignore\s+previous\s+instructions", "prompt_injection"),
        (r"reveal\s+system\s+prompt", "system_prompt_exfiltration"),
        (r"developer\s+message", "prompt_exfiltration"),
        (r"tool\s*:\s*", "tool_channel_injection"),
        (r"rm\s+-rf|sudo\s+|drop\s+table", "unsafe_command_payload"),
        (r"169\.254\.169\.254", "metadata_service_probe"),
    ]

    SECRET_PATTERNS = [
        (r"sk-[A-Za-z0-9_-]{20,}", "openai_api_key"),
        (r"AKIA[0-9A-Z]{16}", "aws_access_key"),
        (r"AIza[0-9A-Za-z_-]{35}", "google_api_key"),
    ]

    def validate_input(self, question: str) -> SecurityDecision:
        text = (question or "").strip()
        threats: list[str] = []
        warnings: list[str] = []

        if not text:
            return SecurityDecision(
                is_safe=False,
                sanitized_question="",
                blocked=True,
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
            sanitized_question=text,
            blocked=blocked,
            threats_found=threats,
            warnings=warnings,
        )

    def validate_output(self, output: str) -> tuple[str, list[str]]:
        """
        Return sanitized output and list of issues found.
        Redacts token-like strings to reduce accidental secret leakage.
        """
        text = output or ""
        issues: list[str] = []

        for pattern, label in self.SECRET_PATTERNS:
            if re.search(pattern, text):
                text = re.sub(pattern, "[REDACTED_SECRET]", text)
                issues.append(label)

        return text, issues
