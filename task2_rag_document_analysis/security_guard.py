"""
Security Guard - Task 2 Resume Intelligence Platform

AC-5: Security controls — adapted from Task 1 SecurityGuard with additions:
  - PII detection and masking (email, phone, SSN, credit card, API keys)
  - Prompt injection detection and blocking
  - Dangerous content detection (SQL injection, shell commands)
  - Role-based access control enforcement (see rag_engine.py ROLE_PERMISSIONS)
  - Output PII scrubbing before returning results to callers
"""

import re
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class SecurityCheckResult:
    """Result of a security validation pass."""

    is_safe: bool
    threats_found: List[str]
    sanitized_input: str
    warnings: List[str]


class SecurityGuard:
    """
    AC-5: Security layer that validates inputs and outputs for the RAG pipeline.

    Threat categories:
      HARD BLOCK  → prompt injection, SQL injection, destructive shell commands
      SOFT WARN   → PII (masked, not blocked; callers may still process safely)
    """

    # ── Threat patterns ───────────────────────────────────────────────────────

    INJECTION_PATTERNS = [
        r"ignore\s+(previous|all|above)\s+instructions",
        r"disregard\s+(your|all)\s+(rules|instructions|guidelines|training)",
        r"\byou\s+are\s+now\b",
        r"\bnew\s+persona\b",
        r"forget\s+(everything|all|your\s+training)",
        r"\bsystem\s+prompt\b",
        r"\bjailbreak\b",
        r"\bDAN\s+mode\b",
        r"pretend\s+(you\s+are|to\s+be)",
        r"act\s+as\s+if\s+you\s+(have\s+no|are\s+not)",
    ]

    PII_PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
        "phone": r"\b(\+\d{1,3}[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b",
        "api_key": r"\b(sk-|pk-|api[-_])[A-Za-z0-9]{15,}\b",
        "password": r"(?i)password\s*[:=]\s*\S+",
    }

    DANGEROUS_PATTERNS = [
        r"rm\s+-rf",
        r"DROP\s+TABLE",
        r"DELETE\s+FROM",
        r"TRUNCATE\s+TABLE",
        r"exec\s*\(",
        r"eval\s*\(",
        r"__import__",
        r"os\.system",
        r"subprocess\.call",
        r"<\s*script",          # XSS
    ]

    def validate_input(self, user_input: str) -> SecurityCheckResult:
        """
        Validate and sanitize user-supplied query before sending to RAG / LLM.

        Returns:
            SecurityCheckResult with is_safe flag, threat list, and sanitized text.
        """
        threats: List[str] = []
        warnings: List[str] = []
        sanitized = user_input

        # Prompt injection (hard block)
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, user_input, re.IGNORECASE):
                threats.append(f"PROMPT_INJECTION: pattern '{pattern}'")

        # Dangerous content (hard block)
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, user_input, re.IGNORECASE):
                threats.append(f"DANGEROUS_CONTENT: pattern '{pattern}'")

        # PII (soft warn + mask)
        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = re.findall(pattern, user_input, re.IGNORECASE)
            if matches:
                warnings.append(f"PII detected ({pii_type}): {len(matches)} instance(s)")
                sanitized = re.sub(
                    pattern,
                    f"[REDACTED_{pii_type.upper()}]",
                    sanitized,
                    flags=re.IGNORECASE,
                )

        return SecurityCheckResult(
            is_safe=len(threats) == 0,
            threats_found=threats,
            sanitized_input=sanitized,
            warnings=warnings,
        )

    def validate_output(self, llm_output: str) -> Tuple[bool, List[str]]:
        """
        Scrub LLM-generated output for PII before returning to callers.

        Returns:
            (is_clean, issues) — issues are non-empty if PII found in output.
        """
        issues: List[str] = []

        for pii_type, pattern in self.PII_PATTERNS.items():
            if re.search(pattern, llm_output, re.IGNORECASE):
                issues.append(f"Output contains {pii_type}")

        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, llm_output, re.IGNORECASE):
                issues.append(f"Output contains injection pattern")

        return len(issues) == 0, issues

    def mask_pii_in_output(self, text: str) -> str:
        """Return text with all PII tokens replaced by redaction placeholders."""
        for pii_type, pattern in self.PII_PATTERNS.items():
            text = re.sub(
                pattern,
                f"[REDACTED_{pii_type.upper()}]",
                text,
                flags=re.IGNORECASE,
            )
        return text

    def print_security_report(self, result: SecurityCheckResult, label: str = "") -> None:
        """Pretty-print a security check result."""
        prefix = f"  [{label}] " if label else "  "
        status = "✅ SAFE" if result.is_safe else "🚫 BLOCKED"
        print(f"{prefix}Status: {status}")
        for threat in result.threats_found:
            print(f"{prefix}  ⛔ Threat: {threat}")
        for warning in result.warnings:
            print(f"{prefix}  ⚠️  Warning: {warning}")
        if result.sanitized_input != result.sanitized_input.replace("[REDACTED_", "XX"):
            print(f"{prefix}  🔒 PII masked in sanitized output")
