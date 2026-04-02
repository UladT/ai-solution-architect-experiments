"""
Security Guard Module
AC-5: Addresses prompt injection, data privacy, and input safety.
"""

import re
from dataclasses import dataclass
from typing import Tuple


@dataclass
class SecurityCheckResult:
    """Result of security validation."""
    is_safe: bool
    threats_found: list
    sanitized_input: str
    warnings: list


class SecurityGuard:
    """
    AC-5: Security layer for prompt engineering.
    Prevents prompt injection, protects privacy,
    ensures safe inputs and outputs.
    """

    # Prompt injection patterns
    INJECTION_PATTERNS = [
        r"ignore (previous|all|above) instructions",
        r"disregard (your|all) (rules|instructions|guidelines)",
        r"you are now",
        r"new persona",
        r"forget (everything|all|your training)",
        r"system prompt",
        r"jailbreak",
        r"DAN mode",
        r"pretend (you are|to be)",
    ]

    # PII patterns
    PII_PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b(\+\d{1,3}[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
        "api_key": r"\b(sk-|pk-|api-)[A-Za-z0-9]{20,}\b",
        "password": r"password\s*[:=]\s*\S+",
    }

    # Dangerous content patterns
    DANGEROUS_PATTERNS = [
        r"rm\s+-rf",
        r"DROP\s+TABLE",
        r"DELETE\s+FROM",
        r"exec\s*\(",
        r"eval\s*\(",
        r"__import__",
        r"os\.system",
    ]

    def validate_input(
        self,
        user_input: str
    ) -> SecurityCheckResult:
        """
        Validate and sanitize user input before sending to LLM.
        AC-5: Prevents prompt injection and PII leakage.

        Args:
            user_input: Raw user input / architecture description

        Returns:
            SecurityCheckResult with safety status and sanitized input
        """
        threats = []
        warnings = []
        sanitized = user_input

        # Check for prompt injection
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, user_input, re.IGNORECASE):
                threats.append(
                    f"PROMPT INJECTION detected: '{pattern}'"
                )

        # Check for PII
        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = re.findall(pattern, user_input, re.IGNORECASE)
            if matches:
                warnings.append(
                    f"PII detected ({pii_type}): "
                    f"{len(matches)} instance(s)"
                )
                # Sanitize PII
                sanitized = re.sub(
                    pattern,
                    f"[REDACTED_{pii_type.upper()}]",
                    sanitized,
                    flags=re.IGNORECASE
                )

        # Check for dangerous content
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, user_input, re.IGNORECASE):
                threats.append(
                    f"DANGEROUS CONTENT detected: '{pattern}'"
                )

        is_safe = len(threats) == 0

        return SecurityCheckResult(
            is_safe=is_safe,
            threats_found=threats,
            sanitized_input=sanitized,
            warnings=warnings
        )

    def validate_output(
        self,
        llm_output: str
    ) -> Tuple[bool, list]:
        """
        Validate LLM output for safety.
        AC-5: Ensures output doesn't contain harmful content.

        Returns:
            Tuple of (is_safe, list_of_issues)
        """
        issues = []

        # Check output doesn't contain injected instructions
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, llm_output, re.IGNORECASE):
                issues.append(
                    f"Output contains injection pattern: {pattern}"
                )

        # Check for accidental PII in output
        for pii_type, pattern in self.PII_PATTERNS.items():
            if re.search(pattern, llm_output, re.IGNORECASE):
                issues.append(
                    f"Output may contain PII: {pii_type}"
                )

        return len(issues) == 0, issues

    def print_security_report(
        self,
        check_result: SecurityCheckResult
    ) -> None:
        """Print security check results."""
        print("\n🔒 SECURITY CHECK REPORT (AC-5)")
        print(f"  Status: {'✅ SAFE' if check_result.is_safe else '❌ UNSAFE'}")

        if check_result.threats_found:
            print("  🚨 THREATS:")
            for threat in check_result.threats_found:
                print(f"    - {threat}")

        if check_result.warnings:
            print("  ⚠️  WARNINGS:")
            for warning in check_result.warnings:
                print(f"    - {warning}")

        if not check_result.threats_found and not check_result.warnings:
            print("  No threats or warnings detected.")
        print()