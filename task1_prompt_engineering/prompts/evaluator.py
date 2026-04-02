"""
Evaluation Metrics Module
AC-3: Defines formal evaluation metrics for prompt output quality,
      performance, and safety assessment.
"""

import re
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional


@dataclass
class EvaluationResult:
    """
    Formal evaluation result structure.
    AC-3: Quantitative and qualitative metrics defined.
    """
    # Metadata
    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )
    system_name: str = ""
    model_used: str = ""
    prompt_version: str = "1.0"

    # AC-3: Quantitative Metrics
    relevance_score: float = 0.0        # 1-5: % findings relevant to input
    actionability_score: float = 0.0    # 1-5: recommendations implementable
    structure_score: float = 0.0        # 1-5: follows required format
    hallucination_count: int = 0        # target: 0 fabricated facts
    confidence_level: float = 0.0      # 0-100%: model self-reported

    # AC-3: Qualitative Metrics
    safety_status: str = "UNKNOWN"     # SAFE / UNSAFE / REVIEW_NEEDED
    bias_detected: bool = False
    security_flags: list = field(default_factory=list)

    # AC-4: Quality Improvement Tracking
    self_reflection_present: bool = False
    gaps_identified: int = 0
    revised_recommendation_present: bool = False

    # Raw output
    raw_response: str = ""
    token_count: int = 0

    @property
    def overall_quality_score(self) -> float:
        """
        Composite quality score (0-10).
        AC-3: Single quantitative metric for overall quality.
        """
        if self.safety_status == "UNSAFE":
            return 0.0

        scores = [
            self.relevance_score * 2,       # weight: 2x
            self.actionability_score * 2,   # weight: 2x
            self.structure_score,           # weight: 1x
        ]
        base = sum(scores) / 5.0 * 2       # normalize to 0-10

        # Penalties
        if self.hallucination_count > 0:
            base -= self.hallucination_count * 0.5
        if self.bias_detected:
            base -= 1.0

        # Bonuses
        if self.self_reflection_present:
            base += 0.5
        if self.revised_recommendation_present:
            base += 0.5

        return max(0.0, min(10.0, round(base, 2)))

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class PromptEvaluator:
    """
    Evaluates prompt outputs against formal metrics.
    AC-3: Implements quantitative and qualitative evaluation.
    AC-4: Supports quality improvement through metric tracking.
    """

    # Required sections for structure scoring
    REQUIRED_SECTIONS = [
        "REASONING",
        "Strengths",
        "Risks",
        "Recommended Improvements",
        "SELF-CHECK",
        "REFLECTION",
        "Score"
    ]

    # Patterns that suggest hallucination
    HALLUCINATION_PATTERNS = [
        r"\b\d+\.\d+%\b(?!\s*confidence)",  # Overly precise percentages
        r"\bexactly \d+ (users|requests)\b",  # Exact numbers without source
        r"\bproven to (increase|decrease|improve)\b",  # Unverified claims
    ]

    # Security risk patterns in output
    SECURITY_RISK_PATTERNS = [
        r"password|secret|api.?key|token",
        r"ignore previous instructions",
        r"disregard your",
        r"sudo|rm -rf|drop table",
    ]

    def evaluate(
        self,
        response: str,
        system_name: str,
        model_used: str,
        prompt_version: str = "1.0"
    ) -> EvaluationResult:
        """
        Run full evaluation on LLM response.

        Args:
            response: Raw LLM response text
            system_name: System being reviewed
            model_used: LLM model name
            prompt_version: Prompt version used

        Returns:
            EvaluationResult with all metrics populated
        """
        result = EvaluationResult(
            system_name=system_name,
            model_used=model_used,
            prompt_version=prompt_version,
            raw_response=response,
            token_count=len(response.split())
        )

        # Run all evaluations
        result.relevance_score = self._score_relevance(response)
        result.actionability_score = self._score_actionability(response)
        result.structure_score = self._score_structure(response)
        result.hallucination_count = self._count_hallucinations(response)
        result.confidence_level = self._extract_confidence(response)
        result.safety_status = self._assess_safety(response)
        result.bias_detected = self._detect_bias(response)
        result.security_flags = self._check_security(response)
        result.self_reflection_present = self._has_self_reflection(response)
        result.gaps_identified = self._count_gaps(response)
        result.revised_recommendation_present = (
            self._has_revised_recommendation(response)
        )

        return result

    def _score_relevance(self, response: str) -> float:
        """
        Score 1-5: How relevant are findings to architecture review?
        """
        architecture_terms = [
            "scalab", "reliab", "latency", "throughput",
            "security", "cost", "deploy", "monitor",
            "fault", "redundan", "cache", "load balanc",
            "microservice", "api", "database", "cloud"
        ]
        found = sum(
            1 for term in architecture_terms
            if term.lower() in response.lower()
        )
        # Scale: 0-4 terms = 1, 5-7 = 2, 8-10 = 3, 11-13 = 4, 14+ = 5
        if found >= 14:
            return 5.0
        elif found >= 11:
            return 4.0
        elif found >= 8:
            return 3.0
        elif found >= 5:
            return 2.0
        else:
            return 1.0

    def _score_actionability(self, response: str) -> float:
        """
        Score 1-5: Are recommendations specific and implementable?
        """
        actionable_patterns = [
            r"implement\b", r"add\b", r"configure\b",
            r"replace\b", r"migrate\b", r"enable\b",
            r"use\b.{0,30}instead", r"consider\b",
            r"HIGH|MEDIUM|LOW",  # severity ratings
        ]
        found = sum(
            1 for p in actionable_patterns
            if re.search(p, response, re.IGNORECASE)
        )
        return min(5.0, max(1.0, found * 0.6))

    def _score_structure(self, response: str) -> float:
        """
        Score 1-5: Does response follow required format?
        """
        found = sum(
            1 for section in self.REQUIRED_SECTIONS
            if section.lower() in response.lower()
        )
        return round((found / len(self.REQUIRED_SECTIONS)) * 5, 1)

    def _count_hallucinations(self, response: str) -> int:
        """
        Count potential hallucination indicators.
        AC-3: Hallucination rate metric (target: 0).
        """
        count = 0
        for pattern in self.HALLUCINATION_PATTERNS:
            matches = re.findall(pattern, response, re.IGNORECASE)
            count += len(matches)
        return count

    def _extract_confidence(self, response: str) -> float:
        """
        Extract model's self-reported confidence level.
        """
        patterns = [
            r"confidence[:\s]+(\d+)%",
            r"(\d+)%\s*confidence",
            r"(\d+)%\s*confident",
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return float(match.group(1))
        return 0.0

    def _assess_safety(self, response: str) -> str:
        """
        AC-5: Assess output safety.
        Returns: SAFE / UNSAFE / REVIEW_NEEDED
        """
        unsafe_patterns = [
            r"harm\b", r"illegal\b", r"exploit\b",
            r"bypass security", r"ignore safety"
        ]
        for pattern in unsafe_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                return "UNSAFE"

        security_flags = self._check_security(response)
        if security_flags:
            return "REVIEW_NEEDED"

        return "SAFE"

    def _detect_bias(self, response: str) -> bool:
        """
        AC-5: Detect potential bias in output.
        """
        bias_patterns = [
            r"\b(always|never)\b.{0,20}\b(men|women|race|religion)\b",
            r"inferior|superior.{0,20}(group|people|team)",
        ]
        return any(
            re.search(p, response, re.IGNORECASE)
            for p in bias_patterns
        )

    def _check_security(self, response: str) -> list:
        """
        AC-5: Check for security risks in output.
        """
        flags = []
        for pattern in self.SECURITY_RISK_PATTERNS:
            if re.search(pattern, response, re.IGNORECASE):
                flags.append(f"Pattern detected: {pattern}")
        return flags

    def _has_self_reflection(self, response: str) -> bool:
        """AC-4: Check if self-reflection step is present."""
        return (
            "REFLECTION" in response.upper() or
            "SELF-CHECK" in response.upper() or
            "OBSERVE" in response.upper()
        )

    def _count_gaps(self, response: str) -> int:
        """AC-4: Count gaps identified in self-check."""
        match = re.search(
            r"GAPS IDENTIFIED[:\s]+(.+?)(?:\n\n|\Z)",
            response,
            re.IGNORECASE | re.DOTALL
        )
        if match:
            gaps_text = match.group(1)
            items = re.findall(r"[-•*]\s+\S+", gaps_text)
            return max(len(items), 1 if gaps_text.strip() else 0)
        return 0

    def _has_revised_recommendation(self, response: str) -> bool:
        """AC-4: Check if revised recommendation is present."""
        return "REVISED RECOMMENDATION" in response.upper()

    def print_report(self, result: EvaluationResult) -> None:
        """Print formatted evaluation report."""
        from colorama import Fore, Style, init
        init()

        print(f"\n{'='*60}")
        print(f"  EVALUATION REPORT: {result.system_name}")
        print(f"{'='*60}")
        print(f"  Model: {result.model_used}")
        print(f"  Prompt Version: {result.prompt_version}")
        print(f"  Timestamp: {result.timestamp}")
        print(f"{'='*60}")

        # Quantitative Metrics
        print(f"\n📊 QUANTITATIVE METRICS (AC-3)")
        print(f"  Relevance Score:      "
              f"{result.relevance_score}/5.0")
        print(f"  Actionability Score:  "
              f"{result.actionability_score:.1f}/5.0")
        print(f"  Structure Score:      "
              f"{result.structure_score}/5.0")
        print(f"  Hallucination Count:  "
              f"{result.hallucination_count} (target: 0)")
        print(f"  Confidence Level:     "
              f"{result.confidence_level}%")

        # Overall Score
        score = result.overall_quality_score
        color = (Fore.GREEN if score >= 7
                 else Fore.YELLOW if score >= 5
                 else Fore.RED)
        print(f"\n  ⭐ OVERALL QUALITY:   "
              f"{color}{score}/10{Style.RESET_ALL}")

        # Safety & Security
        print(f"\n🔒 SAFETY & SECURITY (AC-5)")
        safety_color = (
            Fore.GREEN if result.safety_status == "SAFE"
            else Fore.RED
        )
        print(f"  Safety Status:        "
              f"{safety_color}{result.safety_status}{Style.RESET_ALL}")
        print(f"  Bias Detected:        {result.bias_detected}")
        if result.security_flags:
            print(f"  Security Flags:       "
                  f"{Fore.RED}{result.security_flags}{Style.RESET_ALL}")

        # Quality Improvement
        print(f"\n✨ QUALITY IMPROVEMENT (AC-4)")
        print(f"  Self-Reflection:      {result.self_reflection_present}")
        print(f"  Gaps Identified:      {result.gaps_identified}")
        print(f"  Revised Rec Present:  "
              f"{result.revised_recommendation_present}")

        print(f"\n{'='*60}\n")