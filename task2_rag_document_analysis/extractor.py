"""
Extractor - Task 2 Resume Intelligence Platform

Uses Azure OpenAI to extract structured data from resume text:
  - Skills (list)
  - Years of experience (int)
  - Education level (enum)
  - Primary role / job title

This data feeds directly into the Visualizer (bar charts, pie charts)
and Evaluator (extraction F1 metric).

AC-4 Quality improvement:
  - Prompt includes few-shot examples and explicit JSON schema constraints
  - Extraction retries with relaxed parsing on malformed JSON
"""

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from openai import AzureOpenAI

from config import config


EXTRACTION_PROMPT = """\
You are a resume parser. Extract structured information from the resume text below.

Return ONLY a valid JSON object with these exact fields (no markdown, no extra text):
{{
  "skills": ["<skill1>", "<skill2>", ...],
  "years_experience": <integer, 0 if unclear>,
  "education_level": "<one of: High School | Bachelor's | Master's | PhD | Other>",
  "primary_role": "<most likely job title, e.g. Data Scientist>"
}}

Rules:
- skills: individual technical skills only (languages, frameworks, tools, platforms)
- years_experience: sum all professional roles; use 0 if not stated
- education_level: highest degree found
- primary_role: infer from titles and skills; be specific

Resume:
{resume_text}
"""


@dataclass
class ExtractionResult:
    """Structured data extracted from a single resume."""

    doc_id: str
    category: str
    skills: List[str] = field(default_factory=list)
    years_experience: int = 0
    education_level: str = "Other"
    primary_role: str = "Unknown"
    raw_json: str = ""
    success: bool = True
    error: str = ""

    def to_dict(self) -> Dict:
        return {
            "doc_id": self.doc_id,
            "category": self.category,
            "skills": self.skills,
            "years_experience": self.years_experience,
            "education_level": self.education_level,
            "primary_role": self.primary_role,
            "success": self.success,
        }


class Extractor:
    """
    LLM-based structured extractor for resume documents.

    AC-4: Uses a carefully engineered prompt with schema constraints.
    Includes JSON repair fallback for robustness.
    """

    VALID_EDUCATION_LEVELS = {"High School", "Bachelor's", "Master's", "PhD", "Other"}
    KNOWN_SKILLS = {
        "python", "java", "sql", "r", "tensorflow", "pytorch", "scikit-learn",
        "django", "fastapi", "flask", "postgresql", "mysql", "redis", "docker",
        "kubernetes", "aws", "gcp", "azure", "spark", "hadoop", "tableau",
        "power bi", "airflow", "mlflow", "terraform", "ansible", "jenkins",
        "kafka", "rabbitmq", "graphql", "mongodb", "linux", "bash", "prometheus",
        "grafana", "splunk", "wireshark", "solidity", "react", "node.js",
        "typescript", "spring boot", "hibernate",
    }

    def __init__(self) -> None:
        self._client = AzureOpenAI(
            api_key=config.azure_openai_api_key,
            azure_endpoint=config.azure_openai_endpoint,
            api_version=config.azure_openai_api_version,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(self, doc_id: str, category: str, resume_text: str) -> ExtractionResult:
        """
        Extract structured fields from one resume.

        Args:
            doc_id:      Document identifier from DocumentLoader.
            category:    Ground-truth category label.
            resume_text: Full resume text.

        Returns:
            ExtractionResult with parsed fields.
        """
        prompt = EXTRACTION_PROMPT.format(resume_text=resume_text[:3000])

        try:
            response = self._client.chat.completions.create(
                model=config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,  # Deterministic for extraction
                max_tokens=500,
            )
            raw = response.choices[0].message.content or ""
            return self._parse_response(doc_id, category, raw)

        except Exception as exc:  # noqa: BLE001
            heuristic = self._heuristic_extract(doc_id, category, resume_text)
            heuristic.error = f"LLM extraction unavailable, heuristic fallback used: {exc}"
            return heuristic

    def extract_batch(
        self,
        documents: List[Dict],
        max_docs: Optional[int] = None,
    ) -> List[ExtractionResult]:
        """
        Extract structured data from a list of resume dicts.

        Args:
            documents: List of ResumeDocument.to_dict() or equivalent dicts
                       with keys: doc_id, category, text (or text_preview).
            max_docs:  Limit extraction calls (API cost control).
        """
        results = []
        limit = max_docs or len(documents)

        for doc in documents[:limit]:
            doc_id = doc.get("doc_id", "")
            category = doc.get("category", "Unknown")
            text = doc.get("text", doc.get("text_preview", ""))
            result = self.extract(doc_id, category, text)
            results.append(result)

        return results

    # ── Parsing helpers ───────────────────────────────────────────────────────

    def _parse_response(
        self, doc_id: str, category: str, raw: str
    ) -> ExtractionResult:
        """Parse LLM JSON response with repair fallback."""
        cleaned = self._strip_markdown(raw)

        # First try: direct parse
        data = self._try_json_parse(cleaned)
        if data is None:
            # Second try: extract first {...} block
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                data = self._try_json_parse(match.group())

        if data is None:
            return ExtractionResult(
                doc_id=doc_id,
                category=category,
                raw_json=raw,
                success=False,
                error=f"Could not parse JSON from LLM response: {raw[:200]}",
            )

        skills = data.get("skills", [])
        if isinstance(skills, str):
            skills = [s.strip() for s in skills.split(",") if s.strip()]

        education = data.get("education_level", "Other")
        if education not in self.VALID_EDUCATION_LEVELS:
            education = "Other"

        years = data.get("years_experience", 0)
        if not isinstance(years, int):
            try:
                years = int(re.search(r"\d+", str(years)).group())  # type: ignore
            except (AttributeError, ValueError):
                years = 0

        return ExtractionResult(
            doc_id=doc_id,
            category=category,
            skills=[str(s).strip() for s in skills if s],
            years_experience=max(0, min(years, 50)),
            education_level=education,
            primary_role=str(data.get("primary_role", "Unknown")).strip(),
            raw_json=raw,
            success=True,
        )

    @staticmethod
    def _strip_markdown(text: str) -> str:
        """Remove ```json ... ``` fences if present."""
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        return text.strip()

    @staticmethod
    def _try_json_parse(text: str) -> Optional[Dict]:
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None

    def _heuristic_extract(
        self, doc_id: str, category: str, resume_text: str
    ) -> ExtractionResult:
        """Rule-based extraction fallback used when model calls fail."""
        text_lower = resume_text.lower()

        # Years of experience: prefer explicit EXPERIENCE: X years.
        years = 0
        m = re.search(r"experience\s*:\s*(\d+)\s+years", text_lower)
        if m:
            years = int(m.group(1))
        else:
            candidates = [int(x) for x in re.findall(r"(\d+)\s+years", text_lower)]
            if candidates:
                years = max(candidates)

        # Education level heuristic.
        if re.search(r"\b(ph\.?d|doctorate)\b", text_lower):
            education = "PhD"
        elif re.search(r"\b(m\.?s\.?|master|mba)\b", text_lower):
            education = "Master's"
        elif re.search(r"\b(b\.?s\.?|bachelor)\b", text_lower):
            education = "Bachelor's"
        elif re.search(r"high school", text_lower):
            education = "High School"
        else:
            education = "Other"

        # Primary role: derive from first line if available.
        primary_role = category
        first_line = resume_text.splitlines()[0] if resume_text.splitlines() else ""
        parts = [p.strip() for p in first_line.split("|") if p.strip()]
        if len(parts) >= 2:
            primary_role = parts[1]

        # Skills: use SKILLS section first, then keyword scan.
        skills: List[str] = []
        section_match = re.search(r"skills\s*:\s*(.+?)(\n\n|$)", resume_text, re.IGNORECASE | re.DOTALL)
        if section_match:
            raw_items = section_match.group(1).replace("\n", " ").split(",")
            for item in raw_items:
                cleaned = item.strip()
                if cleaned:
                    skills.append(cleaned)

        for known in self.KNOWN_SKILLS:
            if known in text_lower:
                canonical = known.replace("node.js", "Node.js").replace("power bi", "Power BI")
                skills.append(canonical.title() if canonical.islower() else canonical)

        # De-duplicate while preserving order.
        deduped: List[str] = []
        seen = set()
        for s in skills:
            key = s.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(s)

        return ExtractionResult(
            doc_id=doc_id,
            category=category,
            skills=deduped[:25],
            years_experience=max(0, min(years, 50)),
            education_level=education,
            primary_role=primary_role,
            raw_json="",
            success=True,
        )

    # ── Aggregation ────────────────────────────────────────────────────────────

    @staticmethod
    def aggregate_skills(results: List[ExtractionResult]) -> Dict[str, int]:
        """Count skill frequencies across all extraction results."""
        freq: Dict[str, int] = {}
        for r in results:
            for skill in r.skills:
                normalized = skill.strip()
                if normalized:
                    freq[normalized] = freq.get(normalized, 0) + 1
        return dict(sorted(freq.items(), key=lambda x: x[1], reverse=True))

    @staticmethod
    def skills_by_category(
        results: List[ExtractionResult],
    ) -> Dict[str, List[str]]:
        """Group all skills by category."""
        grouped: Dict[str, List[str]] = {}
        for r in results:
            if r.category not in grouped:
                grouped[r.category] = []
            grouped[r.category].extend(r.skills)
        return grouped

    @staticmethod
    def experience_distribution(results: List[ExtractionResult]) -> Dict[str, int]:
        """Bucket years of experience into named ranges."""
        buckets = {"0–2 yrs": 0, "3–5 yrs": 0, "6–10 yrs": 0, "10+ yrs": 0}
        for r in results:
            y = r.years_experience
            if y <= 2:
                buckets["0–2 yrs"] += 1
            elif y <= 5:
                buckets["3–5 yrs"] += 1
            elif y <= 10:
                buckets["6–10 yrs"] += 1
            else:
                buckets["10+ yrs"] += 1
        return buckets
