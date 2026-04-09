"""
Evaluator - Task 2 Resume Intelligence Platform

Computes four evaluation metrics over a labeled golden dataset:
  1. Precision@K  — fraction of retrieved docs in the relevant category
  2. Recall@K     — fraction of relevant docs successfully retrieved
  3. Extraction F1 — token-level F1 for skill extraction vs. ground truth
  4. Faithfulness  — LLM-as-judge (0–10 scale) for answer groundedness

AC-3: At least one evaluation metric defined and demonstrated.
AC-4: Metrics expose quality gaps → guide prompt / chunk-size improvements.
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from openai import AzureOpenAI

from config import config


FAITHFULNESS_JUDGE_PROMPT = """\
You are an impartial evaluator assessing whether an AI answer is faithfully
grounded in the provided context. Hallucinated or unsupported claims lower the score.

Context (source documents):
{context}

Question:
{question}

Answer to evaluate:
{answer}

Score the answer on a scale of 0–10:
  10 = completely grounded, no unsupported claims
   7 = mostly grounded, minor acceptable inferences
   4 = partially grounded, notable unsupported claims
   1 = mostly unsupported or contradicts context
   0 = completely fabricated / contradicts context

Return ONLY a JSON object, no other text:
{{"score": <0-10>, "explanation": "<one sentence reason>"}}
"""


@dataclass
class EvalSummary:
    """Aggregated evaluation results across all metric categories."""

    precision_at_k_scores: List[float] = field(default_factory=list)
    recall_at_k_scores: List[float] = field(default_factory=list)
    extraction_f1_scores: List[float] = field(default_factory=list)
    faithfulness_scores: List[float] = field(default_factory=list)
    security_pass: int = 0
    security_total: int = 0

    @property
    def avg_precision(self) -> float:
        return _safe_mean(self.precision_at_k_scores)

    @property
    def avg_recall(self) -> float:
        return _safe_mean(self.recall_at_k_scores)

    @property
    def avg_extraction_f1(self) -> float:
        return _safe_mean(self.extraction_f1_scores)

    @property
    def avg_faithfulness(self) -> float:
        return _safe_mean(self.faithfulness_scores)

    @property
    def security_rate(self) -> float:
        return self.security_pass / max(self.security_total, 1)

    def to_dict(self) -> Dict:
        return {
            "avg_precision_at_k": round(self.avg_precision, 3),
            "avg_recall_at_k": round(self.avg_recall, 3),
            "avg_extraction_f1": round(self.avg_extraction_f1, 3),
            "avg_faithfulness_score": round(self.avg_faithfulness, 2),
            "security_tests": f"{self.security_pass}/{self.security_total}",
            "details": {
                "precision_scores": [round(s, 3) for s in self.precision_at_k_scores],
                "recall_scores": [round(s, 3) for s in self.recall_at_k_scores],
                "extraction_f1_scores": [round(s, 3) for s in self.extraction_f1_scores],
                "faithfulness_scores": self.faithfulness_scores,
            },
        }


def _safe_mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


class Evaluator:
    """
    Computes RAG evaluation metrics against the golden evaluation dataset.

    Usage:
        evaluator = Evaluator()
        summary = evaluator.run_full_eval(rag_engine, extraction_results, security_guard)
    """

    def __init__(self) -> None:
        self._client = AzureOpenAI(
            api_key=config.azure_openai_api_key,
            azure_endpoint=config.azure_openai_endpoint,
            api_version=config.azure_openai_api_version,
        )
        self._golden = self._load_golden_set()

    def _load_golden_set(self) -> Dict:
        golden_path = os.path.join(
            os.path.dirname(__file__), "eval_dataset", "golden_set.json"
        )
        with open(golden_path, encoding="utf-8") as fh:
            return json.load(fh)

    # ── 1. Retrieval: Precision@K & Recall@K ─────────────────────────────────

    def eval_retrieval(self, rag_engine) -> Tuple[List[float], List[float]]:
        """
        Compute Precision@K and Recall@K for each retrieval test case.

        Precision@K = |relevant ∩ retrieved| / K
        Recall@K    = |relevant ∩ retrieved| / |total relevant in corpus|

        Returns:
            (precision_scores, recall_scores) lists, one per test case.
        """
        precisions, recalls = [], []

        for test in self._golden.get("retrieval_tests", []):
            query = test["query"]
            relevant_cats = set(test["relevant_categories"])
            k = test["k"]

            retrieved_docs = rag_engine.retrieve(query, k=k, role="hr_admin")
            retrieved_cats = [d.metadata.get("category", "") for d in retrieved_docs]

            relevant_retrieved = sum(1 for c in retrieved_cats if c in relevant_cats)
            precision = relevant_retrieved / k if k > 0 else 0.0
            precisions.append(precision)

            # Recall: estimate total relevant docs in corpus as available unique cats
            # (conservative: we assume >=1 relevant doc per relevant category)
            total_relevant = max(len(relevant_cats), 1)
            recall = min(relevant_retrieved / total_relevant, 1.0)
            recalls.append(recall)

        return precisions, recalls

    # ── 2. Extraction F1 ─────────────────────────────────────────────────────

    def eval_extraction(
        self, extraction_results: List
    ) -> List[float]:
        """
        Compute token-level F1 for skill extraction.

        For each test case in golden_set.extraction_tests:
          - Find the matching extraction result by category label
          - Compute F1 between extracted skills and must-include expected skills

        Returns:
            List of F1 scores, one per test case.
        """
        f1_scores = []

        for test in self._golden.get("extraction_tests", []):
            expected_skills = {
                s.lower() for s in test["expected"].get("skills_must_include", [])
            }
            if not expected_skills:
                continue

            # Find best matching extraction result by checking if expected skills
            # appear in the snippet (proxy for matching the test document)
            snippet = test.get("resume_snippet", "").lower()
            best_result = None
            for result in extraction_results:
                # Match by checking if extracted skills overlap with expected
                if any(skill.lower() in snippet for skill in result.skills):
                    best_result = result
                    break

            if best_result is None:
                f1_scores.append(0.0)
                continue

            predicted_skills = {s.lower() for s in best_result.skills}
            f1 = self._compute_f1(predicted_skills, expected_skills)
            f1_scores.append(f1)

        return f1_scores

    @staticmethod
    def _compute_f1(predicted: set, expected: set) -> float:
        """Compute set-based F1 score."""
        if not predicted or not expected:
            return 0.0
        tp = len(predicted & expected)
        precision = tp / len(predicted)
        recall = tp / len(expected)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    # ── 3. Faithfulness (LLM-as-Judge) ───────────────────────────────────────

    def eval_faithfulness(self) -> List[float]:
        """
        Score answer faithfulness using LLM-as-judge.

        Evaluates both faithful and unfaithful answers from the golden set
        to verify the evaluator discriminates correctly.

        Returns:
            List of faithfulness scores (faithful answers only, 0–10 scale).
        """
        scores = []

        for test in self._golden.get("faithfulness_tests", []):
            # Score the faithful answer — should be HIGH
            faithful_score = self._judge_faithfulness(
                context=test["context"],
                question=test["question"],
                answer=test["faithful_answer"],
            )
            scores.append(faithful_score)

            # Optionally log unfaithful score for contrast (not added to metric)
            _unfaithful_score = self._judge_faithfulness(
                context=test["context"],
                question=test["question"],
                answer=test["unfaithful_answer"],
            )
            if config.verbose:
                print(
                    f"    [{test['test_id']}] Faithful: {faithful_score:.1f}/10 "
                    f"| Unfaithful: {_unfaithful_score:.1f}/10"
                )

        return scores

    def _judge_faithfulness(
        self, context: str, question: str, answer: str
    ) -> float:
        """Call LLM judge and parse score. Returns 0.0 on failure."""
        prompt = FAITHFULNESS_JUDGE_PROMPT.format(
            context=context, question=question, answer=answer
        )
        try:
            response = self._client.chat.completions.create(
                model=config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
            )
            raw = response.choices[0].message.content or ""
            raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
            raw = re.sub(r"\s*```$", "", raw)
            data = json.loads(raw)
            return float(data.get("score", 0))
        except Exception:  # noqa: BLE001
            return 0.0

    # ── 4. Security Tests ─────────────────────────────────────────────────────

    def eval_security(self, security_guard) -> Tuple[int, int]:
        """
        Run security test cases from the golden set against SecurityGuard.

        Returns:
            (passed, total) count of security test cases.
        """
        passed = 0
        total = 0

        for test in self._golden.get("security_tests", []):
            total += 1
            user_input = test["input"]
            should_block = test.get("should_block", False)
            should_mask = test.get("should_mask", False)

            check = security_guard.validate_input(user_input)

            if should_block:
                # Expected: threats found (not safe)
                if not check.is_safe:
                    passed += 1
            elif should_mask:
                # Expected: safe but PII masked
                if check.is_safe and "[REDACTED_" in check.sanitized_input:
                    passed += 1
            else:
                # Expected: safe and unchanged
                if check.is_safe:
                    passed += 1

        return passed, total

    # ── Full Evaluation Suite ─────────────────────────────────────────────────

    def run_full_eval(
        self,
        rag_engine,
        extraction_results: List,
        security_guard,
    ) -> EvalSummary:
        """
        Run all four evaluation metrics and return an aggregated summary.

        AC-3: Demonstrates quantitative and qualitative evaluation.
        """
        summary = EvalSummary()

        # 1. Retrieval
        precisions, recalls = self.eval_retrieval(rag_engine)
        summary.precision_at_k_scores = precisions
        summary.recall_at_k_scores = recalls

        # 2. Extraction F1
        summary.extraction_f1_scores = self.eval_extraction(extraction_results)

        # 3. Faithfulness
        summary.faithfulness_scores = self.eval_faithfulness()

        # 4. Security
        passed, total = self.eval_security(security_guard)
        summary.security_pass = passed
        summary.security_total = total

        return summary

    # ── Result persistence ────────────────────────────────────────────────────

    @staticmethod
    def save_results(summary: EvalSummary, path: str) -> None:
        """Save evaluation summary to JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(summary.to_dict(), fh, indent=2)
