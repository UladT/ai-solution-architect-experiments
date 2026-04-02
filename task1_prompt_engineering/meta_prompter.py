"""
Meta-Prompting Module
AC-1: Demonstrates meta-prompting - using LLM to refine prompts.
AC-4: Supports quality improvement through automated prompt refinement.
"""

from openai import OpenAI
import json


class MetaPrompter:
    """
    Uses LLM to analyze and improve prompts.
    AC-1: Fulfills meta-prompting requirement.
    AC-4: Automated quality improvement mechanism.
    """

    META_PROMPT_TEMPLATE = """You are an expert prompt engineer 
specializing in ReAct and self-reflection prompting patterns.

Analyze the following prompt and provide:

1. WEAKNESSES: List 3-5 specific weaknesses
2. IMPROVEMENTS: Concrete suggestions for each weakness  
3. QUALITY_SCORE: Rate the prompt 1-10 with justification
4. IMPROVED_SECTION: Rewrite the weakest section only
5. MISSING_ELEMENTS: What's missing for production use?

Respond in valid JSON format:
{{
    "weaknesses": ["weakness1", "weakness2", ...],
    "improvements": ["improvement1", "improvement2", ...],
    "quality_score": X,
    "quality_justification": "...",
    "improved_section": "...",
    "missing_elements": ["element1", "element2", ...]
}}

PROMPT TO ANALYZE:
---
{prompt_to_analyze}
---"""

    def __init__(self, client: OpenAI, model: str = "gpt-4"):
        self.client = client
        self.model = model

    def refine_prompt(
        self,
        original_prompt: str,
        verbose: bool = True
    ) -> dict:
        """
        Use LLM to analyze and suggest improvements for a prompt.
        AC-1: Meta-prompting with actual LLM execution.

        Args:
            original_prompt: The prompt to be refined
            verbose: Print results to console

        Returns:
            dict with analysis results
        """
        meta_prompt = self.META_PROMPT_TEMPLATE.format(
            prompt_to_analyze=original_prompt
        )

        if verbose:
            print("\n🔄 META-PROMPTING: Sending prompt for LLM analysis...")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert prompt engineer. "
                               "Always respond with valid JSON only."
                },
                {
                    "role": "user",
                    "content": meta_prompt
                }
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        raw_result = response.choices[0].message.content

        try:
            result = json.loads(raw_result)
        except json.JSONDecodeError:
            result = {"raw_response": raw_result, "parse_error": True}

        if verbose:
            self._print_meta_results(result)

        return result

    def compare_prompts(
        self,
        prompt_v1: str,
        prompt_v2: str,
        test_input: str,
        client: OpenAI
    ) -> dict:
        """
        A/B test two prompt versions on same input.
        AC-2: Demonstrates scalability across prompt versions.
        AC-4: Quality improvement through comparison.

        Args:
            prompt_v1: First prompt version (poor)
            prompt_v2: Second prompt version (improved)
            test_input: Same input to test both prompts
            client: OpenAI client

        Returns:
            Comparison results dict
        """
        print("\n🔬 A/B TESTING: Comparing prompt versions...")

        results = {}
        for version, prompt in [("v1_poor", prompt_v1),
                                 ("v2_improved", prompt_v2)]:
            print(f"  Testing {version}...")
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": test_input}
                ],
                temperature=0.3,
                max_tokens=500
            )
            results[version] = {
                "response": response.choices[0].message.content,
                "tokens_used": response.usage.total_tokens
            }

        # Simple comparison metrics
        v1_len = len(results["v1_poor"]["response"])
        v2_len = len(results["v2_improved"]["response"])

        results["comparison"] = {
            "v1_response_length": v1_len,
            "v2_response_length": v2_len,
            "length_improvement": f"{((v2_len - v1_len) / v1_len * 100):.1f}%",
            "v1_tokens": results["v1_poor"]["tokens_used"],
            "v2_tokens": results["v2_improved"]["tokens_used"],
        }

        print("\n📊 A/B TEST RESULTS:")
        print(f"  V1 (poor) response length:    {v1_len} chars")
        print(f"  V2 (improved) response length: {v2_len} chars")
        print(f"  Improvement: {results['comparison']['length_improvement']}")

        return results

    def _print_meta_results(self, result: dict) -> None:
        """Print meta-prompting results."""
        print("\n📋 META-PROMPT ANALYSIS RESULTS:")
        print(f"  Quality Score: {result.get('quality_score', 'N/A')}/10")
        print(f"  Justification: {result.get('quality_justification', 'N/A')}")

        print("\n  Weaknesses Found:")
        for w in result.get("weaknesses", []):
            print(f"    ❌ {w}")

        print("\n  Suggested Improvements:")
        for i in result.get("improvements", []):
            print(f"    ✅ {i}")

        print("\n  Missing Elements:")
        for m in result.get("missing_elements", []):
            print(f"    ⚠️  {m}")