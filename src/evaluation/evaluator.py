"""Evaluator — self-reflection and quality scoring for generated TikTok descriptions."""

import json

from langchain_core.tools import tool
from openai import OpenAI

from src.config import config

client = OpenAI(api_key=config.OPENAI_API_KEY)


@tool
def evaluate_description(
    description: str,
    video_analysis: str,
) -> str:
    """Evaluate the quality of a generated TikTok description and provide improvement suggestions.

    This implements the self-reflection mechanism: the agent scores its own output
    and decides whether to retry or proceed to publishing.

    Args:
        description: The generated TikTok description to evaluate.
        video_analysis: The original video analysis for relevance checking.

    Returns:
        JSON evaluation with scores and improvement suggestions.
    """
    prompt = f"""You are a TikTok content quality evaluator. Score this TikTok description 
on multiple criteria and provide actionable feedback.

GENERATED DESCRIPTION:
{description}

ORIGINAL VIDEO ANALYSIS:
{video_analysis}

Score each criterion from 1-10 and provide a brief justification:

1. **Hook Quality**: Does the first line grab attention? Is it specific, not generic?
2. **Relevance**: Does the description accurately reflect the video content?
3. **Engagement Potential**: Will this description drive likes, comments, shares?
4. **Hashtag Quality**: Are hashtags relevant, well-mixed (popular + niche), not too many?
5. **CTA Effectiveness**: Is there a clear, natural call-to-action?
6. **Readability**: Is formatting clean? Line breaks? Appropriate length?
7. **Originality**: Does it feel unique, not templated?

RESPOND IN THIS EXACT JSON FORMAT (no markdown, no backticks):
{{
    "scores": {{
        "hook_quality": {{"score": 8, "reason": "..."}},
        "relevance": {{"score": 7, "reason": "..."}},
        "engagement_potential": {{"score": 6, "reason": "..."}},
        "hashtag_quality": {{"score": 8, "reason": "..."}},
        "cta_effectiveness": {{"score": 7, "reason": "..."}},
        "readability": {{"score": 9, "reason": "..."}},
        "originality": {{"score": 6, "reason": "..."}}
    }},
    "overall_score": 7.3,
    "verdict": "PASS or NEEDS_IMPROVEMENT",
    "improvements": ["specific suggestion 1", "specific suggestion 2"],
    "improved_description": "If verdict is NEEDS_IMPROVEMENT, provide a rewritten version here. Otherwise null."
}}
"""

    response = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
        temperature=0.3,  # Lower temperature for more consistent scoring
    )

    raw = response.choices[0].message.content.strip()

    # Clean up potential markdown wrapping
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()

    # Validate JSON
    try:
        evaluation = json.loads(raw)

        # Compute overall score if not provided correctly
        scores = evaluation.get("scores", {})
        if scores:
            avg = sum(s.get("score", 0) for s in scores.values()) / len(scores)
            evaluation["overall_score"] = round(avg, 1)

        # Set verdict based on threshold
        evaluation["verdict"] = (
            "PASS" if evaluation["overall_score"] >= config.MIN_QUALITY_SCORE
            else "NEEDS_IMPROVEMENT"
        )

        return json.dumps(evaluation, indent=2, ensure_ascii=False)

    except json.JSONDecodeError:
        # If parsing fails, return raw text with a warning
        return json.dumps({
            "error": "Could not parse evaluation as JSON",
            "raw_evaluation": raw,
            "overall_score": 5,
            "verdict": "NEEDS_IMPROVEMENT",
        }, indent=2)
