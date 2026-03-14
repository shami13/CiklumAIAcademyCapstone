"""Evaluator — self-reflection and quality scoring for generated descriptions."""

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
    """Evaluate the quality of generated descriptions for TikTok, Instagram Reels, and YouTube Shorts.

    This implements the self-reflection mechanism: the agent scores its own output
    and decides whether to retry or proceed to publishing.

    Args:
        description: The generated multi-platform descriptions to evaluate.
        video_analysis: The original video analysis for relevance checking.

    Returns:
        JSON evaluation with scores and improvement suggestions.
    """
    prompt = f"""You are a short-form video content quality evaluator. Score these descriptions
for TikTok, Instagram Reels, and YouTube Shorts on multiple criteria and provide actionable feedback.

GENERATED DESCRIPTIONS:
{description}

ORIGINAL VIDEO ANALYSIS:
{video_analysis}

Score each criterion from 1-10 and provide a brief justification:

1. **Hook Quality**: Does each platform's first line grab attention? Is it specific, not generic?
2. **Relevance**: Do the descriptions accurately reflect the video content?
3. **Engagement Potential**: Will these descriptions drive likes, comments, shares?
4. **Hashtag Quality**: Are hashtags relevant, well-mixed, platform-appropriate counts?
5. **CTA Effectiveness**: Is there a clear, natural call-to-action on each platform?
6. **Readability**: Is formatting clean? Line breaks? Appropriate length per platform?
7. **Originality**: Do the descriptions feel unique, not templated?
8. **Platform Adaptation**: Are descriptions properly tailored to each platform's style and constraints?

RESPOND IN THIS EXACT JSON FORMAT (no markdown, no backticks):
{{
    "scores": {{
        "hook_quality": {{"score": 8, "reason": "..."}},
        "relevance": {{"score": 7, "reason": "..."}},
        "engagement_potential": {{"score": 6, "reason": "..."}},
        "hashtag_quality": {{"score": 8, "reason": "..."}},
        "cta_effectiveness": {{"score": 7, "reason": "..."}},
        "readability": {{"score": 9, "reason": "..."}},
        "originality": {{"score": 6, "reason": "..."}},
        "platform_adaptation": {{"score": 7, "reason": "..."}}
    }},
    "overall_score": 7.3,
    "verdict": "PASS or NEEDS_IMPROVEMENT",
    "improvements": ["specific suggestion 1", "specific suggestion 2"],
    "improved_description": "If verdict is NEEDS_IMPROVEMENT, provide improved versions for all 3 platforms in the same ---TIKTOK---/---INSTAGRAM REELS---/---YOUTUBE SHORTS--- format. Otherwise null."
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
