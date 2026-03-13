"""Hashtag Researcher Tool — suggests relevant hashtags based on video content analysis."""

from langchain_core.tools import tool
from openai import OpenAI

from src.config import config

client = OpenAI(api_key=config.OPENAI_API_KEY)


@tool
def research_hashtags(video_analysis: str, niche: str = "general") -> str:
    """Research and suggest relevant TikTok hashtags based on video content analysis.

    Args:
        video_analysis: The analysis of the video content from the video analyzer.
        niche: Content niche (e.g., 'tech', 'lifestyle', 'gaming', 'general').

    Returns:
        A curated list of hashtags with explanations.
    """
    prompt = f"""You are a TikTok content strategy expert. Based on this video analysis, suggest 
the best hashtags for maximum reach and engagement.

Video Analysis:
{video_analysis}

Content Niche: {niche}

Provide:
1. 5-7 primary hashtags (high relevance to content)
2. 3-5 trending/discovery hashtags (broader reach)
3. 2-3 niche-specific hashtags (targeted audience)
4. Brief explanation of why each group was chosen

Format the hashtags with # prefix, ready to copy-paste.
Consider: hashtag volume, relevance, competition, and current TikTok trends.
"""

    response = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
    )

    return response.choices[0].message.content
