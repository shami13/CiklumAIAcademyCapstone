"""Hashtag Researcher Tool — suggests relevant hashtags for TikTok, Instagram Reels, and YouTube Shorts."""

from langchain_core.tools import tool
from openai import OpenAI

from src.config import config

client = OpenAI(api_key=config.OPENAI_API_KEY)


@tool
def research_hashtags(video_analysis: str, niche: str = "general") -> str:
    """Research and suggest relevant hashtags for TikTok, Instagram Reels, and YouTube Shorts.

    Args:
        video_analysis: The analysis of the video content from the video analyzer.
        niche: Content niche (e.g., 'tech', 'lifestyle', 'gaming', 'general').

    Returns:
        A curated list of hashtags per platform with explanations.
    """
    prompt = f"""You are a short-form video content strategy expert. Based on this video analysis,
suggest the best hashtags for TikTok, Instagram Reels, and YouTube Shorts.

Video Analysis:
{video_analysis}

Content Niche: {niche}

For EACH platform, provide:

=== TIKTOK (5-7 hashtags) ===
- 3-4 primary hashtags (high relevance)
- 2-3 trending/discovery hashtags (broader reach)
- Avoid oversaturated ones like #fyp

=== INSTAGRAM REELS (5-10 hashtags) ===
- 3-5 content-specific hashtags
- 2-3 community/niche hashtags
- 2 broad discovery hashtags

=== YOUTUBE SHORTS (3 hashtags) ===
- #Shorts (always include)
- 2 SEO-focused keyword hashtags

Format all hashtags with # prefix, ready to copy-paste.
Consider: hashtag volume, relevance, competition, and current platform trends.
"""

    response = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
    )

    return response.choices[0].message.content
