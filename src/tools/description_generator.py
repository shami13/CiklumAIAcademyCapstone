"""Description Generator Tool — creates optimized captions for TikTok, Instagram Reels, and YouTube Shorts."""

from langchain_core.tools import tool
from openai import OpenAI

from src.config import config
from src.rag.knowledge_base import query_knowledge_base
from src.rag.post_importer import query_personal_posts

client = OpenAI(api_key=config.OPENAI_API_KEY)


@tool
def generate_description(
    video_analysis: str,
    hashtags: str,
    style: str = "engaging",
) -> str:
    """Generate optimized descriptions for TikTok, Instagram Reels, and YouTube Shorts.

    Combines video analysis, hashtags, and best practices from RAG to create
    platform-specific captions tailored to each platform's requirements.

    Args:
        video_analysis: Detailed analysis of the video content.
        hashtags: Suggested hashtags from the hashtag researcher.
        style: Desired tone — 'engaging', 'informative', 'funny', 'professional'.

    Returns:
        Descriptions for all three platforms in a structured format.
    """
    # Query RAG for relevant best practices
    rag_context = query_knowledge_base(
        query=f"Best practices for short-form video description: {video_analysis[:200]}",
        n_results=3,
    )

    # Query personal post history for similar content
    personal_context = query_personal_posts(
        query=f"Similar content to: {video_analysis[:200]}",
        n_results=3,
    )

    personal_section = ""
    if personal_context:
        personal_section = f"""
YOUR TOP-PERFORMING SIMILAR POSTS:
{personal_context}

LEARNING FROM YOUR HISTORY:
- Study the caption style, hashtag choices, and tone from your top posts above
- Identify what made these posts successful (hooks, CTAs, hashtag mix)
- Adapt similar patterns for this new video while keeping it fresh
"""

    prompt = f"""You are an expert short-form video content creator. Generate optimized
descriptions for THREE platforms based on the same video. Each platform has different
requirements and audience behavior.

VIDEO ANALYSIS:
{video_analysis}

SUGGESTED HASHTAGS:
{hashtags}

BEST PRACTICES FROM KNOWLEDGE BASE:
{rag_context}
{personal_section}
DESIRED STYLE: {style}

Generate a description for EACH platform following these rules:

=== TIKTOK ===
- Start with a hook (first line grabs attention)
- Main text under 150 characters (visible before "more")
- Clear call-to-action
- 5-7 hashtags (mix of popular and niche)
- Max 2200 characters total
- Line breaks for readability

=== INSTAGRAM REELS ===
- Strong opening hook (first ~125 chars visible before "more")
- More descriptive than TikTok — storytelling works well
- Include a CTA (save, share, comment)
- 5-10 hashtags (placed at the end or in first comment)
- Emojis enhance engagement
- Max 2200 characters total

=== YOUTUBE SHORTS ===
- Title: catchy, SEO-friendly, max 100 characters
- Description: 2-3 sentences with keywords for search
- Include relevant hashtags (max 3, #Shorts is recommended)
- Add a CTA to subscribe or watch the full video
- Max 5000 characters for description

OUTPUT FORMAT (use exactly this structure):
---TIKTOK---
[TikTok caption here]

---INSTAGRAM REELS---
[Instagram Reels caption here]

---YOUTUBE SHORTS---
Title: [YouTube Shorts title here]
Description: [YouTube Shorts description here]

Return ONLY the captions in the format above. No explanations or commentary.
"""

    response = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
    )

    return response.choices[0].message.content
