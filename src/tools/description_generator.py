"""Description Generator Tool — creates optimized TikTok captions using analysis + RAG context."""

from langchain_core.tools import tool
from openai import OpenAI

from src.config import config
from src.rag.knowledge_base import query_knowledge_base

client = OpenAI(api_key=config.OPENAI_API_KEY)


@tool
def generate_description(
    video_analysis: str,
    hashtags: str,
    style: str = "engaging",
) -> str:
    """Generate an optimized TikTok description combining video analysis, hashtags, and best practices from RAG.

    Args:
        video_analysis: Detailed analysis of the video content.
        hashtags: Suggested hashtags from the hashtag researcher.
        style: Desired tone — 'engaging', 'informative', 'funny', 'professional'.

    Returns:
        A complete TikTok description ready to publish.
    """
    # Query RAG for relevant best practices
    rag_context = query_knowledge_base(
        query=f"Best practices for TikTok description: {video_analysis[:200]}",
        n_results=3,
    )

    prompt = f"""You are an expert TikTok content creator. Generate an optimized TikTok video 
description (caption) that maximizes engagement.

VIDEO ANALYSIS:
{video_analysis}

SUGGESTED HASHTAGS:
{hashtags}

BEST PRACTICES FROM KNOWLEDGE BASE:
{rag_context}

DESIRED STYLE: {style}

REQUIREMENTS:
- Start with a hook (first line must grab attention — question, bold statement, or emoji)
- Keep the main text under 150 characters (TikTok shows first ~150 chars before "more")
- Include a clear call-to-action (follow, like, comment, share)
- Add the most relevant hashtags (max 5-7, mix of popular and niche)
- Use line breaks for readability
- Match the tone to the video content and desired style
- TikTok max caption length is 2200 characters

OUTPUT FORMAT:
Return ONLY the final caption text, ready to copy-paste to TikTok. No explanations.
"""

    response = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
    )

    return response.choices[0].message.content
