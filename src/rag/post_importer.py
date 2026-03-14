"""Personal Posts Collection — manages scraped posts in ChromaDB for RAG."""

import hashlib

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from src.config import config

_posts_collection = None


def _get_posts_collection() -> chromadb.Collection:
    """Get or create the personal_posts ChromaDB collection."""
    global _posts_collection
    if _posts_collection is not None:
        return _posts_collection

    embedding_fn = OpenAIEmbeddingFunction(
        api_key=config.OPENAI_API_KEY,
        model_name=config.OPENAI_EMBEDDING_MODEL,
    )

    client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
    _posts_collection = client.get_or_create_collection(
        name="personal_posts",
        embedding_function=embedding_fn,
    )
    return _posts_collection


def _calculate_engagement_rate(post: dict) -> float:
    """Calculate engagement rate from metrics."""
    views = post.get("views", 0)
    if views == 0:
        return 0.0
    engagement = post.get("likes", 0) + post.get("comments", 0) + post.get("shares", 0)
    return round((engagement / views) * 100, 2)


def _post_to_document(post: dict, platform: str) -> str:
    """Convert a post dict into a text document suitable for embedding."""
    views = post.get("views", 0)
    likes = post.get("likes", 0)
    comments = post.get("comments", 0)
    shares = post.get("shares", 0)
    engagement_rate = _calculate_engagement_rate(post)
    content_type = post.get("content_type", "unknown")

    hashtags = post.get("hashtags", [])
    if isinstance(hashtags, str):
        hashtags = [h.strip() for h in hashtags.split(";") if h.strip()]
    hashtag_str = " ".join(hashtags) if hashtags else "none"

    caption = post.get("caption", "").strip()

    return (
        f"[{platform} post, {views} views, {likes} likes, "
        f"{comments} comments, {shares} shares, "
        f"engagement rate: {engagement_rate}%]\n"
        f"Caption: {caption}\n"
        f"Hashtags: {hashtag_str}\n"
        f"Content type: {content_type}"
    )


def _post_to_metadata(post: dict, platform: str) -> dict:
    """Extract metadata dict from a post."""
    return {
        "platform": platform,
        "likes": post.get("likes", 0),
        "comments": post.get("comments", 0),
        "shares": post.get("shares", 0),
        "views": post.get("views", 0),
        "engagement_rate": _calculate_engagement_rate(post),
        "posted_at": post.get("posted_at", ""),
        "content_type": post.get("content_type", "unknown"),
        "source": "personal_import",
    }


def _generate_post_id(caption: str, platform: str) -> str:
    """Generate a stable ID from caption text to avoid duplicates on re-import."""
    caption_hash = hashlib.md5(caption.encode("utf-8")).hexdigest()[:10]
    return f"post_{platform}_{caption_hash}"


def query_personal_posts(query: str, n_results: int = 3, platform: str | None = None) -> str:
    """Query personal posts for similar high-performing content.

    Args:
        query: Search query describing the content.
        n_results: Number of results to return.
        platform: Optional filter by platform ('tiktok' or 'instagram').

    Returns:
        Concatenated relevant personal post documents, or empty string if none.
    """
    collection = _get_posts_collection()

    if collection.count() == 0:
        return ""

    where_filter = {"platform": platform} if platform else None

    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, collection.count()),
        where=where_filter,
    )

    if not results["documents"] or not results["documents"][0]:
        return ""

    return "\n\n---\n\n".join(results["documents"][0])


