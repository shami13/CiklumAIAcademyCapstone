"""RAG Knowledge Base — ChromaDB vector store with TikTok content best practices."""

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from src.config import config

# Best practices knowledge documents
KNOWLEDGE_DOCUMENTS = [
    {
        "id": "hook_strategies",
        "text": (
            "TikTok Hook Strategies: The first 1-3 seconds determine if viewers keep watching. "
            "Effective hooks include: asking a provocative question, making a bold claim, "
            "showing the end result first, using pattern interrupts, starting mid-action. "
            "The caption hook should mirror the video hook — if the video starts with a question, "
            "the caption should too. Avoid generic hooks like 'Wait for it' — be specific."
        ),
        "metadata": {"category": "hooks", "platform": "tiktok"},
    },
    {
        "id": "caption_structure",
        "text": (
            "TikTok Caption Structure: Optimal captions follow a pattern: Hook line (first 150 chars "
            "visible before 'more'), context/value in 1-2 short sentences, call-to-action, then "
            "hashtags on a new line. Use line breaks for readability. Emojis increase engagement by "
            "15-25% when used sparingly (2-4 per caption). Keep total length under 300 chars for "
            "best engagement — longer captions get lower completion rates."
        ),
        "metadata": {"category": "captions", "platform": "tiktok"},
    },
    {
        "id": "hashtag_strategy",
        "text": (
            "TikTok Hashtag Strategy: Use 3-5 hashtags for optimal reach. Mix: 1-2 broad hashtags "
            "(1M+ views), 2-3 niche hashtags (100K-1M views), 1 branded or unique hashtag. "
            "Avoid banned or shadowbanned hashtags. Place hashtags at the end of the caption, "
            "not inline. Trending hashtags boost discovery but only if relevant to content. "
            "Don't use #fyp or #foryou — they're oversaturated and don't help."
        ),
        "metadata": {"category": "hashtags", "platform": "tiktok"},
    },
    {
        "id": "cta_patterns",
        "text": (
            "TikTok Call-to-Action Patterns: Direct CTAs ('Follow for more') work but feel pushy. "
            "Better patterns: question CTAs ('Which one would you pick?'), challenge CTAs "
            "('Try this and tag me'), save CTAs ('Save this for later'), share CTAs "
            "('Send this to someone who needs it'). Comments-focused CTAs drive the algorithm "
            "hardest — the more comments, the more TikTok pushes the video."
        ),
        "metadata": {"category": "engagement", "platform": "tiktok"},
    },
    {
        "id": "posting_timing",
        "text": (
            "TikTok Posting Best Practices: Post when your audience is active — generally "
            "7-9 AM, 12-2 PM, and 7-11 PM in target timezone. Consistency matters more than "
            "timing — 3-5 posts per week minimum. First 30 minutes after posting are critical "
            "for initial engagement signals. Respond to every comment in the first hour. "
            "Cross-post to Instagram Reels and YouTube Shorts for extra reach."
        ),
        "metadata": {"category": "timing", "platform": "tiktok"},
    },
    {
        "id": "content_categories",
        "text": (
            "TikTok Content Categories That Perform Well: Educational content ('Did you know...'), "
            "behind-the-scenes, transformation/before-after, tutorials, day-in-the-life, "
            "trending sounds with original twist, reaction content, storytelling. "
            "Tech content works best as quick tips, gadget reveals, or setup tours. "
            "Smart home content performs well with 'oddly satisfying' automation demos."
        ),
        "metadata": {"category": "content_types", "platform": "tiktok"},
    },
    {
        "id": "engagement_optimization",
        "text": (
            "TikTok Engagement Optimization: Videos with text overlays get 40% more engagement. "
            "Captions should complement, not duplicate the video text. Use curiosity gaps — "
            "hint at something in the caption that requires watching the full video. "
            "Controversy and hot takes drive comments (but stay authentic). "
            "Replying to comments with new videos boosts both old and new content."
        ),
        "metadata": {"category": "engagement", "platform": "tiktok"},
    },
    {
        "id": "description_mistakes",
        "text": (
            "Common TikTok Caption Mistakes: Too many hashtags (10+) looks spammy. "
            "No hook in first line — viewers scroll past. Generic CTAs that don't match content. "
            "Not using line breaks — wall of text is unreadable. Copying other creators' captions "
            "verbatim. Ignoring the caption entirely (empty captions lose 30% potential reach). "
            "Using hashtags mid-sentence breaks reading flow."
        ),
        "metadata": {"category": "mistakes", "platform": "tiktok"},
    },
]

_collection = None


def _get_collection() -> chromadb.Collection:
    """Get or create the ChromaDB collection."""
    global _collection
    if _collection is not None:
        return _collection

    embedding_fn = OpenAIEmbeddingFunction(
        api_key=config.OPENAI_API_KEY,
        model_name=config.OPENAI_EMBEDDING_MODEL,
    )

    client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
    _collection = client.get_or_create_collection(
        name="tiktok_knowledge",
        embedding_function=embedding_fn,
    )
    return _collection


def seed_knowledge_base() -> int:
    """Seed the vector store with knowledge documents. Returns count of documents added."""
    collection = _get_collection()

    # Check if already seeded
    if collection.count() >= len(KNOWLEDGE_DOCUMENTS):
        return collection.count()

    collection.add(
        ids=[doc["id"] for doc in KNOWLEDGE_DOCUMENTS],
        documents=[doc["text"] for doc in KNOWLEDGE_DOCUMENTS],
        metadatas=[doc["metadata"] for doc in KNOWLEDGE_DOCUMENTS],
    )

    return collection.count()


def query_knowledge_base(query: str, n_results: int = 3) -> str:
    """Query the knowledge base for relevant best practices.

    Args:
        query: Search query describing the content or topic.
        n_results: Number of results to return.

    Returns:
        Concatenated relevant knowledge documents.
    """
    collection = _get_collection()

    # Auto-seed if empty
    if collection.count() == 0:
        seed_knowledge_base()

    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, collection.count()),
    )

    if not results["documents"] or not results["documents"][0]:
        return "No relevant best practices found."

    documents = results["documents"][0]
    return "\n\n---\n\n".join(documents)


# Allow running as script to seed the knowledge base
if __name__ == "__main__":
    count = seed_knowledge_base()
    print(f"Knowledge base seeded with {count} documents.")
