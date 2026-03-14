"""RAG Knowledge Base — ChromaDB vector store with video description best practices."""

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from src.config import config

# Best practices for writing video descriptions across platforms
KNOWLEDGE_DOCUMENTS = [
    {
        "id": "tiktok_caption_format",
        "text": (
            "TikTok Caption Format: First 150 characters are visible before 'more' — put the hook there. "
            "Structure: hook line → 1-2 sentences of context → call-to-action → hashtags on a new line. "
            "Use line breaks for readability. 2-4 emojis boost engagement 15-25%. "
            "Keep total length under 300 chars for best performance. Max 2200 chars. "
            "5-7 hashtags: mix 1-2 broad (1M+ views) with 3-4 niche. "
            "Avoid #fyp and #foryou — oversaturated and ineffective."
        ),
        "metadata": {"category": "description_format", "platform": "tiktok"},
    },
    {
        "id": "reels_caption_format",
        "text": (
            "Instagram Reels Caption Format: First ~125 characters visible before 'more'. "
            "Storytelling tone works better than TikTok's punchy style. "
            "Structure: attention-grabbing opener → narrative/value → CTA (save, share, comment) → hashtags. "
            "Emojis are expected and boost engagement. 5-10 hashtags at the end or in first comment. "
            "Mix content-specific, community, and broad hashtags. Max 2200 chars. "
            "Longer captions perform well on Reels if they tell a story."
        ),
        "metadata": {"category": "description_format", "platform": "instagram"},
    },
    {
        "id": "shorts_caption_format",
        "text": (
            "YouTube Shorts Caption Format: Title is king — max 100 chars, must be SEO-friendly and catchy. "
            "Description: 2-3 keyword-rich sentences for search discovery. "
            "Max 3 hashtags — #Shorts is recommended plus 2 topic-relevant tags. "
            "Include a CTA to subscribe or watch full-length content. "
            "YouTube Shorts descriptions are indexed for search — treat them like mini-SEO. "
            "Max 5000 chars for description but keep it concise."
        ),
        "metadata": {"category": "description_format", "platform": "youtube"},
    },
    {
        "id": "hook_writing",
        "text": (
            "Writing Hooks for Video Descriptions: The caption hook should mirror the video hook. "
            "Effective patterns: provocative question ('Did you know X can do Y?'), bold claim "
            "('This changed everything'), result-first ('Here's what happened when...'), "
            "curiosity gap ('The trick no one talks about'). "
            "Avoid generic hooks like 'Wait for it' or 'Watch till the end'. Be specific to the content. "
            "The hook must make sense without watching the video — it's what drives the click."
        ),
        "metadata": {"category": "hooks", "platform": "all"},
    },
    {
        "id": "cta_in_descriptions",
        "text": (
            "Call-to-Action in Video Descriptions: Question CTAs ('Which would you pick?') drive comments. "
            "Save CTAs ('Save this for later') signal value to the algorithm. "
            "Share CTAs ('Send this to someone who needs it') expand reach. "
            "On TikTok, comments-focused CTAs drive the algorithm hardest. "
            "On Reels, save-focused CTAs are weighted heavily. "
            "On Shorts, subscribe CTAs work best ('Subscribe for more X'). "
            "Match the CTA to the content — don't ask 'Which is your favorite?' on a tutorial."
        ),
        "metadata": {"category": "cta", "platform": "all"},
    },
    {
        "id": "description_mistakes",
        "text": (
            "Common Video Description Mistakes: Too many hashtags (10+) looks spammy on TikTok. "
            "No hook in first line — users scroll past. Generic CTAs that don't match the content. "
            "No line breaks — wall of text is unreadable. Empty captions lose ~30% potential reach. "
            "Using the same caption across all platforms — each has different audience expectations. "
            "Hashtags mid-sentence break reading flow — always place at the end. "
            "Duplicating video text overlay in the caption instead of complementing it."
        ),
        "metadata": {"category": "mistakes", "platform": "all"},
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
