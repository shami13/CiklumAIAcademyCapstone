"""Instagram Scraper — fetches public profile posts using instaloader."""

import itertools
import logging
import re
from typing import Optional

import instaloader

from src.config import config
from src.rag.post_importer import (
    _generate_post_id,
    _get_posts_collection,
    _post_to_document,
    _post_to_metadata,
)

logger = logging.getLogger(__name__)

_TYPENAME_MAP = {
    "GraphImage": "image",
    "GraphVideo": "video",
    "GraphSidecar": "carousel",
}


def extract_username(input_str: str) -> str:
    """Extract Instagram username from a URL or plain username string.

    Args:
        input_str: Either 'username' or 'https://www.instagram.com/username/' etc.

    Returns:
        The cleaned username string.

    Raises:
        ValueError: If the input is empty or cannot be parsed.
    """
    input_str = input_str.strip().rstrip("/")
    if not input_str:
        raise ValueError("Instagram username or URL cannot be empty.")

    if "instagram.com" in input_str:
        match = re.search(r"instagram\.com/([A-Za-z0-9._]+)", input_str)
        if match:
            return match.group(1)
        raise ValueError(f"Could not extract username from URL: {input_str}")

    # Treat as plain username — strip leading @
    username = input_str.lstrip("@")
    if not re.match(r"^[A-Za-z0-9._]{1,30}$", username):
        raise ValueError(f"Invalid Instagram username: {username}")
    return username


def _map_typename_to_content_type(typename: str) -> str:
    """Map Instagram GraphQL typename to a content type string."""
    return _TYPENAME_MAP.get(typename, "unknown")


def _instaloader_post_to_dict(post) -> dict:
    """Convert an instaloader.Post to the standard post dict format.

    Args:
        post: An instaloader.Post object.

    Returns:
        Dict with keys: caption, hashtags, likes, comments, shares, views,
        posted_at, content_type.
    """
    caption: str = post.caption or ""
    hashtags = sorted(re.findall(r"#\w+", caption))
    return {
        "caption": caption,
        "hashtags": hashtags,
        "likes": post.likes,
        "comments": post.comments,
        "shares": 0,
        "views": 0,
        "posted_at": post.date_utc.isoformat(),
        "content_type": _map_typename_to_content_type(post.typename),
    }


def fetch_instagram_posts(username: str, post_count: int = 10) -> list[dict]:
    """Fetch recent posts from a public Instagram profile.

    Args:
        username: Instagram username (without @).
        post_count: Maximum number of posts to fetch (default 10).

    Returns:
        List of post dicts in the standard format.

    Raises:
        ValueError: If profile not found, is private, or on network error.
    """
    L = instaloader.Instaloader(
        download_pictures=False,
        download_videos=False,
        download_video_thumbnails=False,
        download_geotags=False,
        download_comments=False,
        save_metadata=False,
        compress_json=False,
        quiet=not config.INSTAGRAM_USERNAME,
    )

    if config.INSTAGRAM_USERNAME:
        try:
            L.load_session_from_file(config.INSTAGRAM_USERNAME)
            logger.info("Loaded saved session for %s", config.INSTAGRAM_USERNAME)
        except FileNotFoundError:
            L.interactive_login(config.INSTAGRAM_USERNAME)
            L.save_session_to_file()

    try:
        profile = instaloader.Profile.from_username(L.context, username)
    except instaloader.exceptions.ProfileNotExistsException:
        raise ValueError(f"Instagram profile '{username}' not found.")
    except instaloader.exceptions.ConnectionException:
        raise ValueError("Could not connect to Instagram. Check your network.")

    if profile.is_private:
        raise ValueError(
            f"Profile '{username}' is private. Only public profiles are supported."
        )

    posts = []
    for post in itertools.islice(profile.get_posts(), post_count):
        try:
            posts.append(_instaloader_post_to_dict(post))
        except Exception as e:
            logger.warning("Skipping post: %s", e)
            continue

    return posts


def scrape_and_import_instagram(username_or_url: str, post_count: int = 10) -> int:
    """Scrape Instagram profile and import posts into ChromaDB.

    Args:
        username_or_url: Instagram username or profile URL.
        post_count: Number of recent posts to fetch.

    Returns:
        Number of posts imported.

    Raises:
        ValueError: If profile cannot be scraped.
    """
    username = extract_username(username_or_url)
    posts = fetch_instagram_posts(username, post_count)

    if not posts:
        return 0

    platform = "instagram"
    collection = _get_posts_collection()

    ids = [_generate_post_id(p.get("caption", ""), platform) for p in posts]
    documents = [_post_to_document(p, platform) for p in posts]
    metadatas = [_post_to_metadata(p, platform) for p in posts]

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    return len(posts)
