"""TikTok Publisher Tool — publishes video + description via TikTok Content Posting API.

TikTok Content Posting API docs:
https://developers.tiktok.com/doc/content-posting-api-get-started

This tool supports two modes:
- Dry run (default): Simulates publishing and returns what would be posted
- Live: Actually publishes to TikTok using the Content Posting API
"""

import json
import os
from datetime import datetime

import requests
from langchain_core.tools import tool

from src.config import config

TIKTOK_API_BASE = "https://open.tiktokapis.com/v2"


def _init_video_upload(video_path: str) -> dict:
    """Initialize a video upload to TikTok (step 1 of Content Posting API)."""
    file_size = os.path.getsize(video_path)

    headers = {
        "Authorization": f"Bearer {config.TIKTOK_ACCESS_TOKEN}",
        "Content-Type": "application/json; charset=UTF-8",
    }

    payload = {
        "post_info": {
            "privacy_level": "SELF_ONLY",  # Safe default; user can change on TikTok
            "disable_comment": False,
            "disable_duet": False,
            "disable_stitch": False,
        },
        "source_info": {
            "source": "FILE_UPLOAD",
            "video_size": file_size,
            "chunk_size": file_size,
            "total_chunk_count": 1,
        },
    }

    response = requests.post(
        f"{TIKTOK_API_BASE}/post/publish/inbox/video/init/",
        headers=headers,
        json=payload,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def _upload_video_chunk(upload_url: str, video_path: str) -> bool:
    """Upload the video file to TikTok (step 2)."""
    file_size = os.path.getsize(video_path)

    headers = {
        "Content-Range": f"bytes 0-{file_size - 1}/{file_size}",
        "Content-Type": "video/mp4",
    }

    with open(video_path, "rb") as f:
        response = requests.put(upload_url, headers=headers, data=f, timeout=120)

    return response.status_code == 201


def _publish_video(publish_id: str, description: str) -> dict:
    """Finalize and publish the video with a description (step 3)."""
    headers = {
        "Authorization": f"Bearer {config.TIKTOK_ACCESS_TOKEN}",
        "Content-Type": "application/json; charset=UTF-8",
    }

    payload = {
        "publish_id": publish_id,
        "post_info": {
            "title": description[:150],  # TikTok title limit
            "description": description,
            "privacy_level": "PUBLIC_TO_EVERYONE",
        },
    }

    response = requests.post(
        f"{TIKTOK_API_BASE}/post/publish/status/fetch/",
        headers=headers,
        json=payload,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


@tool
def publish_to_tiktok(
    video_path: str,
    description: str,
    dry_run: bool = True,
) -> str:
    """Publish a video with description to TikTok.

    Args:
        video_path: Path to the video file.
        description: The generated TikTok description/caption.
        dry_run: If True, simulate publishing without actually posting.

    Returns:
        Publishing result — either simulation output or TikTok API response.
    """
    if dry_run:
        result = {
            "status": "DRY_RUN",
            "timestamp": datetime.now().isoformat(),
            "video_path": video_path,
            "description_preview": description[:150],
            "full_description": description,
            "description_length": len(description),
            "hashtag_count": description.count("#"),
            "message": "Dry run completed. Set dry_run=False to publish for real.",
        }
        return json.dumps(result, indent=2, ensure_ascii=False)

    # Live publishing
    if not config.TIKTOK_ACCESS_TOKEN:
        return "Error: TIKTOK_ACCESS_TOKEN not configured. Set it in .env file."

    if not os.path.exists(video_path):
        return f"Error: Video file not found at {video_path}"

    try:
        # Step 1: Initialize upload
        init_response = _init_video_upload(video_path)
        publish_id = init_response.get("data", {}).get("publish_id")
        upload_url = init_response.get("data", {}).get("upload_url")

        if not publish_id or not upload_url:
            return f"Error: Failed to initialize upload. Response: {init_response}"

        # Step 2: Upload video
        upload_ok = _upload_video_chunk(upload_url, video_path)
        if not upload_ok:
            return "Error: Video upload failed"

        # Step 3: Publish with description
        publish_response = _publish_video(publish_id, description)

        result = {
            "status": "PUBLISHED",
            "timestamp": datetime.now().isoformat(),
            "publish_id": publish_id,
            "description_preview": description[:150],
            "tiktok_response": publish_response,
        }
        return json.dumps(result, indent=2, ensure_ascii=False)

    except requests.exceptions.RequestException as e:
        return f"Error publishing to TikTok: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"
