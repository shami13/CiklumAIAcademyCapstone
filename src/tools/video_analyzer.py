"""Video Analyzer Tool — extracts key frames and analyzes content via GPT-4o Vision."""

import base64
import os
import shutil
from pathlib import Path

import cv2
from langchain_core.tools import tool
from openai import OpenAI

from src.config import config

client = OpenAI(api_key=config.OPENAI_API_KEY)


def _extract_frames(video_path: str, max_frames: int = 5) -> list[str]:
    """Extract evenly-spaced frames from a video file. Returns list of base64 encoded images."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        raise ValueError("Video has no frames")

    # Calculate frame indices to extract (evenly spaced)
    indices = [int(i * total_frames / max_frames) for i in range(max_frames)]

    frames_dir = Path(config.FRAMES_OUTPUT_DIR)
    frames_dir.mkdir(parents=True, exist_ok=True)

    encoded_frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Resize for API efficiency
        height, width = frame.shape[:2]
        scale = config.FRAME_RESIZE_WIDTH / width
        frame = cv2.resize(frame, (config.FRAME_RESIZE_WIDTH, int(height * scale)))

        # Encode to base64
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        encoded = base64.b64encode(buffer).decode("utf-8")
        encoded_frames.append(encoded)

    cap.release()

    # Clean up temp dir
    if frames_dir.exists():
        shutil.rmtree(frames_dir, ignore_errors=True)

    return encoded_frames


def _analyze_frames_with_vision(frames: list[str]) -> str:
    """Send extracted frames to GPT-4o Vision for analysis."""
    content = [
        {
            "type": "text",
            "text": (
                "You are a video content analyst. Analyze these frames extracted from a video "
                "and provide a detailed description including:\n"
                "1. Main subject/topic of the video\n"
                "2. Key visual elements and scenes\n"
                "3. Mood/tone of the content\n"
                "4. Target audience\n"
                "5. Suggested content category (e.g., tech, lifestyle, tutorial, comedy)\n\n"
                "Be specific and detailed — this analysis will be used to generate a TikTok description."
            ),
        }
    ]

    for i, frame in enumerate(frames):
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame}",
                    "detail": "low",
                },
            }
        )

    response = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=[{"role": "user", "content": content}],
        max_tokens=1000,
    )

    return response.choices[0].message.content


@tool
def analyze_video(video_path: str) -> str:
    """Analyze a video file by extracting key frames and understanding its content using GPT-4o Vision.

    Args:
        video_path: Path to the video file to analyze.

    Returns:
        Detailed analysis of the video content including subject, visuals, mood, audience, and category.
    """
    if not os.path.exists(video_path):
        return f"Error: Video file not found at {video_path}"

    try:
        frames = _extract_frames(video_path, max_frames=config.MAX_FRAMES)
        if not frames:
            return "Error: Could not extract any frames from the video"

        analysis = _analyze_frames_with_vision(frames)
        return f"Video Analysis ({len(frames)} frames analyzed):\n\n{analysis}"

    except Exception as e:
        return f"Error analyzing video: {str(e)}"
