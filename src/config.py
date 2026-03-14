import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    # Agent
    MAX_REFLECTION_RETRIES: int = int(os.getenv("MAX_REFLECTION_RETRIES", "3"))
    MIN_QUALITY_SCORE: int = int(os.getenv("MIN_QUALITY_SCORE", "7"))

    # Paths
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    FRAMES_OUTPUT_DIR: str = os.getenv("FRAMES_OUTPUT_DIR", "./temp_frames")

    # Video analysis
    MAX_FRAMES: int = int(os.getenv("MAX_FRAMES", "5"))
    FRAME_RESIZE_WIDTH: int = 512

    # Instagram
    INSTAGRAM_USERNAME: str = os.getenv("INSTAGRAM_USERNAME", "")


config = Config()
