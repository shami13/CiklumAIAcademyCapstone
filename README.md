# Short-Form Video Content Agent 🎬

An AI-powered agentic system that analyzes video content and generates optimized descriptions for **TikTok**, **Instagram Reels**, and **YouTube Shorts** — with self-reflection and quality evaluation. The user publishes content manually on each platform.

Built as part of the **Ciklum AI Academy** final assignment.

## Architecture

See [architecture.mmd](architecture.mmd) for the full system diagram.

### Core Components

| Component | Description |
|---|---|
| **Video Analyzer** | Extracts key frames from video, analyzes content via GPT-4o Vision |
| **RAG Knowledge Base** | ChromaDB vector store with best practices for short-form video content |
| **Post Importer** | Imports personal post history (TikTok/Instagram) from JSON/CSV for RAG personalization |
| **Instagram Scraper** | Fetches posts from public Instagram profiles via `instaloader` |
| **Hashtag Researcher** | Suggests relevant trending hashtags per platform (TikTok, Reels, Shorts) |
| **Description Generator** | Creates optimized captions for all 3 platforms in one pass |
| **Evaluator** | Scores output quality with 8 criteria including platform adaptation |
| **Self-Reflection** | Iteratively improves descriptions until quality score >= 7/10 |

## Tech Stack

- **Python 3.11+**
- **LangChain** — Agent framework (ReAct agent with tools)
- **OpenAI GPT-4o** — Vision analysis + text generation
- **ChromaDB** — Persistent vector store for RAG
- **OpenCV** — Video frame extraction
- **instaloader** — Instagram public profile scraping

## Setup

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/tiktok-content-agent.git
cd tiktok-content-agent
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Seed the RAG knowledge base

```bash
python main.py --seed-kb
```

### 4. Run the agent

```bash
# Analyze video + generate descriptions for TikTok, Reels & Shorts
python main.py --video path/to/video.mp4

# Interactive mode
python main.py --interactive
```

### 5. Import personal post data (optional)

Import your own post history so the agent learns your style and what performs best.

```bash
# Import from JSON (see examples/sample_posts.json for format)
python main.py --import-posts my_tiktok_posts.json

# Import from CSV (see examples/sample_posts.csv for format)
python main.py --import-posts my_instagram_posts.csv

# Scrape from a public Instagram profile
python main.py --scrape-instagram natgeo --post-count 20
python main.py --scrape-instagram https://www.instagram.com/natgeo/

# View import statistics
python main.py --show-stats
```

## Configuration

| Environment Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key (GPT-4o access required) |
| `CHROMA_PERSIST_DIR` | ChromaDB storage path (default: `./chroma_db`) |
| `MAX_REFLECTION_RETRIES` | Max self-reflection loops (default: 3) |
| `MIN_QUALITY_SCORE` | Minimum score to pass evaluation (default: 7) |

## Project Structure

```
tiktok-content-agent/
├── main.py                          # CLI entry point
├── architecture.mmd                 # Mermaid architecture diagram
├── requirements.txt
├── examples/
│   ├── sample_posts.json            # Example JSON format for post import
│   └── sample_posts.csv             # Example CSV format for post import
├── src/
│   ├── agent.py                     # LangChain ReAct agent orchestrator
│   ├── config.py                    # Configuration management
│   ├── tools/
│   │   ├── video_analyzer.py        # Video frame extraction + GPT-4o Vision
│   │   ├── description_generator.py # Multi-platform caption generation
│   │   └── hashtag_researcher.py    # Per-platform hashtag suggestions
│   ├── rag/
│   │   ├── knowledge_base.py        # ChromaDB setup + best practices seeding
│   │   └── post_importer.py         # Personal post import (JSON/CSV) + RAG
│   ├── scrapers/
│   │   └── instagram_scraper.py     # Instagram public profile scraping
│   └── evaluation/
│       └── evaluator.py             # Quality scoring + self-reflection
└── tests/
    ├── test_agent.py                # Core component tests
    └── test_instagram_scraper.py    # Instagram scraper tests
```

## How It Works

1. **Input**: User provides a video file path
2. **Analysis**: Agent extracts frames and sends them to GPT-4o Vision for content understanding
3. **RAG Retrieval**: Queries ChromaDB for relevant best practices + personal post history
4. **Hashtag Research**: Generates platform-specific hashtag suggestions for TikTok, Reels, and Shorts
5. **Generation**: Creates optimized descriptions for all 3 platforms:
   - **TikTok** — hook-first, 150 chars before "more", 5-7 hashtags
   - **Instagram Reels** — storytelling style, emojis, 5-10 hashtags
   - **YouTube Shorts** — SEO-friendly title (100 chars) + keyword-rich description
6. **Self-Reflection**: Agent evaluates its own output against 8 quality criteria, rewrites if score < 7/10
7. **Output**: Presents all 3 platform descriptions ready to copy-paste and publish manually

## Post Import Format

### JSON

```json
{
  "platform": "tiktok",
  "posts": [
    {
      "caption": "Your post caption here",
      "hashtags": ["#tag1", "#tag2"],
      "likes": 1500,
      "comments": 45,
      "shares": 12,
      "views": 25000,
      "posted_at": "2026-02-15T18:30:00",
      "content_type": "tutorial"
    }
  ]
}
```

### CSV

```
caption,hashtags,likes,comments,shares,views,posted_at,content_type,platform
"Your post caption",#tag1;#tag2,1500,45,12,25000,2026-02-15T18:30:00,tutorial,tiktok
```

## Author

**Mykhailo Marchenko** — Ciklum AI Academy, 2026
