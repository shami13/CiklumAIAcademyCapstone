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
| **Instagram Scraper** | Fetches posts from Instagram profiles via `instaloader` (supports login + MFA) and imports into RAG |
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

### 5. Import posts from Instagram (optional)

Scrape posts from a public Instagram profile so the agent learns the creator's style.

To avoid Instagram rate limits, add your username to `.env`:

```env
INSTAGRAM_USERNAME=your_username
```

On first run, instaloader will interactively prompt for your password and MFA code in the terminal. The session is saved locally so subsequent runs won't require re-authentication.

```bash
# Scrape from a public Instagram profile
python main.py --scrape-instagram natgeo --post-count 20
python main.py --scrape-instagram https://www.instagram.com/natgeo/
```

## Configuration

| Environment Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key (GPT-4o access required) |
| `CHROMA_PERSIST_DIR` | ChromaDB storage path (default: `./chroma_db`) |
| `MAX_REFLECTION_RETRIES` | Max self-reflection loops (default: 3) |
| `MIN_QUALITY_SCORE` | Minimum score to pass evaluation (default: 7) |
| `INSTAGRAM_USERNAME` | Instagram username for authenticated scraping (optional, password prompted interactively) |

## Project Structure

```
tiktok-content-agent/
├── main.py                          # CLI entry point
├── architecture.mmd                 # Mermaid architecture diagram
├── requirements.txt
├── src/
│   ├── agent.py                     # LangChain ReAct agent orchestrator
│   ├── config.py                    # Configuration management
│   ├── tools/
│   │   ├── video_analyzer.py        # Video frame extraction + GPT-4o Vision
│   │   ├── description_generator.py # Multi-platform caption generation
│   │   └── hashtag_researcher.py    # Per-platform hashtag suggestions
│   ├── rag/
│   │   ├── knowledge_base.py        # ChromaDB setup + best practices seeding
│   │   └── post_importer.py         # Personal posts ChromaDB collection + RAG queries
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

## Author

**Mykhailo Marchenko** — Ciklum AI Academy, 2026
