# TikTok Content Agent 🎬

An AI-powered agentic system that analyzes video content, generates optimized TikTok descriptions with hashtags, and publishes them — all with self-reflection and quality evaluation.

Built as part of the **Ciklum AI Academy** final assignment.

## Architecture

See [architecture.mmd](architecture.mmd) for the full system diagram.

### Core Components

| Component | Description |
|---|---|
| **Video Analyzer** | Extracts key frames from video, analyzes content via GPT-4o Vision |
| **RAG Knowledge Base** | ChromaDB vector store with TikTok best practices, hashtag strategies |
| **Hashtag Researcher** | Suggests relevant trending hashtags based on video content |
| **Description Generator** | Combines all inputs to create an optimized TikTok caption |
| **Self-Reflection** | Evaluates generated description and iteratively improves it |
| **TikTok Publisher** | Posts video + description via TikTok Content Posting API |
| **Evaluator** | Scores output quality (relevance, engagement, clarity) |

## Tech Stack

- **Python 3.11+**
- **LangChain** — Agent framework (ReAct agent with tools)
- **OpenAI GPT-4o** — Vision analysis + text generation
- **ChromaDB** — Vector store for RAG
- **OpenCV** — Video frame extraction
- **TikTok Content Posting API** — Publishing

## Setup

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/tiktok-content-agent.git
cd tiktok-content-agent
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Seed the RAG knowledge base

```bash
python -m src.rag.knowledge_base
```

### 4. Run the agent

```bash
# Analyze video + generate description (dry run)
python main.py --video path/to/video.mp4

# Analyze + publish to TikTok
python main.py --video path/to/video.mp4 --publish

# Interactive mode
python main.py --interactive
```

## Configuration

| Environment Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key (GPT-4o access required) |
| `TIKTOK_CLIENT_KEY` | TikTok app client key |
| `TIKTOK_CLIENT_SECRET` | TikTok app client secret |
| `TIKTOK_ACCESS_TOKEN` | TikTok user access token |

## Project Structure

```
tiktok-content-agent/
├── main.py                    # Entry point
├── architecture.mmd           # Mermaid architecture diagram
├── requirements.txt
├── .env.example
├── src/
│   ├── agent.py               # LangChain ReAct agent
│   ├── config.py              # Configuration
│   ├── tools/
│   │   ├── video_analyzer.py  # Video frame extraction + GPT-4o Vision
│   │   ├── description_generator.py  # Caption generation
│   │   ├── hashtag_researcher.py     # Hashtag suggestions
│   │   └── tiktok_publisher.py       # TikTok API integration
│   ├── rag/
│   │   ├── knowledge_base.py  # ChromaDB setup + seeding
│   │   └── data/              # Raw knowledge documents
│   └── evaluation/
│       └── evaluator.py       # Quality scoring
└── tests/
    └── test_agent.py
```

## How It Works

1. **Input**: User provides a video file path
2. **Analysis**: Agent extracts frames and sends them to GPT-4o Vision for content understanding
3. **RAG Retrieval**: Queries ChromaDB for relevant best practices based on video content
4. **Generation**: Creates an optimized TikTok description with hashtags
5. **Self-Reflection**: Agent evaluates its own output against quality criteria, rewrites if score < 7/10
6. **Publishing**: Posts to TikTok via Content Posting API (with user confirmation)
7. **Evaluation**: Final quality report with scores

## Author

**Mykhailo Marchenko** — Ciklum AI Academy, 2026
