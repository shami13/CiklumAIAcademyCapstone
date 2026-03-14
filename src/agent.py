"""Content Agent — LangChain ReAct agent with tools for video analysis and multi-platform description generation."""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config import config
from src.evaluation.evaluator import evaluate_description
from src.rag.knowledge_base import seed_knowledge_base
from src.tools.description_generator import generate_description
from src.tools.hashtag_researcher import research_hashtags
from src.tools.video_analyzer import analyze_video

# All tools available to the agent
TOOLS = [
    analyze_video,
    research_hashtags,
    generate_description,
    evaluate_description,
]

SYSTEM_PROMPT = """You are an AI content agent specialized in creating short-form video content \
for multiple platforms: TikTok, Instagram Reels, and YouTube Shorts.

Your job is to take a video, analyze it, create optimized descriptions for all three platforms, \
and evaluate their quality. The user will publish the content manually.

You have access to the following tools:
1. analyze_video — Extract frames from a video and analyze its content using GPT-4o Vision
2. research_hashtags — Find relevant trending hashtags based on the video content
3. generate_description — Create optimized descriptions for TikTok, Instagram Reels, and YouTube Shorts
4. evaluate_description — Score the descriptions quality and get improvement suggestions

WORKFLOW (follow this order):
1. ANALYZE: Use analyze_video to understand the video content
2. RESEARCH: Use research_hashtags to find relevant hashtags
3. GENERATE: Use generate_description to create captions for all 3 platforms
4. EVALUATE: Use evaluate_description to score quality of all descriptions
5. REFLECT: If evaluation verdict is "NEEDS_IMPROVEMENT", take the improved_description \
   from the evaluation and evaluate it again. Repeat up to 3 times.
6. OUTPUT: Present the final descriptions for all 3 platforms to the user, ready to copy-paste.

IMPORTANT:
- Always analyze the video FIRST before doing anything else
- generate_description produces descriptions for ALL 3 platforms in one call
- Always evaluate the descriptions before presenting the final result
- If evaluation fails 3 times, present the best version anyway
- Be transparent about what you're doing at each step
- In your final output, clearly present all 3 platform descriptions so the user can copy-paste them
"""


def create_agent():
    """Create and return the LangChain ReAct agent."""
    # Seed RAG knowledge base
    seed_knowledge_base()

    llm = ChatOpenAI(
        model=config.OPENAI_MODEL,
        api_key=config.OPENAI_API_KEY,
        temperature=0.7,
    )

    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(TOOLS)

    return llm_with_tools, TOOLS


def run_agent(video_path: str) -> str:
    """Run the full agent workflow for a video.

    Args:
        video_path: Path to the video file.

    Returns:
        Final descriptions for TikTok, Instagram Reels, and YouTube Shorts.
    """
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    seed_knowledge_base()

    llm = ChatOpenAI(
        model=config.OPENAI_MODEL,
        api_key=config.OPENAI_API_KEY,
        temperature=0.7,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, TOOLS, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=TOOLS,
        verbose=True,
        max_iterations=15,
        handle_parsing_errors=True,
    )

    user_input = (
        f"Process this video and create optimized descriptions for TikTok, "
        f"Instagram Reels, and YouTube Shorts. "
        f"Video path: {video_path}. "
        f"Follow the full workflow: analyze → research hashtags → generate descriptions "
        f"for all 3 platforms → evaluate → reflect/improve → present final result."
    )

    result = executor.invoke({"input": user_input})
    return result["output"]


def run_interactive():
    """Run the agent in interactive chat mode."""
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    seed_knowledge_base()

    llm = ChatOpenAI(
        model=config.OPENAI_MODEL,
        api_key=config.OPENAI_API_KEY,
        temperature=0.7,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, TOOLS, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=TOOLS,
        verbose=True,
        max_iterations=15,
        handle_parsing_errors=True,
    )

    print("\n🎬 Content Agent — Interactive Mode")
    print("=" * 50)
    print("Commands: 'quit' to exit, 'help' for usage\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        if user_input.lower() == "help":
            print(
                "\nUsage:\n"
                "  Process a video:  process /path/to/video.mp4\n"
                "  Ask anything:     How should I optimize my TikTok captions?\n"
                "  Quit:             quit\n"
            )
            continue

        result = executor.invoke({"input": user_input})
        print(f"\nAgent: {result['output']}\n")
