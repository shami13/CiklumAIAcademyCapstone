"""TikTok Content Agent — CLI entry point."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="TikTok Content Agent — Analyze videos, generate descriptions, publish to TikTok",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --video my_video.mp4                  # Dry run
  python main.py --video my_video.mp4 --publish        # Publish to TikTok
  python main.py --interactive                          # Interactive chat mode
  python main.py --seed-kb                              # Seed knowledge base only
        """,
    )

    parser.add_argument(
        "--video",
        type=str,
        help="Path to the video file to process",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        default=False,
        help="Actually publish to TikTok (default: dry run)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        default=False,
        help="Run in interactive chat mode",
    )
    parser.add_argument(
        "--seed-kb",
        action="store_true",
        default=False,
        help="Seed the RAG knowledge base and exit",
    )

    args = parser.parse_args()

    if args.seed_kb:
        from src.rag.knowledge_base import seed_knowledge_base
        count = seed_knowledge_base()
        print(f"Knowledge base seeded with {count} documents.")
        return

    if args.interactive:
        from src.agent import run_interactive
        run_interactive()
        return

    if args.video:
        from src.agent import run_agent
        print(f"\n🎬 Processing video: {args.video}")
        print(f"📤 Publish mode: {'LIVE' if args.publish else 'DRY RUN'}")
        print("=" * 50)

        result = run_agent(args.video, publish=args.publish)
        print(f"\n{'=' * 50}")
        print("📋 FINAL RESULT:")
        print(result)
        return

    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
