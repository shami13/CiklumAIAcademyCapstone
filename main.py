"""Content Agent — CLI entry point."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Content Agent — Analyze videos, generate descriptions for TikTok, Reels & Shorts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --video my_video.mp4                  # Generate descriptions
  python main.py --interactive                          # Interactive chat mode
  python main.py --seed-kb                              # Seed knowledge base only
  python main.py --scrape-instagram natgeo --post-count 20  # Scrape Instagram
        """,
    )

    parser.add_argument(
        "--video",
        type=str,
        help="Path to the video file to process",
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
    parser.add_argument(
        "--scrape-instagram",
        type=str,
        metavar="USERNAME",
        help="Scrape posts from a public Instagram profile (username or URL)",
    )
    parser.add_argument(
        "--post-count",
        type=int,
        default=10,
        help="Number of posts to fetch when scraping (default: 10)",
    )

    args = parser.parse_args()

    if args.seed_kb:
        from src.rag.knowledge_base import seed_knowledge_base
        count = seed_knowledge_base()
        print(f"Knowledge base seeded with {count} documents.")
        return

    if args.scrape_instagram:
        from src.scrapers.instagram_scraper import scrape_and_import_instagram
        try:
            count = scrape_and_import_instagram(
                args.scrape_instagram,
                post_count=args.post_count,
            )
            print(f"Successfully scraped and imported {count} Instagram posts.")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
        return

    if args.interactive:
        from src.agent import run_interactive
        run_interactive()
        return

    if args.video:
        from src.agent import run_agent
        print(f"\n🎬 Processing video: {args.video}")
        print("=" * 50)

        result = run_agent(args.video)
        print(f"\n{'=' * 50}")
        print("📋 FINAL DESCRIPTIONS:")
        print(result)
        return

    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
