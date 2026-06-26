"""Fetch PubMed articles for a journal and save them as CSV."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from journal_alignment.data import PubMedClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch PubMed article metadata and abstracts for a journal."
    )
    parser.add_argument("--journal-name", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--max-results", type=int, default=200)
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--email", default=None)
    parser.add_argument("--api-key", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = PubMedClient(email=args.email, api_key=args.api_key)

    if args.start_year is None and args.end_year is None:
        df = client.fetch_journal_articles_last_10_years(
            journal_name=args.journal_name,
            max_results=args.max_results,
        )
    else:
        current_year = date.today().year
        start_year = (
            args.start_year if args.start_year is not None else current_year - 10
        )
        end_year = args.end_year if args.end_year is not None else current_year
        pmids = client.search_articles(
            journal_name=args.journal_name,
            start_year=start_year,
            end_year=end_year,
            max_results=args.max_results,
        )
        df = client.fetch_articles(pmids)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} PubMed articles to {output_path}")


if __name__ == "__main__":
    main()
