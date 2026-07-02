"""Run the local journal alignment analysis pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from journal_alignment.config import ProjectConfig
from journal_alignment.pipeline import AlignmentPipeline


def parse_args() -> argparse.Namespace:
    default_config = ProjectConfig()
    parser = argparse.ArgumentParser(
        description="Run thematic alignment analysis on a local PubMed CSV."
    )
    parser.add_argument(
        "--articles-path",
        type=Path,
        default=default_config.articles_path,
        help="Path to the local PubMed article CSV.",
    )
    parser.add_argument(
        "--aims-scope-path",
        type=Path,
        default=default_config.aims_scope_path,
        help="Path to the manually provided Aims & Scope TXT file.",
    )
    parser.add_argument(
        "--model-name",
        default=default_config.model_name,
        help="SentenceTransformer model name used for semantic embeddings.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=default_config.top_n,
        help="Number of top and least aligned articles to save.",
    )
    parser.add_argument(
        "--outlier-method",
        choices=["zscore", "iqr"],
        default=default_config.outlier_method,
        help="Outlier detection method for alignment scores.",
    )
    parser.add_argument(
        "--z-threshold",
        type=float,
        default=default_config.z_threshold,
        help="Absolute z-score threshold used when --outlier-method zscore.",
    )
    parser.add_argument(
        "--skip-topic-model",
        action="store_true",
        help="Skip the optional BERTopic corpus-structure analysis.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    config = ProjectConfig(
        articles_path=args.articles_path,
        aims_scope_path=args.aims_scope_path,
        model_name=args.model_name,
        top_n=args.top_n,
        outlier_method=args.outlier_method,
        z_threshold=args.z_threshold,
        run_topic_model=not args.skip_topic_model,
    )

    print("Starting journal alignment pipeline")
    print(f"Articles CSV: {config.articles_path}")
    print(f"Aims & Scope TXT: {config.aims_scope_path}")
    print(f"Embedding model: {config.model_name}")
    print(f"BERTopic enabled: {config.run_topic_model}")

    pipeline = AlignmentPipeline(config)
    results = pipeline.run()

    print("\nPipeline completed")
    print(f"Analyzed articles: {len(results['alignment_scores'])}")
    print(f"Alignment scores saved to: {config.results_dir / 'alignment_scores.csv'}")
    print(f"Tables saved to: {config.tables_dir}")
    print(f"Figures saved to: {config.figures_dir}")
    if config.run_topic_model and "bertopic_topic_diagnostics" in results:
        topic_diagnostics = results["bertopic_topic_diagnostics"]
        noise_size = topic_diagnostics.loc[
            topic_diagnostics["metric"] == "noise_cluster_size",
            "value",
        ].iloc[0]
        noise_share = topic_diagnostics.loc[
            topic_diagnostics["metric"] == "noise_cluster_share",
            "value",
        ].iloc[0]
        print(f"BERTopic noise cluster size: {noise_size} ({float(noise_share):.1%})")


if __name__ == "__main__":
    main()
