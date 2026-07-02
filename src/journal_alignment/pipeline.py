"""Pipeline orchestration utilities for the journal alignment project."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from journal_alignment.analysis import AlignmentAnalyzer
from journal_alignment.config import ProjectConfig
from journal_alignment.data import ArticleDataset
from journal_alignment.embeddings import EmbeddingModel
from journal_alignment.preprocessing import clean_text, preprocess_dataframe
from journal_alignment.topic_model import (
    build_outlier_topic_assignments,
    check_bertopic_dependencies,
    run_bertopic_analysis,
)
from journal_alignment.visualization import (
    plot_alignment_boxplot,
    plot_alignment_by_year,
    plot_alignment_histogram,
    plot_article_scores_by_year,
)


LOGGER = logging.getLogger(__name__)


class AlignmentPipeline:
    """Run the complete local journal alignment workflow."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.dataset = ArticleDataset(
            articles_path=config.articles_path,
            aims_scope_path=config.aims_scope_path,
            min_abstract_length=config.min_abstract_length,
        )

    def run(self) -> dict[str, pd.DataFrame]:
        """Execute the full analysis and return all result DataFrames."""

        LOGGER.info("Loading articles from %s", self.config.articles_path)
        articles_df = self.dataset.load_articles()

        LOGGER.info("Loading Aims & Scope text from %s", self.config.aims_scope_path)
        aims_scope = clean_text(self.dataset.load_aims_scope())

        LOGGER.info("Validating article records")
        valid_articles = self.dataset.validate_articles(articles_df)
        LOGGER.info("Validated %s article records", len(valid_articles))

        LOGGER.info("Preprocessing abstracts")
        processed_articles = preprocess_dataframe(
            valid_articles,
            text_column="abstract",
        )

        if self.config.run_topic_model:
            LOGGER.info("Checking BERTopic dependencies")
            check_bertopic_dependencies()

        LOGGER.info("Loading embedding model: %s", self.config.model_name)
        analyzer = AlignmentAnalyzer(
            embedding_model=EmbeddingModel(model_name=self.config.model_name)
        )

        LOGGER.info("Computing embedding alignment scores")
        (
            abstract_embeddings,
            aims_scope_embedding,
            chunking_report,
        ) = analyzer.compute_embeddings(
            df=processed_articles,
            aims_scope=aims_scope,
        )
        scored_df = analyzer.add_embedding_alignment_scores(
            df=processed_articles,
            abstract_embeddings=abstract_embeddings,
            aims_scope_embedding=aims_scope_embedding,
        )

        LOGGER.info("Computing TF-IDF baseline scores")
        scored_df = analyzer.compute_tfidf_baseline(
            df=scored_df,
            aims_scope=aims_scope,
        )

        topic_results = None
        if self.config.run_topic_model:
            LOGGER.info("Fitting BERTopic model using pre-computed embeddings")
            topic_results = run_bertopic_analysis(
                df=scored_df,
                abstract_embeddings=abstract_embeddings,
                topics_over_time_path=self.config.figures_dir
                / "bertopic_topics_over_time.png",
                random_state=self.config.bertopic_random_state,
                min_cluster_size=self.config.hdbscan_min_cluster_size,
            )
            scored_df = topic_results.article_topics

        LOGGER.info("Building summary tables")
        results = analyzer.build_summary_tables(
            scored_df,
            top_n=self.config.top_n,
            outlier_method=self.config.outlier_method,
            z_threshold=self.config.z_threshold,
        )
        results["chunk_count_report"] = chunking_report.to_dataframe()

        if topic_results is not None:
            results["bertopic_topic_summary"] = topic_results.topic_summary
            results["bertopic_topic_diagnostics"] = topic_results.topic_diagnostics
            results["bertopic_noise_diagnostics"] = topic_results.noise_diagnostics
            results["bertopic_topics_over_time"] = topic_results.topics_over_time
            results["outlier_topic_assignments"] = build_outlier_topic_assignments(
                results["alignment_scores"]
            )

        LOGGER.info("Saving CSV outputs")
        self.save_results(results)

        LOGGER.info("Generating figures")
        self.generate_figures(results["alignment_scores"])

        return results

    def save_results(
        self,
        results: dict[str, pd.DataFrame],
    ) -> None:
        """Save result DataFrames to the project output directories."""

        required_output_paths = {
            "alignment_scores": self.config.results_dir / "alignment_scores.csv",
            "summary_statistics": self.config.tables_dir / "summary_statistics.csv",
            "yearly_alignment": self.config.tables_dir / "yearly_alignment.csv",
            "top_aligned_articles": self.config.tables_dir
            / "top_aligned_articles.csv",
            "least_aligned_articles": self.config.tables_dir
            / "least_aligned_articles.csv",
            "outlier_articles": self.config.tables_dir / "outlier_articles.csv",
            "chunk_count_report": self.config.tables_dir / "chunk_count_report.csv",
        }
        optional_output_paths = {
            "bertopic_topic_summary": self.config.tables_dir
            / "bertopic_topic_summary.csv",
            "bertopic_topic_diagnostics": self.config.tables_dir
            / "bertopic_topic_diagnostics.csv",
            "bertopic_noise_diagnostics": self.config.tables_dir
            / "bertopic_noise_diagnostics.csv",
            "bertopic_topics_over_time": self.config.tables_dir
            / "bertopic_topics_over_time.csv",
            "outlier_topic_assignments": self.config.tables_dir
            / "outlier_topic_assignments.csv",
        }

        for result_name, output_path in required_output_paths.items():
            if result_name not in results:
                raise KeyError(f"Missing result DataFrame: {result_name}")
            self._save_csv(results[result_name], output_path)

        for result_name, output_path in optional_output_paths.items():
            if result_name in results:
                self._save_csv(results[result_name], output_path)

    def generate_figures(
        self,
        scored_df: pd.DataFrame,
    ) -> None:
        """Generate the standard alignment figures."""

        figure_paths = {
            "alignment_histogram.png": plot_alignment_histogram,
            "alignment_boxplot.png": plot_alignment_boxplot,
            "alignment_by_year.png": plot_alignment_by_year,
            "article_scores_by_year.png": plot_article_scores_by_year,
        }

        for file_name, plot_function in figure_paths.items():
            plot_function(scored_df, self.config.figures_dir / file_name)

    @staticmethod
    def _save_csv(df: pd.DataFrame, output_path: str | Path) -> None:
        """Save one DataFrame as CSV, creating parent directories when needed."""

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)


__all__ = ["AlignmentPipeline"]
