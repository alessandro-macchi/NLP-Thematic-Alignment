"""BERTopic utilities for complementary corpus structure analysis."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BERTopicAnalysisResult:
    """Container for BERTopic article assignments and summary tables."""

    article_topics: pd.DataFrame
    topic_summary: pd.DataFrame
    topic_diagnostics: pd.DataFrame
    noise_diagnostics: pd.DataFrame
    topics_over_time: pd.DataFrame
    noise_cluster_size: int


def run_bertopic_analysis(
    df: pd.DataFrame,
    abstract_embeddings: np.ndarray,
    topics_over_time_path: str | Path,
    text_column: str = "abstract",
    year_column: str = "year",
    random_state: int = 6,
    min_cluster_size: int = 15,
    top_n_keywords: int = 5,
    max_topics_in_plot: int = 10,
) -> BERTopicAnalysisResult:
    """Fit BERTopic using pre-computed embeddings and save a time plot."""

    check_bertopic_dependencies()
    documents = _valid_documents(df, text_column)
    years = _valid_years(df, year_column)
    embeddings = _valid_embeddings(abstract_embeddings, n_rows=len(df))

    topic_model = _fit_topic_model(
        documents=documents,
        embeddings=embeddings,
        random_state=random_state,
        min_cluster_size=min_cluster_size,
    )
    topics = np.asarray(topic_model.topics_, dtype=int)
    topic_summary = summarize_topics(
        topic_model=topic_model,
        topics=topics,
        top_n_keywords=top_n_keywords,
    )
    article_topics = add_topic_assignments(df, topics, topic_summary)

    nr_bins = years.nunique()
    topics_over_time = topic_model.topics_over_time(
        documents,
        years.tolist(),
        topics=topics.tolist(),
        nr_bins=nr_bins,
    )
    topics_over_time = label_topics_over_time_with_publication_years(
        topics_over_time=topics_over_time,
        publication_years=years,
    )
    plot_topics_over_time(
        topics_over_time=topics_over_time,
        topic_summary=topic_summary,
        output_path=topics_over_time_path,
        max_topics=max_topics_in_plot,
    )

    noise_cluster_size = int((topics == -1).sum())
    noise_cluster_share = noise_cluster_size / len(topics)
    LOGGER.info("BERTopic noise cluster size: %s articles", noise_cluster_size)
    topic_diagnostics = pd.DataFrame(
        {
            "metric": [
                "noise_cluster_size",
                "noise_cluster_share",
                "topics_excluding_noise",
                "distinct_years",
                "topics_over_time_nr_bins",
                "umap_random_state",
                "hdbscan_min_cluster_size",
            ],
            "value": [
                noise_cluster_size,
                noise_cluster_share,
                int((topic_summary["topic_id"] != -1).sum()),
                int(nr_bins),
                int(nr_bins),
                int(random_state),
                int(min_cluster_size),
            ],
        }
    )

    return BERTopicAnalysisResult(
        article_topics=article_topics,
        topic_summary=topic_summary,
        topic_diagnostics=topic_diagnostics,
        noise_diagnostics=build_noise_diagnostics(article_topics),
        topics_over_time=topics_over_time,
        noise_cluster_size=noise_cluster_size,
    )


def label_topics_over_time_with_publication_years(
    topics_over_time: pd.DataFrame,
    publication_years: pd.Series,
) -> pd.DataFrame:
    """Replace BERTopic bin labels with integer publication years."""

    if "Timestamp" not in topics_over_time.columns:
        raise ValueError("Column 'Timestamp' is missing from topics_over_time.")

    years = sorted(pd.Series(publication_years).dropna().astype(int).unique())
    if not years:
        raise ValueError("publication_years must contain at least one year.")

    result = topics_over_time.copy()
    raw_timestamps = sorted(
        result["Timestamp"].dropna().unique().tolist(),
        key=_timestamp_sort_key,
    )
    result["bertopic_timestamp"] = result["Timestamp"]

    if len(raw_timestamps) == len(years):
        timestamp_to_year = dict(zip(raw_timestamps, years))
        result["Timestamp"] = result["Timestamp"].map(timestamp_to_year).astype(int)
        return result

    numeric_timestamps = pd.to_numeric(result["Timestamp"], errors="coerce")
    if numeric_timestamps.isna().any():
        raise ValueError(
            "BERTopic timestamps could not be mapped to publication years."
        )

    result["Timestamp"] = numeric_timestamps.round().astype(int)
    return result


def check_bertopic_dependencies() -> None:
    """Raise a clear error if BERTopic dependencies are not importable."""

    try:
        import bertopic  # noqa: F401
        import umap  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "BERTopic analysis requires the optional topic-modeling dependencies. "
            "Install them with `pip install -r requirements.txt`, or run the "
            "pipeline with `--skip-topic-model`."
        ) from exc

    try:
        import hdbscan  # noqa: F401
    except ImportError:
        try:
            from sklearn.cluster import HDBSCAN  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "HDBSCAN is required for BERTopic clustering. Install hdbscan "
                "or use a scikit-learn version that provides sklearn.cluster.HDBSCAN."
            ) from exc


def summarize_topics(
    topic_model: Any,
    topics: np.ndarray,
    top_n_keywords: int = 5,
) -> pd.DataFrame:
    """Build a compact topic summary table."""

    if top_n_keywords < 1:
        raise ValueError("top_n_keywords must be at least 1.")

    topic_counts = pd.Series(topics, name="topic_id").value_counts().sort_index()
    rows = []
    for topic_id, article_count in topic_counts.items():
        keywords = _topic_keywords(topic_model, int(topic_id), top_n_keywords)
        rows.append(
            {
                "topic_id": int(topic_id),
                f"top_{top_n_keywords}_keywords": "; ".join(keywords),
                "article_count": int(article_count),
            }
        )

    return pd.DataFrame(rows).sort_values("topic_id").reset_index(drop=True)


def add_topic_assignments(
    df: pd.DataFrame,
    topics: np.ndarray,
    topic_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Add BERTopic assignment columns to the article-level DataFrame."""

    if len(topics) != len(df):
        raise ValueError("topics must have one assignment per article.")
    keyword_column = _keyword_column(topic_summary)
    keyword_lookup = topic_summary.set_index("topic_id")[keyword_column].to_dict()

    result = df.copy()
    result["bertopic_topic_id"] = topics.astype(int)
    result["bertopic_is_noise"] = result["bertopic_topic_id"].eq(-1)
    result["bertopic_topic_keywords"] = (
        result["bertopic_topic_id"].map(keyword_lookup).fillna("")
    )
    result["bertopic_topic_label"] = [
        _topic_label(topic_id, keywords)
        for topic_id, keywords in zip(
            result["bertopic_topic_id"],
            result["bertopic_topic_keywords"],
        )
    ]
    return result


def build_outlier_topic_assignments(
    df: pd.DataFrame,
    score_column: str = "alignment_score",
    outlier_column: str = "is_outlier",
) -> pd.DataFrame:
    """Return outlier rows with their BERTopic assignments."""

    required_columns = [
        "title",
        score_column,
        outlier_column,
        "bertopic_topic_id",
        "bertopic_is_noise",
        "bertopic_topic_keywords",
        "bertopic_topic_label",
    ]
    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]
    if missing_columns:
        missing = "', '".join(missing_columns)
        raise ValueError(f"Column '{missing}' is missing from the DataFrame.")

    outliers = df[df[outlier_column]].copy()
    if outliers.empty:
        return pd.DataFrame(columns=[*required_columns, "outlier_direction"])

    scores = pd.to_numeric(df[score_column], errors="coerce")
    score_mean = scores.mean()
    outliers["outlier_direction"] = np.where(
        pd.to_numeric(outliers[score_column], errors="coerce") >= score_mean,
        "high_alignment_outlier",
        "low_alignment_outlier",
    )
    preferred_columns = [
        column
        for column in [
            "pmid",
            "title",
            "year",
            score_column,
            "tfidf_alignment_score",
            "outlier_direction",
            "bertopic_topic_id",
            "bertopic_is_noise",
            "bertopic_topic_keywords",
            "bertopic_topic_label",
            "doi",
        ]
        if column in outliers.columns
    ]
    return outliers[preferred_columns].reset_index(drop=True)


def build_noise_diagnostics(
    df: pd.DataFrame,
    text_column: str = "abstract",
    noise_column: str = "bertopic_is_noise",
    score_column: str = "alignment_score",
    keywords_column: str = "keywords",
    top_n_keywords: int = 8,
) -> pd.DataFrame:
    """Summarize BERTopic noise articles against clustered articles."""

    required_columns = [text_column, noise_column]
    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]
    if missing_columns:
        missing = "', '".join(missing_columns)
        raise ValueError(f"Column '{missing}' is missing from the DataFrame.")
    if top_n_keywords < 1:
        raise ValueError("top_n_keywords must be at least 1.")

    working = df.copy()
    working["_abstract_characters"] = working[text_column].astype("string").str.len()
    working["_abstract_words"] = (
        working[text_column].astype("string").str.split().str.len()
    )
    noise_flags = _boolean_series(working[noise_column])

    rows = []
    for is_noise, group_label in [
        (False, "clustered_articles"),
        (True, "noise_articles"),
    ]:
        group = working[noise_flags.eq(is_noise)]
        rows.append(
            {
                "bertopic_group": group_label,
                "article_count": int(len(group)),
                "article_share": _safe_share(len(group), len(working)),
                "mean_abstract_words": _safe_mean(group["_abstract_words"]),
                "median_abstract_words": _safe_median(group["_abstract_words"]),
                "mean_abstract_characters": _safe_mean(group["_abstract_characters"]),
                "median_abstract_characters": _safe_median(
                    group["_abstract_characters"]
                ),
                "mean_alignment_score": _optional_mean(group, score_column),
                "frequent_author_keywords": _top_semicolon_terms(
                    group,
                    column=keywords_column,
                    n=top_n_keywords,
                ),
            }
        )

    return pd.DataFrame(rows)


def plot_topics_over_time(
    topics_over_time: pd.DataFrame,
    topic_summary: pd.DataFrame,
    output_path: str | Path,
    max_topics: int = 10,
) -> None:
    """Save a readable Matplotlib version of BERTopic topics over time."""

    if max_topics < 1:
        raise ValueError("max_topics must be at least 1.")
    required_columns = {"Topic", "Timestamp", "Frequency"}
    missing_columns = required_columns.difference(topics_over_time.columns)
    if missing_columns:
        missing = "', '".join(sorted(missing_columns))
        raise ValueError(f"Column '{missing}' is missing from topics_over_time.")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    try:
        working = topics_over_time.copy()
        working = working[working["Topic"] != -1]
        if working.empty:
            ax.text(
                0.5,
                0.5,
                "No non-noise BERTopic topics were found.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            _style_topic_axes(ax)
            fig.tight_layout()
            fig.savefig(path, dpi=300, bbox_inches="tight")
            return

        top_topic_ids = (
            topic_summary[topic_summary["topic_id"] != -1]
            .sort_values("article_count", ascending=False)
            .head(max_topics)["topic_id"]
            .tolist()
        )
        working = working[working["Topic"].isin(top_topic_ids)].copy()
        working["Frequency"] = pd.to_numeric(
            working["Frequency"],
            errors="coerce",
        ).fillna(0)
        working["Timestamp"] = working["Timestamp"].astype(str)

        pivot = working.pivot_table(
            index="Timestamp",
            columns="Topic",
            values="Frequency",
            aggfunc="sum",
            fill_value=0,
        )
        sorted_timestamps = sorted(pivot.index.tolist(), key=_timestamp_sort_key)
        pivot = pivot.loc[sorted_timestamps]
        x_positions = np.arange(len(pivot.index))
        for topic_id in top_topic_ids:
            if topic_id not in pivot.columns:
                continue
            ax.plot(
                x_positions,
                pivot[topic_id].to_numpy(),
                marker="o",
                linewidth=1.8,
                markersize=4,
                label=f"Topic {int(topic_id)}",
            )

        ax.set_title("BERTopic Topics Over Time", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Publication year", fontsize=10)
        ax.set_ylabel("Article count", fontsize=10)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(pivot.index.tolist(), rotation=45, ha="right")
        ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        _style_topic_axes(ax)
        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
    finally:
        plt.close(fig)


def _fit_topic_model(
    documents: list[str],
    embeddings: np.ndarray,
    random_state: int,
    min_cluster_size: int,
) -> Any:
    """Create and fit BERTopic with reproducible UMAP and HDBSCAN."""

    if len(documents) < min_cluster_size:
        raise ValueError(
            "BERTopic requires at least min_cluster_size documents. "
            f"Received {len(documents)} documents and min_cluster_size={min_cluster_size}."
        )

    try:
        from bertopic import BERTopic
        from umap import UMAP
    except ImportError as exc:
        raise ImportError(
            "BERTopic analysis requires the optional topic-modeling dependencies. "
            "Install them with `pip install -r requirements.txt`."
        ) from exc

    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=random_state,
    )
    hdbscan_model = _make_hdbscan_model(min_cluster_size=min_cluster_size)
    vectorizer_model = CountVectorizer(stop_words="english")

    topic_model = BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=False,
        verbose=False,
    )
    topic_model.fit_transform(documents, embeddings=embeddings)
    return topic_model


def _make_hdbscan_model(min_cluster_size: int) -> Any:
    """Create an HDBSCAN clusterer, preferring the original hdbscan package."""

    try:
        from hdbscan import HDBSCAN

        return HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=False,
        )
    except ImportError:
        try:
            from sklearn.cluster import HDBSCAN
        except ImportError as exc:
            raise ImportError(
                "HDBSCAN is required for BERTopic clustering. Install hdbscan "
                "or use a scikit-learn version that provides sklearn.cluster.HDBSCAN."
            ) from exc

        return HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")


def _valid_documents(df: pd.DataFrame, text_column: str) -> list[str]:
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' is missing from the DataFrame.")

    documents = df[text_column].astype("string")
    if documents.isna().any():
        raise ValueError(f"Column '{text_column}' must not contain missing values.")

    document_list = documents.tolist()
    if not document_list:
        raise ValueError("df must contain at least one article.")
    if any(not document.strip() for document in document_list):
        raise ValueError(f"Column '{text_column}' must not contain empty strings.")
    return document_list


def _valid_years(df: pd.DataFrame, year_column: str) -> pd.Series:
    if year_column not in df.columns:
        raise ValueError(f"Column '{year_column}' is missing from the DataFrame.")

    years = pd.to_numeric(df[year_column], errors="coerce")
    if years.isna().any():
        raise ValueError(f"Column '{year_column}' must contain numeric years.")
    return years.astype(int)


def _valid_embeddings(embeddings: np.ndarray, n_rows: int) -> np.ndarray:
    matrix = np.asarray(embeddings, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("abstract_embeddings must be a two-dimensional array.")
    if matrix.shape[0] != n_rows:
        raise ValueError("abstract_embeddings must have one row per article.")
    if matrix.shape[1] == 0:
        raise ValueError("abstract_embeddings must contain at least one feature.")
    if not np.isfinite(matrix).all():
        raise ValueError("abstract_embeddings must contain only finite values.")
    return matrix


def _topic_keywords(
    topic_model: Any,
    topic_id: int,
    top_n_keywords: int,
) -> list[str]:
    words = topic_model.get_topic(topic_id) or []
    return [word for word, _ in words[:top_n_keywords]]


def _keyword_column(topic_summary: pd.DataFrame) -> str:
    keyword_columns = [
        column for column in topic_summary.columns if column.startswith("top_")
    ]
    if len(keyword_columns) != 1:
        raise ValueError("topic_summary must contain exactly one top keyword column.")
    return keyword_columns[0]


def _topic_label(topic_id: int, keywords: str) -> str:
    if int(topic_id) == -1:
        return "Noise"
    compact_keywords = ", ".join(keywords.split("; ")[:3])
    if compact_keywords:
        return f"Topic {int(topic_id)}: {compact_keywords}"
    return f"Topic {int(topic_id)}"


def _topic_label_lookup(topic_summary: pd.DataFrame) -> dict[int, str]:
    keyword_column = _keyword_column(topic_summary)
    return {
        int(row.topic_id): _topic_label(int(row.topic_id), getattr(row, keyword_column))
        for row in topic_summary.itertuples(index=False)
    }


def _safe_share(part: int, total: int) -> float:
    if total == 0:
        return 0.0
    return float(part / total)


def _safe_mean(values: pd.Series) -> float:
    if values.empty:
        return float("nan")
    return float(pd.to_numeric(values, errors="coerce").mean())


def _safe_median(values: pd.Series) -> float:
    if values.empty:
        return float("nan")
    return float(pd.to_numeric(values, errors="coerce").median())


def _optional_mean(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns or df.empty:
        return float("nan")
    return float(pd.to_numeric(df[column], errors="coerce").mean())


def _boolean_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    normalized = series.astype("string").str.strip().str.lower()
    valid_values = {"true", "false", "1", "0", "yes", "no"}
    invalid_values = normalized.dropna()[~normalized.dropna().isin(valid_values)]
    if not invalid_values.empty:
        raise ValueError("noise_column must contain boolean-like values.")
    return normalized.isin({"true", "1", "yes"})


def _top_semicolon_terms(df: pd.DataFrame, column: str, n: int) -> str:
    if column not in df.columns or df.empty:
        return ""

    terms = (
        df[column]
        .dropna()
        .astype("string")
        .str.split(";")
        .explode()
        .dropna()
        .str.strip()
    )
    terms = terms[terms.ne("")]
    if terms.empty:
        return ""

    return "; ".join(terms.str.lower().value_counts().head(n).index.tolist())


def _timestamp_sort_key(timestamp: object) -> tuple[float, str]:
    numeric_timestamp = pd.to_numeric(pd.Series([timestamp]), errors="coerce").iloc[0]
    if pd.notna(numeric_timestamp):
        return float(numeric_timestamp), str(timestamp)

    years = re.findall(r"(?:18|19|20|21)\d{2}", str(timestamp))
    if years:
        return float(years[-1]), str(timestamp)
    return 9999.0, str(timestamp)


def _style_topic_axes(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.set_facecolor("#FAFAFA")
    ax.tick_params(axis="both", labelsize=9, colors="#4A4A4A")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.65)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#BFBFBF")
        ax.spines[spine].set_linewidth(0.8)


__all__ = [
    "BERTopicAnalysisResult",
    "add_topic_assignments",
    "build_noise_diagnostics",
    "build_outlier_topic_assignments",
    "check_bertopic_dependencies",
    "label_topics_over_time_with_publication_years",
    "plot_topics_over_time",
    "run_bertopic_analysis",
    "summarize_topics",
]
