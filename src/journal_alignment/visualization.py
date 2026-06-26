"""Visualization utilities for alignment results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TITLE_KWARGS = {"fontsize": 13, "fontweight": "bold", "pad": 12}
LABEL_KWARGS = {"fontsize": 10}
PRIMARY_COLOR = "#2F6F9F"
SECONDARY_COLOR = "#6AA84F"
ACCENT_COLOR = "#C04F4A"
POINT_COLOR = "#3F7FBF"


def _style_axes(ax: plt.Axes, grid_axis: str = "both") -> None:
    """Apply consistent non-data styling to a Matplotlib axis."""

    ax.set_axisbelow(True)
    ax.set_facecolor("#FAFAFA")
    ax.tick_params(axis="both", labelsize=9, colors="#4A4A4A")
    ax.grid(axis=grid_axis, color="#D9D9D9", linewidth=0.8, alpha=0.65)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#BFBFBF")
        ax.spines[spine].set_linewidth(0.8)


def _prepare_output_path(output_path: str | Path) -> Path:
    """Create parent directories and return ``output_path`` as a Path."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _score_values(df: pd.DataFrame, score_column: str) -> pd.Series:
    """Return a non-empty numeric score series."""

    if score_column not in df.columns:
        raise ValueError(f"Column '{score_column}' is missing from the DataFrame.")

    scores = pd.to_numeric(df[score_column], errors="coerce").dropna()
    if scores.empty:
        raise ValueError(f"Column '{score_column}' does not contain numeric scores.")

    return scores


def _year_score_frame(df: pd.DataFrame, score_column: str) -> pd.DataFrame:
    """Return complete numeric year and score observations."""

    missing_columns = [
        column for column in ("year", score_column) if column not in df.columns
    ]
    if missing_columns:
        missing = "', '".join(missing_columns)
        raise ValueError(f"Column '{missing}' is missing from the DataFrame.")

    working = pd.DataFrame(
        {
            "year": pd.to_numeric(df["year"], errors="coerce"),
            score_column: pd.to_numeric(df[score_column], errors="coerce"),
        }
    ).dropna()

    if working.empty:
        raise ValueError(
            f"Columns 'year' and '{score_column}' do not contain complete numeric observations."
        )

    return working.sort_values("year")


def _embedding_matrix(embeddings: np.ndarray, n_rows: int) -> np.ndarray:
    """Return a valid two-dimensional embedding matrix for PCA."""

    matrix = np.asarray(embeddings, dtype=float)

    if matrix.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional array.")
    if matrix.shape[0] != n_rows:
        raise ValueError("embeddings must have the same number of rows as df.")
    if matrix.shape[0] < 2:
        raise ValueError("embeddings must contain at least two rows for PCA.")
    if matrix.shape[1] < 2:
        raise ValueError("embeddings must contain at least two features for PCA.")
    if not np.isfinite(matrix).all():
        raise ValueError("embeddings must contain only finite numeric values.")

    return matrix


def plot_alignment_histogram(
    df: pd.DataFrame,
    output_path: str | Path,
    score_column: str = "alignment_score",
) -> None:
    """Save a histogram of article alignment scores."""

    scores = _score_values(df, score_column)
    path = _prepare_output_path(output_path)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    try:
        ax.hist(
            scores,
            bins=20,
            color=PRIMARY_COLOR,
            edgecolor="white",
            linewidth=0.9,
        )
        ax.set_title("Distribution of Alignment Scores", **TITLE_KWARGS)
        ax.set_xlabel("Alignment score", **LABEL_KWARGS)
        ax.set_ylabel("Number of articles", **LABEL_KWARGS)
        _style_axes(ax, grid_axis="y")
        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
    finally:
        plt.close(fig)


def plot_alignment_boxplot(
    df: pd.DataFrame,
    output_path: str | Path,
    score_column: str = "alignment_score",
) -> None:
    """Save a boxplot of article alignment scores."""

    scores = _score_values(df, score_column)
    path = _prepare_output_path(output_path)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    try:
        ax.boxplot(
            scores,
            vert=True,
            patch_artist=True,
            boxprops={
                "facecolor": SECONDARY_COLOR,
                "edgecolor": "#333333",
                "linewidth": 1.1,
            },
            medianprops={"color": ACCENT_COLOR, "linewidth": 2},
            whiskerprops={"color": "#333333", "linewidth": 1.1},
            capprops={"color": "#333333", "linewidth": 1.1},
            flierprops={
                "marker": "o",
                "markerfacecolor": POINT_COLOR,
                "markeredgecolor": "#333333",
                "markersize": 4,
                "alpha": 0.7,
            },
        )
        ax.set_title("Alignment Score Spread", **TITLE_KWARGS)
        ax.set_ylabel("Alignment score", **LABEL_KWARGS)
        ax.set_xticks([1])
        ax.set_xticklabels(["Articles"])
        _style_axes(ax, grid_axis="y")
        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
    finally:
        plt.close(fig)


def plot_alignment_by_year(
    df: pd.DataFrame,
    output_path: str | Path,
    score_column: str = "alignment_score",
) -> None:
    """Save a line plot of average alignment score by publication year."""

    working = _year_score_frame(df, score_column)
    yearly_scores = (
        working.groupby("year", as_index=False)[score_column]
        .mean()
        .sort_values("year")
    )
    path = _prepare_output_path(output_path)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    try:
        ax.plot(
            yearly_scores["year"],
            yearly_scores[score_column],
            color=PRIMARY_COLOR,
            marker="o",
            markerfacecolor="white",
            markeredgewidth=1.2,
            linewidth=2,
        )
        ax.set_title("Average Alignment Score by Year", **TITLE_KWARGS)
        ax.set_xlabel("Year", **LABEL_KWARGS)
        ax.set_ylabel("Average alignment score", **LABEL_KWARGS)
        _style_axes(ax)
        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
    finally:
        plt.close(fig)


def plot_article_scores_by_year(
    df: pd.DataFrame,
    output_path: str | Path,
    score_column: str = "alignment_score",
) -> None:
    """Save a scatter plot of article-level alignment scores by year."""

    working = _year_score_frame(df, score_column)
    path = _prepare_output_path(output_path)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    try:
        ax.scatter(
            working["year"],
            working[score_column],
            color=POINT_COLOR,
            alpha=0.7,
            edgecolors="white",
            linewidths=0.35,
            s=28,
        )
        ax.set_title("Article Alignment Scores by Year", **TITLE_KWARGS)
        ax.set_xlabel("Year", **LABEL_KWARGS)
        ax.set_ylabel("Alignment score", **LABEL_KWARGS)
        _style_axes(ax)
        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
    finally:
        plt.close(fig)


def plot_embedding_pca(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Save a two-dimensional PCA scatter plot of article embeddings."""

    if "year" not in df.columns:
        raise ValueError("Column 'year' is missing from the DataFrame.")

    matrix = _embedding_matrix(embeddings, len(df))
    years = pd.to_numeric(df["year"], errors="coerce")
    if years.isna().any():
        raise ValueError("Column 'year' must contain only numeric values for PCA plotting.")

    from sklearn.decomposition import PCA

    coordinates = PCA(n_components=2).fit_transform(matrix)
    path = _prepare_output_path(output_path)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    try:
        scatter = ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=years,
            cmap="viridis",
            alpha=0.75,
            edgecolors="none",
            s=28,
        )
        colorbar = fig.colorbar(scatter, ax=ax)
        colorbar.set_label("Year", **LABEL_KWARGS)
        ax.set_title("PCA Projection of Article Embeddings", **TITLE_KWARGS)
        ax.set_xlabel("Principal component 1", **LABEL_KWARGS)
        ax.set_ylabel("Principal component 2", **LABEL_KWARGS)
        _style_axes(ax)
        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
    finally:
        plt.close(fig)


__all__ = [
    "plot_alignment_histogram",
    "plot_alignment_boxplot",
    "plot_alignment_by_year",
    "plot_article_scores_by_year",
    "plot_embedding_pca",
]
