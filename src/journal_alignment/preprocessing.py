"""Text preprocessing utilities for article abstracts and Aims & Scope text."""

from __future__ import annotations

import re

import pandas as pd


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace while preserving the original casing."""

    if text is None:
        return ""
    if not isinstance(text, str):
        if pd.isna(text):
            return ""
        text = str(text)
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str) -> str:
    """Apply conservative text cleaning suitable for later embedding models."""

    return normalize_whitespace(text)


def preprocess_dataframe(
    df: pd.DataFrame,
    text_column: str = "abstract",
) -> pd.DataFrame:
    """Return a copy of ``df`` with normalized text in ``text_column``."""

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' is missing from the DataFrame.")

    processed = df.copy()
    processed[text_column] = processed[text_column].map(clean_text)
    return processed
