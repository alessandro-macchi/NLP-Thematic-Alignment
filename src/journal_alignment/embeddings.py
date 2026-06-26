"""Embedding utilities for representing journal content."""

from __future__ import annotations

from typing import Any

import numpy as np


class EmbeddingModel:
    """Small wrapper around a SentenceTransformer embedding model."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
    ) -> None:
        """Initialize the sentence embedding model.

        Parameters
        ----------
        model_name:
            Name of the SentenceTransformer model to load.
        batch_size:
            Number of texts encoded in each batch.
        """

        if not model_name.strip():
            raise ValueError("model_name must be a non-empty string.")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required to use EmbeddingModel. "
                "Install it with `pip install sentence-transformers`."
            ) from exc

        self.model_name = model_name
        self.batch_size = batch_size
        self.model: Any = SentenceTransformer(model_name)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode a non-empty list of texts into a NumPy array."""

        if not texts:
            raise ValueError("texts must contain at least one text.")
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("All items in texts must be strings.")
        if any(not text.strip() for text in texts):
            raise ValueError("texts must not contain empty strings.")

        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def encode_single(self, text: str) -> np.ndarray:
        """Encode one text into a single embedding vector."""

        if not isinstance(text, str):
            raise TypeError("text must be a string.")
        if not text.strip():
            raise ValueError("text must be a non-empty string.")

        return self.encode_texts([text])[0]
