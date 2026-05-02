"""Text cleaning and narrative relevance detection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


CANONICAL_NARRATIVE = (
    "HBM supply constraints are limiting AI accelerator production and creating "
    "bottlenecks in the AI hardware supply chain."
)

NARRATIVE_KEYWORDS = [
    "HBM",
    "High Bandwidth Memory",
    "memory bottleneck",
    "memory shortage",
    "AI supply chain",
    "AI infrastructure bottleneck",
    "CoWoS",
    "advanced packaging",
    "GPU supply constraint",
    "AI accelerator supply",
    "capacity constraint",
    "allocation",
]

ECONOMIC_SALIENCE_KEYWORDS = [
    "pricing",
    "price increase",
    "margin",
    "gross margin",
    "allocation",
    "capacity",
    "supply constraint",
    "demand",
    "lead time",
    "backlog",
    "capex",
    "earnings",
    "revenue",
    "guidance",
    "shortage",
    "bottleneck",
    "market share",
    "production",
    "shipment",
]


def normalize_text(text: str) -> str:
    """Lowercase and compact whitespace while preserving financial tokens."""
    text = str(text or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def keyword_hit(text: str, keywords: Iterable[str]) -> bool:
    lowered = normalize_text(text).lower()
    return any(keyword.lower() in lowered for keyword in keywords)


@dataclass
class NarrativeDetector:
    """Hybrid keyword + semantic detector with optional sentence-transformers.

    The TF-IDF backend is deterministic and lightweight. The sentence-transformer
    backend is used only when installed and explicitly requested or available in
    auto mode.
    """

    backend: str = "auto"
    semantic_threshold: float = 0.22
    model_name: str = "all-MiniLM-L6-v2"

    def __post_init__(self) -> None:
        self.backend_used_: str | None = None
        self._st_model = None
        self._tfidf: TfidfVectorizer | None = None
        self._canonical_embedding: np.ndarray | None = None

    def fit(self, texts: Iterable[str]) -> "NarrativeDetector":
        clean_texts = [normalize_text(text) for text in texts]
        backend = self._resolve_backend()
        self.backend_used_ = backend
        if backend == "sentence-transformers":
            self._st_model = self._load_sentence_transformer()
            self._canonical_embedding = self._st_model.encode(
                [CANONICAL_NARRATIVE], normalize_embeddings=True
            )[0]
        else:
            self._tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
            self._tfidf.fit([CANONICAL_NARRATIVE] + clean_texts)
            self._canonical_embedding = self._tfidf.transform([CANONICAL_NARRATIVE]).toarray()[0]
        return self

    def transform(self, texts: Iterable[str]) -> dict[str, np.ndarray]:
        if self.backend_used_ is None:
            raise RuntimeError("NarrativeDetector must be fitted before transform().")

        clean_texts = [normalize_text(text) for text in texts]
        keyword_flags = np.array([keyword_hit(text, NARRATIVE_KEYWORDS) for text in clean_texts])
        embeddings = self.embed(clean_texts)
        canonical = np.asarray(self._canonical_embedding).reshape(1, -1)
        semantic_scores = cosine_similarity(embeddings, canonical).ravel()
        semantic_flags = semantic_scores >= self.semantic_threshold
        relevant = keyword_flags | semantic_flags
        return {
            "keyword_relevant": keyword_flags.astype(bool),
            "semantic_score": semantic_scores.astype(float),
            "narrative_relevant": relevant.astype(bool),
            "embeddings": embeddings,
        }

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        clean_texts = [normalize_text(text) for text in texts]
        if self.backend_used_ == "sentence-transformers":
            return np.asarray(
                self._st_model.encode(clean_texts, normalize_embeddings=True),
                dtype=float,
            )
        if self._tfidf is None:
            raise RuntimeError("TF-IDF vectorizer is not fitted.")
        return self._tfidf.transform(clean_texts).toarray().astype(float)

    def _resolve_backend(self) -> str:
        requested = self.backend.lower()
        if requested in {"tfidf", "tf-idf"}:
            return "tfidf"
        if requested in {"sentence-transformers", "sentence_transformers"}:
            self._load_sentence_transformer()
            return "sentence-transformers"
        if requested != "auto":
            raise ValueError("backend must be one of: auto, tfidf, sentence-transformers")
        try:
            self._load_sentence_transformer()
            return "sentence-transformers"
        except Exception:
            return "tfidf"

    def _load_sentence_transformer(self):
        if self._st_model is not None:
            return self._st_model
        from sentence_transformers import SentenceTransformer

        self._st_model = SentenceTransformer(self.model_name)
        return self._st_model

