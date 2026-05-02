"""Weekly source-level feature construction for latent narrative stage modeling."""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from text_preprocessing import ECONOMIC_SALIENCE_KEYWORDS, NarrativeDetector, keyword_hit, normalize_text


def build_feature_matrix(
    documents: pd.DataFrame,
    detector: NarrativeDetector,
    source_types: list[str],
    lag_weeks: int = 3,
    persistence_window: int = 3,
    activation_threshold: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build source-week features and the future composite target."""
    docs = prepare_document_features(documents, detector)
    weekly = aggregate_weekly_features(
        docs,
        source_types=source_types,
        persistence_window=persistence_window,
        activation_threshold=activation_threshold,
    )
    weekly = add_temporal_features(weekly, lag_weeks=lag_weeks)
    weekly = add_future_target(weekly)
    return weekly, docs


def prepare_document_features(documents: pd.DataFrame, detector: NarrativeDetector) -> pd.DataFrame:
    docs = documents.copy()
    docs["timestamp"] = pd.to_datetime(docs["timestamp"])
    docs["week"] = docs["timestamp"].dt.to_period("W-SUN").dt.start_time
    docs["clean_text"] = (docs["title"].fillna("") + " " + docs["text"].fillna("")).map(normalize_text)
    if detector.backend_used_ is None:
        detector.fit(docs["clean_text"].tolist())
    transformed = detector.transform(docs["clean_text"].tolist())
    docs["keyword_relevant"] = transformed["keyword_relevant"]
    docs["semantic_score"] = transformed["semantic_score"]
    docs["narrative_relevant"] = transformed["narrative_relevant"]
    docs["economic_salient"] = [
        bool(rel and keyword_hit(text, ECONOMIC_SALIENCE_KEYWORDS))
        for rel, text in zip(docs["narrative_relevant"], docs["clean_text"], strict=False)
    ]
    docs["embedding"] = list(transformed["embeddings"])
    return docs


def aggregate_weekly_features(
    docs: pd.DataFrame,
    source_types: list[str],
    persistence_window: int = 3,
    activation_threshold: float = 0.05,
) -> pd.DataFrame:
    min_week = docs["week"].min()
    max_week = docs["week"].max()
    weeks = pd.date_range(min_week, max_week, freq="W-MON")
    if len(weeks) == 0 or weeks[0] != min_week:
        weeks = sorted(docs["week"].unique())

    rows = []
    grouped = docs.groupby(["source_type", "week"], observed=True)
    for source_type, week in product(source_types, weeks):
        try:
            group = grouped.get_group((source_type, week))
        except KeyError:
            group = docs.iloc[0:0]
        total_docs = int(len(group))
        relevant = group[group["narrative_relevant"]] if total_docs else group
        relevant_docs = int(len(relevant))
        intensity = relevant_docs / total_docs if total_docs else 0.0
        coherence = _average_pairwise_cosine(relevant["embedding"].tolist())
        salience = float(relevant["economic_salient"].mean()) if relevant_docs else 0.0
        rows.append(
            {
                "week": pd.Timestamp(week),
                "source_type": source_type,
                "total_docs": total_docs,
                "relevant_docs": relevant_docs,
                "intensity": float(np.clip(intensity, 0.0, 1.0)),
                "coherence": float(np.clip(coherence, 0.0, 1.0)),
                "salience": float(np.clip(salience, 0.0, 1.0)),
            }
        )

    weekly = pd.DataFrame(rows).sort_values(["source_type", "week"]).reset_index(drop=True)
    weekly["persistence"] = (
        weekly.groupby("source_type", group_keys=False)["intensity"]
        .rolling(window=persistence_window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    active = weekly.assign(active_source=weekly["intensity"] > activation_threshold)
    breadth = (
        active.groupby("week", as_index=False)["active_source"]
        .sum()
        .rename(columns={"active_source": "active_source_count"})
    )
    breadth["diffusion_breadth"] = breadth["active_source_count"] / max(1, len(source_types))
    weekly = weekly.merge(breadth, on="week", how="left")
    return weekly


def add_temporal_features(weekly: pd.DataFrame, lag_weeks: int = 3) -> pd.DataFrame:
    out = weekly.sort_values(["source_type", "week"]).copy()
    source_features = ["intensity", "coherence", "salience", "persistence"]
    global_features = ["diffusion_breadth"]

    for feature in source_features:
        out[f"delta_{feature}"] = out.groupby("source_type")[feature].diff().fillna(0.0)
        for lag in range(1, lag_weeks + 1):
            out[f"{feature}_lag_{lag}"] = (
                out.groupby("source_type")[feature].shift(lag).fillna(0.0)
            )

    global_by_week = out[["week", "diffusion_breadth"]].drop_duplicates("week").sort_values("week")
    global_by_week["delta_diffusion_breadth"] = global_by_week["diffusion_breadth"].diff().fillna(0.0)
    for lag in range(1, lag_weeks + 1):
        global_by_week[f"diffusion_breadth_lag_{lag}"] = (
            global_by_week["diffusion_breadth"].shift(lag).fillna(0.0)
        )

    out = out.drop(columns=[col for col in out.columns if col.startswith("diffusion_breadth_lag_")], errors="ignore")
    out = out.merge(
        global_by_week[["week", "delta_diffusion_breadth"] + [f"diffusion_breadth_lag_{lag}" for lag in range(1, lag_weeks + 1)]],
        on="week",
        how="left",
    )
    return out.fillna(0.0)


def add_future_target(weekly: pd.DataFrame) -> pd.DataFrame:
    out = weekly.sort_values(["source_type", "week"]).copy()
    for feature in ["intensity", "coherence", "salience", "persistence"]:
        out[f"{feature}_lead_1"] = out.groupby("source_type")[feature].shift(-1)

    out["target_stage_proxy"] = (
        0.40 * out["intensity_lead_1"]
        + 0.25 * out["coherence_lead_1"]
        + 0.20 * out["salience_lead_1"]
        + 0.15 * out["persistence_lead_1"]
    )
    return out


def feature_columns(weekly: pd.DataFrame) -> list[str]:
    excluded = {
        "week",
        "source_type",
        "target_stage_proxy",
        "intensity_lead_1",
        "coherence_lead_1",
        "salience_lead_1",
        "persistence_lead_1",
    }
    return [
        col
        for col in weekly.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(weekly[col])
    ]


def _average_pairwise_cosine(embeddings: list[np.ndarray]) -> float:
    if len(embeddings) < 2:
        return 0.0
    matrix = np.vstack(embeddings)
    sim = cosine_similarity(matrix)
    upper = sim[np.triu_indices_from(sim, k=1)]
    if upper.size == 0:
        return 0.0
    return float(np.nanmean(upper))
