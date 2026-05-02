"""Source subset ablation and information sufficiency analysis."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from feature_engineering import build_feature_matrix
from stage_model import (
    attach_stage_predictions,
    detect_emergence_events,
    global_stage,
    train_stage_model,
)
from text_preprocessing import NarrativeDetector


def default_source_subsets(source_types: list[str]) -> list[tuple[str, ...]]:
    subsets: list[tuple[str, ...]] = []
    for r in range(1, len(source_types) + 1):
        subsets.extend(tuple(combo) for combo in combinations(source_types, r))
    return subsets


def run_ablation_analysis(
    documents: pd.DataFrame,
    detector: NarrativeDetector,
    source_types: list[str],
    all_source_global: pd.DataFrame,
    lag_weeks: int = 3,
    activation_threshold: float = 0.05,
    model_name: str = "ridge",
    random_state: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train source-subset models and compare trajectories to all-source output."""
    rows = []
    trajectories = []
    for subset in default_source_subsets(source_types):
        subset_docs = documents[documents["source_type"].isin(subset)].copy()
        subset_weekly, _ = build_feature_matrix(
            subset_docs,
            detector=detector,
            source_types=list(subset),
            lag_weeks=lag_weeks,
            activation_threshold=activation_threshold,
        )
        model, metrics = train_stage_model(subset_weekly, model_name=model_name, random_state=random_state)
        source_state = attach_stage_predictions(subset_weekly, model)
        global_df = global_stage(source_state)
        events = detect_emergence_events(
            global_df,
            source_state,
            percentiles=[80],
            activation_threshold=activation_threshold,
        )
        emergence_week = events["week"].min() if not events.empty else pd.NaT
        deviation = _aligned_rmse(
            all_source_global[["week", "GlobalStage"]],
            global_df[["week", "GlobalStage"]],
        )
        stability_variance = float(global_df["GlobalDeltaStage"].var(ddof=0))
        subset_name = " + ".join(subset)
        rows.append(
            {
                "source_subset": subset_name,
                "n_sources": len(subset),
                "emergence_week": emergence_week,
                "stability_variance": stability_variance,
                "deviation_from_all_source": deviation,
                "target_rmse_holdout": metrics["rmse_holdout"],
            }
        )
        trajectories.append(global_df.assign(source_subset=subset_name))

    summary = pd.DataFrame(rows)
    singleton_baseline = summary.loc[summary["n_sources"] == 1, "deviation_from_all_source"].mean()
    if not np.isfinite(singleton_baseline) or singleton_baseline == 0:
        singleton_baseline = summary["deviation_from_all_source"].max()
    if not np.isfinite(singleton_baseline) or singleton_baseline == 0:
        summary["marginal_information_gain"] = 0.0
    else:
        summary["marginal_information_gain"] = (
            singleton_baseline - summary["deviation_from_all_source"]
        ).clip(lower=0.0)
    trajectory_df = pd.concat(trajectories, ignore_index=True) if trajectories else pd.DataFrame()
    return summary.sort_values(["n_sources", "source_subset"]).reset_index(drop=True), trajectory_df


def _aligned_rmse(reference: pd.DataFrame, candidate: pd.DataFrame) -> float:
    merged = reference.merge(candidate, on="week", suffixes=("_ref", "_candidate"))
    if merged.empty:
        return float("nan")
    err = merged["GlobalStage_ref"] - merged["GlobalStage_candidate"]
    return float(np.sqrt(np.mean(np.square(err))))
