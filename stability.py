"""Bootstrap stability analysis for document-level sampling uncertainty."""

from __future__ import annotations

import numpy as np
import pandas as pd

from feature_engineering import build_feature_matrix
from stage_model import FittedStageModel, attach_stage_predictions, global_stage
from text_preprocessing import NarrativeDetector


def bootstrap_stage_proxy(
    documents: pd.DataFrame,
    detector: NarrativeDetector,
    source_types: list[str],
    fitted_model: FittedStageModel,
    lag_weeks: int = 3,
    activation_threshold: float = 0.05,
    n_bootstrap: int = 100,
    random_state: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Resample documents within each source-week and recompute StageProxy."""
    rng = np.random.default_rng(random_state)
    docs = documents.copy()
    docs["timestamp"] = pd.to_datetime(docs["timestamp"])
    docs["week"] = docs["timestamp"].dt.to_period("W-SUN").dt.start_time

    source_records = []
    global_records = []
    grouped = docs.groupby(["source_type", "week"], observed=True)
    group_keys = list(grouped.groups.keys())

    for iteration in range(n_bootstrap):
        sampled_groups = []
        for key in group_keys:
            group = grouped.get_group(key)
            sampled_idx = rng.choice(group.index.to_numpy(), size=len(group), replace=True)
            sampled_groups.append(docs.loc[sampled_idx].copy())
        sample_docs = pd.concat(sampled_groups, ignore_index=True)
        weekly, _ = build_feature_matrix(
            sample_docs,
            detector=detector,
            source_types=source_types,
            lag_weeks=lag_weeks,
            activation_threshold=activation_threshold,
        )
        source_state = attach_stage_predictions(weekly, fitted_model)
        global_df = global_stage(source_state)
        source_records.append(source_state.assign(iteration=iteration))
        global_records.append(global_df.assign(iteration=iteration))

    source_boot = pd.concat(source_records, ignore_index=True)
    global_boot = pd.concat(global_records, ignore_index=True)
    source_summary = summarize_bootstrap(
        source_boot,
        group_cols=["source_type", "week"],
        value_col="StageProxy",
    )
    global_summary = summarize_bootstrap(
        global_boot,
        group_cols=["week"],
        value_col="GlobalStage",
    )
    return source_summary, global_summary, global_boot


def summarize_bootstrap(
    boot: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
) -> pd.DataFrame:
    summary = (
        boot.groupby(group_cols, as_index=False)[value_col]
        .agg(
            mean="mean",
            lower=lambda x: x.quantile(0.05),
            upper=lambda x: x.quantile(0.95),
            variance=lambda x: x.var(ddof=0),
        )
        .rename(
            columns={
                "mean": f"{value_col}_mean",
                "lower": f"{value_col}_lower",
                "upper": f"{value_col}_upper",
                "variance": f"{value_col}_variance",
            }
        )
    )
    return summary
