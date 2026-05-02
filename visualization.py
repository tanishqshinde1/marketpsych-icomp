"""Visualization utilities for narrative diffusion outputs."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from semantic_frames import FRAME_KEYWORDS


def save_all_visualizations(
    output_dir: str | Path,
    global_state: pd.DataFrame,
    source_state: pd.DataFrame,
    activation_table: pd.DataFrame,
    lead_lag: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    ablation_trajectories: pd.DataFrame,
    bootstrap_global: pd.DataFrame | None,
    frame_global: pd.DataFrame,
    emergence_events: pd.DataFrame,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "global_stage": fig_dir / "01_global_stage_proxy.png",
        "global_delta": fig_dir / "02_global_delta_stage_emergence.png",
        "source_stage": fig_dir / "03_source_stage_proxy.png",
        "activation_timeline": fig_dir / "04_source_activation_timeline.png",
        "lead_lag_heatmap": fig_dir / "05_lead_lag_heatmap.png",
        "ablation": fig_dir / "06_ablation_comparison.png",
        "bootstrap": fig_dir / "07_bootstrap_stability.png",
        "semantic_frames": fig_dir / "08_semantic_frame_share.png",
    }
    plot_global_stage(global_state, bootstrap_global, paths["global_stage"])
    plot_global_delta(global_state, emergence_events, paths["global_delta"])
    plot_source_stage(source_state, paths["source_stage"])
    plot_activation_timeline(activation_table, paths["activation_timeline"])
    plot_lead_lag_heatmap(lead_lag, paths["lead_lag_heatmap"])
    plot_ablation(ablation_summary, ablation_trajectories, paths["ablation"])
    plot_bootstrap_stability(bootstrap_global, paths["bootstrap"])
    plot_semantic_frames(frame_global, paths["semantic_frames"])
    return paths


def plot_global_stage(global_state: pd.DataFrame, bootstrap_global: pd.DataFrame | None, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(global_state["week"], global_state["GlobalStage"], color="#1f77b4", linewidth=2.2, label="Global StageProxy")
    if bootstrap_global is not None and not bootstrap_global.empty:
        merged = global_state[["week"]].merge(bootstrap_global, on="week", how="left")
        ax.fill_between(
            merged["week"],
            merged["GlobalStage_lower"],
            merged["GlobalStage_upper"],
            color="#1f77b4",
            alpha=0.18,
            label="90% bootstrap band",
        )
    ax.set_title("Global Narrative StageProxy Over Time")
    ax.set_xlabel("Week")
    ax.set_ylabel("StageProxy [0, 1]")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_global_delta(global_state: pd.DataFrame, emergence_events: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.bar(global_state["week"], global_state["GlobalDeltaStage"], width=5, color="#6c8ebf", alpha=0.8)
    if not emergence_events.empty:
        flagged = emergence_events[emergence_events["threshold_percentile"] == 80]
        ax.scatter(flagged["week"], flagged["GlobalDeltaStage"], color="#d62728", s=60, label="Emergence candidate, p80")
    ax.set_title("GlobalDeltaStage With Emergence Spikes")
    ax.set_xlabel("Week")
    ax.set_ylabel("Weekly change in Global StageProxy")
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_source_stage(source_state: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    for source_type, group in source_state.groupby("source_type", observed=True):
        ax.plot(group["week"], group["StageProxy"], linewidth=2, marker="o", markersize=2.5, label=source_type)
    ax.set_title("Source-Specific StageProxy Trajectories")
    ax.set_xlabel("Week")
    ax.set_ylabel("StageProxy [0, 1]")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_activation_timeline(activation: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 3.8))
    y_positions = np.arange(len(activation))
    for y, row in zip(y_positions, activation.itertuples(index=False), strict=False):
        ax.scatter(pd.Timestamp(row.first_activation_week), y, color="#2ca02c", s=80, label="First activation" if y == 0 else None)
        if pd.notna(row.first_infected_proxy_week_quantile):
            ax.scatter(pd.Timestamp(row.first_infected_proxy_week_quantile), y, color="#d62728", marker="D", s=70, label="First infected proxy, q75" if y == 0 else None)
        if pd.notna(row.first_infected_proxy_week_absolute):
            ax.scatter(pd.Timestamp(row.first_infected_proxy_week_absolute), y, color="#9467bd", marker="x", s=80, label="First infected proxy, 0.65" if y == 0 else None)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(activation["source_type"])
    ax.set_title("Source Activation and Infected-Proxy Timeline")
    ax.set_xlabel("Week")
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_lead_lag_heatmap(matrix: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    values = matrix.to_numpy(dtype=float)
    finite_abs_max = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 1.0
    if finite_abs_max == 0:
        finite_abs_max = 1.0
    im = ax.imshow(values, cmap="coolwarm", vmin=-finite_abs_max, vmax=finite_abs_max)
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_title("Lead-Lag Matrix, Weeks\npositive means row source leads column source")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            label = "" if np.isnan(values[i, j]) else f"{values[i, j]:.0f}"
            ax.text(j, i, label, ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, label="Weeks")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_ablation(summary: pd.DataFrame, trajectories: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ordered = summary.sort_values("deviation_from_all_source")
    axes[0].barh(ordered["source_subset"], ordered["deviation_from_all_source"], color="#7f7f7f")
    axes[0].set_title("Deviation From All-Source Trajectory")
    axes[0].set_xlabel("RMSE vs all-source GlobalStage")
    if not trajectories.empty:
        for subset, group in trajectories.groupby("source_subset", observed=True):
            alpha = 0.8 if " + " in subset else 0.45
            axes[1].plot(group["week"], group["GlobalStage"], linewidth=1.6, alpha=alpha, label=subset)
    axes[1].set_title("Ablation GlobalStage Trajectories")
    axes[1].set_xlabel("Week")
    axes[1].set_ylabel("GlobalStage")
    axes[1].legend(fontsize=7, loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_bootstrap_stability(bootstrap_global: pd.DataFrame | None, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    if bootstrap_global is None or bootstrap_global.empty:
        ax.text(0.5, 0.5, "Bootstrap was not run", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.plot(bootstrap_global["week"], bootstrap_global["GlobalStage_variance"], color="#ff7f0e", linewidth=2)
    ax.set_title("Bootstrap Stability: Global StageProxy Variance")
    ax.set_xlabel("Week")
    ax.set_ylabel("Bootstrap variance")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_semantic_frames(frame_global: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    share_cols = [f"{frame}_share" for frame in FRAME_KEYWORDS]
    y = [frame_global[col].to_numpy(dtype=float) for col in share_cols]
    labels = [frame.replace("_", " ") for frame in FRAME_KEYWORDS]
    ax.stackplot(frame_global["week"], y, labels=labels, alpha=0.85)
    ax.set_title("Semantic Frame Share Over Time")
    ax.set_xlabel("Week")
    ax.set_ylabel("Share of assigned narrative frames")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
