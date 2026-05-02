"""Run the HBM narrative diffusion research prototype."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ablation import run_ablation_analysis
from data_loader import infer_source_types, load_documents, save_demo_documents
from feature_engineering import build_feature_matrix
from semantic_frames import aggregate_frame_shares, assign_document_frames
from stability import bootstrap_stage_proxy
from stage_model import (
    attach_stage_predictions,
    compare_models,
    detect_emergence_events,
    global_stage,
    lead_lag_matrix,
    source_activation_table,
)
from text_preprocessing import NarrativeDetector
from visualization import save_all_visualizations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Model HBM shortage narrative formation and diffusion.")
    parser.add_argument("--input", type=str, default=None, help="CSV/JSON/JSONL document file. Uses demo corpus if omitted.")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory for tables and figures.")
    parser.add_argument("--lag-weeks", type=int, default=3, help="Number of temporal lags to include.")
    parser.add_argument("--activation-threshold", type=float, default=0.05, help="Intensity threshold for active source.")
    parser.add_argument("--semantic-backend", choices=["auto", "tfidf", "sentence-transformers"], default="tfidf")
    parser.add_argument("--semantic-threshold", type=float, default=0.22)
    parser.add_argument("--primary-model", choices=["ridge", "mlp"], default="ridge")
    parser.add_argument("--bootstrap", type=int, default=100, help="Number of bootstrap resamples. Use 0 to skip.")
    parser.add_argument("--random-state", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    docs = load_documents(args.input)
    if args.input is None:
        save_demo_documents(output_dir / "demo_documents.csv")

    source_types = infer_source_types(docs)
    detector = NarrativeDetector(
        backend=args.semantic_backend,
        semantic_threshold=args.semantic_threshold,
    )
    weekly, doc_features = build_feature_matrix(
        docs,
        detector=detector,
        source_types=source_types,
        lag_weeks=args.lag_weeks,
        activation_threshold=args.activation_threshold,
    )

    models, comparison = compare_models(weekly, random_state=args.random_state)
    primary_model = models[args.primary_model]
    source_state = attach_stage_predictions(weekly, primary_model)
    global_state = global_stage(source_state)
    emergence_events = detect_emergence_events(
        global_state,
        source_state,
        percentiles=[75, 80, 90],
        activation_threshold=args.activation_threshold,
    )
    emergence_80 = emergence_events[emergence_events["threshold_percentile"] == 80][["week"]].drop_duplicates()
    global_state = global_state.merge(emergence_80.assign(emergence_flag=True), on="week", how="left")
    global_state["emergence_flag"] = global_state["emergence_flag"].eq(True)

    framed_docs = assign_document_frames(doc_features)
    source_frames, global_frames = aggregate_frame_shares(framed_docs, source_types=source_types)
    global_state = global_state.merge(
        global_frames[["week", "dominant_semantic_frame"]],
        on="week",
        how="left",
    )

    activation = source_activation_table(source_state, activation_threshold=args.activation_threshold)
    lead_lag = lead_lag_matrix(activation)

    ablation_summary, ablation_trajectories = run_ablation_analysis(
        docs,
        detector=detector,
        source_types=source_types,
        all_source_global=global_state,
        lag_weeks=args.lag_weeks,
        activation_threshold=args.activation_threshold,
        model_name=args.primary_model,
        random_state=args.random_state,
    )

    if args.bootstrap > 0:
        bootstrap_source, bootstrap_global, bootstrap_global_raw = bootstrap_stage_proxy(
            docs,
            detector=detector,
            source_types=source_types,
            fitted_model=primary_model,
            lag_weeks=args.lag_weeks,
            activation_threshold=args.activation_threshold,
            n_bootstrap=args.bootstrap,
            random_state=args.random_state,
        )
    else:
        bootstrap_source = pd.DataFrame()
        bootstrap_global = pd.DataFrame()
        bootstrap_global_raw = pd.DataFrame()

    weekly.to_csv(table_dir / "engineered_source_week_features.csv", index=False)
    doc_features.drop(columns=["embedding"]).to_csv(table_dir / "document_relevance_features.csv", index=False)
    comparison.to_csv(table_dir / "model_comparison.csv", index=False)
    global_state.to_csv(table_dir / "weekly_global_narrative_state.csv", index=False)
    source_state[
        [
            "week",
            "source_type",
            "StageProxy",
            "DeltaStage",
            "intensity",
            "coherence",
            "salience",
            "persistence",
            "total_docs",
            "relevant_docs",
        ]
    ].to_csv(table_dir / "source_specific_state.csv", index=False)
    emergence_events.to_csv(table_dir / "emergence_sensitivity.csv", index=False)
    activation.to_csv(table_dir / "source_activation_infected_timeline.csv", index=False)
    lead_lag.to_csv(table_dir / "lead_lag_matrix.csv")
    source_frames.to_csv(table_dir / "semantic_frame_share_by_source_week.csv", index=False)
    global_frames.to_csv(table_dir / "semantic_frame_share_global.csv", index=False)
    ablation_summary.to_csv(table_dir / "ablation_summary.csv", index=False)
    ablation_trajectories.to_csv(table_dir / "ablation_trajectories.csv", index=False)
    bootstrap_source.to_csv(table_dir / "bootstrap_source_stage_summary.csv", index=False)
    bootstrap_global.to_csv(table_dir / "bootstrap_global_stage_summary.csv", index=False)
    bootstrap_global_raw.to_csv(table_dir / "bootstrap_global_stage_raw.csv", index=False)

    save_all_visualizations(
        output_dir=output_dir,
        global_state=global_state,
        source_state=source_state,
        activation_table=activation,
        lead_lag=lead_lag,
        ablation_summary=ablation_summary,
        ablation_trajectories=ablation_trajectories,
        bootstrap_global=bootstrap_global,
        frame_global=global_frames,
        emergence_events=emergence_events,
    )

    print(f"Completed HBM narrative diffusion run with detector={detector.backend_used_}, model={args.primary_model}")
    print(f"Tables: {table_dir}")
    print(f"Figures: {output_dir / 'figures'}")


if __name__ == "__main__":
    main()
