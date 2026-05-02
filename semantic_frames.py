"""Semantic frame assignment for narrative-relevant documents."""

from __future__ import annotations

import pandas as pd

from text_preprocessing import keyword_hit, normalize_text


FRAME_KEYWORDS = {
    "technical_constraint": [
        "hbm",
        "high bandwidth memory",
        "cowos",
        "advanced packaging",
        "memory bandwidth",
        "substrate",
        "packaging",
        "technical",
    ],
    "supply_bottleneck": [
        "shortage",
        "allocation",
        "lead time",
        "unable to meet demand",
        "supply constraint",
        "capacity constraint",
        "bottleneck",
        "supply chain",
    ],
    "economic_consequence": [
        "pricing",
        "price increase",
        "margin",
        "gross margin",
        "earnings",
        "revenue",
        "guidance",
        "capex",
        "backlog",
        "shipment",
        "production",
    ],
    "market_investor_framing": [
        "winner",
        "loser",
        "valuation",
        "stock reaction",
        "market share",
        "investor",
        "multiple",
        "share price",
    ],
}


def assign_document_frames(docs: pd.DataFrame) -> pd.DataFrame:
    """Assign one or more coarse semantic frames to narrative documents."""
    rows = []
    for row in docs.itertuples(index=False):
        text = normalize_text(f"{getattr(row, 'title', '')} {getattr(row, 'text', '')}")
        is_relevant = bool(getattr(row, "narrative_relevant", False))
        hits = {
            frame: int(is_relevant and keyword_hit(text, keywords))
            for frame, keywords in FRAME_KEYWORDS.items()
        }
        if is_relevant and not any(hits.values()):
            hits["technical_constraint"] = 1
        rows.append(hits)

    frame_df = pd.DataFrame(rows)
    out = docs.reset_index(drop=True).copy()
    for frame in FRAME_KEYWORDS:
        out[frame] = frame_df[frame].astype(int)
    out["assigned_frame_count"] = out[list(FRAME_KEYWORDS)].sum(axis=1)
    return out


def aggregate_frame_shares(
    framed_docs: pd.DataFrame,
    source_types: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return source-week and global-week semantic frame shares."""
    docs = framed_docs.copy()
    docs["week"] = pd.to_datetime(docs["week"])
    if source_types is None:
        source_types = sorted(docs["source_type"].unique())

    frames = list(FRAME_KEYWORDS)
    relevant = docs[docs["narrative_relevant"]].copy()
    if relevant.empty:
        source_week = _empty_source_frame_table(docs, source_types, frames)
        global_week = _global_from_source(source_week, frames)
        return source_week, global_week

    grouped = (
        relevant.groupby(["source_type", "week"], observed=True)[frames]
        .sum()
        .reset_index()
    )
    grouped["frame_total"] = grouped[frames].sum(axis=1).replace(0, pd.NA)
    for frame in frames:
        grouped[f"{frame}_share"] = (grouped[frame] / grouped["frame_total"]).fillna(0.0)

    full_index = pd.MultiIndex.from_product(
        [source_types, sorted(docs["week"].unique())],
        names=["source_type", "week"],
    )
    source_week = (
        grouped.set_index(["source_type", "week"])
        .reindex(full_index)
        .fillna(0.0)
        .reset_index()
    )
    share_cols = [f"{frame}_share" for frame in frames]
    source_week["dominant_semantic_frame"] = source_week[share_cols].idxmax(axis=1).str.replace(
        "_share", "", regex=False
    )
    source_week.loc[source_week[share_cols].sum(axis=1) == 0, "dominant_semantic_frame"] = "none"
    global_week = _global_from_source(source_week, frames)
    return source_week, global_week


def _empty_source_frame_table(docs: pd.DataFrame, source_types: list[str], frames: list[str]) -> pd.DataFrame:
    full_index = pd.MultiIndex.from_product(
        [source_types, sorted(docs["week"].unique())],
        names=["source_type", "week"],
    )
    table = full_index.to_frame(index=False)
    for frame in frames:
        table[frame] = 0.0
        table[f"{frame}_share"] = 0.0
    table["frame_total"] = 0.0
    table["dominant_semantic_frame"] = "none"
    return table


def _global_from_source(source_week: pd.DataFrame, frames: list[str]) -> pd.DataFrame:
    share_cols = [f"{frame}_share" for frame in frames]
    global_week = source_week.groupby("week", as_index=False)[share_cols].mean()
    global_week["dominant_semantic_frame"] = global_week[share_cols].idxmax(axis=1).str.replace(
        "_share", "", regex=False
    )
    global_week.loc[global_week[share_cols].sum(axis=1) == 0, "dominant_semantic_frame"] = "none"
    return global_week

