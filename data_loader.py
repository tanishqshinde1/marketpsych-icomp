"""Data loading and demo corpus generation for narrative diffusion modeling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "document_id",
    "timestamp",
    "source_type",
    "company",
    "title",
    "text",
]

DEFAULT_SOURCE_TYPES = [
    "specialist_media",
    "earnings_call_transcripts",
    "general_news",
]

SOURCE_TYPE_REGISTRY = DEFAULT_SOURCE_TYPES + [
    "earnings_call_transcript",
    "company_ir",
    "sec_filing",
    "supply_chain_vendor",
    "general_web_unknown",
    "analyst_or_institutional_commentary",
    "social_high_frequency_reserved",
    "social_high_frequency",
]


@dataclass(frozen=True)
class CorpusConfig:
    start_week: str = "2023-10-02"
    n_weeks: int = 36
    random_seed: int = 7


def load_documents(path: str | Path | None = None) -> pd.DataFrame:
    """Load documents from CSV/JSONL, or create a deterministic demo corpus."""
    if path is None:
        return generate_demo_documents()

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix.lower() == ".csv":
        docs = pd.read_csv(path)
    elif path.suffix.lower() in {".jsonl", ".ndjson"}:
        docs = pd.read_json(path, lines=True)
    elif path.suffix.lower() == ".json":
        docs = pd.read_json(path)
    else:
        raise ValueError("Supported input formats are .csv, .json, .jsonl, .ndjson")

    return validate_documents(docs)


def validate_documents(docs: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the required document schema."""
    docs = normalize_ingested_schema(docs)
    missing = [col for col in REQUIRED_COLUMNS if col not in docs.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    clean = docs.copy()
    clean["timestamp"] = pd.to_datetime(clean["timestamp"], utc=False, errors="coerce")
    if clean["timestamp"].isna().any():
        bad = clean.loc[clean["timestamp"].isna(), "document_id"].head(5).tolist()
        raise ValueError(f"Invalid timestamps for document_id values: {bad}")

    for col in ["document_id", "source_type", "company", "title", "text"]:
        clean[col] = clean[col].fillna("").astype(str)

    return clean.sort_values(["timestamp", "source_type", "document_id"]).reset_index(drop=True)


def normalize_ingested_schema(docs: pd.DataFrame) -> pd.DataFrame:
    """Map ingestion-layer accepted_documents.csv into the modeling schema."""
    clean = docs.copy()
    rename_map = {
        "doc_id": "document_id",
        "company_entity": "company",
    }
    for old, new in rename_map.items():
        if old in clean.columns and new not in clean.columns:
            clean[new] = clean[old]
    if "timestamp" not in clean.columns:
        if "timestamp_published" in clean.columns:
            clean["timestamp"] = clean["timestamp_published"]
        elif "timestamp_modeling_fallback" in clean.columns:
            clean["timestamp"] = clean["timestamp_modeling_fallback"]
        elif "timestamp_discovered" in clean.columns:
            clean["timestamp"] = clean["timestamp_discovered"]
    if "timestamp" in clean.columns:
        for fallback_col in ["timestamp_modeling_fallback", "timestamp_discovered"]:
            if fallback_col in clean.columns:
                clean["timestamp"] = clean["timestamp"].fillna(clean[fallback_col])
    if "company" not in clean.columns:
        clean["company"] = ""
    if "document_id" not in clean.columns:
        clean["document_id"] = clean.index.map(lambda idx: f"doc_{idx:08d}")
    if "text" not in clean.columns:
        clean["text"] = ""
    if "title" not in clean.columns:
        clean["title"] = ""
    if "source_type" in clean.columns:
        clean["source_type"] = clean["source_type"].replace(
            {
                "earnings_call_transcript": "earnings_call_transcripts",
                "social_high_frequency_reserved": "social_high_frequency",
            }
        )
    return clean


def infer_source_types(docs: pd.DataFrame, include_future_registry: bool = False) -> list[str]:
    """Return known source types in stable order, preserving extensibility."""
    observed = list(dict.fromkeys(docs["source_type"].tolist()))
    base = SOURCE_TYPE_REGISTRY if include_future_registry else DEFAULT_SOURCE_TYPES
    ordered = [source for source in base if source in observed]
    ordered.extend(source for source in sorted(observed) if source not in ordered)
    return ordered or DEFAULT_SOURCE_TYPES.copy()


def generate_demo_documents(config: CorpusConfig | None = None) -> pd.DataFrame:
    """Create a small synthetic corpus with plausible narrative diffusion dynamics.

    The demo corpus is intentionally simple and deterministic. It exists so the
    full research pipeline can be executed before proprietary data is connected.
    """
    cfg = config or CorpusConfig()
    rng = np.random.default_rng(cfg.random_seed)
    weeks = pd.date_range(cfg.start_week, periods=cfg.n_weeks, freq="W-MON")

    source_specs = {
        "specialist_media": {"mean_docs": 5, "shift": 8, "scale": 2.4},
        "earnings_call_transcripts": {"mean_docs": 3, "shift": 13, "scale": 2.8},
        "general_news": {"mean_docs": 4, "shift": 16, "scale": 3.2},
    }
    companies = ["NVDA", "AMD", "MU", "TSMC", "SK Hynix", "Samsung", "ASML"]

    rows: list[dict[str, str]] = []
    doc_id = 1
    for week_idx, week in enumerate(weeks):
        for source_type, spec in source_specs.items():
            n_docs = max(1, int(rng.poisson(spec["mean_docs"])))
            prob = _logistic((week_idx - spec["shift"]) / spec["scale"])
            # Simulate some late-cycle saturation and replacement by adjacent themes.
            if week_idx > 27:
                prob *= max(0.35, 1.0 - 0.035 * (week_idx - 27))

            for local_idx in range(n_docs):
                is_narrative = rng.random() < prob
                company = rng.choice(companies)
                timestamp = week + pd.Timedelta(days=int(rng.integers(0, 5)))
                frame = _frame_for_week(week_idx, rng)
                text, title = (
                    _narrative_doc(source_type, company, frame, week_idx, rng)
                    if is_narrative
                    else _background_doc(source_type, company, rng)
                )
                rows.append(
                    {
                        "document_id": f"demo_{doc_id:05d}",
                        "timestamp": timestamp.isoformat(),
                        "source_type": source_type,
                        "company": company,
                        "title": title,
                        "text": text,
                    }
                )
                doc_id += 1

    return validate_documents(pd.DataFrame(rows))


def save_demo_documents(path: str | Path) -> Path:
    """Write the demo corpus to disk for inspection or modification."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    docs = generate_demo_documents()
    docs.to_csv(path, index=False)
    return path


def _logistic(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _frame_for_week(week_idx: int, rng: np.random.Generator) -> str:
    if week_idx < 10:
        choices = ["technical_constraint", "supply_bottleneck"]
        probs = [0.75, 0.25]
    elif week_idx < 18:
        choices = ["technical_constraint", "supply_bottleneck", "economic_consequence"]
        probs = [0.35, 0.45, 0.20]
    elif week_idx < 27:
        choices = ["supply_bottleneck", "economic_consequence", "market_investor_framing"]
        probs = [0.30, 0.45, 0.25]
    else:
        choices = ["economic_consequence", "market_investor_framing", "technical_constraint"]
        probs = [0.45, 0.35, 0.20]
    return str(rng.choice(choices, p=probs))


def _narrative_doc(
    source_type: str,
    company: str,
    frame: str,
    week_idx: int,
    rng: np.random.Generator,
) -> tuple[str, str]:
    frame_text = {
        "technical_constraint": (
            "HBM and High Bandwidth Memory capacity remain tightly linked to CoWoS, "
            "advanced packaging, memory bandwidth, substrate availability, and GPU supply constraint."
        ),
        "supply_bottleneck": (
            "Customers describe allocation, long lead time, and shortage conditions as AI accelerator "
            "supply struggles to meet demand across the AI hardware supply chain."
        ),
        "economic_consequence": (
            "Pricing, gross margin, capex, revenue guidance, backlog, shipment timing, and production "
            "plans are increasingly framed around HBM supply constraints."
        ),
        "market_investor_framing": (
            "Investors debate winners, losers, market share, valuation, stock reaction, and whether "
            "memory bottlenecks change the economics of AI infrastructure."
        ),
    }[frame]

    source_voice = {
        "specialist_media": "A semiconductor trade analysis notes",
        "earnings_call_transcripts": "Management said during prepared remarks and Q&A that",
        "general_news": "A broad technology report says",
    }[source_type]
    title = f"{company} HBM bottleneck narrative update week {week_idx:02d}"
    noise = rng.choice(
        [
            "The discussion also mentioned foundry utilization and networking components.",
            "Executives cautioned that qualification cycles can differ by customer.",
            "Analysts said supply agreements may smooth some quarter-to-quarter volatility.",
        ]
    )
    text = (
        f"{source_voice} HBM shortage is becoming a binding bottleneck in AI infrastructure. "
        f"{frame_text} {noise}"
    )
    return text, title


def _background_doc(source_type: str, company: str, rng: np.random.Generator) -> tuple[str, str]:
    topics = [
        "automotive semiconductor demand and industrial inventory correction",
        "smartphone recovery, PC channel normalization, and consumer device seasonality",
        "data center networking, optical modules, and server refresh timing",
        "foundry node migration, export controls, and regional manufacturing incentives",
        "software ecosystem updates around AI model deployment and cloud spending",
    ]
    topic = str(rng.choice(topics))
    title = f"{company} sector note on {topic}"
    source_voice = {
        "specialist_media": "The technical note reviewed",
        "earnings_call_transcripts": "The company discussed",
        "general_news": "The business article covered",
    }[source_type]
    text = f"{source_voice} {topic}. The piece focused on adjacent sector context rather than the target narrative."
    return text, title
