"""Query generation for HBM narrative discovery."""

from __future__ import annotations

from dataclasses import dataclass, field


CORE_NARRATIVE_TERMS = [
    "HBM shortage",
    "High Bandwidth Memory shortage",
    "HBM supply constraint",
    "HBM allocation",
    "HBM capacity shortage",
    "HBM bottleneck AI",
    "AI memory bottleneck",
    "AI supply chain bottleneck",
    "CoWoS bottleneck",
    "advanced packaging constraint",
    "HBM DRAM capacity reallocation",
]

COMPANY_EXPANSIONS = [
    "NVIDIA",
    "AMD",
    "Micron",
    "SK Hynix",
    "Samsung",
    "TSMC",
    "Broadcom",
    "Marvell",
]

SOURCE_INTENT_VARIANTS = [
    "earnings call transcript",
    "investor relations",
    "earnings call prepared remarks",
    "quarterly results",
    "supply constraint",
    "capacity expansion",
    "capex",
    "pricing",
]


@dataclass(frozen=True)
class QueryPlan:
    start_date: str = "2023-01-01"
    end_date: str | None = None
    core_terms: list[str] = field(default_factory=lambda: CORE_NARRATIVE_TERMS.copy())
    companies: list[str] = field(default_factory=lambda: COMPANY_EXPANSIONS.copy())
    intents: list[str] = field(default_factory=lambda: SOURCE_INTENT_VARIANTS.copy())
    max_queries: int = 80


def generate_queries(plan: QueryPlan) -> list[str]:
    """Generate query strings from narrative terms, companies, and source intent."""
    queries: list[str] = []
    for term in plan.core_terms:
        queries.append(term)
        for company in plan.companies:
            queries.append(f"{term} {company}")
            queries.append(f"{term} {company} earnings call")
            queries.append(f"{term} {company} investor relations")
            queries.append(f"{term} {company} quarterly results")
        queries.extend(
            [
                f"{term} semiconductor trade press",
                f"{term} TrendForce",
                f"{term} SemiAnalysis",
                f"{term} Tom's Hardware",
                f"{term} AnandTech",
                f"{term} EE Times",
                f"{term} Data Center Knowledge",
                f"{term} Business Korea",
            ]
        )

    for company in plan.companies:
        queries.extend(
            [
                f"HBM shortage {company} earnings call transcript",
                f"{company} HBM earnings call transcript",
                f"{company} high bandwidth memory earnings call",
                f"{company} HBM supply constraint transcript",
                f"{company} AI memory bottleneck earnings call",
            ]
        )

    queries.extend(
        [
            "CoWoS bottleneck TSMC earnings call",
            "AI memory bottleneck semiconductor trade press",
            "HBM shortage semiconductor trade press",
            "advanced packaging constraint AI accelerator supply",
        ]
    )
    return list(dict.fromkeys(queries))[: plan.max_queries]
