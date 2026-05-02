"""CLI for compliance-aware real-data document discovery."""

from __future__ import annotations

import argparse
from datetime import date

from ingestion.pipeline import IngestionConfig, IngestionPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover compliant real-world documents for HBM narrative modeling.")
    parser.add_argument("--output", default="ingestion_outputs", help="Directory for raw/accepted/rejected/error CSVs.")
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument("--refresh-mode", choices=["weekly", "monthly"], default="monthly")
    parser.add_argument("--max-queries", type=int, default=80)
    parser.add_argument("--max-queries-per-run", type=int, default=30)
    parser.add_argument("--max-results-per-query", type=int, default=10)
    parser.add_argument("--max-pages-per-query", type=int, default=2)
    parser.add_argument("--relevance-threshold", type=float, default=0.35)
    parser.add_argument("--minimum-text-chars", type=int, default=500)
    parser.add_argument("--timeout-seconds", type=int, default=10)
    parser.add_argument("--extraction-timeout-seconds", type=int, default=10)
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--max-candidates-to-process", type=int, default=None)
    parser.add_argument("--skip-extraction", action="store_true", help="Run discovery only and write metadata-only raw candidates.")
    parser.add_argument("--rate-limit-seconds", type=float, default=1.0)
    parser.add_argument("--user-agent", default=IngestionConfig().user_agent)
    parser.add_argument("--adapters", default="serpapi_google,rss_sitemap,investor_relations,sec_edgar,transcript_candidate")
    parser.add_argument("--feed-url", action="append", default=[], help="Public RSS/Atom feed URL. May be repeated.")
    parser.add_argument("--sitemap-url", action="append", default=[], help="Public sitemap URL. May be repeated.")
    parser.add_argument("--search-results-file", default=None, help="Offline CSV/JSON export from SerpAPI or another structured SERP API.")
    parser.add_argument("--serpapi-key", default=None, help="SerpAPI key. Defaults to SERPAPI_KEY environment variable.")
    parser.add_argument("--verbose", action="store_true", help="Print query, extraction, acceptance, and rejection diagnostics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = IngestionConfig(
        output_dir=args.output,
        start_date=args.start_date,
        end_date=args.end_date,
        refresh_mode=args.refresh_mode,
        relevance_threshold=args.relevance_threshold,
        minimum_text_chars=args.minimum_text_chars,
        max_queries=args.max_queries,
        max_queries_per_run=args.max_queries_per_run,
        max_results_per_query=args.max_results_per_query,
        max_pages_per_query=args.max_pages_per_query,
        timeout_seconds=args.timeout_seconds,
        extraction_timeout_seconds=args.extraction_timeout_seconds,
        max_workers=args.max_workers,
        max_candidates_to_process=args.max_candidates_to_process,
        skip_extraction=args.skip_extraction,
        rate_limit_seconds=args.rate_limit_seconds,
        user_agent=args.user_agent,
        enabled_adapters=[name.strip() for name in args.adapters.split(",") if name.strip()],
        feed_urls=args.feed_url,
        sitemap_urls=args.sitemap_url,
        search_results_file=args.search_results_file,
        serpapi_api_key=args.serpapi_key,
        verbose=args.verbose,
    )
    paths = IngestionPipeline(config).run()
    print("Completed compliance-aware ingestion.")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
