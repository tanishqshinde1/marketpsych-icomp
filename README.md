# HBM Narrative Diffusion Prototype

This project is a research-grade prototype for modeling how the narrative
"HBM shortage as a binding bottleneck in the AI hardware supply chain" forms,
spreads, accelerates, and stabilizes across heterogeneous information sources.

It is not a sentiment model and it is not a return prediction model. Phase 1
focuses only on narrative diffusion, SEIR-inspired latent stage inference,
source contribution, semantic framing, ablation, and stability.

## Research Motivation

Financial narratives do not appear everywhere at once. A technical constraint
may first surface in specialist semiconductor media, then become economically
salient in earnings calls, and only later stabilize into broad market coverage.
This prototype treats that process as a latent diffusion problem over source
populations and weekly observations.

The target narrative is:

> HBM supply constraints are limiting AI accelerator production and creating
> bottlenecks in the AI hardware supply chain.

## SEIR Mapping

The SEIR analogy is used as an interpretive framework, not as a system of
differential equations.

- `S = Susceptible`: a source population is not meaningfully discussing the narrative.
- `E = Exposed`: isolated, low-intensity, low-coherence mentions appear.
- `I = Infected`: the narrative is coherent, persistent, and economically salient.
- `R = Recovered`: novelty fades as the narrative is absorbed into consensus,
  declines, or is replaced by an adjacent narrative.

The model does not observe true SEIR labels. It estimates a continuous
`StageProxy` in `[0, 1]`.

## Data Schema

Each document must contain:

- `document_id`
- `timestamp`
- `source_type`
- `company`
- `title`
- `text`

Phase 1 source populations are:

- `specialist_media`
- `earnings_call_transcripts`
- `general_news`

The source registry also reserves `social_high_frequency` for a future extension.

## Feature Definitions

The core unit is `(source_type, week)`.

- `intensity`: share of documents in the source-week that mention the narrative.
- `coherence`: average pairwise cosine similarity among relevant documents.
- `salience`: fraction of relevant documents with economic consequence language.
- `persistence`: rolling 3-week mean of intensity.
- `diffusion_breadth`: fraction of source populations active in the week.
- lag features: 1 to `k` week lags for intensity, coherence, salience,
  persistence, and diffusion breadth.
- first differences: weekly changes in each state feature.

Narrative relevance uses a hybrid detector:

- keyword matching for terms such as `HBM`, `High Bandwidth Memory`, `CoWoS`,
  `advanced packaging`, `GPU supply constraint`, `allocation`, and related terms.
- optional semantic similarity to the canonical narrative. `sentence-transformers`
  can be used when installed; otherwise the default TF-IDF backend is used.

## Learning Target

The target is a future composite proxy for next-period narrative maturity:

```text
Y[t,s] =
  0.40 * intensity[t+1,s]
+ 0.25 * coherence[t+1,s]
+ 0.20 * salience[t+1,s]
+ 0.15 * persistence[t+1,s]
```

Rows without future values are dropped from training. The fitted model estimates:

```text
StageProxy[t,s] = f_theta(X[t,s])
```

Low values are closer to susceptible/exposed conditions. Rising values indicate
movement toward active propagation. High values indicate an infected-stage proxy.
Declines after high values suggest saturation or recovery.

## Models

Two models are trained and compared:

- `ridge`: regularized linear baseline for interpretability.
- `mlp`: small feedforward neural network with one hidden layer.

The default primary model for downstream tables and figures is `ridge`.

## Semantic Frames

Narrative-relevant documents are assigned to one or more frames:

- `technical_constraint`: HBM capacity, CoWoS, packaging, memory bandwidth.
- `supply_bottleneck`: shortage, allocation, lead times, inability to meet demand.
- `economic_consequence`: pricing, margins, earnings, revenue, capex.
- `market_investor_framing`: winners, losers, valuation, stock reaction, market share.

Outputs include frame shares and dominant frame by source-week.

## Ablation and Sufficiency

The pipeline trains source-subset models for:

- each single source
- each two-source combination
- all sources

It reports emergence week, trajectory deviation from the all-source model,
stability variance, holdout RMSE, and a marginal information gain proxy.

The goal is to ask which smaller source subset is sufficient to detect emergence
with stable results.

## Stability

Bootstrap analysis resamples documents with replacement within each source-week,
recomputes features, and predicts StageProxy with the fitted model. Outputs
include mean, 5th percentile, 95th percentile, and variance by week and source.

## How to Run

Install ordinary scientific Python dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib google-search-results
```

Optional semantic backend:

```bash
pip install sentence-transformers
```

Run the demo corpus:

```bash
python main.py --output outputs --bootstrap 100
```

Run with your own document file:

```bash
python main.py --input path/to/documents.csv --output outputs --semantic-backend tfidf
```

Use the small neural net as the primary downstream model:

```bash
python main.py --primary-model mlp --output outputs_mlp
```

## Real Data Ingestion

The ingestion layer is query-driven and compliance-aware. It discovers candidate
URLs, checks access constraints and robots policy where feasible, extracts full
text only when permitted, and otherwise keeps metadata-only records for manual
review.

Run ingestion:

```bash
python ingest_main.py --output ingestion_outputs --start-date 2023-01-01
```

Use SerpAPI as the search discovery provider:

```bash
export SERPAPI_KEY="..."
python ingest_main.py --output ingestion_outputs --adapters serpapi_google,investor_relations,transcript_candidate,sec_edgar
```

Or use an offline search-results export with `url`, `title`, and optional
`query` columns from SerpAPI or another structured, officially supported SERP
API:

```bash
python ingest_main.py --output ingestion_outputs --search-results-file search_results.csv
```

RSS and sitemap discovery can be added without changing code:

```bash
python ingest_main.py \
  --output ingestion_outputs \
  --feed-url https://example.com/feed.xml \
  --sitemap-url https://example.com/sitemap.xml
```

The ingestion outputs are:

- `raw_candidates.csv`: every discovered item, including restricted and low-relevance candidates.
- `accepted_documents.csv`: only accessible, permitted, relevant full-text items for modeling.
- `rejected_documents.csv`: duplicates, restricted items, parse failures, and irrelevant candidates.
- `ingestion_errors.csv`: explicit error states such as `paywalled`, `robots_blocked`,
  `timeout`, `parse_failed`, `missing_timestamp`, and `unknown_source_type`.
- `acceptance_diagnostics.csv`: aggregate acceptance/rejection counts, missing
  metadata counts, and source-type/rejection-reason breakdowns.

Restricted domains and transcript providers are not bypassed. If content is
paywalled, login-gated, robots-disallowed, or otherwise restricted, the pipeline
stores metadata and URL only with `text_permission_status = metadata_only`.
The search adapter never scrapes Google or Bing result pages directly and does
not treat SERP snippets as document text.

Acceptance is relevance-first: accessible, permitted full-text documents are not
rejected solely because `timestamp_published`, `source_type`, or
`company_entity` is imperfect. Metadata issues are tracked with
`timestamp_quality`, `timestamp_modeling_fallback`, `source_type_quality`, and
`metadata_quality_notes`.

Handoff to modeling:

```bash
python main.py --input ingestion_outputs/accepted_documents.csv --output real_outputs
```

The modeling loader maps ingestion fields such as `doc_id`,
`timestamp_published`, and `company_entity` into the modeling schema.

Useful ingestion debug controls:

```bash
python ingest_main.py \
  --output ingestion_outputs_debug \
  --adapters serpapi_google \
  --max-queries-per-run 2 \
  --max-results-per-query 5 \
  --max-pages-per-query 1 \
  --max-candidates-to-process 10 \
  --max-workers 5 \
  --timeout-seconds 8 \
  --extraction-timeout-seconds 8 \
  --verbose
```

Use `--skip-extraction` to validate SerpAPI discovery output without fetching
candidate pages. During extraction, verbose mode prints progress every five
processed URLs and partial CSVs are flushed every 25 processed candidates.

## Outputs

Tables are written to `outputs/tables/`:

- `weekly_global_narrative_state.csv`
- `source_specific_state.csv`
- `ablation_summary.csv`
- `emergence_sensitivity.csv`
- `source_activation_infected_timeline.csv`
- `lead_lag_matrix.csv`
- `semantic_frame_share_global.csv`
- `semantic_frame_share_by_source_week.csv`
- bootstrap summaries

Figures are written to `outputs/figures/`:

1. Global StageProxy with confidence bands
2. GlobalDeltaStage with emergence spikes
3. Source-specific StageProxy trajectories
4. Source activation and infected-proxy timeline
5. Lead-lag heatmap
6. Ablation comparison
7. Bootstrap stability
8. Semantic frame share over time

## Interpretation

- A large positive `GlobalDeltaStage` is a narrative acceleration signal.
- Sustained high `GlobalStage` suggests active propagation.
- A falling stage after a high plateau suggests saturation or recovery.
- Lead-lag tables show which source populations cross activation or infected
  proxy thresholds first.
- Bootstrap variance shows whether inference is stable or document-sample
  dependent.

## Limitations

- SEIR states are latent proxies, not observed truth.
- `StageProxy` is trained on future narrative maturity, not true infection labels.
- Coherence can be unstable with few documents.
- Market pricing is not modeled in Phase 1.
- X / social media is reserved for a future high-frequency extension.
- The model detects narrative structure, not guaranteed tradable alpha.
- The demo corpus is synthetic and exists only to exercise the pipeline.
