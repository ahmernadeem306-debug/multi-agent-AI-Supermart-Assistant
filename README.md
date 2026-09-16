# BizAgent — AI Supermart Operations Assistant

BizAgent is a multi-agent AI operations assistant for a supermarket / retail store. It answers
operational questions, diagnoses *why* stockouts, shrinkage and supply-chain delays happen with
evidence-backed root-cause reports, forecasts demand and perishable expiry, and answers policy
questions from a real knowledge base — all through a LangGraph multi-agent orchestrator, a real MCP
tool server, a RAG pipeline over ChromaDB, and an XGBoost forecasting model, surfaced through a
six-page Streamlit dashboard and a FastAPI backend.

**The problem it solves** (from the project proposal): supermarket managers face unpredicted
stockouts and inventory shrinkage, manual tracking of SOPs/vendor agreements/return policies that
causes compliance errors, and siloed POS/stockroom/supplier data with no automated root-cause
diagnostics tying it together. BizAgent is a single assistant that unifies structured operational data,
policy documents and predictive analytics behind one multi-agent interface.

> Requirements source: *BizAgent Capstone Project Proposal* (Soojal Kumar & Ahmer Nadeem).
> Implementation source of truth: [`BIZAGENT_BUILD_PLAN.md`](BIZAGENT_BUILD_PLAN.md).

---

## Table of contents

- [Key features](#key-features)
- [Architecture](#architecture)
- [Technology stack](#technology-stack)
- [Project structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Environment variables](#environment-variables)
- [Getting a Groq API key](#getting-a-groq-api-key)
- [Database setup and seeding](#database-setup-and-seeding)
- [Knowledge-base ingestion](#knowledge-base-ingestion)
- [Model training](#model-training)
- [Running the app](#running-the-app)
- [Running the tests](#running-the-tests)
- [Example usage](#example-usage)
- [API documentation](#api-documentation)
- [MCP tool catalogue](#mcp-tool-catalogue-13-tools)
- [Agent catalogue](#agent-catalogue)
- [Screenshots](#screenshots)
- [Deployment](#deployment)
- [Known limitations](#known-limitations)
- [Future improvements](#future-improvements)
- [Credits and licence](#credits-and-licence)

---

## Key features

Mapped to the proposal's five major functionalities (§4 of the proposal / §1.5 of the build plan):

1. **RAG Knowledge Engine** — 7 real documents (shelf-stocking SOP, returns & refunds policy,
   employee handbook, two supplier contracts, shrinkage/loss-prevention SOP, inventory count &
   discrepancy SOP) chunked, embedded and stored in ChromaDB; every policy answer carries a document
   title, section and similarity score.
2. **MCP Tool Server Integration** — a real Model Context Protocol server (13 read-only, typed,
   result-capped tools) is the *only* path from an agent to the database; agents never see raw SQL or
   an open connection.
3. **Multi-Agent Orchestration** — a LangGraph Supervisor routes each question to up to five
   specialist agents (Sales, Inventory & Stock, Finance, Demand Forecasting, Policy), running the
   relevant ones in parallel and synthesising one coherent answer.
4. **Predictive Analytics** — an XGBoost demand model (with an always-available seasonal-naive
   fallback) drives stockout-risk dates, reorder-point recommendations and perishable expiry alerts.
5. **Interactive Dashboard** — a six-page Streamlit app: aisle-level metrics, a chat page with a full
   agent decision trace, root-cause reports, forecasting charts, the knowledge base, and a searchable
   agent decision log.

Plus the differentiator the proposal calls out explicitly: **automated root-cause diagnostics** — a
dedicated LangGraph subgraph (not a single prompt) that gathers sales, inventory, shrinkage, supply
and forecast evidence in parallel, retrieves governing policy text, and produces a structured report
where every cause carries real, code-validated evidence.

## Architecture

```
                       ┌───────────────────────────┐
                       │   Streamlit Dashboard     │
                       │  metrics · chat · RCA ·   │
                       │  forecasts · agent logs   │
                       └────────────┬──────────────┘
                                    │ HTTP (httpx)
                       ┌────────────▼──────────────┐
                       │        FastAPI API        │
                       │ /query /rca /metrics      │
                       │ /forecast /documents /logs│
                       └────────────┬──────────────┘
                                    │
                       ┌────────────▼──────────────┐
                       │   LangGraph Orchestrator  │
                       │      Supervisor Agent     │
                       └───┬───┬───┬───┬───┬───────┘
             ┌─────────────┘   │   │   │   └──────────────┐
             ▼                 ▼   ▼   ▼                  ▼
        Inventory           Sales  Finance  Forecast    Policy
          Agent             Agent   Agent    Agent      Agent
             │                 │      │        │           │
             └────────┬────────┴──────┴────────┘           │
                      ▼                                    ▼
          ┌───────────────────────┐            ┌───────────────────────┐
          │   MCP Client Adapter  │            │    RAG Retriever      │
          │  (ToolProvider ABC)   │            │  ChromaDB + local     │
          └───────────┬───────────┘            │      embeddings       │
                      │ stdio (MCP)            └───────────┬───────────┘
          ┌───────────▼───────────┐                        │
          │    MCP Tool Server    │                  data/knowledge/
          │  read-only, typed,    │                  SOPs · policies ·
          │  parameter-validated  │                  handbooks · contracts
          └───────────┬───────────┘
                      │
          ┌───────────▼───────────┐        ┌──────────────────────────┐
          │  Service / Repository │◄───────│  Forecasting Service     │
          │        layer          │        │  XGBoost + feature eng.  │
          └───────────┬───────────┘        └──────────────────────────┘
                      │
          ┌───────────▼───────────┐
          │  SQLite (SQLAlchemy)  │
          │ POS · stockroom ·     │
          │ suppliers · shrinkage │
          └───────────────────────┘
```

**Hard rule:** agents never touch the database directly —
`Agent → MCP tool → service → repository → DB`, enforced by a static import guard, not just
convention. A root-cause diagnosis is a separate LangGraph subgraph that reuses the same MCP tools and
the same RAG retriever (see the sequence diagram below); it does not duplicate any analytics.

**Full component breakdown, data-flow sequence diagrams for `/query` and `/rca`, and every design
decision with its rationale:** [`docs/architecture.md`](docs/architecture.md).

## Technology stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | Required by the proposal |
| Agent orchestration | LangGraph + LangChain | Genuine parallel multi-agent fan-out and a dedicated RCA subgraph — a single prompt or if/else router can't do either |
| LLM | Groq (`groq`, OpenAI-compatible chat completions) | Fast, low-cost inference; isolated behind `app/llm/provider.py::LLMProvider` so it's swappable |
| Embeddings | `sentence-transformers` local backend | Groq has no embeddings API, so RAG embeddings run fully offline |
| Vector DB | ChromaDB (persistent) | Metadata filtering, simple on Windows, no external service to run |
| Tool protocol | Model Context Protocol (`mcp` SDK, FastMCP, stdio) | Satisfies "safe structured retrieval, no raw DB access" as a real protocol boundary, not a wrapper |
| API layer | FastAPI + Pydantic v2 | Async-capable, typed request/response validation for free |
| Forecasting | XGBoost (primary), seasonal-naive baseline | Installs cleanly everywhere (Prophet's toolchain is the #1 way this kind of project loses a day on Windows); baseline guarantees the feature works before training |
| Database | SQLite + SQLAlchemy 2.x ORM | Zero setup for an evaluator; a `DATABASE_URL` edit reaches Postgres later |
| Frontend | Streamlit (multipage) | Fast to build a real operational dashboard without a JS framework |
| Config | `pydantic-settings` + `.env` | Centralised, validated, fails fast on a missing key |
| Testing | `pytest`, `pytest-asyncio`, `pytest-cov`, `httpx` | Full offline coverage; `mcp`/live-Groq tests isolated behind `-m live` |
| Packaging | `requirements.txt` (pinned) + `Dockerfile` + `docker-compose.yml` | Reproducible install; a documented container path |

## Project structure

```
BizAgent/
├── app/
│   ├── agents/            # Supervisor, 5 specialists, the query graph, the RCA subgraph + schemas
│   ├── api/                # FastAPI app, routes (health/query/metrics/forecast/rca/documents/logs), schemas
│   ├── core/               # Exception hierarchy, agent decision log
│   ├── db/                 # SQLAlchemy models, engine/session, repositories (the only ORM callers)
│   ├── llm/                 # LLMProvider protocol, GroqProvider, structured-output parsing, prompts/
│   ├── mcp_client/          # ToolProvider protocol: DirectToolProvider + MCPToolProvider (stdio)
│   ├── mcp_server/          # The FastMCP server + 13 read-only tool modules + schemas
│   ├── ml/                  # Feature engineering, forecast backends, training, evaluation
│   ├── rag/                 # Chunking, embeddings, Chroma store, ingestion, cited retriever
│   ├── services/            # Business logic: inventory, sales, finance, supplier, analytics, forecasting
│   ├── config.py            # pydantic-settings Settings (every env var, with defaults)
│   └── logging_config.py    # structlog → stderr (stdout is reserved for the MCP JSON-RPC stream)
├── data/
│   ├── knowledge/            # 7 policy/SOP/handbook/contract Markdown documents (committed)
│   └── anomalies.json        # The 4 planted anomalies written by the seeder (committed)
├── docs/
│   ├── architecture.md, demo_script.md, requirements_traceability.md,
│   │   security_review.md, viva_questions.md
│   └── screenshots/           # Six dashboard page captures (see docs/screenshots/README.md)
├── models/                    # forecast_xgb.meta.json committed; the .joblib model is gitignored
├── scripts/                    # seed_db, ingest_docs, train_forecast, run_mcp_server, mcp_smoke_test,
│                                 run_demo_scenarios, smoke_groq
├── tests/                       # 30 test modules, see "Running the tests"
├── ui/
│   ├── components/            # common (API client + sidebar), charts, report_view
│   ├── pages/                  # 1_Dashboard … 6_Agent_Logs
│   └── streamlit_app.py        # Landing page + shared sidebar
├── Dockerfile, docker-compose.yml, .dockerignore
├── requirements.txt (pinned), .env.example, .gitignore
├── LICENSE (MIT)
└── BIZAGENT_BUILD_PLAN.md      # The single source of truth this project was built from
```

## Prerequisites

- **Python 3.11+** (developed and tested on 3.10.9 and 3.11; both work — the build plan targets
  3.11+, use whichever is available if 3.11 isn't installed).
- **OS:** Windows, macOS or Linux. Developed primarily on Windows; no OS-specific code paths.
- ~2 GB of free disk for the `sentence-transformers`/PyTorch dependency, used by the default (and only
  supported) `EMBEDDING_BACKEND=local`.
- A **Groq API key** — see below. The app runs and degrades gracefully without one, but every AI
  feature needs it for full functionality.

## Installation

```bash
git clone <this-repository-url> bizagent
cd bizagent

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Environment variables

Copy `.env.example` to `.env` and fill in `GROQ_API_KEY`. Every other variable has a sensible
default — change only what you need to.

| Variable | Default | Required | Purpose |
|---|---|---|---|
| `GROQ_API_KEY` | *(none)* | **Yes** | Your Groq key. App fails fast with `ConfigError` if missing. |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | No | Chat/completion model. |
| `GROQ_TIMEOUT_SECONDS` | `60` | No | Hard timeout per Groq call. |
| `DATABASE_URL` | `sqlite:///./data/bizagent.db` | No | SQLAlchemy connection string. |
| `LOG_LEVEL` | `INFO` | No | stdlib/structlog level. |
| `API_HOST` | `0.0.0.0` | No | uvicorn bind host. |
| `API_PORT` | `8000` | No | uvicorn bind port. |
| `API_BASE_URL` | `http://localhost:8000` | No | Where Streamlit looks for the API. |
| `CHROMA_PERSIST_DIR` | `./data/chroma` | No | ChromaDB persistence directory. |
| `TOOL_PROVIDER` | `mcp` | No | `mcp` (real stdio subprocess) or `direct` (in-process). |
| `TOOL_PROVIDER_FALLBACK` | `false` | No | If `true`, fall back to `direct` when the MCP handshake fails. |
| `EMBEDDING_BACKEND` | `local` | No | `local` (only supported backend; `groq` fails fast — Groq has no embeddings API). |
| `LOCAL_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | No | Used only when `EMBEDDING_BACKEND=local`. |
| `KNOWLEDGE_DIR` | `./data/knowledge` | No | Source directory for `ingest_docs.py` and uploads. |
| `RAG_COLLECTION_PREFIX` | `bizagent_kb` | No | Chroma collection name prefix (backend + dimension are appended automatically). |
| `RAG_CHUNK_SIZE` | `800` | No | Target characters per chunk. |
| `RAG_CHUNK_OVERLAP` | `120` | No | Overlap characters between consecutive chunks. |
| `RAG_TOP_K` | `5` | No | Chunks retrieved per query. |
| `RAG_MIN_SCORE` | `0.15` | No | Minimum cosine similarity to accept a chunk. |
| `AGENT_MAX_TOOL_CALLS` | `3` | No | Max tool calls per specialist agent per turn. |
| `GRAPH_RECURSION_LIMIT` | `25` | No | LangGraph step/recursion cap for the query graph. |
| `LLM_TEMPERATURE_ROUTING` | `0.0` | No | Temperature for routing/argument synthesis. |
| `LLM_TEMPERATURE_SYNTHESIS` | `0.35` | No | Temperature for final narrative synthesis. |
| `FORECAST_BACKEND` | `xgboost` | No | `xgboost` (falls back to `seasonal_naive` if untrained) or `seasonal_naive`. |
| `FORECAST_HORIZON_DAYS` | `14` | No | Default forecast horizon. |
| `FORECAST_MAX_HORIZON_DAYS` | `30` | No | Maximum allowed horizon (API-validated). |
| `FORECAST_MIN_HISTORY_DAYS` | `60` | No | Below this, forecasts are flagged low-confidence. |
| `MODEL_DIR` | `./models` | No | Where the trained model bundle + metadata live. |
| `REORDER_Z_SCORE` | `1.65` | No | Service-level multiplier (~95%) for reorder safety stock. |
| `STOCKOUT_CRITICAL_DAYS` | `3` | No | `days_of_cover` below this ⇒ `critical`. |
| `STOCKOUT_HIGH_DAYS` | `7` | No | ⇒ `high`. |
| `STOCKOUT_MEDIUM_DAYS` | `14` | No | ⇒ `medium`; above ⇒ `low`. |
| `MAX_UPLOAD_BYTES` | `10485760` (10 MB) | No | Knowledge-base upload size cap. |
| `APP_ENV` | `development` | No | Informational only. |

## Getting a Groq API key

1. Go to [Groq Console](https://console.groq.com/keys) and sign in.
2. Click **Create API Key**.
3. Copy the key and put it in `.env`:
   ```
   GROQ_API_KEY=your_actual_key_here
   ```
4. Verify it works: `python scripts/smoke_groq.py` — prints one real Groq response, or a clear
   error naming the problem (missing key, invalid key, network) if something is wrong.

`.env` is gitignored — never commit it. `.env.example` documents every variable with a placeholder.

## Database setup and seeding

```bash
python scripts/seed_db.py --reset
```

Creates the 9-table SQLite schema and populates it with a deterministic (`seed=42`), reproducible
synthetic dataset: 6 suppliers, 60 SKUs across 8 aisles, 540 days of daily sales, daily stock
snapshots, perishable batches, ~120 purchase orders (~20% late), ~80 shrinkage events, and **exactly
four planted anomalies** written to `data/anomalies.json`:

| ID | SKU | Anomaly |
|---|---|---|
| A | SKU-1035 | Stockout caused by a purchase order delivered 9 days late |
| B | SKU-1002 | Abnormal spike in theft-reason shrinkage |
| C | SKU-1001 | A large perishable batch about to expire, causing a write-off and a follow-on stockout |
| D | SKU-1003 | Sales roughly doubled while the reorder point stayed at the old level |

Run without `--reset` to seed once and refuse to double-seed. There are no migrations (Alembic is out
of scope for this project) — re-run with `--reset` after any schema change.

## Knowledge-base ingestion

```bash
python scripts/ingest_docs.py            # ingest data/knowledge/, skip unchanged files
python scripts/ingest_docs.py --force    # re-embed every file regardless of change
```

Chunks, embeds and stores all 7 documents in ChromaDB, and records each in the `documents` table with
a sha256 hash so re-running is idempotent (unchanged files are skipped; a changed file has its old
chunks replaced, not duplicated).

## Model training

```bash
python scripts/train_forecast.py [--valid-days 28]
```

Trains the global XGBoost demand model on a **time-based** holdout (never shuffled), prints an
MAE/RMSE/MAPE/WAPE comparison against the seasonal-naive baseline, and saves
`models/forecast_xgb.joblib` + `models/forecast_xgb.meta.json`. Forecasting works even before this is
run — `FORECAST_BACKEND=xgboost` automatically falls back to the baseline if no model file exists.

Latest recorded backtest (28-day holdout, 1,680 rows):

| model | MAE | RMSE | MAPE | WAPE |
|---|---|---|---|---|
| seasonal_naive | 3.85 | 6.07 | 14.41% | 13.93% |
| **xgboost** | **3.66** | **5.76** | **13.73%** | **13.22%** |

XGBoost beats the baseline by ~5% MAE — reported honestly either way; see `docs/architecture.md` §5.

## Running the app

Run these **in order**, each in its own terminal, after seeding + ingesting + (optionally) training:

```bash
# Terminal 1 — MCP tool server (stdio; the API spawns this automatically if you skip
# this terminal and TOOL_PROVIDER=mcp, but running it standalone here also lets you
# use `python scripts/mcp_smoke_test.py` independently)
python scripts/run_mcp_server.py

# Terminal 2 — FastAPI backend
uvicorn app.api.main:app --reload --port 8000

# Terminal 3 — Streamlit dashboard
streamlit run ui/streamlit_app.py
```

Open `http://localhost:8501`. The sidebar shows API health, the active tool provider and forecast
backend. API docs (Swagger UI) are at `http://localhost:8000/docs`.

Verification scripts:

```bash
python scripts/smoke_groq.py           # confirm the Groq key works
python scripts/mcp_smoke_test.py       # list + call all 13 MCP tools over real stdio
python scripts/run_demo_scenarios.py   # diagnose all 4 planted anomalies — must print 4/4
```

A `Makefile` wraps all of the above (`make seed`, `make ingest`, `make train`, `make demo`,
`make run-mcp`, `make run-api`, `make run-ui`, `make test`, `make test-live`).

## Running the tests

```bash
pytest -q                                     # 183 passed, 11 deselected — no API key, no network
pytest --cov=app --cov-report=term-missing    # 91% coverage on app/
pytest -m live                                 # 11 passed — spawns a real MCP subprocess
```

- The default run needs **no Groq API key and no network** — the `groq` SDK is fully mocked in
  `tests/test_llm_provider.py` and every agent/graph/RCA test uses a scripted or stub LLM
  (`tests/fakes.py`). A dummy key (`test-key-for-pytest-do-not-use`) is set in `tests/conftest.py`
  purely so `Settings` validates.
- Tests marked `@pytest.mark.live` (`tests/test_tool_provider_parity.py`) spawn a real MCP subprocess
  to prove `MCPToolProvider` and `DirectToolProvider` return identical results; they need no Groq
  key either, only the `mcp` package (already a dependency).
- `tests/test_smoke_e2e.py` is a single runnable regression check: seed → ingest (fake embeddings, no
  network) → `/query` → `/rca` → `/forecast`, all through the live FastAPI app.
- Coverage breakdown: `app/services` 95%, `app/agents` 92%, `app/mcp_server` 94%, `app/ml` 96% — all
  above the 80% package target; overall `app/` is 91%, well above the 70% floor.

## Example usage

Six operational questions on the **Ask BizAgent** page and what each demonstrates:

| # | Question | Demonstrates |
|---|---|---|
| 1 | "Which SKUs are low on stock in the Produce aisle?" | Inventory agent → `list_low_stock` MCP tool, real seeded data |
| 2 | "What were the top-selling SKUs in Beverages last 30 days?" | Sales agent → `get_top_sellers` |
| 3 | "What is the gross margin on the Dairy aisle?" | Finance agent → `get_margin_report` |
| 4 | "Which SKUs will stock out next week?" | Forecast agent → `get_demand_forecast`, XGBoost-backed |
| 5 | "How long do customers have to return frozen goods?" | Policy agent → RAG retrieval with a document citation |
| 6 | "Why did the Dairy margin drop while stock held?" | Multi-agent fan-out: Finance + Inventory run in parallel, one synthesised answer |

The four demo RCA scenarios (Root Cause Analysis page, or the sidebar "run demo scenario" shortcut):

| Scenario | SKU | Expected primary cause |
|---|---|---|
| Late-delivery stockout | SKU-1035 | `supply_delay` |
| Theft shrinkage spike | SKU-1002 | `shrinkage_theft` |
| Expiry write-off + stockout | SKU-1001 | `expiry_writeoff` |
| Demand shift, stale reorder point | SKU-1003 | `reorder_point` |

Run `python scripts/run_demo_scenarios.py` to see all four diagnosed end-to-end with their full
evidence and confidence in one command.

## API documentation

Interactive docs (Swagger UI): `http://localhost:8000/docs` (ReDoc: `/redoc`).

| Method | Path | Request | Response | Example |
|---|---|---|---|---|
| GET | `/health` | — | `{status, version, db_connected, groq_configured, tool_provider, forecast_backend, forecast_model_trained}` | `curl localhost:8000/health` |
| POST | `/query` | `{question, run_id?}` | `{run_id, answer, route, agents_invoked, tool_calls[], citations[], confidence, plan_reasoning, latency_ms, status}` | `curl -X POST localhost:8000/query -d '{"question":"Which SKUs are low on stock?"}'` |
| GET | `/metrics/aisles` | `?days=` | `{days, aisles: [...]}` | `curl localhost:8000/metrics/aisles?days=30` |
| GET | `/metrics/aisle/{aisle}` | `?days=` | per-aisle metrics dict | `curl localhost:8000/metrics/aisle/Produce` |
| GET | `/metrics/low-stock` | `?aisle=&limit=` | `{items[], count}` | `curl localhost:8000/metrics/low-stock` |
| GET | `/metrics/expiring` | `?days_ahead=&aisle=` | `{items[], count}` | `curl localhost:8000/metrics/expiring?days_ahead=7` |
| GET | `/metrics/suppliers` | — | `{suppliers[]}` | `curl localhost:8000/metrics/suppliers` |
| GET | `/metrics/anomalies` | `?kind=&days=` | `{days, results: {kind: [...]}}` | `curl localhost:8000/metrics/anomalies?kind=stockout` |
| GET | `/forecast/skus` | — | `{skus: [{sku, name, aisle, is_perishable}]}` | `curl localhost:8000/forecast/skus` |
| GET | `/forecast/{sku}` | `?horizon_days=&include_risk=` | `{sku, backend, forecast[], stockout_risk, reorder_recommendation, expiry_risk[]}` | `curl localhost:8000/forecast/SKU-1035?horizon_days=14` |
| GET | `/forecast/alerts` | `?limit=` | `{generated_at, count, alerts[]}` | `curl localhost:8000/forecast/alerts` |
| GET | `/forecast/reorder-recommendations` | `?limit=` | `{generated_at, count, recommendations[]}` | `curl localhost:8000/forecast/reorder-recommendations` |
| POST | `/rca` | `{sku? , aisle?, anomaly_type, start_date?, end_date?}` | `RootCauseReport` (summary, ranked_causes[], timeline[], policy_findings[], recommended_actions[], data_gaps[], evidence[], status, llm_calls) | `curl -X POST localhost:8000/rca -d '{"sku":"SKU-1035","anomaly_type":"auto"}'` |
| GET | `/documents` | — | `{documents[], count, total_chunks}` | `curl localhost:8000/documents` |
| POST | `/documents/search` | `{query, doc_type?}` | `{found, citations[]}` (or `{found:false, message}`) | `curl -X POST localhost:8000/documents/search -d '{"query":"return window for frozen goods"}'` |
| POST | `/documents/reingest` | `{force}` | `{results[], total_chunks}` | `curl -X POST localhost:8000/documents/reingest -d '{"force":false}'` |
| POST | `/documents/upload` | multipart file (`.pdf`/`.md`/`.txt`/`.docx`, ≤10 MB) | `{filename, ingested, total_chunks}` | `curl -F file=@policy.md localhost:8000/documents/upload` |
| GET | `/logs` | `?limit=&offset=` | `{total, limit, offset, runs[]}` | `curl localhost:8000/logs?limit=20` |
| GET | `/logs/{run_id}` | — | full `agent_runs` row | `curl localhost:8000/logs/<run_id>` |

Every error response is `{error_code, message, run_id}` with the matching HTTP status (`400`
validation, `404` not found, `422` schema validation, `429` rate limit, `500`/`503` upstream/internal)
— never a raw traceback.

## MCP tool catalogue (13 tools)

All read-only, all Pydantic-validated, all result-capped (`limit` default 100, max 500 where
applicable). Full input/output schemas: `app/mcp_server/schemas.py`.

| # | Tool | Arguments | Purpose |
|---|---|---|---|
| 1 | `get_product_info` | `sku?`, `name_query?` | Resolve a SKU or partial product name to canonical product info |
| 2 | `get_stock_level` | `sku`, `include_batches=false` | Current shelf/backroom/on-hand quantity, optionally with perishable batches |
| 3 | `list_low_stock` | `aisle?`, `limit=50` | SKUs at or below their reorder point, worst shortfall first |
| 4 | `get_sales_history` | `sku`, `start_date`, `end_date`, `granularity=daily` | Units + revenue for a SKU, bucketed daily or weekly |
| 5 | `get_top_sellers` | `aisle?`, `days=30`, `limit=10` | Best-selling SKUs by units over a trailing window |
| 6 | `get_shrinkage_report` | `sku?`, `aisle?`, `days=90` | Shrinkage cost broken down by reason (damage/theft/expiry/admin_error) |
| 7 | `get_expiring_batches` | `days_ahead=7`, `aisle?` | Perishable batches expiring soon |
| 8 | `get_supplier_status` | `supplier_id?`, `supplier_name?` | Supplier reliability: on-time rate, avg delay, open/late PO counts |
| 9 | `get_open_purchase_orders` | `sku?`, `supplier?`, `late_only=false` | Open POs, or (with `late_only`) POs received after the promised date |
| 10 | `get_margin_report` | `sku?`, `aisle?`, `days=30` | Gross margin for a SKU, an aisle, or store-wide |
| 11 | `get_aisle_metrics` | `aisle`, `days=30` | Dashboard metrics for one aisle: revenue, units, stock/margin/shrinkage/expiry |
| 12 | `detect_anomalies` | `kind` (shrinkage\|late_delivery\|demand_shift\|stockout), `days=90` | Run one anomaly detector, evidence records not prose |
| 13 | `get_demand_forecast` | `sku`, `horizon_days=14`, `include_risk=true` | Forecast series + stockout risk + reorder recommendation + expiry risk |

## Agent catalogue

| Agent | Responsibility | Permitted tools |
|---|---|---|
| **Supervisor** | Classifies each question into `sales\|inventory\|finance\|forecast\|policy\|rca\|smalltalk` and plans which specialists run; falls back to a deterministic keyword router if the LLM call fails | — (routes only, calls no tools) |
| **Sales** | Sales history, velocity, top sellers, demand shifts | `get_product_info`, `get_sales_history`, `get_top_sellers`, `detect_anomalies` |
| **Inventory & Stock** | Stock levels, low stock, batches, expiring stock, discrepancies | `get_product_info`, `get_stock_level`, `list_low_stock`, `get_expiring_batches`, `detect_anomalies` |
| **Finance** | Margins, revenue/COGS, shrinkage valuation, margin erosion | `get_product_info`, `get_margin_report`, `get_shrinkage_report`, `get_aisle_metrics` |
| **Demand Forecasting** | Future demand, stockout risk, reorder points, days of cover | `get_product_info`, `get_demand_forecast`, `get_sales_history`, `get_stock_level`, `get_top_sellers` |
| **Policy** | Answers strictly from retrieved policy chunks, always cites sources; calls no MCP tools | *(RAG retriever only)* |
| **RCA workflow** | Not a chat agent — a dedicated LangGraph subgraph invoked by `POST /rca`; gathers sales/inventory/shrinkage/supply/forecast evidence and cites policy, then ranks causes | All 13 tools (via the six evidence nodes) + the RAG retriever |

## Screenshots

Capture instructions and exact filenames: [`docs/screenshots/README.md`](docs/screenshots/README.md).
Once captured, they render here:

### Dashboard
![Dashboard](docs/screenshots/01_dashboard.png)

### Ask BizAgent
![Ask BizAgent](docs/screenshots/02_ask_bizagent.png)

### Root Cause Analysis
![Root Cause Analysis](docs/screenshots/03_root_cause.png)

### Forecasting
![Forecasting](docs/screenshots/04_forecasting.png)

### Knowledge Base
![Knowledge Base](docs/screenshots/05_knowledge_base.png)

### Agent Logs
![Agent Logs](docs/screenshots/06_agent_logs.png)

## Deployment

### Local

Exactly as in [Running the app](#running-the-app) — three processes, no containers.

### Docker

```bash
docker build -t bizagent .
docker run --rm -p 8000:8000 --env-file .env \
  -v "$PWD/data:/app/data" -v "$PWD/models:/app/models" bizagent

# one-time setup inside the running container:
docker exec -it <container_name> python scripts/seed_db.py --reset
docker exec -it <container_name> python scripts/ingest_docs.py
docker exec -it <container_name> python scripts/train_forecast.py
```

Or both API + UI together:

```bash
docker compose up --build
docker compose exec api python scripts/seed_db.py --reset
docker compose exec api python scripts/ingest_docs.py
docker compose exec api python scripts/train_forecast.py
```

The MCP server runs as a subprocess of the `api` container automatically (`TOOL_PROVIDER=mcp`); no
separate container is needed. **Limitation:** the image installs `sentence-transformers`/PyTorch for
the local embedding backend, making the image large (~2–3 GB); trimming that is future
work, not attempted here to keep the Dockerfile low-risk.

### Streamlit Community Cloud

Streamlit Community Cloud can host the **UI only** — it has no facility to run a second FastAPI
process or a stdio MCP subprocess alongside it. To deploy there:

1. Deploy the FastAPI backend somewhere that can run a long-lived process (a small VM, Render, Fly.io,
   etc.) and note its public URL.
2. On Streamlit Community Cloud, point the app at `ui/streamlit_app.py`, set `requirements.txt` as the
   dependency file, and add `API_BASE_URL=https://<your-api-host>` and `GROQ_API_KEY` as app
   secrets.
3. **Limitation:** `TOOL_PROVIDER` must be `direct` on whatever host runs the API in this split
   deployment unless that host can also spawn a stdio subprocess reliably; Streamlit Cloud itself
   never runs the API or the MCP server. This is a genuine gap between "runs on my machine" and a
   fully managed multi-process deployment, called out here rather than glossed over.

## Known limitations

Honest, not hidden — several are deliberate scope decisions documented in
`docs/architecture.md` (§4, Assumptions A-01…A-10):

- **POS, stockroom and supplier data are simulated**, not live integrations — the proposal names
  these systems without providing one to integrate with; `scripts/seed_db.py` generates a
  deterministic, realistic dataset instead (Assumption A-02/A-03).
- **No authentication.** Single-store, single-tenant demo scope (Assumption A-07).
- **Batch, not real-time, data.** The UI refreshes on demand against seeded/batch data, not a live
  feed (Assumption A-08).
- **Single-store scope.** The schema and every service assume one store; multi-store is future work.
- **Synthetic dataset only** — `random.seed(42)`, no real customer, supplier or personal data anywhere.
- **The seeded data window is date-relative** ("today minus N days"); re-run `seed_db.py --reset` and
  `train_forecast.py` together after a long gap so the planted anomalies stay inside the detectors'
  windows (see `docs/architecture.md`).
- **No Alembic migrations** — a schema change requires `seed_db.py --reset`.
- **RCA ranking is deterministic-primary by design**: Groq refines ordering and writes the
  narrative, but a strong rule-based cause is never demoted below a weaker LLM-proposed one. This is a
  documented reliability trade-off (`docs/architecture.md` D-05), not an oversight.
- **`app/llm/groq_client.py` and `app/mcp_client/client.py`** are only exercised by the `live` test
  tier (real Groq calls / a real MCP subprocess), so their line coverage (61%/51%) is lower than the
  rest of the codebase by design — see `docs/requirements_traceability.md`.
- **The Docker image is large** (PyTorch is installed for the local embedding path) — see
  Deployment above.
- **Groq has no embeddings API** — `EMBEDDING_BACKEND=groq` is accepted by config but fails fast with
  a clear error at call time; `local` (sentence-transformers) is the only supported embedding backend.

## Future improvements

- Prophet as a second forecasting backend behind the existing `ForecastBackend` interface, compared
  honestly against XGBoost rather than replacing it.
- A FAISS vector-store backend behind the existing `ChromaVectorStore` interface.
- Multi-store support and a lightweight role selector (Manager / Staff / Supply Chain) wired to
  per-role default dashboard views.
- A real POS/webhook or scheduled-polling ingestion path to replace the batch-seeded dataset.
- Role-based authentication.
- A smaller Docker image (split the local-embedding extras into an optional stage).

## Credits and licence

Built by **Soojal Kumar** and **Ahmer Nadeem** as a capstone project, following
[`BIZAGENT_BUILD_PLAN.md`](BIZAGENT_BUILD_PLAN.md) as the implementation source of truth against the
*BizAgent Capstone Project Proposal*.

Licensed under the [MIT License](LICENSE).
