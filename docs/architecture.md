# Architecture

## 1. Components

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
          │  (ToolProvider ABC)   │            │   ChromaDB + Gemini   │
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

**Hard rule:** agents never touch the database or SQLAlchemy directly.
`Agent → MCP tool → service → repository → DB`. This is what satisfies "without exposing raw
database access" and is the single most important architectural constraint in the project — enforced
by a static import guard (`tests/test_graph_e2e.py::test_agents_package_has_no_forbidden_imports`),
not just convention.

| Layer | Package | Responsibility |
|---|---|---|
| Presentation | `ui/` | Streamlit pages + shared components. Talks to the API only, over HTTP. |
| API | `app/api/` | FastAPI routes, request/response schemas, one exception handler. No business logic. |
| Orchestration | `app/agents/` | Supervisor routing, 5 specialist agents, the LangGraph query graph, the RCA subgraph. |
| Prompting | `app/llm/prompts/` | One module per agent: `SYSTEM` + `build_user_prompt(...)` constants. |
| LLM | `app/llm/` | `LLMProvider` protocol; `GeminiProvider` is the only module importing the Gemini SDK. |
| Tool protocol | `app/mcp_server/`, `app/mcp_client/` | The MCP server (13 read-only tools) and the `ToolProvider` abstraction (`mcp` \| `direct`). |
| RAG | `app/rag/` | Chunking, embeddings, Chroma store, ingestion, cited retrieval. |
| ML | `app/ml/` | Feature engineering, forecast backends, training, evaluation. |
| Business logic | `app/services/` | Inventory, sales, finance, supplier, analytics, forecasting — no LLM, no ORM. |
| Data access | `app/db/repositories/` | The *only* code permitted to import `app/db/models.py`. Parameterised queries only. |
| Data model | `app/db/` | SQLAlchemy models, engine/session factory, `create_all()` (no migrations). |
| Core | `app/core/` | Exception hierarchy, the agent decision log. |

## 2. Data flow — `POST /query`

```
User (Streamlit "Ask BizAgent")
   │ httpx POST /query {question}
   ▼
FastAPI /query route ──────────────────────────────────────────────┐
   │ start_run() → agent_runs row (status=pending)                 │
   ▼                                                                │
LangGraph.invoke(question)                                         │
   │                                                                │
   ├─▶ supervisor_node                                              │
   │      Supervisor.route(question)                                │
   │        LLM generate_structured(RoutingPlan) ──▶ Gemini          │
   │        on failure → keyword_route() (deterministic fallback)   │
   │      returns {route, agents[], reasoning, needs_policy_check}  │
   │                                                                │
   ├─▶ conditional fan-out to the selected specialists, IN PARALLEL │
   │      SpecialistAgent.run(state):                               │
   │        1. plan: LLM → ToolPlan {tool_calls[]}                  │
   │        2. execute: ToolProvider.call_tool(name, args)          │
   │             ToolProvider = MCPToolProvider (stdio subprocess)  │
   │                          or DirectToolProvider (in-process)    │
   │             tool → service → repository → SQLite (read-only)  │
   │           (TTL cache: identical (tool, args) inside one run    │
   │            served from cache, not re-called)                  │
   │        3. answer: LLM → AgentAnswer {answer, confidence}       │
   │      PolicyAgent.run(state) (if routed):                       │
   │        Retriever.retrieve(question) → ChromaDB                │
   │        LLM answers strictly from the cited chunks              │
   │                                                                │
   ├─▶ synthesis_node                                                │
   │      merges every specialist's AgentOutput                     │
   │      LLM → {final_answer, confidence}                          │
   │      (Gemini unavailable → concatenates the specialist answers) │
   │                                                                │
   ▼                                                                │
FastAPI /query route                                                │
   │ finish_run() → agent_runs row updated: route, agents_invoked,  │
   │   full tool_calls[] (args, row_count, duration_ms, errors),    │
   │   retrieved_docs[] (citations), final_answer, confidence,      │
   │   latency_ms, status (success|partial)                         │
   ▼                                                                │
QueryResponse {answer, route, agents_invoked, tool_calls,           │
   citations, confidence, plan_reasoning, latency_ms} ◀─────────────┘
   │
   ▼
Streamlit "Ask BizAgent" — answer + expandable agent trace + citations
```

## 3. Data flow — `POST /rca`

```
User (Streamlit "Root Cause Analysis", or a "run demo scenario" shortcut)
   │ httpx POST /rca {sku | aisle, anomaly_type, [date range]}
   ▼
FastAPI /rca route
   │ start_run() → agent_runs row (route will be "rca")
   ▼
RcaWorkflow.run(request)  — a DEDICATED LangGraph subgraph, not one prompt
   │
   ├─▶ 6 evidence nodes run IN PARALLEL, each deterministic (no LLM):
   │     sales_evidence      → get_sales_history, detect_anomalies(demand_shift), get_top_sellers
   │     inventory_evidence  → get_stock_level(+batches), detect_anomalies(stockout)
   │     shrinkage_evidence  → get_shrinkage_report, detect_anomalies(shrinkage)
   │     supply_evidence     → get_product_info, get_open_purchase_orders (open + late),
   │                           get_supplier_status
   │     forecast_evidence   → get_demand_forecast (stockout risk, reorder delta, expiry)
   │     policy_evidence     → Retriever.retrieve(anomaly-specific query) → ChromaDB citations
   │   each node: on a tool/retriever failure, records a data_gap and continues —
   │   one failing step never aborts the run
   │
   ▼
rank_node
   │ 1. _rule_rank(evidence_pool) → deterministic candidate causes, each with
   │      concrete evidence_ids (supply_delay / shrinkage_theft / expiry_writeoff /
   │      reorder_point / demand_shift / ...) and a confidence score
   │ 2. ONE generate_structured call → Gemini ranks/refines the candidates,
   │      writes the narrative summary and recommended actions
   │      (2 calls only if a parse repair fires — capped, reported)
   │ 3. _validate_causes(): drop any LLM cause whose evidence_ids do not
   │      resolve in the evidence pool — code enforces this, not the prompt
   │ 4. _merge_causes(): a deterministic cause with confidence ≥ 0.8 is never
   │      demoted below a weaker LLM-ranked cause
   │ 5. Gemini unavailable/rate-limited → deterministic causes are used as-is,
   │      status="partial", summary prefixed "[Gemini unavailable ...]"
   ▼
RootCauseReport {summary, ranked_causes[], timeline[], policy_findings[],
   recommended_actions[], data_gaps[], evidence[], tool_trace[], status,
   gemini_calls}
   │
FastAPI /rca route
   │ finish_run() → agent_runs row: route="rca", tool_calls=evidence trace,
   │   retrieved_docs=policy citations, final_answer=summary,
   │   confidence=top cause's confidence, status
   ▼
Streamlit "Root Cause Analysis" — summary, causes with confidence bars and
   evidence expanders, timeline table, policy findings with citations,
   recommended actions
```

## 4. Design decisions (from `BIZAGENT_BUILD_PLAN.md` §2) and their rationale

The project proposal is silent or offers a choice on the points below. Each decision is reversible;
the "how to reverse it" column shows the seam.

| ID | Ambiguity | Decision made | Rationale | Reverse via |
|---|---|---|---|---|
| A-01 | No LLM provider named | Gemini for all generation | Project constraint | `app/llm/provider.py` is a `Protocol`; add another `LLMProvider` implementation |
| A-02 | "POS"/"stockroom" named but don't exist | Seeded SQLite tables shaped like a real POS/WMS | No integration to build against; shapes are realistic enough to demonstrate the architecture | Replace the repository layer only — services and MCP tools are unchanged |
| A-03 | "Supplier APIs" | `supplier_service` shaped like an API client | Same reasoning as A-02, isolated behind one seam | Swap the service body for `httpx` calls |
| A-04 | ChromaDB or FAISS | ChromaDB (persistent, metadata filtering, simpler on Windows) | Lower setup friction for a Windows-first dev environment | `app/rag/vector_store.py` is a thin interface; add a FAISS backend (O7, not built) |
| A-05 | Prophet or XGBoost | XGBoost — see D-03 | — | Add a Prophet backend behind `ForecastBackend` |
| A-06 | No data volume given | 540 days, 60 SKUs, 8 aisles | Enough history for weekly seasonality + lag features; small enough to seed in seconds | `scripts/seed_db.py` constants |
| A-07 | 3 roles named, no auth stated | No authentication | Out of scope for a single-store capstone demo | Add a FastAPI auth dependency |
| A-08 | "Real-time" claimed | Batch/seeded; UI refreshes on demand | No live POS/WMS feed exists to integrate with (A-02/A-03) | Add a polling or webhook ingestion job |
| A-09 | No deployment target named | Local run + Dockerfile + Streamlit Community Cloud notes | Covers the two realistic grading paths | — |
| A-10 | "Agent decision logs" undefined | `agent_runs` rows: route, agents invoked, every tool call (args, timing, errors), citations, final answer | Matches the dashboard requirement precisely and gives the RCA/query UI a real audit trail | — |

**Decisions worth defending in the viva** (see also `docs/viva_questions.md`):

- **D-01 — Real MCP, not a fake wrapper.** The MCP server is an actual `mcp` SDK server over stdio,
  launched as a subprocess (`scripts/run_mcp_server.py`). Agents reach it through `MCPToolProvider`,
  which implements the `ToolProvider` protocol; `DirectToolProvider` implements the identical
  interface in-process. Selected by `TOOL_PROVIDER=mcp|direct`. MCP is the default and the one
  demonstrated; `direct` exists so a stdio failure on demo day cannot take the whole project down,
  and so the default test suite runs fast without spawning a subprocess. Parity between the two is
  proven by `tests/test_tool_provider_parity.py` (`@pytest.mark.live`).
- **D-02 — Gemini embeddings with a local fallback.** One API key for the whole project.
  `EMBEDDING_BACKEND=local` switches to `sentence-transformers/all-MiniLM-L6-v2` if the embeddings
  quota is exhausted or the network is down. The Chroma collection name encodes backend + dimension
  (e.g. `bizagent_kb_gemini_3072` vs `bizagent_kb_local_384`) so a backend switch opens a fresh
  collection and can never silently corrupt an existing one; a genuine dimension mismatch on an
  existing collection raises a clear "re-ingest" error instead of failing at query time.
- **D-03 — XGBoost before Prophet.** Prophet's install (cmdstan/pystan toolchain) is the single most
  common way a short Python project loses a day, especially on Windows. XGBoost with lag/rolling/
  calendar features installs cleanly and trains in seconds at this data volume. A seasonal-naive
  baseline ships first and is the automatic fallback if no model is trained, so forecasting works
  even before `scripts/train_forecast.py` is ever run.
- **D-04 — SQLite over Postgres.** Zero setup for an evaluator cloning the repo. The SQLAlchemy ORM
  means the change to Postgres is a `DATABASE_URL` connection-string edit — no model or repository
  code changes.
- **D-05 (Day 3/4 addition) — Deterministic RCA ranking, LLM for refinement + narrative only.** The
  build plan asks Gemini to "rank candidate causes" but also requires every cause to carry resolvable
  evidence and caps Gemini calls per RCA run at ≤4. Rather than trust the prompt to rank correctly
  under time pressure, a rule-based ranker derives causes with concrete evidence first; Gemini may
  re-rank, add contributing factors and write the narrative, but code (`_merge_causes`) refuses to
  demote a rule-based cause with confidence ≥0.8 below a weaker LLM-proposed one, and
  `_validate_causes` drops any cause whose evidence doesn't resolve. This trades a small amount of
  ranking flexibility for a guarantee that the four planted anomalies (and any similarly clear-cut
  real case) are diagnosed correctly even if Gemini is down, rate-limited, or simply wrong that day —
  the exact failure mode a live demo cannot risk.
- **D-06 (Day 4 addition) — Velocity pre-screen before a full forecast scan.** `/forecast/alerts` and
  `/forecast/reorder-recommendations` would otherwise run a 30-day recursive XGBoost forecast for
  every SKU on every request. Both first rank all SKUs by a cheap velocity-based days-of-cover
  estimate and only run the full model for the worst `2×limit` candidates, bounding the request to a
  few seconds regardless of catalogue size.

## 5. Performance

- **Singletons, not per-request construction:** the compiled LangGraph query graph (`app/api/routes/query.py::get_graph`),
  the RCA workflow (`app/api/routes/rca.py::get_rca_workflow`), the trained forecast model bundle
  (`app/ml/backends.py::load_bundle`, cached by path + mtime) and the RAG retriever/embedding client
  (`app/rag/retriever.py::build_retriever`, fixed in this review — see `docs/security_review.md` item 8's
  sibling performance note below) are all built once per process and reused.
- **Caching within a run:** `app/agents/state.py::_TTLCache` deduplicates identical `(tool, arguments)`
  calls inside one graph run (60 s TTL); embedding queries are LRU-cached
  (`app/rag/embeddings.py::_cached_query_embedding`); Streamlit GET calls use `st.cache_data(ttl=30)`
  (`ui/components/common.py::api_get`).
- **Performance fix applied in this review:** `app/rag/retriever.py::build_retriever()` previously
  constructed a brand-new embedding backend (for `EMBEDDING_BACKEND=local`, a fresh
  `sentence-transformers` model load) and a fresh Chroma collection handle on **every**
  `POST /documents/search` call — the one caller that had no singleton around it (`/query` and `/rca`
  each cache their own graph/workflow, which internally cached the retriever they were built with, but
  `/documents/search` called the factory function directly). Fixed by caching the single process-wide
  `Retriever` inside `build_retriever()` itself, so all three callers now share one instance. Verified
  by the full test suite (no behavioural change — `tests/test_error_paths.py`'s Chroma-deleted test
  now explicitly resets the singleton to prove the "no knowledge base" path still degrades correctly).
- **Indexes:** `sales_transactions(sku, ts)`, `stock_levels(sku, snapshot_date)`,
  `purchase_orders(sku, status)`, `shrinkage_events(sku, event_date)`, `products.aisle`,
  `stock_batches.sku`, `stock_batches.expiry_date` were already present from Day 1–2 and cover every
  hot query path used by the services, the anomaly detectors and the forecasting feature builder.
  Profiling the forecasting alert scan and the RCA evidence queries against the current 60-SKU /
  540-day seeded dataset found no query missing an index; no new index was added. `agent_runs.ts`
  (used to order `/logs`) and `documents.source_path` are unindexed but hold at most a few hundred and
  a handful of rows respectively at this data scale, so an index there would not be measurable.
- **Measured latency** (mocked LLM, 15 runs each, in-process tools, no network):

  | Scenario | p50 | p95 |
  |---|---|---|
  | Simple query (1 specialist) — pipeline only | 8 ms | 17 ms |
  | Multi-agent query (3 specialists in parallel) — pipeline only | 13 ms | 14 ms |
  | RCA run (deterministic ranking, ~15 tool calls incl. a forecast) | 1.85 s | 1.97 s |

  These isolate the graph/tool/DB overhead from Gemini's own latency, which this environment cannot
  measure live (no network egress). With a valid key, each `generate`/`generate_structured` call
  observed in earlier live smoke testing (Days 3–4) takes roughly 1–3 s, so a realistic end-to-end
  estimate is: simple query ≈ 4 calls → ~4–8 s; a 3-agent query ≈ 8 calls → ~8–16 s; an RCA run ≈
  pipeline (1.85 s) + 1 ranking call → ~3–5 s. The RCA workflow's parallel evidence fan-out already
  keeps the non-LLM portion flat regardless of how many evidence steps run, which is why it does not
  scale with agent count the way the query graph's per-specialist LLM calls do.
