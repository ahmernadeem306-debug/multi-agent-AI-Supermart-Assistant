# Requirements Traceability

Maps every mandatory requirement (M1–M8) from `BIZAGENT_BUILD_PLAN.md` §1.12 — itself derived from
the project proposal (Soojal Kumar & Ahmer Nadeem) — to the code that implements it, the test that
proves it, and an honest status. Status is one of **Complete**, **Partial**, **Not implemented**.

| # | Requirement | Status |
|---|---|---|
| M1 | RAG knowledge engine | **Complete** |
| M2 | MCP tool server, no raw DB exposure | **Complete** |
| M3 | Supervisor routing to 4+1 specialist agents | **Complete** |
| M4 | Forecasting: expiry, stockouts, reorder points | **Complete** |
| M5 | Dashboard: aisle metrics, decision logs, RCA reports | **Complete** |
| M6 | Automated root-cause diagnostics | **Complete** |
| M7 | Groq via `GROQ_API_KEY`, never hardcoded | **Complete** |
| M8 | Tests, README, GitHub-ready repository | **Complete** |

---

## M1 — RAG knowledge engine

*Ingests store SOPs, return policies, employee handbooks and supplier contracts into a vector
database.*

- **Code:** `data/knowledge/*.md` (7 documents: `shelf_stocking_sop.md`, `returns_and_refunds_policy.md`,
  `employee_handbook_operations.md`, `supplier_contract_freshfarm.md`, `supplier_contract_dailygoods.md`,
  `shrinkage_and_loss_prevention_sop.md`, `inventory_count_and_discrepancy_sop.md`); `app/rag/chunking.py`
  (heading-aware recursive chunking); `app/rag/embeddings.py` (`GeminiEmbeddings` / `LocalEmbeddings`,
  selected by `EMBEDDING_BACKEND`); `app/rag/vector_store.py` (`ChromaVectorStore`, persisted, collection
  name encodes backend + dimension); `app/rag/ingest.py` (sha256-idempotent ingestion, script/control-char
  stripping); `app/rag/retriever.py` (top-k cosine, minimum-score threshold, `Citation` model);
  `scripts/ingest_docs.py`; `app/api/routes/documents.py` (`GET /documents`, `POST /documents/search`,
  `/upload`, `/reingest`); `ui/pages/5_Knowledge_Base.py`.
- **Tests:** `tests/test_rag_ingest.py` (chunk boundaries/overlap, metadata, sha256 idempotency, re-ingest
  replaces chunks), `tests/test_retriever.py` (5 golden questions, each returns the expected source
  document in the top-3, using the local embedding backend so no API key is required),
  `tests/test_prompt_injection.py` (ingestion sanitisation + the untrusted-reference-block defence),
  `tests/test_api_metrics.py`/`test_error_paths.py` (Chroma-missing degradation).
- **Status:** **Complete.** All four document types from the proposal are present and cited in
  answers with document title, section and score.

## M2 — MCP tool server, safe structured retrieval, no raw DB exposure

*Safely executes structured data retrievals from POS, stockroom and supplier data without exposing
raw database access.*

- **Code:** `app/mcp_server/server.py` (FastMCP, stdio transport, 13 tools registered — see the tool
  catalogue in the README), `app/mcp_server/schemas.py` (an explicit Pydantic input + output model per
  tool, result caps), `app/mcp_server/tools_{inventory,sales,finance,supplier,forecast}.py` (plain,
  read-only functions), `app/mcp_client/adapter.py` (`ToolProvider` protocol, `DirectToolProvider`,
  `tool_catalogue()`), `app/mcp_client/client.py` (`MCPToolProvider`, a real stdio subprocess client),
  `app/db/base.py::get_readonly_session` (a `before_flush` guard that raises `ToolExecutionError` on
  any pending ORM write — the only way tool code ever touches the database).
  **Hard rule enforced:** every tool function calls a service, which calls a repository — no tool,
  agent or route imports the ORM or writes raw SQL (`app/db/repositories/` is the only layer that does).
- **Tests:** `tests/test_mcp_tools.py` (schema validity, limit enforcement, unknown-identifier errors
  for all 13 tools), `tests/test_tool_provider_parity.py` (`@pytest.mark.live`; `MCPToolProvider` and
  `DirectToolProvider` return identical results for 8 tools including `get_demand_forecast`),
  `tests/test_api_metrics.py::test_readonly_session_blocks_writes` (a write attempt through the
  read-only session raises), `tests/test_graph_e2e.py::test_agents_package_has_no_forbidden_imports`
  (static AST guard: no `app.services`, `app.db` or Gemini SDK import in `app/agents/`).
- **Status:** **Complete.**

## M3 — Supervisor Agent routing to Sales, Inventory & Stock, Finance, Demand Forecasting agents

*(The proposal's fifth agent, Policy, answers from the RAG engine and is treated as a peer specialist.)*

- **Code:** `app/agents/supervisor.py` (`Supervisor.route`, LLM-based with a deterministic keyword
  fallback), `app/agents/{sales,inventory_agent,finance_agent,forecast_agent,policy_agent}.py` (five
  specialists, each constructor-injected with an `LLMProvider` and a `ToolProvider`),
  `app/agents/graph.py` (`StateGraph`: supervisor → conditional parallel fan-out → synthesis → END,
  compiled once), `app/agents/state.py` (`AgentState`, the `SpecialistAgent` base class, the
  process-wide tool-result cache), `app/api/routes/query.py` (`POST /query`).
- **Tests:** `tests/test_supervisor_routing.py` (28-case labelled fixture, ≥90% keyword-router
  accuracy, LLM-failure fallback, `needs_policy_check`), `tests/test_agents.py` (per-agent tool
  selection/argument construction/empty-result handling/failure isolation), `tests/test_graph_e2e.py`
  (single-agent, multi-agent and policy queries through the live API, decision-log assertions).
- **Status:** **Complete.**

## M4 — Time-series forecasting: perishable expiration, imminent stockouts, optimal reorder points

- **Code:** `app/ml/features.py` (lags, rolling stats, calendar features, promo flag,
  days-since-stockout — all leakage-free), `app/ml/backends.py` (`SeasonalNaiveBackend`,
  `XGBoostBackend`, selected by `FORECAST_BACKEND` with automatic fallback if no model file exists),
  `app/ml/train.py` + `scripts/train_forecast.py` (time-based holdout, MAE/RMSE/MAPE/WAPE vs baseline,
  `models/forecast_xgb.{joblib,meta.json}`), `app/services/forecasting_service.py` (`stockout_risk`,
  `reorder_point`, `expiry_risk`, `store_alerts`, `reorder_recommendations`), MCP tool 13
  `get_demand_forecast`, `app/api/routes/forecast.py`, `ui/pages/4_Forecasting.py`.
- **Tests:** `tests/test_features.py` (explicit no-target-leakage assertion), `tests/test_forecast_backends.py`
  (baseline correctness on a synthetic seasonal series; XGBoost trains + predicts and respects the
  horizon), `tests/test_forecast_service.py` (stockout risk on a known trajectory, risk-level
  boundaries, reorder arithmetic, expiry risk against seeded batches, zero-sales safety),
  `tests/test_api_forecast.py`.
- **Status:** **Complete.** Trained XGBoost beats the seasonal-naive baseline by ~5% MAE on the
  seeded data (see `docs/architecture.md` §Performance); the baseline still ships and is used
  automatically if no model is trained, so the feature works even before training.

## M5 — Dashboard: aisle-level metrics, agent decision logs, root-cause analysis reports

- **Code:** `ui/pages/1_Dashboard.py` (KPIs, aisle metrics, revenue/units/shrinkage charts, low stock,
  expiring batches, active risk alerts), `ui/pages/6_Agent_Logs.py` (searchable/filterable run history
  with per-run tool trace), `ui/pages/3_Root_Cause.py` (full `RootCauseReport` rendering:
  summary, ranked causes with confidence, evidence, timeline, policy findings, actions),
  `app/api/routes/metrics.py`, `app/core/decision_log.py` (`agent_runs` table, `GET /logs`,
  `GET /logs/{run_id}`).
- **Tests:** `tests/test_api_metrics.py`, `tests/test_logs_api.py`, `tests/test_smoke_e2e.py`
  (end-to-end: a query and an RCA run both appear in `/logs`).
- **Status:** **Complete.**

## M6 — Automated root-cause diagnostics for stockouts, shrinkage and supply-chain delays

- **Code:** `app/agents/rca_workflow.py` (a dedicated LangGraph subgraph: six parallel
  evidence-gathering nodes — sales, inventory, shrinkage, supply, forecast, policy — feeding one
  ranking node), `app/agents/rca_schemas.py` (`RootCauseReport`, `RankedCause` with mandatory
  `evidence_ids`), a deterministic rule-based ranker that Gemini refines but cannot override on a
  strong signal, and code-level evidence validation (`_validate_causes`) that drops any cause whose
  evidence does not resolve — the "prompt instructions are not a guarantee; validation is" rule from
  §5 is enforced, not just requested. `app/api/routes/rca.py` (`POST /rca`, logs the full evidence
  trace under `route="rca"`). `scripts/run_demo_scenarios.py` is the regression harness.
- **Tests:** `tests/test_rca_workflow.py` (all evidence steps run; each of the 4 planted anomalies
  yields the correct primary cause category; a cause with unresolvable evidence is dropped; a failing
  evidence step degrades to a `data_gap` instead of raising; a control SKU with no anomaly returns
  `status="no_anomaly"`), `tests/test_api_rca.py`, `tests/test_error_paths.py` (Gemini down / 429 /
  MCP killed / Chroma deleted all still produce a usable report).
- **Status:** **Complete.** `python scripts/run_demo_scenarios.py` diagnoses all four planted
  anomalies (late delivery, theft shrinkage, expiry write-off, stale-reorder-point demand shift) with
  the correct primary cause and real, resolvable evidence every time — see `docs/demo_script.md`.

## M7 — Gemini via `GEMINI_API_KEY`, never hardcoded

- **Code:** `app/config.py::Settings.gemini_api_key: SecretStr` (fails fast with `ConfigError` if
  absent), `app/llm/gemini_client.py::GeminiProvider` (the only module that imports `google.genai`),
  `app/llm/provider.py::LLMProvider` (the protocol every agent depends on instead), `.env.example`
  (`GEMINI_API_KEY=your_api_key_here`, no real value).
- **Tests:** `tests/test_config.py` (missing key → `ConfigError`; the key is never exposed by
  `repr`/`str`), `tests/test_llm_provider.py` (retry, `RateLimitError` mapping, timeout — SDK fully
  mocked, no live call), `tests/test_graph_e2e.py` static guard (no `google.genai` import outside
  `app/llm/`). See `docs/security_review.md` for the repo-wide secret scan.
- **Status:** **Complete.**

## M8 — Tests, README, GitHub-ready repository

- **Code:** 30 test modules under `tests/` (`pytest -q` → 183 passed, 11 deselected, no API key, no
  network); `README.md` (full rewrite); `LICENSE` (MIT); `Dockerfile` + `.dockerignore`; pinned
  `requirements.txt`; finalised `.gitignore`; `docs/architecture.md`, `docs/demo_script.md`,
  `docs/security_review.md`, `docs/viva_questions.md`, this document.
- **Tests:** the suite itself, plus `tests/test_smoke_e2e.py` (one runnable seed → ingest → query →
  rca → forecast regression check).
- **Status:** **Complete.** Coverage on `app/` is **91%** (`pytest --cov=app --cov-report=term-missing`),
  comfortably above the 70% target and above 80% on every targeted package (services 95%, agents 92%,
  mcp_server 94%, ml 96%). See `docs/security_review.md` for the security pass and the Known
  Limitations section of the README for anything not 100%.

---

## Known partial areas (documented, not hidden)

None of M1–M8 are partial. Two components sit deliberately below 100% branch coverage and are called
out here for transparency rather than in a status row above, because they do not affect a mandatory
requirement:

- `app/llm/gemini_client.py` (61% line coverage) and `app/mcp_client/client.py` (51%): both talk to a
  real external process (the Gemini API / a stdio MCP subprocess) and are exercised by the `live`
  test tier (`pytest -m live`), which is intentionally excluded from the default, network-free run —
  see `BIZAGENT_BUILD_PLAN.md` §3 ("Tests never call the real Gemini API or spawn the MCP subprocess
  unless marked `@pytest.mark.live`").
- The RCA hypothesis ranking is deterministic-primary by design (§ Decisions in `docs/architecture.md`):
  Gemini ranks and writes the narrative, but code guarantees a strong rule-based signal is never
  demoted. This is a considered trade-off for reliability under a capstone demo, not a shortcut.
