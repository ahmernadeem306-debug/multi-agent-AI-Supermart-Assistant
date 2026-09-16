# Demo Script (8–10 minutes)

Run through this twice before the real demo. Each beat lists the exact click path / text to type, the
expected output, and a fallback if a live Gemini call fails or is slow.

## Pre-demo checklist

Run in order, in a terminal at the repo root, **before** the audience joins:

- [ ] `.env` exists with a **valid** `GEMINI_API_KEY` (`python scripts/smoke_gemini.py` prints a real
      response — if it errors, fix this first, everything else still works in degraded mode but the
      demo is much better with a live key).
- [ ] Network check: the machine can reach `generativelanguage.googleapis.com` (same smoke script
      covers this).
- [ ] `python scripts/seed_db.py --reset` — prints row counts and writes `data/anomalies.json`.
- [ ] `python scripts/ingest_docs.py` — prints 7 "ingested" or "unchanged" rows and a chunk count.
- [ ] `models/forecast_xgb.joblib` exists (`python scripts/train_forecast.py` if not — prints the
      backtest table and a verdict).
- [ ] `python scripts/run_demo_scenarios.py` — must print `4/4 scenarios passed.` If it doesn't, fix
      it before the demo; do not demo RCA on a broken build.
- [ ] Three terminals running, **in this order**: `python scripts/run_mcp_server.py`,
      `uvicorn app.api.main:app --reload`, `streamlit run ui/streamlit_app.py`.
- [ ] Open `http://localhost:8501` and confirm the sidebar shows **API: connected**, a tool provider
      and a forecast backend.
- [ ] Close any other window/tab that might steal focus or notification sound during screen share.

---

## 0:00 — Problem and solution framing

**Say:** "Supermart stores drown in three problems the proposal names: unpredicted stockouts and
shrinkage, manual SOP/compliance overhead, and siloed POS/stockroom/supplier data with no automated
root-cause diagnostics. BizAgent is a multi-agent operations assistant that answers operational
questions, forecasts demand, and — its centrepiece — diagnoses *why* an anomaly happened, with
evidence, not guesswork."

**Show:** nothing yet — this is framing only. Have the Dashboard page already loaded behind you.

**Fallback:** none needed, this is narration.

## 1:00 — Architecture in 60 seconds

**Say:** "Streamlit talks only to a FastAPI layer. A LangGraph Supervisor routes each question to up
to five specialist agents — Sales, Inventory, Finance, Forecasting, Policy. Agents never touch the
database directly: they call a real MCP tool server, which calls services, which call repositories.
Policy questions go through a RAG pipeline over seven real store documents in ChromaDB. Forecasting is
a trained XGBoost model with a safe baseline fallback. Root-cause analysis is a dedicated LangGraph
subgraph that gathers evidence deterministically and only uses Gemini to rank causes and write the
narrative — every cause it reports must carry real, resolvable evidence."

**Show:** `docs/architecture.md` §1 diagram (have it open in a second window, or draw it live).

**Fallback:** none needed.

## 2:00 — Dashboard: aisle metrics, alerts

**Click path:** sidebar → **Dashboard**.

**Show:** the four KPI tiles, the aisle-metrics table, the revenue/units/shrinkage bar charts, the
"Active risk alerts" table, low-stock and expiring-batches tables.

**Say:** "This is served entirely by `/metrics` and `/forecast/alerts` — real seeded POS and stockroom
data, aggregated by aisle, with a store-wide risk ranking underneath it."

**Fallback:** if a chart is slow to render, the tables below it still load — narrate over it. If the
API is down, the page shows a clear "Could not reach the BizAgent API" banner, never a crash — point
that out as a deliberate design choice.

## 3:00 — Policy question → RAG answer with citations

**Click path:** sidebar → **Ask BizAgent**.

**Type:** `How long do customers have to return frozen goods?`

**Expected output:** an answer stating the frozen-goods return window (7 days, kept frozen), with a
**Citations** section showing `Returns and Refunds Policy`, the matched section and a snippet.

**Say:** "The Policy agent answers strictly from retrieved chunks — never invents policy — and always
cites the document and section."

**Fallback:** if retrieval returns nothing (e.g. Chroma wasn't ingested), the page shows "no supporting
policy document was found" instead of a fabricated answer — say this is the intended degraded
behaviour, then run `python scripts/ingest_docs.py` in a terminal and retry.

## 4:00 — Operational question → agent trace, supervisor routing, MCP tool calls

**Type:** `Which SKUs are low on stock in the Produce aisle?`

**Expected output:** an answer naming specific SKUs, with the **Agent trace** expander open showing
route = `inventory`, the tool call (`list_low_stock`, arguments, row count, timing).

**Say:** "That table is a real MCP tool call over stdio — you can see the exact arguments the LLM
synthesised and how long the call took. This is the 'agent decision log' the proposal asks for, live."

**Fallback:** if the MCP server was killed or is slow, `TOOL_PROVIDER_FALLBACK=true` means the app
falls back to the in-process `direct` provider automatically — the answer still arrives, just note
that the tool-provider indicator in the sidebar would show `direct`.

## 5:30 — Forecasting: stockout risk and reorder recommendation

**Click path:** sidebar → **Forecasting** → select SKU `SKU-1035`.

**Show:** the history + forecast chart (shaded prediction band), the stockout-risk badge, days of
cover, the reorder-point recommendation (current → recommended, with the delta), and the backtest
metrics table at the bottom.

**Say:** "The backend is XGBoost, trained on 540 days of history with lag and rolling features; it
beat the seasonal-naive baseline by about 5% MAE in the last backtest — shown right here, honestly,
including the comparison. If no model were trained, this page would say so and use the baseline
automatically."

**Fallback:** if the chart is empty, the SKU selector still lists all 60 SKUs — pick another (e.g.
`SKU-1001`). If `models/forecast_xgb.joblib` is missing, the backend badge will read
`seasonal_naive` — say that's the designed fallback, not a bug.

## 7:00 — ROOT CAUSE ANALYSIS on the late-delivery anomaly (the centrepiece)

**Click path:** sidebar → "Run demo scenario" → pick **Late-delivery stockout (SKU-1035)** → **Load
into Root Cause →** (or navigate to **Root Cause Analysis** and select SKU `SKU-1035` directly) → **Run
root-cause analysis**.

**Expected output (≈2–5 s):**
- Summary: a supplier PO arrived ~9 days late, depleting stock before replenishment.
- Ranked causes: **supply_delay** primary, confidence ~0.9, with an **Evidence** expander listing the
  late PO, the recorded stockout, and the supplier's reliability record.
- Timeline: PO promised/received dates and the first/last zero-on-hand days.
- Policy findings: the FreshFarm/DailyGoods contract clause on late-delivery penalties, with a real
  citation.
- Recommended actions: escalate to the supplier under the contract, expedite the outstanding order.

**Say:** "Every cause you see is backed by a concrete tool result or document citation — the code
drops any cause the model can't back up, it doesn't just ask nicely in the prompt. If I killed Gemini
right now this would still produce the same primary cause, just without the narrative — watch the
status badge, it says 'partial' when that happens, never a fabricated answer."

**Fallback:** if the live call is slow or fails, say so and immediately show
`python scripts/run_demo_scenarios.py`'s pre-captured output (keep a terminal tab open with it already
run) — narrate that it demonstrates all four planted anomalies, not just this one.

## 9:00 — Agent decision log, then limitations and future work

**Click path:** sidebar → **Agent Logs**. Filter by route = `rca`, select the run just created.

**Show:** the full tool trace (every evidence-gathering call, arguments, timing), the retrieved policy
documents, the final summary, latency and status — the same run just demonstrated, now as a permanent,
searchable audit record.

**Say (limitations, honestly):** "POS, stockroom and supplier systems are simulated with a
deterministic seeded dataset, not live integrations — that's a stated assumption, not a hidden gap.
There's no authentication; this is a single-store, single-tenant demo. Data is batch-seeded, not
real-time. Everything else — RAG, the MCP tool server, the multi-agent graph, forecasting, and root-
cause analysis — is real and running live."

**Say (future work):** "With more time: a FAISS backend alternative, Prophet as a second forecasting
model, multi-store support, a real POS/webhook ingestion path, and role-based auth."

---

## Troubleshooting — five most likely failures

| Symptom | Likely cause | Fix during the demo |
|---|---|---|
| Sidebar shows "API: unreachable" | `uvicorn` not running or crashed | Check its terminal for a traceback; restart with `uvicorn app.api.main:app --reload` |
| `/query` or `/rca` hangs / times out | MCP subprocess died or Gemini is slow/rate-limited | Wait ~5 s (retry/backoff is automatic); if it returns `status: partial`, say that's the designed degraded path, not a failure |
| Ask BizAgent gives a generic/wrong answer | Supervisor mis-routed the question | Rephrase using a keyword from the intended domain (e.g. "stock", "margin", "policy", "forecast", "why did") — the keyword fallback router is exactly this vocabulary |
| Forecasting page shows `seasonal_naive` unexpectedly | `models/forecast_xgb.joblib` missing or stale | Run `python scripts/train_forecast.py` in a spare terminal; refresh the page after it finishes |
| Root Cause page shows `no_anomaly` for a planted SKU | Wrong SKU selected, or the seeded data was regenerated with a different date window since the last seed | Re-select from `data/anomalies.json`; if genuinely stale, re-run `seed_db.py --reset` then `train_forecast.py` before the demo, not during it |
