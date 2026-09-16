# Viva Questions

Fifteen likely examiner questions with concise, defensible answers. Cross-references point to where
the full reasoning lives.

**1. Why MCP instead of just calling your services directly from the agents?**
The proposal requires agents to "safely execute structured queries ... without exposing raw database
access." MCP gives us a protocol boundary with typed, validated, capped tools instead of an open
database connection or ad-hoc function calls. It's also transport-agnostic — the same tool functions
run over a real stdio subprocess (`MCPToolProvider`) or in-process (`DirectToolProvider`), proven
identical by a parity test, so a stdio failure on demo day degrades instead of breaking the project.
See `docs/architecture.md` D-01.

**2. Why LangGraph instead of a single long prompt or a simple if/else router?**
Two reasons: (a) genuine multi-agent fan-out — a question like "why did the dairy margin drop while
stock held?" needs Finance *and* Inventory to run and be merged, which a single prompt can't do
cleanly; (b) the RCA workflow needs a fixed evidence-gathering sequence with several independent steps
run in parallel and one ranking step at the end — that's a graph, not a chain. LangGraph gives us
explicit state, reducers for merging parallel branch output, and a recursion limit for safety.

**3. Why Gemini and not OpenAI/Claude/a local model?**
Project constraint (the proposal names no provider, only permits "OpenAI / Hugging Face" for
embeddings). Gemini is isolated entirely behind `app/llm/provider.py::LLMProvider`, a `Protocol` — no
other module imports the Gemini SDK. Swapping providers is one new class and a config change, not a
rewrite (see `docs/architecture.md` A-01).

**4. Why XGBoost over Prophet, given the proposal names both?**
Prophet's install chain (cmdstan/pystan) is the single most common way a short project loses a day,
especially on Windows. XGBoost with lag/rolling/calendar features installs cleanly and trains in
seconds at our data volume, and the proposal explicitly permits either. A seasonal-naive baseline
ships first and is the automatic fallback, so the feature exists even before a model is trained. See
`docs/architecture.md` D-03 and the backtest table (XGBoost beat baseline ~5% MAE).

**5. How do you prevent the LLM from hallucinating numbers, SKUs or policy text?**
Three layers: (a) every agent prompt states an explicit anti-hallucination rule — use only the tool
results/reference blocks given, never invent, say so if nothing relevant was returned; (b) structured
output is validated against a Pydantic schema with one repair retry, so malformed output can't leak
through as prose; (c) for RCA specifically, code — not the prompt — drops any cause whose evidence
doesn't resolve to a real tool result or citation (`_validate_causes`). Prompt instructions are a
request; code validation is a guarantee.

**6. How do you prevent SQL injection / LLM-generated SQL?**
The LLM never sees a SQL string and never constructs one. Agents call named MCP tools with
Pydantic-validated arguments; tools call services; services call repositories, which are the only code
that imports the ORM and always use SQLAlchemy's parameterised `select()` API — no f-string or
`%`-formatted SQL anywhere in the codebase (verified by grep in `docs/security_review.md` item 3).

**7. What's real versus simulated in this project?**
Real: the MCP server (an actual `mcp` SDK stdio server), the RAG pipeline (real ChromaDB + real
embeddings), the LangGraph orchestration and RCA subgraph, the trained XGBoost model, the FastAPI/
Streamlit stack. Simulated: POS transactions, stockroom snapshots and supplier/PO data — there is no
real point-of-sale or warehouse system to integrate with, so `scripts/seed_db.py` generates a
deterministic, realistic dataset (the proposal names "POS systems" and "stockroom databases" without
providing one). This is documented, not hidden — see A-02/A-03 in `docs/architecture.md` and Known
Limitations in the README.

**8. How does the read-only guarantee on the MCP server actually work — is it enforced or just a
convention?**
Enforced. `app/db/base.py::get_readonly_session()` installs a SQLAlchemy `before_flush` event listener
that raises `ToolExecutionError` if the session has any pending new/dirty/deleted objects — the only
way an ORM write can happen. Every MCP tool, every `/metrics` and `/forecast` route, and the RCA
workflow use this session. A test attempts a write through it and asserts it raises
(`tests/test_api_metrics.py::test_readonly_session_blocks_writes`).

**9. What happens if Gemini is down or rate-limited during a demo?**
Every LLM call goes through `tenacity` retry with exponential backoff for transient errors, and a 429
is mapped to `RateLimitError`. The Supervisor falls back to a deterministic keyword router. Specialist
agent failures are isolated — one agent's failure never fails the whole request, and synthesis
concatenates whatever answers *did* come back with an honest "degraded" status. RCA falls back to its
deterministic ranker entirely, still producing the correct primary cause for all four planted
anomalies with zero Gemini calls. This is tested, not just claimed — see `tests/test_error_paths.py`.

**10. Why no authentication?**
The proposal names three user roles but states no auth requirement, and this is a single-store,
single-tenant capstone demo (Assumption A-07 in `docs/architecture.md`). Adding role-based auth is a
FastAPI dependency away and explicitly listed as future work — it wasn't a blocker for demonstrating
the required functionality.

**11. How do you keep the vector store from getting corrupted if you switch embedding backends?**
The Chroma collection name encodes both the backend and its embedding dimension (e.g.
`bizagent_kb_gemini_3072` vs `bizagent_kb_local_384`), so switching `EMBEDDING_BACKEND` opens a
different, empty collection rather than mixing incompatible vectors in one. If a collection's recorded
dimension ever disagrees with the active backend (a corrupted or mismatched persist directory),
`ChromaVectorStore.__init__` raises a clear "delete `CHROMA_PERSIST_DIR` and re-ingest" error instead
of returning garbage similarity scores.

**12. How is root-cause analysis different from just asking the LLM "why did this happen?"**
It's a fixed, auditable pipeline, not a single prompt. Six evidence-gathering steps run against real
tool results (sales velocity, stock trajectory, shrinkage events, purchase-order timeliness, the
forecast model's own expectation, and cited policy text) — all deterministic, all logged. Only the
final ranking step touches the LLM, and even there a rule-based ranker derives evidence-backed
candidates first; Gemini can refine and narrate but cannot demote a strong deterministic signal, and
any cause it proposes without resolvable evidence is dropped by code. This is why the same four
planted anomalies are diagnosed correctly whether or not Gemini is available that day.

**13. How did you validate the forecasting model isn't overfitting or leaking future data?**
A time-based train/validation split — the last N days are held out, never shuffled, never randomly
sampled. Every lag and rolling-window feature is shifted by at least one day before being used, and
`tests/test_features.py::test_no_target_leakage` explicitly asserts that changing the target value on
the final day of a series does not change any feature value computed for that or any earlier row.
Accuracy is then reported honestly against a seasonal-naive baseline (MAE/RMSE/MAPE/WAPE), and the
training script would report it even if XGBoost lost to the baseline — it happened to win by ~5% MAE.

**14. What's your test strategy, and how do you know the suite isn't just testing itself?**
Unit tests on the deterministic layers (config, models, repositories, services, analytics, feature
engineering, structured parsing); integration tests on the MCP tools, provider parity, the full graph
end-to-end through the live FastAPI app, and the RCA workflow with mocked LLM + real (in-process)
tools against the actual seeded data; a routing-accuracy measurement against a 28-case labelled
fixture; a retrieval-quality measurement against 5 golden questions; and an explicit fault-injection
matrix (`tests/test_error_paths.py`) covering Gemini down/429, MCP killed, Chroma deleted, no trained
model, a simulated database error, zero-sales SKUs, malformed input, and unsafe uploads. The whole
suite (`pytest -q`) runs offline, with no API key and no network — only tests marked `@pytest.mark.live`
touch the real Gemini API or spawn the real MCP subprocess. Coverage on `app/` is 91%.

**15. What would you do with one more week?**
Prophet as a second forecasting backend behind the existing `ForecastBackend` interface (an honest
comparison, not a replacement); a FAISS vector-store backend behind the same `ChromaVectorStore`
interface; multi-store support (the schema is single-store today); a lightweight role selector wired
to the dashboard views (Assumption A-07/O2); async rewrites of the FastAPI routes for higher
concurrency under load (deliberately out of scope for Day 5 per the build plan — "no async rewrites or
architectural changes"); and a real webhook/polling ingestion path to replace the batch-seeded data,
which is the single biggest gap between this demo and a production deployment.
