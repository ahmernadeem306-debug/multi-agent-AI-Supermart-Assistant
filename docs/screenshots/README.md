# Screenshots to capture

This directory is committed empty except for this file. Before submission, capture the following
PNGs at a normal desktop resolution (≈1600×900 or your screen's native size) with the sidebar visible,
and save them under these exact names so `README.md` can link them:

1. `01_dashboard.png` — **Dashboard** page: KPI tiles, aisle-metrics table, the three bar charts, and
   the "Active risk alerts" table all visible (scroll to fit if needed, or take two stacked shots and
   keep the more informative one).
2. `02_ask_bizagent.png` — **Ask BizAgent** page after asking an operational question (e.g. "Which
   SKUs are low on stock in the Produce aisle?"), with the **Agent trace** expander open showing the
   route and at least one tool call.
3. `03_root_cause.png` — **Root Cause Analysis** page after running the late-delivery scenario
   (SKU-1035): summary, the primary ranked cause with its confidence bar and an open evidence
   expander, and the policy findings section.
4. `04_forecasting.png` — **Forecasting** page for SKU-1035 (or another): the history + forecast
   chart, the stockout-risk / days-of-cover / reorder-point metrics row, and the backtest metrics
   table.
5. `05_knowledge_base.png` — **Knowledge Base** page: the ingested-documents table and the retrieval
   test box with a result showing a citation.
6. `06_agent_logs.png` — **Agent Logs** page: the run-history table with the route/status filters
   applied, and a selected run's detail (tool trace + retrieved documents) visible below it.

Optional but useful for the viva:

7. `07_api_docs.png` — the FastAPI auto-generated docs at `http://localhost:8000/docs`, showing the
   full route list grouped by tag.

After capturing, add each image to `README.md` under "Screenshots" as:

```markdown
### Dashboard
![Dashboard](docs/screenshots/01_dashboard.png)
```
