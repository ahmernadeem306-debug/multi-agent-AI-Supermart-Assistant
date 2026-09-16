"""One end-to-end regression check: seed -> ingest -> query -> rca -> forecast.

Runs entirely offline against the FastAPI app via TestClient, with the LLM
mocked and the real seeded database, real MCP tool functions (in-process),
and a real (fake-embedding) ChromaDB retriever. This is the single command a
reviewer can run to confirm the whole system still wires together:

    pytest -q tests/test_smoke_e2e.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.agents.graph import build_graph
from app.agents.rca_schemas import RcaRequest
from app.agents.rca_workflow import RcaWorkflow
from app.api.main import app
from app.api.routes.query import get_graph
from app.api.routes.rca import get_rca_workflow
from app.mcp_client.adapter import DirectToolProvider
from app.rag.chunking import chunk_text
from app.rag.ingest import infer_doc_type
from app.rag.retriever import Retriever
from app.rag.vector_store import ChromaVectorStore
from tests.fakes import RaisingLLM, ScriptedLLM

pytestmark = pytest.mark.usefixtures("seeded_db")

_KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "data" / "knowledge"


class _FakeEmbeddings:
    """Deterministic, offline stand-in for a real embedding backend."""

    name = "fake"
    dimension = 16

    def _vec(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        return [b / 255.0 for b in digest[: self.dimension]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def _ingest_corpus_with_fake_embeddings(tmp_path) -> Retriever:
    backend = _FakeEmbeddings()
    store = ChromaVectorStore(str(tmp_path / "chroma"), "smoke_fake_16", backend.dimension)
    for path in sorted(_KNOWLEDGE_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        chunks = chunk_text(text, chunk_size=800, overlap=120)
        if not chunks:
            continue
        doc_id = hashlib.sha256(str(path).encode()).hexdigest()[:16]
        store.add(
            ids=[f"{doc_id}:{c.chunk_index}" for c in chunks],
            embeddings=backend.embed_documents([c.text for c in chunks]),
            documents=[c.text for c in chunks],
            metadatas=[
                {"doc_id": doc_id, "title": path.stem.replace("_", " ").title(),
                 "doc_type": infer_doc_type(path), "source_path": str(path),
                 "chunk_index": c.chunk_index, "section": c.section or ""}
                for c in chunks
            ],
        )
    return Retriever(backend, store, top_k=5, min_score=0.0)


def _query_responder(prompt: str, system: str | None) -> str:
    low = prompt.lower()
    if "return json with: route" in low:
        return json.dumps({"route": "inventory", "agents": ["inventory"],
                            "reasoning": "smoke test", "needs_policy_check": False})
    if '"tool_calls":' in prompt and "Return an empty list" in prompt:
        return json.dumps({"tool_calls": [{"tool": "get_stock_level", "arguments": {"sku": "SKU-1035"}}],
                            "reasoning": "smoke test"})
    if '"answer":' in prompt:
        return json.dumps({"answer": "SKU-1035 is below its reorder point.", "confidence": 0.8,
                            "used_tool_data": True})
    if '"final_answer":' in prompt:
        return json.dumps({"final_answer": "SKU-1035 is below its reorder point.", "confidence": 0.8})
    return "ok"


@pytest.fixture()
def client(tmp_path):
    retriever = _ingest_corpus_with_fake_embeddings(tmp_path)
    tools = DirectToolProvider()

    graph = build_graph(ScriptedLLM(_query_responder), tools, retriever)
    rca_workflow = RcaWorkflow(RaisingLLM(), tools, retriever)  # deterministic ranking path

    app.dependency_overrides[get_graph] = lambda: graph
    app.dependency_overrides[get_rca_workflow] = lambda: rca_workflow
    yield TestClient(app)
    app.dependency_overrides.pop(get_graph, None)
    app.dependency_overrides.pop(get_rca_workflow, None)


def test_seed_ingest_query_rca_forecast_end_to_end(client):
    # seed -> already done by the seeded_db fixture (4 planted anomalies)
    # ingest -> already done by the client fixture (real corpus, fake embeddings)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["db_connected"] is True

    query = client.post("/query", json={"question": "Is SKU-1035 low on stock?"})
    assert query.status_code == 200
    query_body = query.json()
    assert query_body["status"] == "success"
    assert query_body["agents_invoked"] == ["inventory"]
    assert query_body["tool_calls"], "expected the inventory agent to call a tool"

    rca = client.post("/rca", json={"sku": "SKU-1035", "anomaly_type": "auto"})
    assert rca.status_code == 200
    rca_body = rca.json()
    assert rca_body["ranked_causes"][0]["category"] == "supply_delay"
    assert rca_body["ranked_causes"][0]["evidence_ids"]

    forecast = client.get("/forecast/SKU-1035", params={"horizon_days": 7})
    assert forecast.status_code == 200
    forecast_body = forecast.json()
    assert forecast_body["backend"] in {"xgboost", "seasonal_naive"}
    assert len(forecast_body["forecast"]) == 7
    assert forecast_body["stockout_risk"]["risk_level"] in {"critical", "high", "medium", "low"}

    # every run above is in the decision log
    logs = client.get("/logs", params={"limit": 10}).json()
    routes = {row["route"] for row in logs["runs"]}
    assert {"inventory", "rca"} <= routes
