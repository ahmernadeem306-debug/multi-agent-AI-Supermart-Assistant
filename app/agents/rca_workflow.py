"""Root-Cause Analysis workflow (Workflow W2) — a dedicated LangGraph subgraph.

Six evidence-gathering steps run in parallel and feed a single ranking step.
Evidence is assembled deterministically through the ToolProvider and the RAG
retriever; the LLM is used only to rank the already-evidence-backed candidate
causes and write the narrative. Every cause in the final report is guaranteed
by code to carry resolvable evidence.
"""
from __future__ import annotations

import datetime as dt
import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.core.exceptions import BizAgentError, LLMError, LLMParseError
from app.llm.parsing import generate_structured
from app.llm.prompts import rca as prompts
from app.llm.provider import LLMProvider
from app.logging_config import get_logger
from app.mcp_client.adapter import ToolProvider
from app.agents.rca_schemas import (
    EvidenceItem,
    LlmRanking,
    PolicyFinding,
    RankedCause,
    RcaRequest,
    RecommendedAction,
    RootCauseReport,
    TimelineEvent,
)

logger = get_logger(__name__)

_LOOKBACK_DAYS = 90
_LARGE_BATCH_RATIO = 0.6
_STRONG_CONFIDENCE = 0.8

_POLICY_QUERY = {
    "stockout": "stockout after supplier late delivery lead time reorder point escalation",
    "supply_delay": "late delivery penalty lead time supplier contract escalation",
    "shrinkage": "theft shrinkage investigation threshold write-off approval discrepancy",
    "margin_drop": "shrinkage cost margin erosion write-off approval",
    "auto": "stockout shrinkage late delivery reorder point escalation investigation",
}

_DEFAULT_ACTIONS = {
    "supply_delay": RecommendedAction(
        action="Raise the late delivery with the supplier under the contract penalty clause and expedite the outstanding order.",
        owner_role="Inventory and Supply Chain Manager",
        urgency="high",
        expected_impact="Restores stock cover and recovers a contractual credit.",
    ),
    "shrinkage_theft": RecommendedAction(
        action="Open a loss-prevention investigation for the SKU: review CCTV for the spike window, secure the bay, and reconcile counts.",
        owner_role="Loss Prevention Lead",
        urgency="high",
        expected_impact="Stops ongoing theft loss and corrects the shelf/backroom discrepancy.",
    ),
    "expiry_writeoff": RecommendedAction(
        action="Start markdowns on the near-expiry batch now and reduce the next order quantity for this perishable SKU.",
        owner_role="Store Manager",
        urgency="high",
        expected_impact="Recovers part of the batch value and prevents the follow-on stockout.",
    ),
    "reorder_point": RecommendedAction(
        action="Update the reorder point and safety stock to the forecast-based recommendation and place a catch-up order.",
        owner_role="Inventory and Supply Chain Manager",
        urgency="high",
        expected_impact="Aligns replenishment with the new demand level and prevents repeat stockouts.",
    ),
    "demand_shift": RecommendedAction(
        action="Review the demand step-up, confirm whether it is promotional or structural, and adjust the replenishment plan.",
        owner_role="Store Manager",
        urgency="medium",
        expected_impact="Keeps a fast-moving line in stock during the elevated demand period.",
    ),
    "other": RecommendedAction(
        action="Investigate on the floor: confirm the physical facts and reconcile the recent deliveries, counts and shrinkage records.",
        owner_role="Shift Lead",
        urgency="medium",
        expected_impact="Establishes the true cause before corrective action.",
    ),
}


class RcaState(TypedDict, total=False):
    sku: str | None
    aisle: str | None
    anomaly_type: str
    supplier_id: int | None
    evidence: Annotated[list[dict], operator.add]
    tool_trace: Annotated[list[dict], operator.add]
    data_gaps: Annotated[list[str], operator.add]
    timeline: Annotated[list[dict], operator.add]
    policy_findings: Annotated[list[dict], operator.add]
    checks: Annotated[list[str], operator.add]
    llm_calls: Annotated[int, operator.add]
    report: dict


class RcaWorkflow:
    def __init__(self, llm: LLMProvider, tools: ToolProvider, retriever) -> None:
        self.llm = llm
        self.tools = tools
        self.retriever = retriever
        self._graph = self._build()

    # ---------------------------------------------------------------- tools
    def _call(self, trace: list, name: str, arguments: dict) -> tuple[dict, bool]:
        try:
            result = self.tools.call_tool(name, arguments)
        except BizAgentError as exc:
            trace.append({"tool": name, "arguments": arguments, "error": exc.error_code})
            return {"error": exc.message}, False
        except Exception as exc:  # noqa: BLE001 - one bad step must not abort RCA
            trace.append({"tool": name, "arguments": arguments, "error": str(exc)})
            return {"error": str(exc)}, False
        trace.append(
            {
                "tool": name,
                "arguments": arguments,
                "summary": result.summary,
                "is_error": result.is_error,
            }
        )
        return result.data, not result.is_error

    def _window(self) -> tuple[str, str]:
        end = dt.date.today()
        return (end - dt.timedelta(days=_LOOKBACK_DAYS)).isoformat(), end.isoformat()

    # -------------------------------------------------------- evidence nodes
    def _sales_node(self, state: RcaState) -> dict:
        sku = state.get("sku")
        trace: list = []
        evidence: list = []
        gaps: list = []
        checks = ["sales velocity and history", "demand shift vs baseline", "top-seller status"]
        if not sku:
            return {"checks": checks, "tool_trace": trace}
        start, end = self._window()
        hist, ok = self._call(trace, "get_sales_history", {"sku": sku, "start_date": start, "end_date": end})
        if ok:
            evidence.append(
                EvidenceItem(
                    id="sales.history",
                    source="tool:get_sales_history",
                    title="Recent sales history",
                    detail=f"{hist.get('total_units', 0)} units over the last {_LOOKBACK_DAYS} days.",
                    data={"total_units": hist.get("total_units", 0), "points": len(hist.get("points", []))},
                ).model_dump()
            )
        shifts, ok = self._call(trace, "detect_anomalies", {"kind": "demand_shift", "days": _LOOKBACK_DAYS})
        if ok:
            match = next((i for i in shifts.get("items", []) if i.get("sku") == sku), None)
            if match:
                evidence.append(
                    EvidenceItem(
                        id="sales.demand_shift",
                        source="detector:demand_shift",
                        title="Demand shift detected",
                        detail=(
                            f"Recent daily sales {match.get('recent_avg_daily')} vs baseline "
                            f"{match.get('baseline_avg_daily')} (x{match.get('ratio')})."
                        ),
                        data=match,
                    ).model_dump()
                )
        tops, ok = self._call(trace, "get_top_sellers", {"days": 30, "limit": 20})
        if ok:
            ranked = [i.get("sku") for i in tops.get("items", [])]
            if sku in ranked:
                evidence.append(
                    EvidenceItem(
                        id="sales.top_seller",
                        source="tool:get_top_sellers",
                        title="High-velocity SKU",
                        detail=f"Ranked #{ranked.index(sku) + 1} by units over the last 30 days.",
                        data={"rank": ranked.index(sku) + 1},
                    ).model_dump()
                )
        return {"evidence": evidence, "tool_trace": trace, "data_gaps": gaps, "checks": checks}

    def _inventory_node(self, state: RcaState) -> dict:
        sku = state.get("sku")
        trace: list = []
        evidence: list = []
        timeline: list = []
        gaps: list = []
        checks = ["stock trajectory", "shelf/backroom discrepancy", "batch and expiry history"]
        if not sku:
            return {"checks": checks, "tool_trace": trace}
        stock, ok = self._call(trace, "get_stock_level", {"sku": sku, "include_batches": True})
        if ok and stock.get("stock"):
            s = stock["stock"]
            batches = stock.get("batches", [])
            evidence.append(
                EvidenceItem(
                    id="inventory.stock",
                    source="tool:get_stock_level",
                    title="Current stock position",
                    detail=(
                        f"{s.get('on_hand_qty')} on hand vs reorder point {s.get('reorder_point')}"
                        f"{' (below reorder point)' if s.get('below_reorder_point') else ''}."
                    ),
                    data={"stock": s, "batches": batches},
                ).model_dump()
            )
            expected = (s.get("shelf_qty") or 0) + (s.get("backroom_qty") or 0)
            if expected != s.get("on_hand_qty"):
                evidence.append(
                    EvidenceItem(
                        id="inventory.discrepancy",
                        source="tool:get_stock_level",
                        title="Shelf/backroom discrepancy",
                        detail=f"Shelf+backroom {expected} != recorded on-hand {s.get('on_hand_qty')}.",
                        data={"expected": expected, "on_hand": s.get("on_hand_qty")},
                    ).model_dump()
                )
            for b in batches:
                timeline.append(
                    TimelineEvent(
                        date=str(b.get("expiry_date")),
                        event=f"Batch {b.get('batch_no')} expires ({b.get('qty_remaining')} units remaining)",
                    ).model_dump()
                )
        else:
            gaps.append("inventory: current stock level could not be retrieved")

        outs, ok = self._call(trace, "detect_anomalies", {"kind": "stockout", "days": _LOOKBACK_DAYS})
        if ok:
            match = next((i for i in outs.get("items", []) if i.get("sku") == sku), None)
            if match:
                evidence.append(
                    EvidenceItem(
                        id="inventory.stockout",
                        source="detector:stockout",
                        title="Recorded stockout",
                        detail=(
                            f"{match.get('stockout_days')} zero on-hand day(s) between "
                            f"{match.get('first_stockout')} and {match.get('last_stockout')}."
                        ),
                        data=match,
                    ).model_dump()
                )
                timeline.append(
                    TimelineEvent(date=str(match.get("first_stockout")), event="First zero on-hand day").model_dump()
                )
                timeline.append(
                    TimelineEvent(date=str(match.get("last_stockout")), event="Most recent zero on-hand day").model_dump()
                )
        return {"evidence": evidence, "tool_trace": trace, "timeline": timeline, "data_gaps": gaps, "checks": checks}

    def _shrinkage_node(self, state: RcaState) -> dict:
        sku = state.get("sku")
        aisle = state.get("aisle")
        trace: list = []
        evidence: list = []
        timeline: list = []
        checks = ["shrinkage events by reason", "shrinkage spike detection", "loss valuation"]
        args = {"days": _LOOKBACK_DAYS}
        args.update({"sku": sku} if sku else {"aisle": aisle})
        report, ok = self._call(trace, "get_shrinkage_report", args)
        if ok:
            evidence.append(
                EvidenceItem(
                    id="shrinkage.report",
                    source="tool:get_shrinkage_report",
                    title="Shrinkage valuation",
                    detail=(
                        f"{report.get('total_qty', 0)} units lost / "
                        f"{report.get('total_cost', 0)} cost over {_LOOKBACK_DAYS} days."
                    ),
                    data={"total_qty": report.get("total_qty", 0), "total_cost": report.get("total_cost", 0),
                          "by_reason": report.get("by_reason", {})},
                ).model_dump()
            )
        spikes, ok = self._call(trace, "detect_anomalies", {"kind": "shrinkage", "days": _LOOKBACK_DAYS})
        if ok and sku:
            match = next((i for i in spikes.get("items", []) if i.get("sku") == sku), None)
            if match:
                evidence.append(
                    EvidenceItem(
                        id="shrinkage.spike",
                        source="detector:shrinkage",
                        title="Shrinkage spike detected",
                        detail=(
                            f"{match.get('recent_qty')} units lost recently vs baseline {match.get('baseline_qty')}; "
                            f"dominant reason '{match.get('dominant_reason')}'."
                        ),
                        data=match,
                    ).model_dump()
                )
                timeline.append(
                    TimelineEvent(date=str(match.get("first_event")), event="First event in the shrinkage spike").model_dump()
                )
                timeline.append(
                    TimelineEvent(date=str(match.get("last_event")), event="Latest event in the shrinkage spike").model_dump()
                )
        return {"evidence": evidence, "tool_trace": trace, "timeline": timeline, "checks": checks}

    def _supply_node(self, state: RcaState) -> dict:
        sku = state.get("sku")
        trace: list = []
        evidence: list = []
        timeline: list = []
        gaps: list = []
        checks = ["open and late purchase orders", "promised vs received dates", "supplier on-time rate", "lead-time breaches"]
        if not sku:
            return {"checks": checks, "tool_trace": trace}

        info, ok = self._call(trace, "get_product_info", {"sku": sku})
        supplier_id = None
        if ok and info.get("matches"):
            supplier_id = info["matches"][0].get("supplier_id")

        open_pos, ok = self._call(trace, "get_open_purchase_orders", {"sku": sku, "late_only": False})
        if ok and open_pos.get("items"):
            evidence.append(
                EvidenceItem(
                    id="supply.open_pos",
                    source="tool:get_open_purchase_orders",
                    title="Open purchase orders",
                    detail=f"{open_pos.get('count', 0)} open purchase order(s) for this SKU.",
                    data={"items": open_pos.get("items", [])},
                ).model_dump()
            )
        late_pos, ok = self._call(trace, "get_open_purchase_orders", {"sku": sku, "late_only": True})
        if ok and late_pos.get("items"):
            items = late_pos["items"]
            max_delay = max((po.get("delay_days") or 0) for po in items)
            evidence.append(
                EvidenceItem(
                    id="supply.late_pos",
                    source="tool:get_open_purchase_orders",
                    title="Late purchase orders",
                    detail=f"{len(items)} late delivery(ies); worst was {max_delay} days late.",
                    data={"items": items, "max_delay_days": max_delay},
                ).model_dump()
            )
            for po in items:
                timeline.append(
                    TimelineEvent(date=str(po.get("promised_date")), event=f"PO {po.get('po_id')} promised").model_dump()
                )
                timeline.append(
                    TimelineEvent(
                        date=str(po.get("received_date")),
                        event=f"PO {po.get('po_id')} received ({po.get('delay_days')} days late)",
                    ).model_dump()
                )
        if supplier_id is not None:
            status, ok = self._call(trace, "get_supplier_status", {"supplier_id": supplier_id})
            if ok and status.get("supplier"):
                sup = status["supplier"]
                evidence.append(
                    EvidenceItem(
                        id="supply.supplier",
                        source="tool:get_supplier_status",
                        title="Supplier reliability",
                        detail=(
                            f"{sup.get('name')}: {round((sup.get('on_time_delivery_rate') or 0) * 100)}% on time, "
                            f"avg delay {sup.get('avg_delay_days')} days."
                        ),
                        data=sup,
                    ).model_dump()
                )
        else:
            gaps.append("supply: could not resolve the SKU's supplier")
        return {
            "evidence": evidence,
            "tool_trace": trace,
            "timeline": timeline,
            "data_gaps": gaps,
            "checks": checks,
            "supplier_id": supplier_id,
        }

    def _forecast_node(self, state: RcaState) -> dict:
        sku = state.get("sku")
        trace: list = []
        evidence: list = []
        gaps: list = []
        checks = ["forecast vs actual — was this predictable?"]
        if not sku:
            return {"checks": checks, "tool_trace": trace}
        fc, ok = self._call(
            trace, "get_demand_forecast", {"sku": sku, "horizon_days": 14, "include_risk": True}
        )
        if not ok:
            gaps.append("forecast: demand forecast unavailable")
            return {"tool_trace": trace, "data_gaps": gaps, "checks": checks}
        risk = fc.get("stockout_risk") or {}
        reorder = fc.get("reorder_recommendation") or {}
        expiry_units = sum(e.get("projected_units_expiring", 0) for e in fc.get("expiry_risk", []))
        evidence.append(
            EvidenceItem(
                id="forecast.summary",
                source="tool:get_demand_forecast",
                title="Forecast and derived risk",
                detail=(
                    f"~{fc.get('avg_daily_forecast')} units/day ({fc.get('backend')}); "
                    f"stockout risk {risk.get('risk_level')} (cover {risk.get('days_of_cover')}); "
                    f"reorder point {reorder.get('current_reorder_point')} -> "
                    f"{reorder.get('recommended_reorder_point')} (delta {reorder.get('delta')})."
                ),
                data={
                    "avg_daily_forecast": fc.get("avg_daily_forecast"),
                    "backend": fc.get("backend"),
                    "stockout_risk_level": risk.get("risk_level"),
                    "projected_stockout_date": risk.get("projected_stockout_date"),
                    "reorder_delta": reorder.get("delta"),
                    "current_reorder_point": reorder.get("current_reorder_point"),
                    "recommended_reorder_point": reorder.get("recommended_reorder_point"),
                    "expiry_units": expiry_units,
                },
            ).model_dump()
        )
        return {"evidence": evidence, "tool_trace": trace, "data_gaps": gaps, "checks": checks}

    def _policy_node(self, state: RcaState) -> dict:
        anomaly = state.get("anomaly_type", "auto")
        query = _POLICY_QUERY.get(anomaly, _POLICY_QUERY["auto"])
        checks = ["SOP and contract clauses governing this situation (RAG)"]
        try:
            result = self.retriever.retrieve(query)
        except BizAgentError as exc:
            return {"data_gaps": [f"policy: {exc.message}"], "checks": checks}
        except Exception as exc:  # noqa: BLE001
            return {"data_gaps": [f"policy: {exc}"], "checks": checks}
        if not result.found:
            return {"data_gaps": ["policy: no governing SOP or contract clause was found"], "checks": checks}
        findings = [
            PolicyFinding(
                clause=c.snippet.split(". ")[0][:200],
                citation=c.model_dump(mode="json"),
                relevance=f"Governs {anomaly.replace('_', ' ')} handling and escalation.",
            ).model_dump()
            for c in result.citations[:3]
        ]
        return {"policy_findings": findings, "checks": checks}

    # ------------------------------------------------------------- ranking
    def _rule_rank(self, pool: dict[str, dict]) -> list[RankedCause]:
        causes: list[RankedCause] = []
        stockout = "inventory.stockout" in pool
        late = pool.get("supply.late_pos")
        if late and late["data"].get("items"):
            delay = late["data"].get("max_delay_days", 0)
            # A late delivery is only a *root cause* if stock actually ran out;
            # otherwise it is at most a contributing factor.
            if delay >= 4 or stockout:
                conf = round(min(0.92, 0.55 + delay / 25), 2) if stockout else 0.5
                ids = ["supply.late_pos"]
                ids += ["inventory.stockout"] if stockout else []
                ids += ["supply.supplier"] if "supply.supplier" in pool else []
                causes.append(
                    RankedCause(
                        cause=f"A supplier purchase order arrived {delay} days late, depleting stock before it could be replenished.",
                        category="supply_delay",
                        confidence=conf,
                        evidence_ids=ids,
                        contributing_factor=not stockout,
                    )
                )

        spike = pool.get("shrinkage.spike")
        if spike:
            reason = spike["data"].get("dominant_reason", "")
            cat = {
                "theft": "shrinkage_theft",
                "damage": "shrinkage_damage",
                "admin_error": "shrinkage_admin_error",
                "expiry": "expiry_writeoff",
            }.get(reason, "other")
            ids = ["shrinkage.spike"] + (["shrinkage.report"] if "shrinkage.report" in pool else [])
            ids += ["inventory.discrepancy"] if "inventory.discrepancy" in pool else []
            causes.append(
                RankedCause(
                    cause=f"An abnormal spike in '{reason}'-reason shrinkage events is driving unaccounted loss.",
                    category=cat,
                    confidence=0.82 if reason == "theft" else 0.72,
                    evidence_ids=ids,
                )
            )

        fc = pool.get("forecast.summary")
        stock_ev = pool.get("inventory.stock")
        batches = (stock_ev or {}).get("data", {}).get("batches", []) if stock_ev else []
        large_near_expiry = [
            b
            for b in batches
            if (b.get("days_until_expiry", 999) <= 7)
            and b.get("qty_remaining", 0)
            >= _LARGE_BATCH_RATIO * (b.get("qty_received") or b.get("qty_remaining", 1))
        ]
        if large_near_expiry:
            conf = 0.72 + (0.1 if fc else 0.0) + (0.1 if stockout else 0.0)
            ids = ["inventory.stock"]
            ids += ["forecast.summary"] if fc else []
            ids += ["inventory.stockout"] if stockout else []
            causes.append(
                RankedCause(
                    cause="A large perishable batch is projected to expire before it can sell, forcing a write-off and a follow-on stockout.",
                    category="expiry_writeoff",
                    confidence=round(min(conf, 1.0), 2),
                    evidence_ids=ids,
                )
            )

        shift = pool.get("sales.demand_shift")
        if shift and (shift["data"].get("ratio") or 0) >= 1.4:
            reorder_delta = fc["data"].get("reorder_delta") if fc else None
            current_rop = (fc["data"].get("current_reorder_point") if fc else None) or 1
            stale = reorder_delta is not None and reorder_delta > max(0.2 * current_rop, 5)
            ids = ["sales.demand_shift"] + (["forecast.summary"] if fc else [])
            causes.append(
                RankedCause(
                    cause=(
                        "Demand has stepped up sharply while the reorder point stayed at its old level, so replenishment is sized for the old demand."
                        if stale
                        else "A sustained step-up in demand is outpacing the current replenishment plan."
                    ),
                    category="reorder_point" if stale else "demand_shift",
                    confidence=0.82 if stale else 0.62,
                    evidence_ids=ids,
                )
            )

        causes.sort(key=lambda c: c.confidence, reverse=True)
        return causes

    def _rank_node(self, state: RcaState) -> dict:
        pool = {e["id"]: e for e in state.get("evidence", [])}
        sku = state.get("sku")
        aisle = state.get("aisle")
        anomaly = state.get("anomaly_type", "auto")
        gaps = list(dict.fromkeys(state.get("data_gaps", [])))
        timeline = _dedupe_timeline(state.get("timeline", []))
        policy = state.get("policy_findings", [])
        checks = list(dict.fromkeys(state.get("checks", [])))

        deterministic = self._rule_rank(pool)
        now = dt.datetime.utcnow().isoformat()
        tool_trace = state.get("tool_trace", [])

        if not deterministic:
            report = RootCauseReport(
                sku=sku,
                aisle=aisle,
                anomaly_type=anomaly,
                status="no_anomaly",
                summary=(
                    "No significant anomaly was detected for this target. "
                    "The checks below were performed and none crossed their thresholds."
                ),
                ranked_causes=[],
                timeline=[TimelineEvent(**t) for t in timeline],
                policy_findings=[PolicyFinding(**p) for p in policy],
                recommended_actions=[],
                data_gaps=gaps,
                evidence=[EvidenceItem(**e) for e in pool.values()],
                checks_performed=checks,
                tool_trace=tool_trace,
                generated_at=now,
                llm_calls=state.get("llm_calls", 0),
            )
            return {"report": report.model_dump(mode="json"), "llm_calls": 0}

        llm_used = 0
        llm_causes: list[RankedCause] = []
        summary = ""
        actions: list[RecommendedAction] = []
        try:
            candidates = [c.model_dump() for c in deterministic]
            ranking: LlmRanking = generate_structured(
                self.llm,
                prompts.build_user_prompt(
                    sku=sku,
                    aisle=aisle,
                    anomaly_type=anomaly,
                    evidence=list(pool.values()),
                    candidates=candidates,
                    policy_findings=policy,
                    timeline=timeline,
                    data_gaps=gaps,
                ),
                prompts.SYSTEM,
                LlmRanking,
                temperature=0.2,
            )
            llm_used = 1
            llm_causes = _validate_causes(ranking.ranked_causes, pool)
            summary = ranking.summary.strip()
            actions = ranking.recommended_actions
            gaps = list(dict.fromkeys(gaps + [g for g in ranking.data_gaps if g]))
        except (LLMError, LLMParseError) as exc:
            logger.warning("rca_llm_unavailable_using_deterministic", error=str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.error("rca_ranking_failed", error=str(exc))

        final_causes = _merge_causes(deterministic, llm_causes)
        status = "partial" if (llm_used == 0 or gaps) else "complete"
        if not summary:
            summary = _fallback_summary(final_causes, llm_used == 0)
        if not actions:
            actions = [_DEFAULT_ACTIONS.get(final_causes[0].category, _DEFAULT_ACTIONS["other"])]

        report = RootCauseReport(
            sku=sku,
            aisle=aisle,
            anomaly_type=anomaly,
            status=status,
            summary=summary,
            ranked_causes=final_causes,
            timeline=[TimelineEvent(**t) for t in timeline],
            policy_findings=[PolicyFinding(**p) for p in policy],
            recommended_actions=actions,
            data_gaps=gaps,
            evidence=[EvidenceItem(**e) for e in pool.values()],
            checks_performed=checks,
            tool_trace=tool_trace,
            generated_at=now,
            llm_calls=state.get("llm_calls", 0) + llm_used,
        )
        return {"report": report.model_dump(mode="json"), "llm_calls": llm_used}

    # ------------------------------------------------------------- graph
    def _build(self):
        graph = StateGraph(RcaState)
        nodes = {
            "sales_evidence": self._sales_node,
            "inventory_evidence": self._inventory_node,
            "shrinkage_evidence": self._shrinkage_node,
            "supply_evidence": self._supply_node,
            "forecast_evidence": self._forecast_node,
            "policy_evidence": self._policy_node,
        }
        for name, fn in nodes.items():
            graph.add_node(name, fn)
            graph.add_edge(START, name)
            graph.add_edge(name, "rank")
        graph.add_node("rank", self._rank_node)
        graph.add_edge("rank", END)
        return graph.compile()

    def run(self, request: RcaRequest) -> RootCauseReport:
        initial: RcaState = {
            "sku": request.sku,
            "aisle": request.aisle,
            "anomaly_type": request.anomaly_type,
            "evidence": [],
            "tool_trace": [],
            "data_gaps": [],
            "timeline": [],
            "policy_findings": [],
            "checks": [],
            "llm_calls": 0,
        }
        try:
            final = self._graph.invoke(initial, config={"recursion_limit": 25})
        except Exception as exc:  # noqa: BLE001 - never crash the caller
            logger.error("rca_workflow_failed", error=str(exc))
            return RootCauseReport(
                sku=request.sku,
                aisle=request.aisle,
                anomaly_type=request.anomaly_type,
                status="partial",
                summary=f"The root-cause workflow could not complete: {exc}",
                generated_at=dt.datetime.utcnow().isoformat(),
            )
        return RootCauseReport(**final["report"])


# ------------------------------------------------------------------- helpers
def _validate_causes(causes: list[RankedCause], pool: dict[str, dict]) -> list[RankedCause]:
    """Drop causes whose evidence does not resolve; keep only resolvable ids."""
    kept: list[RankedCause] = []
    for c in causes:
        valid = [eid for eid in c.evidence_ids if eid in pool]
        if not valid:
            logger.info("rca_cause_dropped_no_evidence", cause=c.cause[:80])
            continue
        c.evidence_ids = valid
        kept.append(c)
    return kept


def _merge_causes(deterministic: list[RankedCause], llm: list[RankedCause]) -> list[RankedCause]:
    """LLM ranking wins for ambiguous cases; a strong deterministic primary is
    never demoted below a weaker cause."""
    if not llm:
        for i, c in enumerate(deterministic):
            c.contributing_factor = i > 0
        return deterministic
    primary = deterministic[0]
    llm_categories = {c.category for c in llm}
    ordered = list(llm)
    if primary.category not in llm_categories:
        ordered.insert(0, primary)
    elif primary.confidence >= _STRONG_CONFIDENCE and ordered[0].category != primary.category:
        ordered = [c for c in ordered if c.category != primary.category]
        ordered.insert(0, primary)
        logger.info("rca_rank_override", forced=primary.category)
    for i, c in enumerate(ordered):
        c.contributing_factor = i > 0
    return ordered


def _dedupe_timeline(events: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for e in sorted(events, key=lambda x: str(x.get("date"))):
        key = (str(e.get("date")), e.get("event"))
        if key in seen or e.get("date") in (None, "None", ""):
            continue
        seen.add(key)
        out.append(e)
    return out


def _fallback_summary(causes: list[RankedCause], degraded: bool) -> str:
    prefix = "[LLM unavailable — deterministic analysis] " if degraded else ""
    if not causes:
        return prefix + "No cause could be established from the available evidence."
    top = causes[0]
    return (
        prefix
        + f"Most likely cause: {top.cause} (confidence {top.confidence:.0%}). "
        + (f"{len(causes) - 1} contributing factor(s) also identified." if len(causes) > 1 else "")
    ).strip()
