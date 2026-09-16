"""Typed, parameterised data access for shrinkage events (finance-relevant loss data)."""
from __future__ import annotations

import datetime as dt

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import ShrinkageEvent


def _event_to_dict(e: ShrinkageEvent) -> dict:
    return {
        "id": e.id,
        "sku": e.sku,
        "event_date": e.event_date,
        "qty": e.qty,
        "reason": e.reason,
        "notes": e.notes,
    }


class FinanceRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add_shrinkage_event(self, **fields) -> dict:
        event = ShrinkageEvent(**fields)
        self.session.add(event)
        self.session.flush()
        return _event_to_dict(event)

    def list_shrinkage_events(
        self,
        sku: str | None = None,
        start_date: dt.date | None = None,
        end_date: dt.date | None = None,
    ) -> list[dict]:
        stmt = select(ShrinkageEvent)
        if sku is not None:
            stmt = stmt.where(ShrinkageEvent.sku == sku)
        if start_date is not None:
            stmt = stmt.where(ShrinkageEvent.event_date >= start_date)
        if end_date is not None:
            stmt = stmt.where(ShrinkageEvent.event_date <= end_date)
        stmt = stmt.order_by(ShrinkageEvent.event_date.asc())
        return [_event_to_dict(e) for e in self.session.scalars(stmt)]
