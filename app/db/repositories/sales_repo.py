"""Typed, parameterised data access for sales transactions."""
from __future__ import annotations

import datetime as dt

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import SalesTransaction


def _txn_to_dict(t: SalesTransaction) -> dict:
    return {
        "txn_id": t.txn_id,
        "ts": t.ts,
        "sku": t.sku,
        "qty": t.qty,
        "unit_price": t.unit_price,
        "discount": t.discount,
        "register_id": t.register_id,
    }


class SalesRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add_transaction(self, **fields) -> dict:
        txn = SalesTransaction(**fields)
        self.session.add(txn)
        self.session.flush()
        return _txn_to_dict(txn)

    def bulk_add_transactions(self, rows: list[dict]) -> None:
        """Bulk-insert many transactions at once (used by the seeder)."""
        if rows:
            self.session.execute(SalesTransaction.__table__.insert(), rows)

    def list_transactions(
        self, sku: str, start_ts: dt.datetime, end_ts: dt.datetime
    ) -> list[dict]:
        stmt = (
            select(SalesTransaction)
            .where(SalesTransaction.sku == sku)
            .where(SalesTransaction.ts >= start_ts)
            .where(SalesTransaction.ts <= end_ts)
            .order_by(SalesTransaction.ts.asc())
        )
        return [_txn_to_dict(t) for t in self.session.scalars(stmt)]

    def total_units_sold(self, sku: str, start_ts: dt.datetime, end_ts: dt.datetime) -> int:
        stmt = (
            select(func.coalesce(func.sum(SalesTransaction.qty), 0))
            .where(SalesTransaction.sku == sku)
            .where(SalesTransaction.ts >= start_ts)
            .where(SalesTransaction.ts <= end_ts)
        )
        return int(self.session.scalar(stmt) or 0)

    def count_transactions(self) -> int:
        return int(self.session.scalar(select(func.count()).select_from(SalesTransaction)) or 0)
