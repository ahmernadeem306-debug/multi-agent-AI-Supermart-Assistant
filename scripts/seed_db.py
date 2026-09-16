"""Deterministic synthetic data seeder for BizAgent.

Generates 6 suppliers, 60 products across 8 aisles, 540 days of daily sales
with weekly seasonality and a mild upward trend, daily stock snapshots,
perishable batches, ~120 purchase orders (~20% late), ~80 shrinkage events,
and plants exactly four anomalies for later root-cause-analysis validation.

Usage:
    python scripts/seed_db.py           # seed once; refuses if already seeded
    python scripts/seed_db.py --reset   # drop all tables and reseed from scratch
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.base import Base, create_all, get_engine, get_session  # noqa: E402
from app.db.models import Product  # noqa: E402
from app.db.repositories.finance_repo import FinanceRepository  # noqa: E402
from app.db.repositories.inventory_repo import InventoryRepository  # noqa: E402
from app.db.repositories.sales_repo import SalesRepository  # noqa: E402
from app.db.repositories.supplier_repo import SupplierRepository  # noqa: E402

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

N_SUPPLIERS = 6
N_PRODUCTS = 60
N_DAYS = 540
LATE_DELIVERY_RATE = 0.20
N_SHRINKAGE_EVENTS = 80

AISLES = [
    "Produce",
    "Dairy",
    "Bakery",
    "Frozen",
    "Beverages",
    "Snacks",
    "Household",
    "Personal Care",
]
PERISHABLE_AISLES = {"Produce", "Dairy", "Bakery", "Frozen"}
SHELF_LIFE_BY_AISLE = {"Produce": 7, "Dairy": 14, "Bakery": 5, "Frozen": 180}
CATEGORY_BY_AISLE = {
    "Produce": ["Fruits", "Vegetables"],
    "Dairy": ["Milk", "Cheese", "Yogurt"],
    "Bakery": ["Bread", "Pastries"],
    "Frozen": ["Frozen Meals", "Ice Cream"],
    "Beverages": ["Soft Drinks", "Juices", "Water"],
    "Snacks": ["Chips", "Cookies", "Candy"],
    "Household": ["Cleaning", "Paper Goods"],
    "Personal Care": ["Toiletries", "Health"],
}
SHRINKAGE_REASONS = ["damage", "theft", "expiry", "admin_error"]

END_DATE = dt.date.today() - dt.timedelta(days=1)
START_DATE = END_DATE - dt.timedelta(days=N_DAYS - 1)


def build_suppliers(repo: SupplierRepository) -> list[dict]:
    suppliers = []
    for i in range(1, N_SUPPLIERS + 1):
        s = repo.add_supplier(
            name=f"Supplier {i}",
            lead_time_days=int(random.randint(3, 14)),
            reliability_score=round(random.uniform(0.75, 0.99), 2),
            contact_email=f"supplier{i}@example-vendors.test",
            contract_ref=f"CTR-{1000 + i}",
        )
        suppliers.append(s)
    return suppliers


def build_products(repo: InventoryRepository, suppliers: list[dict]) -> list[dict]:
    products = []
    sku_counter = 1000
    per_aisle = N_PRODUCTS // len(AISLES)
    remainder = N_PRODUCTS % len(AISLES)

    for aisle_idx, aisle in enumerate(AISLES):
        count = per_aisle + (1 if aisle_idx < remainder else 0)
        categories = CATEGORY_BY_AISLE[aisle]
        is_perishable = aisle in PERISHABLE_AISLES
        for n in range(count):
            sku_counter += 1
            sku = f"SKU-{sku_counter}"
            category = categories[n % len(categories)]
            base_daily_demand = round(random.uniform(2, 45), 1)
            unit_cost = round(random.uniform(0.8, 20.0), 2)
            margin = random.uniform(0.25, 0.55)
            unit_price = round(unit_cost / (1 - margin), 2)
            supplier = suppliers[(sku_counter) % len(suppliers)]

            product = repo.add_product(
                sku=sku,
                name=f"{category} {sku_counter}",
                category=category,
                aisle=aisle,
                unit_cost=unit_cost,
                unit_price=unit_price,
                is_perishable=is_perishable,
                shelf_life_days=SHELF_LIFE_BY_AISLE.get(aisle) if is_perishable else None,
                reorder_point=int(round(base_daily_demand * 5)),
                safety_stock=int(round(base_daily_demand * 2)),
                supplier_id=supplier["id"],
            )
            product["_base_daily_demand"] = base_daily_demand
            products.append(product)
    return products


def _daily_demand_series(base_demand: float, boost_from_day: int | None = None, boost_factor: float = 2.0) -> list[int]:
    """Weekly-seasonal, mildly trending, noisy daily demand for N_DAYS."""
    series = []
    for day_index in range(N_DAYS):
        current_date = START_DATE + dt.timedelta(days=day_index)
        weekday = current_date.weekday()  # 0=Mon ... 6=Sun
        seasonality = 1.3 if weekday >= 5 else (0.9 if weekday == 0 else 1.0)
        trend = 1 + 0.0003 * day_index
        noise = float(np.random.normal(loc=1.0, scale=0.15))
        demand = base_demand * seasonality * trend * max(noise, 0.1)
        if boost_from_day is not None and day_index >= boost_from_day:
            demand *= boost_factor
        series.append(max(int(round(demand)), 0))
    return series


def build_sales_and_stock(
    sales_repo: SalesRepository,
    inv_repo: InventoryRepository,
    product: dict,
    demand_series: list[int],
    starve_days: set[int] | None = None,
) -> None:
    """Populate sales_transactions and stock_levels for one product.

    `starve_days` (day indices) get their replenishment skipped, which is how
    the stockout anomalies are produced without a full inventory-events engine.
    """
    starve_days = starve_days or set()
    sku = product["sku"]
    reorder_point = product["reorder_point"]
    safety_stock = product["safety_stock"]

    on_hand = safety_stock * 3
    sales_rows: list[dict] = []
    stock_rows: list[dict] = []

    for day_index, qty_demanded in enumerate(demand_series):
        current_date = START_DATE + dt.timedelta(days=day_index)
        qty_sold = min(qty_demanded, on_hand)
        on_hand -= qty_sold

        sales_rows.append(
            {
                "txn_id": f"TXN-{sku}-{day_index:04d}",
                "ts": dt.datetime.combine(current_date, dt.time(hour=12)),
                "sku": sku,
                "qty": qty_sold,
                "unit_price": product["unit_price"],
                "discount": 0.0,
                "register_id": "REG-1",
            }
        )

        if on_hand <= reorder_point and day_index not in starve_days:
            on_hand += reorder_point * 2

        shelf_qty = min(on_hand, max(reorder_point, 1))
        backroom_qty = on_hand - shelf_qty
        stock_rows.append(
            {
                "sku": sku,
                "snapshot_date": current_date,
                "shelf_qty": shelf_qty,
                "backroom_qty": backroom_qty,
                "on_hand_qty": on_hand,
            }
        )

    sales_repo.bulk_add_transactions(sales_rows)
    inv_repo.bulk_add_stock_levels(stock_rows)


def build_purchase_orders(
    supplier_repo: SupplierRepository,
    product: dict,
    lead_time_days: int,
    po_counter: list[int],
    forced_late_days: int | None = None,
) -> None:
    """Generate ~2 purchase orders spread across the seed window for one product."""
    n_orders = 2
    span = N_DAYS // (n_orders + 1)
    for i in range(1, n_orders + 1):
        order_day = min(span * i, N_DAYS - 1)
        order_date = START_DATE + dt.timedelta(days=order_day)
        promised_date = order_date + dt.timedelta(days=lead_time_days)

        is_forced_late = forced_late_days is not None and i == 1
        if is_forced_late:
            delay = forced_late_days
        else:
            delay = random.randint(3, 10) if random.random() < LATE_DELIVERY_RATE else random.randint(-1, 1)
        received_date = promised_date + dt.timedelta(days=max(delay, 0))

        if received_date > END_DATE:
            received_date = None
            status = "open"
            qty_received = None
        else:
            status = "late" if received_date > promised_date else "received"
            qty_received = product["reorder_point"] * 2

        po_counter[0] += 1
        supplier_repo.add_purchase_order(
            po_id=f"PO-{po_counter[0]:05d}",
            supplier_id=product["supplier_id"],
            sku=product["sku"],
            qty_ordered=product["reorder_point"] * 2,
            order_date=order_date,
            promised_date=promised_date,
            received_date=received_date,
            qty_received=qty_received,
            status=status,
        )


def build_batches(inv_repo: InventoryRepository, product: dict, forced_expiring: bool = False) -> None:
    shelf_life = product["shelf_life_days"] or 14
    n_batches = 3
    for i in range(n_batches):
        received_offset = N_DAYS - 1 - (i * (shelf_life * 2))
        received_offset = max(received_offset, 0)
        received_date = START_DATE + dt.timedelta(days=received_offset)
        expiry_date = received_date + dt.timedelta(days=shelf_life)
        qty_received = product["reorder_point"] * 2

        if forced_expiring and i == 0:
            expiry_date = END_DATE + dt.timedelta(days=3)
            received_date = expiry_date - dt.timedelta(days=shelf_life)
            qty_remaining = int(qty_received * 0.8)
        else:
            qty_remaining = max(qty_received - int(qty_received * random.uniform(0.6, 1.0)), 0)

        inv_repo.add_stock_batch(
            sku=product["sku"],
            batch_no=f"BATCH-{product['sku']}-{i}",
            received_date=received_date,
            expiry_date=expiry_date,
            qty_received=qty_received,
            qty_remaining=qty_remaining,
        )


def build_shrinkage_events(
    finance_repo: FinanceRepository,
    products: list[dict],
    theft_spike_sku: str | None,
) -> int:
    count = 0
    baseline_events = N_SHRINKAGE_EVENTS - 10  # reserve ~10 for the planted spike
    for _ in range(baseline_events):
        product = random.choice(products)
        event_day = random.randint(0, N_DAYS - 1)
        finance_repo.add_shrinkage_event(
            sku=product["sku"],
            event_date=START_DATE + dt.timedelta(days=event_day),
            qty=random.randint(1, 8),
            reason=random.choice(SHRINKAGE_REASONS),
            notes=None,
        )
        count += 1

    if theft_spike_sku:
        spike_start = N_DAYS - 21
        for i in range(10):
            finance_repo.add_shrinkage_event(
                sku=theft_spike_sku,
                event_date=START_DATE + dt.timedelta(days=spike_start + i),
                qty=random.randint(8, 20),
                reason="theft",
                notes="planted anomaly: theft spike",
            )
            count += 1
    return count


def already_seeded(session) -> bool:
    return session.query(Product).first() is not None


def seed(reset: bool) -> dict:
    engine = get_engine()

    if reset:
        Base.metadata.drop_all(engine)
    create_all()

    with get_session() as session:
        if not reset and already_seeded(session):
            print("Database already seeded. Re-run with --reset to reseed from scratch.")
            sys.exit(1)

        supplier_repo = SupplierRepository(session)
        inv_repo = InventoryRepository(session)
        sales_repo = SalesRepository(session)
        finance_repo = FinanceRepository(session)

        suppliers = build_suppliers(supplier_repo)
        products = build_products(inv_repo, suppliers)

        high_velocity = max(products, key=lambda p: p["_base_daily_demand"])
        perishables = [p for p in products if p["is_perishable"]]
        expiry_sku_product = perishables[0]
        remaining = [p for p in products if p["sku"] not in {high_velocity["sku"], expiry_sku_product["sku"]}]
        theft_sku_product = remaining[0]
        demand_shift_product = remaining[1]

        anomalies = {
            "generated_at": dt.datetime.utcnow().isoformat(),
            "anomalies": [
                {
                    "id": "A",
                    "sku": high_velocity["sku"],
                    "type": "stockout_late_delivery",
                    "description": (
                        f"{high_velocity['sku']} is a high-velocity SKU that stocks out because a purchase "
                        "order was delivered 9 days late."
                    ),
                },
                {
                    "id": "B",
                    "sku": theft_sku_product["sku"],
                    "type": "shrinkage_theft_spike",
                    "description": (
                        f"{theft_sku_product['sku']} has normal purchase orders but an abnormal spike in "
                        "theft-reason shrinkage events, producing a shelf/backroom discrepancy."
                    ),
                },
                {
                    "id": "C",
                    "sku": expiry_sku_product["sku"],
                    "type": "expiry_writeoff_stockout",
                    "description": (
                        f"{expiry_sku_product['sku']} is a perishable SKU with a large batch expiring soon, "
                        "causing an expiry-driven write-off and a subsequent stockout."
                    ),
                },
                {
                    "id": "D",
                    "sku": demand_shift_product["sku"],
                    "type": "demand_shift_stale_reorder_point",
                    "description": (
                        f"{demand_shift_product['sku']} sales suddenly doubled (promotion-like demand shift) "
                        "in the last 60 days while its reorder_point stayed at the old level."
                    ),
                },
            ],
        }

        po_counter = [0]
        for product in products:
            supplier = next(s for s in suppliers if s["id"] == product["supplier_id"])

            starve_days: set[int] = set()
            forced_late_days = None
            if product["sku"] == high_velocity["sku"]:
                forced_late_days = 9
                starve_days = set(range(N_DAYS - 30, N_DAYS - 21))

            boost_from_day = N_DAYS - 60 if product["sku"] == demand_shift_product["sku"] else None
            demand_series = _daily_demand_series(product["_base_daily_demand"], boost_from_day=boost_from_day)

            if product["sku"] == expiry_sku_product["sku"]:
                # Long enough non-replenishment run that on-hand reliably
                # reaches zero regardless of day-of-week noise, so the
                # expiry-driven write-off always produces a real stockout.
                starve_days |= set(range(N_DAYS - 20, N_DAYS))

            build_sales_and_stock(sales_repo, inv_repo, product, demand_series, starve_days=starve_days)
            build_purchase_orders(
                supplier_repo,
                product,
                lead_time_days=supplier["lead_time_days"],
                po_counter=po_counter,
                forced_late_days=forced_late_days,
            )
            if product["is_perishable"]:
                build_batches(
                    inv_repo,
                    product,
                    forced_expiring=(product["sku"] == expiry_sku_product["sku"]),
                )

        n_shrinkage = build_shrinkage_events(finance_repo, products, theft_spike_sku=theft_sku_product["sku"])

        counts = {
            "suppliers": len(suppliers),
            "products": len(products),
            "sales_transactions": N_DAYS * len(products),
            "stock_levels": N_DAYS * len(products),
            "stock_batches": 3 * len(perishables),
            "purchase_orders": po_counter[0],
            "shrinkage_events": n_shrinkage,
        }

    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    with open(data_dir / "anomalies.json", "w", encoding="utf-8") as f:
        json.dump(anomalies, f, indent=2)

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the BizAgent database with synthetic retail data.")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate all tables before seeding.")
    args = parser.parse_args()

    counts = seed(reset=args.reset)

    print("\nSeed complete. Row counts:")
    for table, count in counts.items():
        print(f"  {table:<20} {count}")
    print("\nPlanted anomalies written to data/anomalies.json")


if __name__ == "__main__":
    main()
