"""Manage Data — CSV bulk import + manual quick-entry for a real store's own data.

Writes go through ``components.data_entry``, which uses the *same* SQLite
database and the *same* repository methods the seed script already uses.
The FastAPI app, the agents and the MCP tool server are untouched and keep
using their existing read-only session for every AI-facing code path.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from components import common, data_entry, theme  # noqa: E402

st.set_page_config(page_title="BizAgent · Manage Data", page_icon="🗂️", layout="wide")
theme.inject_global_css()
common.sidebar("manage_data")

theme.page_header(
    "Manage Data",
    "Load your own store data once via CSV, then keep it current with quick daily entries. "
    "Everything here is saved permanently to the same database BizAgent already reads from.",
    icon="cart",
)

tab_bulk, tab_manual = st.tabs(["📥 Bulk import (CSV)", "✍️ Manual entry"])

_ENTITY_LABELS = {
    "suppliers": "Suppliers",
    "products": "Products",
    "stock_levels": "Stock snapshot",
    "stock_batches": "Stock received (batches)",
    "sales_transactions": "Sales",
    "shrinkage_events": "Shrinkage / loss",
}

# ---------------------------------------------------------------------------
# Bulk CSV import
# ---------------------------------------------------------------------------
with tab_bulk:
    st.caption(
        "Import order matters the first time: **Suppliers → Products** first (products need a "
        "supplier), then Stock / Sales / Shrinkage in any order."
    )
    entity_tabs = st.tabs(list(_ENTITY_LABELS.values()))
    for (entity, label), etab in zip(_ENTITY_LABELS.items(), entity_tabs):
        with etab:
            spec = data_entry.ENTITY_TEMPLATES[entity]
            c1, c2 = st.columns([1, 2])
            with c1:
                st.download_button(
                    f"⬇️ Download {label} CSV template",
                    data=data_entry.csv_template(entity),
                    file_name=f"{entity}_template.csv",
                    mime="text/csv",
                    key=f"tmpl_{entity}",
                )
                st.caption("Required columns: " + ", ".join(spec["required"]))
            uploaded = st.file_uploader(f"Upload {label} CSV", type=["csv"], key=f"upl_{entity}")
            if uploaded is not None:
                try:
                    frame = pd.read_csv(uploaded)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Could not read this CSV: {exc}")
                    frame = None
                if frame is not None:
                    missing = data_entry.validate_columns(entity, frame)
                    if missing:
                        st.error(f"Missing required column(s): {', '.join(missing)}")
                    else:
                        st.write(f"Preview — {len(frame)} row(s):")
                        st.dataframe(frame.head(20), use_container_width=True, hide_index=True)
                        if st.button(f"Import {len(frame)} row(s)", type="primary", key=f"import_{entity}"):
                            with st.spinner("Saving to the database…"):
                                ok, errors = data_entry.import_csv(entity, frame)
                            if ok:
                                st.success(f"Imported {ok} of {len(frame)} row(s) into {label}.")
                            if errors:
                                with st.expander(f"{len(errors)} row(s) skipped — details"):
                                    for e in errors[:50]:
                                        st.write(e)

# ---------------------------------------------------------------------------
# Manual quick entry
# ---------------------------------------------------------------------------
with tab_manual:
    manual_tabs = st.tabs(list(_ENTITY_LABELS.values()))

    # --- Suppliers ---
    with manual_tabs[0]:
        with st.form("form_supplier", clear_on_submit=True):
            name = st.text_input("Supplier name")
            c1, c2 = st.columns(2)
            lead_time = c1.number_input("Lead time (days)", 1, 90, 7)
            reliability = c2.slider("Reliability score", 0.0, 1.0, 0.9, 0.01)
            email = st.text_input("Contact email")
            contract = st.text_input("Contract reference", value="")
            if st.form_submit_button("Save supplier", type="primary"):
                if not name or not email:
                    st.warning("Supplier name and contact email are required.")
                else:
                    res = data_entry.add_supplier(
                        name=name, lead_time_days=int(lead_time), reliability_score=float(reliability),
                        contact_email=email, contract_ref=contract or f"CTR-{name[:3].upper()}",
                    )
                    st.success(f"Saved supplier **{res['name']}** (id {res['id']}).")

    # --- Products ---
    with manual_tabs[1]:
        suppliers = data_entry.list_suppliers()
        if not suppliers:
            st.info("Add at least one supplier first (Suppliers tab) before adding products.")
        else:
            supplier_choice = {f"{s['name']} (id {s['id']})": s["id"] for s in suppliers}
            with st.form("form_product", clear_on_submit=True):
                c1, c2 = st.columns(2)
                sku = c1.text_input("SKU (unique code)")
                pname = c2.text_input("Product name")
                c3, c4 = st.columns(2)
                category = c3.text_input("Category")
                aisle = c4.selectbox("Aisle", data_entry.list_aisles() + ["Other (type below)"])
                if aisle == "Other (type below)":
                    aisle = st.text_input("New aisle name")
                c5, c6 = st.columns(2)
                unit_cost = c5.number_input("Unit cost", 0.0, step=0.01, format="%.2f")
                unit_price = c6.number_input("Unit price", 0.0, step=0.01, format="%.2f")
                is_perishable = st.checkbox("Perishable item")
                shelf_life = st.number_input("Shelf life (days)", 0, 720, 0, disabled=not is_perishable)
                c7, c8 = st.columns(2)
                reorder_point = c7.number_input("Reorder point", 0, 10000, 10)
                safety_stock = c8.number_input("Safety stock", 0, 10000, 5)
                supplier_label = st.selectbox("Supplier", list(supplier_choice))
                if st.form_submit_button("Save product", type="primary"):
                    if not sku or not pname or not category or not aisle:
                        st.warning("SKU, name, category and aisle are required.")
                    else:
                        res = data_entry.add_product(
                            sku=sku, name=pname, category=category, aisle=aisle,
                            unit_cost=float(unit_cost), unit_price=float(unit_price),
                            is_perishable=bool(is_perishable),
                            shelf_life_days=int(shelf_life) if is_perishable else None,
                            reorder_point=int(reorder_point), safety_stock=int(safety_stock),
                            supplier_id=supplier_choice[supplier_label],
                        )
                        st.success(f"Saved product **{res['sku']} — {res['name']}**.")

    # --- Stock snapshot ---
    with manual_tabs[2]:
        products = data_entry.list_products()
        if not products:
            st.info("Add at least one product first before logging stock.")
        else:
            sku_choice = [p["sku"] for p in products]
            with st.form("form_stock_level", clear_on_submit=True):
                sku = st.selectbox("SKU", sku_choice)
                snap_date = st.date_input("Snapshot date", value=dt.date.today())
                c1, c2 = st.columns(2)
                shelf_qty = c1.number_input("Shelf quantity", 0, 100000, 0)
                backroom_qty = c2.number_input("Backroom quantity", 0, 100000, 0)
                on_hand = st.number_input("On-hand total", 0, 200000, int(shelf_qty + backroom_qty))
                if st.form_submit_button("Save stock snapshot", type="primary"):
                    res = data_entry.add_stock_level(
                        sku=sku, snapshot_date=snap_date, shelf_qty=int(shelf_qty),
                        backroom_qty=int(backroom_qty), on_hand_qty=int(on_hand),
                    )
                    st.success(f"Saved stock snapshot for **{res['sku']}** on {res['snapshot_date']}.")

            st.divider()
            st.caption("New delivery arrived? Log it as a batch (also tracks expiry):")
            with st.form("form_stock_batch", clear_on_submit=True):
                sku_b = st.selectbox("SKU ", sku_choice, key="batch_sku")
                batch_no = st.text_input("Batch / delivery number",
                                          value=f"B-{dt.date.today():%Y%m%d}")
                c1, c2 = st.columns(2)
                received = c1.date_input("Received date", value=dt.date.today())
                expiry = c2.date_input("Expiry date", value=dt.date.today() + dt.timedelta(days=14))
                qty_received = st.number_input("Quantity received", 1, 100000, 1)
                if st.form_submit_button("Save batch", type="primary"):
                    res = data_entry.add_stock_batch(
                        sku=sku_b, batch_no=batch_no, received_date=received, expiry_date=expiry,
                        qty_received=int(qty_received), qty_remaining=int(qty_received),
                    )
                    st.success(f"Saved batch **{res['batch_no']}** for {res['sku']}.")

    # --- Sales ---
    with manual_tabs[3]:
        products = data_entry.list_products()
        if not products:
            st.info("Add at least one product first before logging a sale.")
        else:
            price_by_sku = {p["sku"]: p["unit_price"] for p in products}
            sku_choice = list(price_by_sku)
            with st.form("form_sale", clear_on_submit=True):
                sku = st.selectbox("SKU", sku_choice)
                c1, c2 = st.columns(2)
                sale_date = c1.date_input("Date", value=dt.date.today())
                sale_time = c2.time_input("Time", value=dt.datetime.now().time())
                c3, c4, c5 = st.columns(3)
                qty = c3.number_input("Quantity sold", 1, 100000, 1)
                unit_price = c4.number_input("Unit price", 0.0, step=0.01,
                                              value=float(price_by_sku.get(sku, 0.0)), format="%.2f")
                discount = c5.number_input("Discount", 0.0, step=0.01, format="%.2f")
                register_id = st.text_input("Register / till ID", value="REG-1")
                if st.form_submit_button("Save sale", type="primary"):
                    res = data_entry.add_transaction(
                        ts=dt.datetime.combine(sale_date, sale_time), sku=sku, qty=int(qty),
                        unit_price=float(unit_price), discount=float(discount), register_id=register_id,
                    )
                    st.success(f"Saved sale **{res['txn_id']}** — {qty} × {sku}.")

    # --- Shrinkage ---
    with manual_tabs[4]:
        products = data_entry.list_products()
        if not products:
            st.info("Add at least one product first before logging shrinkage.")
        else:
            sku_choice = [p["sku"] for p in products]
            with st.form("form_shrinkage", clear_on_submit=True):
                sku = st.selectbox("SKU", sku_choice, key="shrink_sku")
                event_date = st.date_input("Date", value=dt.date.today(), key="shrink_date")
                qty = st.number_input("Quantity lost", 1, 100000, 1, key="shrink_qty")
                reason = st.selectbox("Reason", ["damage", "theft", "expiry", "admin_error"])
                notes = st.text_area("Notes (optional)")
                if st.form_submit_button("Save shrinkage event", type="primary"):
                    res = data_entry.add_shrinkage_event(
                        sku=sku, event_date=event_date, qty=int(qty), reason=reason,
                        notes=notes or None,
                    )
                    st.success(f"Logged shrinkage of {qty} for **{res['sku']}**.")
