# Inventory Count and Discrepancy SOP

**Document type:** SOP · **Owner:** Inventory and Supply Chain Manager · **Review cycle:** quarterly

## Purpose

This procedure defines cycle-count frequency, the tolerance thresholds that
separate normal variance from a reportable discrepancy, and how discrepancies
are escalated and resolved. The operations assistant references it when it
detects a shelf/backroom discrepancy or an unexplained stock movement.

## Cycle count frequency

Counting is continuous and risk-based:

- **High-velocity SKUs** — weekly. These are mostly Beverages, Snacks and
  fresh Dairy, where a full facing can sell between counts.
- **Mid-velocity SKUs** — every two weeks.
- **Low-velocity SKUs** — monthly. Much of Household and Personal Care.
- **Full aisle count** — one aisle per week on rotation, so all eight aisles
  are fully counted at least once per quarter.
- **Trigger counts** — any SKU flagged by a stockout, a shrinkage event, a
  customer quality complaint, or a delivery discrepancy is counted that shift
  regardless of its schedule.

Counts are entered into the stockroom system the same shift. The system
compares the count to the recorded on-hand (shelf plus backroom) figure.

## Tolerance thresholds

A difference within tolerance is recorded and cleared. A difference outside
tolerance is a **discrepancy** and is escalated.

| SKU type | Tolerance (whichever is greater) |
|---|---|
| Low unit cost (< 5.00) | 3% of on-hand, or 2 units |
| Standard | 2% of on-hand, or 1 unit |
| High unit cost (> 25.00) | 1% of on-hand, or 1 unit |
| Perishable, any cost | 1 unit — perishables are counted tightly because loss is expected to show as expiry, not variance |

Two consecutive counts outside tolerance for the same SKU is always a
discrepancy even if each single count is marginal.

## Discrepancy escalation

1. **Recount.** A different associate recounts within the shift. Most
   discrepancies are miscounts and clear here.
2. **Reconcile paperwork.** Check recent deliveries (short or over shipment),
   transfers, returns, and register voids for an admin_error explanation.
3. **Record the loss.** If the recount confirms a real shortage and no
   paperwork explains it, record it as shrinkage with the best-supported
   reason code — usually admin_error, or theft if there is corroborating
   evidence — following the Shrinkage and Loss Prevention SOP.
4. **Escalate.** Discrepancies above tolerance go Shift Lead → Store Manager.
   A discrepancy above 250.00 at cost, or a third discrepancy on the same SKU
   in a quarter, also goes to the Inventory and Supply Chain Manager and Loss
   Prevention.

## Root-cause expectations

When investigating a discrepancy or a stockout, work the evidence in this
order: sales history and velocity; stock trajectory and recent deliveries;
shrinkage events by reason; supplier on-time performance and lead-time
breaches; and finally the forecast versus what actually happened. A stockout
that follows an on-time delivery, or a discrepancy with no shrinkage events
and no paperwork error, is escalated with all of that evidence attached rather
than closed as "unknown".

## Records

Every count, recount, discrepancy and resolution is retained for 12 months and
reviewed monthly for SKUs that discrepate repeatedly, which usually indicates
a process problem (receiving, scanning, or planogram) rather than theft.
