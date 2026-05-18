# P01 ⭐ — Retail EDA
## The Darko Method 2026 | Student Project

---

## Your Brief

**Company:** ShopSmart Retail Group
**Your role:** Junior Data Analyst

You have `processed-data.csv` from your Module 05 ETL pipeline. The Head of
Merchandising has three urgent questions before the quarterly board meeting.

**Question 1 — Revenue performance:** Which product categories and stores generate
the highest revenue per sale? Use groupby to rank them. Which stores are
underperforming relative to the category average?

**Question 2 — Discount impact:** Does offering higher discounts actually correlate
with higher sales volume? Compute the Pearson correlation between `discount_pct`
and `total_amount`. A negative correlation means higher discounts produce lower totals
— the company is losing margin for nothing.

**Question 3 — Anomaly detection:** Use IQR and Z-score consensus to find
transactions that are statistically unusual. Save these to `reports/anomalies.csv`.

**Input:** `data/processed-data.csv` (from Module 05)
**Output:** `reports/analysis_report.txt` + `reports/anomalies.csv`

---

## Success Criteria

- [ ] `EDAEngine` with load, profile, group_analysis, correlation, report methods
- [ ] `AnomalyDetector` using IQR + Z-score consensus
- [ ] `analysis_report.txt` shows revenue ranked by category and store
- [ ] `anomalies.csv` contains flagged transactions
- [ ] 8+ unit tests pass
- [ ] Project pushed to GitHub

---

> Build from scratch using the teaching project as your reference.
