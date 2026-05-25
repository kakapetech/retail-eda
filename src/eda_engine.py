import pathlib
import sys

import pandas as pd

_root = pathlib.Path(__file__).resolve().parent
while not (_root / "config.py").exists() and _root != _root.parent:
    _root = _root.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import (
    ANALYSIS_REPORT_PATH,
    ANOMALIES_PATH,
    COMPANY,
    CORRELATION_THRESHOLD,
    DATA_PATH,
    REVENUE_RANKING_PATH,
    TOP_N_GROUPS,
    UNDERPERFORMING_STORES_PATH,
    logger,
)
from src.anomaly_detector import AnomalyDetector


class _ProfileResults(dict):
    """Dictionary of profile results that also supports the old .profile() call."""

    def __init__(self, engine: "EDAEngine"):
        super().__init__()
        self._engine = engine

    def __call__(self) -> "EDAEngine":
        return self._engine.profile_data()


class EDAEngine:
    """Run retail EDA for revenue, discount impact, and transaction anomalies."""

    def __init__(self, data_path=DATA_PATH):
        self.data_path = pathlib.Path(data_path)
        self.df = pd.DataFrame()
        self.profile = _ProfileResults(self)
        self.results = {}
        self.num_cols = []
        self.cat_cols = []
        self.revenue_rankings = {}
        self.underperforming_stores = pd.DataFrame()
        self.discount_result = {}
        self.anomalies = pd.DataFrame()
        self.anomaly_summary = {}

    def load(self) -> "EDAEngine":
        if not self.data_path.exists():
            raise FileNotFoundError(f"Processed retail data not found: {self.data_path}")

        self.df = pd.read_csv(self.data_path, low_memory=False)

        for col in ["sale_date", "last_restocked", "return_date"]:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors="coerce")

        for col in ["store_is_active", "has_return", "needs_reorder"]:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.lower().isin(
                    ["true", "1", "yes", "y"]
                )

        if "sale_year" not in self.df.columns and "sale_date" in self.df.columns:
            self.df["sale_year"] = self.df["sale_date"].dt.year
        if "sale_month" not in self.df.columns and "sale_date" in self.df.columns:
            self.df["sale_month"] = self.df["sale_date"].dt.month

        analysis_df = self.df.drop(
            columns=[c for c in self.df.columns if c.startswith("_")],
            errors="ignore",
        )
        self.num_cols = analysis_df.select_dtypes(include="number").columns.tolist()
        self.cat_cols = analysis_df.select_dtypes(include=["object", "string", "bool"]).columns.tolist()

        logger.info(f"Loaded {len(self.df):,} rows from {self.data_path}")
        return self

    def profile_data(self) -> "EDAEngine":
        profile = {
            "rows": len(self.df),
            "columns": len(self.df.columns),
            "numeric_cols": len(self.num_cols),
            "categorical_cols": len(self.cat_cols),
            "total_nulls": int(self.df.isna().sum().sum()),
            "null_pct": round(self.df.isna().sum().sum() / self.df.size * 100, 2),
            "memory_mb": round(self.df.memory_usage(deep=True).sum() / 1024**2, 2),
            "duplicates": int(self.df.duplicated().sum()),
            "unique_sales": int(self.df["sale_id"].nunique()),
            "unique_stores": int(self.df["store_id"].nunique()),
            "unique_products": int(self.df["product_id"].nunique()),
            "total_revenue": float(self._amount_series().sum()),
            "avg_revenue_per_sale": float(self._amount_series().mean()),
            "total_returns": int(self.df["has_return"].sum()) if "has_return" in self.df.columns else 0,
            "return_rate": float(self.df["has_return"].mean()) if "has_return" in self.df.columns else 0.0,
            "avg_discount_pct": float(pd.to_numeric(self.df["discount_pct"], errors="coerce").mean()),
        }
        self.profile.clear()
        self.profile.update(profile)
        self.results["profile"] = dict(self.profile)
        return self

    def group_analysis(self) -> "EDAEngine":
        amount_col = self._amount_col()
        group_specs = {
            "category": "category",
            "store": "store_name",
            "category_store": ["category", "store_name"],
        }

        for name, group_cols in group_specs.items():
            table = (
                self.df.groupby(group_cols, dropna=False)
                .agg(
                    sale_count=("sale_id", "count"),
                    total_revenue=(amount_col, "sum"),
                    avg_revenue_per_sale=(amount_col, "mean"),
                    median_revenue_per_sale=(amount_col, "median"),
                    total_quantity=("sale_quantity", "sum"),
                    avg_discount_pct=("discount_pct", "mean"),
                    return_count=("has_return", "sum") if "has_return" in self.df.columns else ("sale_id", "count"),
                )
                .reset_index()
            )
            table["revenue_rank"] = table["avg_revenue_per_sale"].rank(
                ascending=False, method="dense"
            ).astype(int)
            self.revenue_rankings[name] = table.sort_values(
                ["avg_revenue_per_sale", "total_revenue"], ascending=False
            )

        self._find_underperforming_stores()
        self.results["group_analysis"] = {}
        for name, table in self.revenue_rankings.items():
            notebook_table = table.copy()
            notebook_table["mean"] = notebook_table["avg_revenue_per_sale"]
            notebook_table["median"] = notebook_table["median_revenue_per_sale"]
            notebook_table["count"] = notebook_table["sale_count"]
            notebook_table["rank"] = notebook_table["revenue_rank"]
            self.results["group_analysis"][name] = {
                "avg_revenue_per_sale": notebook_table.to_dict(orient="records")
            }
        return self

    def correlation(self) -> "EDAEngine":
        discount = pd.to_numeric(self.df["discount_pct"], errors="coerce")
        total = pd.to_numeric(self.df["total_amount"], errors="coerce")
        quantity = pd.to_numeric(self.df["sale_quantity"], errors="coerce")

        self.discount_result = {
            "discount_total_amount_corr": float(discount.corr(total)),
            "discount_quantity_corr": float(discount.corr(quantity)),
            "threshold": CORRELATION_THRESHOLD,
            "interpretation": self._discount_interpretation(discount.corr(total)),
        }

        numeric_df = self.df[self.num_cols] if self.num_cols else self.df.select_dtypes(include="number")
        corr_matrix = numeric_df.corr(numeric_only=True).round(3)
        strong_pairs = []
        numeric_cols = corr_matrix.columns.tolist()
        for i, col_a in enumerate(numeric_cols):
            for j, col_b in enumerate(numeric_cols):
                if i >= j:
                    continue

                correlation = corr_matrix.loc[col_a, col_b]
                if pd.isna(correlation) or abs(correlation) < CORRELATION_THRESHOLD:
                    continue

                if abs(correlation) >= 0.7:
                    strength = "STRONG"
                elif abs(correlation) >= 0.5:
                    strength = "MODERATE"
                else:
                    strength = "WEAK"

                strong_pairs.append({
                    "col_a": col_a,
                    "col_b": col_b,
                    "correlation": float(correlation),
                    "strength": strength,
                    "direction": "positive" if correlation > 0 else "negative",
                })

        strong_pairs.sort(key=lambda pair: abs(pair["correlation"]), reverse=True)

        self.results["correlation"] = {
            "matrix": corr_matrix.to_dict(),
            "strong_pairs": strong_pairs,
            "discount_impact": self.discount_result,
        }
        return self

    def detect_anomalies(self) -> "EDAEngine":
        detector = AnomalyDetector(self.df)
        detector.add_amount_scores().flag_retail_anomalies()
        self.anomalies = detector.get_anomalies()
        self.anomaly_summary = detector.summary_dict()
        return self

    def save_outputs(self) -> "EDAEngine":
        revenue_export = self.revenue_rankings.get("category_store", pd.DataFrame())
        revenue_export.to_csv(REVENUE_RANKING_PATH, index=False)
        self.underperforming_stores.to_csv(UNDERPERFORMING_STORES_PATH, index=False)
        self.anomalies.to_csv(ANOMALIES_PATH, index=False)
        self.write_report()
        return self

    def write_report(self) -> None:
        category_rank = self.revenue_rankings.get("category", pd.DataFrame())
        store_rank = self.revenue_rankings.get("store", pd.DataFrame())

        lines = [
            "SHOPSMART RETAIL EDA REPORT",
            "=" * 60,
            f"Company: {COMPANY}",
            f"Data source: {self.data_path}",
            "",
            "DATA PROFILE",
            f"Rows: {self.profile['rows']:,}",
            f"Unique sales: {self.profile['unique_sales']:,}",
            f"Unique stores: {self.profile['unique_stores']:,}",
            f"Unique products: {self.profile['unique_products']:,}",
            f"Total revenue: {self.profile['total_revenue']:,.2f}",
            f"Average revenue per sale: {self.profile['avg_revenue_per_sale']:,.2f}",
            f"Return rate: {self.profile['return_rate']:.2%}",
            "",
            "DISCOUNT IMPACT",
            f"Pearson correlation discount_pct vs total_amount: {self.discount_result['discount_total_amount_corr']:.4f}",
            f"Pearson correlation discount_pct vs sale_quantity: {self.discount_result['discount_quantity_corr']:.4f}",
            f"Interpretation: {self.discount_result['interpretation']}",
            "",
            "ANOMALY SUMMARY",
        ]

        for key, value in self.anomaly_summary.items():
            lines.append(f"{key}: {value:,}" if isinstance(value, int) else f"{key}: {value}")

        lines.extend(["", "TOP CATEGORIES BY AVERAGE REVENUE PER SALE"])
        lines.append(category_rank.head(TOP_N_GROUPS).to_string(index=False))

        lines.extend(["", "TOP STORES BY AVERAGE REVENUE PER SALE"])
        lines.append(store_rank.head(TOP_N_GROUPS).to_string(index=False))

        lines.extend(["", "UNDERPERFORMING STORES RELATIVE TO CATEGORY AVERAGE"])
        if self.underperforming_stores.empty:
            lines.append("No underperforming category-store combinations found.")
        else:
            lines.append(self.underperforming_stores.head(25).to_string(index=False))

        ANALYSIS_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Wrote report to {ANALYSIS_REPORT_PATH}")

    def run_all(self) -> "EDAEngine":
        return (
            self.load()
            .profile_data()
            .group_analysis()
            .correlation()
            .detect_anomalies()
            .save_outputs()
        )

    def _find_underperforming_stores(self) -> None:
        if "category_store" not in self.revenue_rankings:
            self.underperforming_stores = pd.DataFrame()
            return

        category_store = self.revenue_rankings["category_store"].copy()
        category_avg = self.revenue_rankings["category"][
            ["category", "avg_revenue_per_sale"]
        ].rename(columns={"avg_revenue_per_sale": "category_avg_revenue_per_sale"})

        merged = category_store.merge(category_avg, on="category", how="left")
        merged["revenue_gap_vs_category_avg"] = (
            merged["avg_revenue_per_sale"] - merged["category_avg_revenue_per_sale"]
        )
        merged["pct_gap_vs_category_avg"] = (
            merged["revenue_gap_vs_category_avg"]
            / merged["category_avg_revenue_per_sale"]
            * 100
        )
        self.underperforming_stores = merged[
            merged["revenue_gap_vs_category_avg"] < 0
        ].sort_values("revenue_gap_vs_category_avg")

    def _amount_col(self) -> str:
        return "net_sales_amount" if "net_sales_amount" in self.df.columns else "total_amount"

    def _amount_series(self) -> pd.Series:
        return pd.to_numeric(self.df[self._amount_col()], errors="coerce")

    @staticmethod
    def _discount_interpretation(correlation: float) -> str:
        if correlation < -0.3:
            return "Higher discounts are associated with lower transaction totals."
        if correlation > 0.3:
            return "Higher discounts are associated with higher transaction totals."
        return "Discounts have weak linear relationship with transaction totals."
