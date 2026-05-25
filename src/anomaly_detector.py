import pathlib
import sys

import pandas as pd

_root = pathlib.Path(__file__).resolve().parent
while not (_root / "config.py").exists() and _root != _root.parent:
    _root = _root.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import IQR_MULTIPLIER, Z_SCORE_THRESHOLD


class AnomalyDetector:
    """Detect unusual retail transactions with IQR + Z-score consensus."""

    def __init__(self, df: pd.DataFrame, amount_col: str = "net_sales_amount"):
        self.df = df.copy()
        if amount_col in self.df.columns:
            self.amount_col = amount_col
        elif "total_amount" in self.df.columns:
            self.amount_col = "total_amount"
        else:
            self.amount_col = self.df.select_dtypes(include="number").columns[0]
        self.anomalies = pd.DataFrame()
        self.confirmed = pd.DataFrame()
        self.results = {}
        self._n_checked = 0
        self.bounds = {}

    def add_amount_scores(self) -> "AnomalyDetector":
        values = pd.to_numeric(self.df[self.amount_col], errors="coerce")

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - IQR_MULTIPLIER * iqr
        upper = q3 + IQR_MULTIPLIER * iqr

        mean = values.mean()
        std = values.std(ddof=0)
        z_score = pd.Series(0.0, index=self.df.index) if std == 0 else (values - mean) / std

        self.df["amount_iqr_outlier"] = (values < lower) | (values > upper)
        self.df["amount_z_score"] = z_score
        self.df["amount_z_outlier"] = z_score.abs() > Z_SCORE_THRESHOLD
        self.df["amount_anomaly"] = (
            self.df["amount_iqr_outlier"] & self.df["amount_z_outlier"]
        )
        anomaly_count = int(self.df["amount_anomaly"].sum())
        self.results[self.amount_col] = {
            "iqr_flagged": int(self.df["amount_iqr_outlier"].sum()),
            "zscore_flagged": int(self.df["amount_z_outlier"].sum()),
            "confirmed_anomalies": anomaly_count,
            "anomaly_pct": round(anomaly_count / len(self.df) * 100, 2),
        }
        self._n_checked = 1

        self.bounds = {
            "amount_col": self.amount_col,
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr),
            "lower": float(lower),
            "upper": float(upper),
            "mean": float(mean),
            "std": float(std),
        }
        return self

    def flag_retail_anomalies(self) -> "AnomalyDetector":
        if "amount_anomaly" not in self.df.columns:
            self.add_amount_scores()

        has_return = (
            self.df["has_return"].astype(bool)
            if "has_return" in self.df.columns
            else pd.Series(False, index=self.df.index)
        )
        needs_reorder = (
            self.df["needs_reorder"].astype(bool)
            if "needs_reorder" in self.df.columns
            else pd.Series(False, index=self.df.index)
        )

        self.df["return_with_high_value"] = has_return & self.df["amount_anomaly"]
        self.df["reorder_high_value_sale"] = needs_reorder & self.df["amount_anomaly"]

        self.anomalies = self.df[
            self.df["amount_anomaly"]
            | self.df["return_with_high_value"]
            | self.df["reorder_high_value_sale"]
        ].copy()
        self.confirmed = self.anomalies.copy()
        return self

    def run(self, columns: list = None) -> "AnomalyDetector":
        """Backward-compatible notebook API for IQR + Z-score detection."""
        if columns:
            usable_columns = [
                col for col in columns
                if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col])
            ]
            if usable_columns:
                self.amount_col = usable_columns[0]
        return self.add_amount_scores().flag_retail_anomalies()

    def save_anomalies(self) -> "AnomalyDetector":
        """Backward-compatible helper used by older notebooks/run scripts."""
        from config import ANOMALIES_PATH

        if "amount_anomaly" not in self.df.columns:
            self.run()
        self.confirmed.to_csv(ANOMALIES_PATH, index=False)
        return self

    def get_anomalies(self) -> pd.DataFrame:
        if self.anomalies.empty and "amount_anomaly" not in self.df.columns:
            self.flag_retail_anomalies()
        return self.anomalies.copy()

    def summary(self) -> pd.DataFrame:
        if "amount_anomaly" not in self.df.columns:
            self.flag_retail_anomalies()

        return pd.DataFrame([{
            "rows_scanned": len(self.df),
            "amount_column": self.amount_col,
            "iqr_lower_bound": round(self.bounds.get("lower", 0), 2),
            "iqr_upper_bound": round(self.bounds.get("upper", 0), 2),
            "amount_anomalies": int(self.df["amount_anomaly"].sum()),
            "return_with_high_value": int(self.df["return_with_high_value"].sum()),
            "reorder_high_value_sale": int(self.df["reorder_high_value_sale"].sum()),
            "total_flagged_rows": len(self.get_anomalies()),
        }])

    def summary_dict(self) -> dict:
        return self.summary().iloc[0].to_dict()
