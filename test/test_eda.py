import pathlib
import sys
import tempfile

import pandas as pd

_root = pathlib.Path(__file__).resolve().parent
while not (_root / "config.py").exists() and _root != _root.parent:
    _root = _root.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.anomaly_detector import AnomalyDetector
from src.eda_engine import EDAEngine


def make_df() -> pd.DataFrame:
    return pd.DataFrame({
        "sale_id": range(1, 13),
        "store_id": [1, 1, 2, 2, 3, 3, 1, 2, 3, 1, 2, 3],
        "product_id": range(101, 113),
        "sale_date": ["2024-01-01"] * 12,
        "sale_quantity": [2, 3, 1, 5, 2, 4, 9, 1, 2, 3, 4, 1],
        "sale_unit_price": [100, 120, 80, 40, 300, 60, 999, 35, 25, 50, 70, 90],
        "discount_pct": [0, 5, 10, 0, 20, 15, 50, 0, 5, 10, 0, 30],
        "total_amount": [200, 360, 80, 200, 600, 240, 8991, 35, 50, 150, 280, 90],
        "category": ["Electronics", "Electronics", "Beauty", "Beauty", "Sports", "Sports", "Electronics", "Beauty", "Sports", "Food", "Food", "Food"],
        "store_name": ["Accra", "Accra", "Kumasi", "Kumasi", "Tamale", "Tamale", "Accra", "Kumasi", "Tamale", "Accra", "Kumasi", "Tamale"],
        "has_return": [False, False, True, False, False, True, False, False, False, False, True, False],
        "net_sales_amount": [200, 360, 60, 200, 600, 200, 8991, 35, 50, 150, 200, 90],
        "gross_profit": [80, 150, 20, 70, 250, 90, 3000, 10, 8, 35, 60, 20],
        "gross_margin_pct": [40, 41.7, 25, 35, 41.7, 37.5, 33.4, 28.6, 16, 23.3, 21.4, 22.2],
        "needs_reorder": [False, False, True, False, True, False, False, False, True, False, False, True],
        "sale_year": [2024] * 12,
        "sale_month": [1] * 12,
    })


def test_anomaly_detector_adds_amount_scores():
    detector = AnomalyDetector(make_df()).add_amount_scores()
    assert "amount_anomaly" in detector.df.columns


def test_anomaly_detector_summary_has_total_flagged_rows():
    summary = AnomalyDetector(make_df()).flag_retail_anomalies().summary()
    assert "total_flagged_rows" in summary


def test_engine_loads_csv():
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "processed-data.csv"
        make_df().to_csv(path, index=False)
        engine = EDAEngine(data_path=path).load()
        assert len(engine.df) == 12


def test_engine_profiles_data():
    engine = EDAEngine()
    engine.df = make_df()
    engine.profile_data()
    assert engine.profile["unique_sales"] == 12


def test_engine_group_analysis_creates_category_rankings():
    engine = EDAEngine()
    engine.df = make_df()
    engine.group_analysis()
    assert "category" in engine.revenue_rankings


def test_engine_group_analysis_creates_store_rankings():
    engine = EDAEngine()
    engine.df = make_df()
    engine.group_analysis()
    assert "store" in engine.revenue_rankings


def test_engine_finds_underperforming_stores():
    engine = EDAEngine()
    engine.df = make_df()
    engine.group_analysis()
    assert isinstance(engine.underperforming_stores, pd.DataFrame)


def test_engine_correlation_returns_discount_result():
    engine = EDAEngine()
    engine.df = make_df()
    engine.correlation()
    assert "discount_total_amount_corr" in engine.discount_result


def test_engine_detect_anomalies_sets_dataframe():
    engine = EDAEngine()
    engine.df = make_df()
    engine.detect_anomalies()
    assert isinstance(engine.anomalies, pd.DataFrame)


def test_pipeline_runs_end_to_end_with_temp_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "processed-data.csv"
        make_df().to_csv(path, index=False)
        engine = EDAEngine(data_path=path)
        engine.load().profile_data().group_analysis().correlation().detect_anomalies()
        assert engine.profile["rows"] == 12


if __name__ == "__main__":
    tests = [
        test_anomaly_detector_adds_amount_scores,
        test_anomaly_detector_summary_has_total_flagged_rows,
        test_engine_loads_csv,
        test_engine_profiles_data,
        test_engine_group_analysis_creates_category_rankings,
        test_engine_group_analysis_creates_store_rankings,
        test_engine_finds_underperforming_stores,
        test_engine_correlation_returns_discount_result,
        test_engine_detect_anomalies_sets_dataframe,
        test_pipeline_runs_end_to_end_with_temp_file,
    ]
    passed = 0
    for test_fn in tests:
        test_fn()
        print(f"PASS: {test_fn.__name__}")
        passed += 1
    print(f"{passed} test(s) passed.")
