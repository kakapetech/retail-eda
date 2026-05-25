import pathlib
import sys

_root = pathlib.Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import (
    ANALYSIS_REPORT_PATH,
    ANOMALIES_PATH,
    REVENUE_RANKING_PATH,
    UNDERPERFORMING_STORES_PATH,
    logger,
)
from src.eda_engine import EDAEngine


def main() -> None:
    logger.info("=" * 60)
    logger.info("  P01 RETAIL EDA STARTING")
    logger.info("=" * 60)

    engine = EDAEngine().run_all()

    logger.info("Retail EDA complete.")
    logger.info(f"Report: {ANALYSIS_REPORT_PATH}")
    logger.info(f"Anomalies: {ANOMALIES_PATH}")
    logger.info(f"Revenue rankings: {REVENUE_RANKING_PATH}")
    logger.info(f"Underperforming stores: {UNDERPERFORMING_STORES_PATH}")

    print()
    print("RETAIL EDA COMPLETE")
    print(f"Rows analyzed: {engine.profile['rows']:,}")
    print(f"Average revenue per sale: {engine.profile['avg_revenue_per_sale']:,.2f}")
    print(f"Discount vs total_amount correlation: {engine.discount_result['discount_total_amount_corr']:.4f}")
    print(f"Anomalies flagged: {len(engine.anomalies):,}")


if __name__ == "__main__":
    main()
