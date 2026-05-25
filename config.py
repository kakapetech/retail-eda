import logging
import os
import pathlib

from dotenv import load_dotenv


load_dotenv()

PROJECT_NAME = "P01 Retail EDA"
COMPANY = "ShopSmart Retail Group"
INDUSTRY = os.getenv("INDUSTRY", "retail")

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

LOCAL_DATA_PATH = DATA_DIR / "processed-data.csv"
ATTACHED_DATA_PATH = pathlib.Path(
    r"C:\retail-etl-pipeline\data\processed\processed-data.csv"
)
DATA_PATH = LOCAL_DATA_PATH if LOCAL_DATA_PATH.exists() else ATTACHED_DATA_PATH
PROCESSED_DATA_PATH = DATA_PATH

ANALYSIS_REPORT_PATH = REPORTS_DIR / "analysis_report.txt"
ANOMALIES_PATH = REPORTS_DIR / "anomalies.csv"
REVENUE_RANKING_PATH = REPORTS_DIR / "revenue_rankings.csv"
UNDERPERFORMING_STORES_PATH = REPORTS_DIR / "underperforming_stores.csv"

DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

TOP_N_GROUPS = 10
CORRELATION_THRESHOLD = 0.30
IQR_MULTIPLIER = 1.5
Z_SCORE_THRESHOLD = 3.0


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("retail_eda")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)

    return logger


logger = _setup_logger()
