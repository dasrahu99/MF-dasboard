"""
fetcher.py — AMFI NAV data fetcher
Fetches daily NAV for all Indian mutual funds from AMFI's public endpoint.
"""

import requests
import pandas as pd
import sqlite3
import logging
from datetime import date, datetime
from io import StringIO
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

AMFI_URL = "https://www.amfiindia.com/spages/NAVAll.txt"
MFAPI_BASE = "https://api.mfapi.in/mf"
DB_PATH = Path("data/mf_navs.db")


# ── Database setup ─────────────────────────────────────────────────────────────

def init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Create tables if they don't exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nav_history (
            scheme_code   TEXT NOT NULL,
            scheme_name   TEXT,
            category      TEXT,
            nav           REAL NOT NULL,
            nav_date      TEXT NOT NULL,
            fetched_at    TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (scheme_code, nav_date)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fund_metadata (
            scheme_code   TEXT PRIMARY KEY,
            scheme_name   TEXT,
            amc           TEXT,
            category      TEXT,
            sub_category  TEXT,
            isin_growth   TEXT,
            isin_idcw     TEXT,
            updated_at    TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_nav_date ON nav_history(nav_date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_scheme ON nav_history(scheme_code)")
    conn.commit()
    log.info("Database initialised at %s", db_path)
    return conn


# ── AMFI bulk NAV fetch ────────────────────────────────────────────────────────

def parse_amfi_nav(raw_text: str) -> pd.DataFrame:
    """
    Parse AMFI NAVAll.txt format into a clean DataFrame.
    Format: Scheme Code;ISIN Div Payout/ ISIN Growth;ISIN Div Reinvestment;
            Scheme Name;Net Asset Value;Date
    """
    rows = []
    current_amc = None

    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        # AMC header lines have no semicolons
        if ";" not in line:
            current_amc = line
            continue
        parts = line.split(";")
        if len(parts) < 6:
            continue
        try:
            scheme_code = parts[0].strip()
            isin_growth = parts[1].strip()
            scheme_name = parts[3].strip()
            nav_str     = parts[4].strip()
            nav_date    = parts[5].strip()

            if not scheme_code.isdigit():
                continue
            nav = float(nav_str)
            # Normalise date to ISO
            for fmt in ("%d-%b-%Y", "%d/%m/%Y", "%Y-%m-%d"):
                try:
                    nav_date = datetime.strptime(nav_date, fmt).strftime("%Y-%m-%d")
                    break
                except ValueError:
                    pass

            rows.append({
                "scheme_code": scheme_code,
                "amc":         current_amc,
                "isin_growth": isin_growth,
                "scheme_name": scheme_name,
                "nav":         nav,
                "nav_date":    nav_date,
            })
        except (ValueError, IndexError):
            continue

    df = pd.DataFrame(rows)
    log.info("Parsed %d NAV records from AMFI", len(df))
    return df


def fetch_todays_nav() -> pd.DataFrame:
    """Download and parse today's NAV from AMFI."""
    log.info("Fetching NAV data from AMFI...")
    try:
        resp = requests.get(AMFI_URL, timeout=30)
        resp.raise_for_status()
        resp.encoding = "utf-8"
        df = parse_amfi_nav(resp.text)
        return df
    except requests.RequestException as e:
        log.error("Failed to fetch AMFI data: %s", e)
        raise


def save_nav_to_db(df: pd.DataFrame, conn: sqlite3.Connection) -> int:
    """Insert new NAV records; skip duplicates. Returns rows inserted."""
    records = df[["scheme_code", "scheme_name", "nav", "nav_date"]].to_dict("records")
    inserted = 0
    for r in records:
        try:
            conn.execute(
                "INSERT OR IGNORE INTO nav_history (scheme_code, scheme_name, nav, nav_date) "
                "VALUES (:scheme_code, :scheme_name, :nav, :nav_date)",
                r
            )
            inserted += conn.total_changes
        except sqlite3.Error:
            pass
    conn.commit()
    log.info("Saved %d new NAV records to DB", inserted)
    return inserted


# ── Historical NAV via MFApi ───────────────────────────────────────────────────

def fetch_historical_nav(scheme_code: str, conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Fetch full NAV history for a single fund from mfapi.in.
    Stores results in DB. Returns DataFrame sorted by date.
    """
    url = f"{MFAPI_BASE}/{scheme_code}"
    log.info("Fetching history for scheme %s", scheme_code)
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.error("Error fetching history for %s: %s", scheme_code, e)
        return pd.DataFrame()

    name = data.get("meta", {}).get("scheme_name", "")
    records = []
    for entry in data.get("data", []):
        try:
            nav_date = datetime.strptime(entry["date"], "%d-%m-%Y").strftime("%Y-%m-%d")
            records.append({
                "scheme_code": str(scheme_code),
                "scheme_name": name,
                "nav":         float(entry["nav"]),
                "nav_date":    nav_date,
            })
        except (ValueError, KeyError):
            continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).sort_values("nav_date").reset_index(drop=True)
    df["nav_date"] = pd.to_datetime(df["nav_date"])

    # Bulk insert
    conn.executemany(
        "INSERT OR IGNORE INTO nav_history (scheme_code, scheme_name, nav, nav_date) "
        "VALUES (:scheme_code, :scheme_name, :nav, :nav_date)",
        records
    )
    conn.commit()
    log.info("Saved %d historical records for %s", len(records), scheme_code)
    return df


def get_nav_history(scheme_code: str, conn: sqlite3.Connection,
                    from_date: str = None) -> pd.DataFrame:
    """Load NAV history from DB for a given fund."""
    query = "SELECT nav_date, nav FROM nav_history WHERE scheme_code = ?"
    params = [scheme_code]
    if from_date:
        query += " AND nav_date >= ?"
        params.append(from_date)
    query += " ORDER BY nav_date ASC"
    df = pd.read_sql_query(query, conn, params=params)
    df["nav_date"] = pd.to_datetime(df["nav_date"])
    return df