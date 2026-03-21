"""
fetcher.py — AMFI NAV data fetcher
Fetches daily NAV for all Indian mutual funds from AMFI's public endpoint.

Fix 1 applied: fetch_historical_nav() now has retry-with-exponential-backoff
and a polite inter-request delay to avoid rate-limiting mfapi.in.
"""

import random
import time
import requests
import pandas as pd
import sqlite3
import logging
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

AMFI_URL   = "https://www.amfiindia.com/spages/NAVAll.txt"
MFAPI_BASE = "https://api.mfapi.in/mf"
DB_PATH    = Path("data/mf_navs.db")

# Minimum NAV records required before we trust a fund's history
MIN_HISTORY_RECORDS = 250   # ~1 year of trading days

# Polite delay between mfapi requests (seconds)
REQUEST_DELAY     = 0.35
REQUEST_DELAY_JITTER = 0.15   # random extra 0–0.15s


# ── Database setup ────────────────────────────────────────────────────────────

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
    # Discovery tracking table — records which codes have been attempted
    conn.execute("""
        CREATE TABLE IF NOT EXISTS discovery_log (
            scheme_code   TEXT PRIMARY KEY,
            scheme_name   TEXT,
            attempted_at  TEXT DEFAULT (datetime('now')),
            records_saved INTEGER DEFAULT 0,
            status        TEXT DEFAULT 'pending'
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_nav_date ON nav_history(nav_date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_scheme   ON nav_history(scheme_code)")
    conn.commit()
    log.info("Database initialised at %s", db_path)
    return conn


# ── AMFI bulk NAV fetch ───────────────────────────────────────────────────────

def parse_amfi_nav(raw_text: str) -> pd.DataFrame:
    """
    Parse AMFI NAVAll.txt format into a clean DataFrame.
    Format: Scheme Code;ISIN Growth;ISIN IDCW;Scheme Name;NAV;Date
    Also infers category from scheme name at parse time (Fix 2).
    """
    rows = []
    current_amc = None

    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
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

            for fmt in ("%d-%b-%Y", "%d/%m/%Y", "%Y-%m-%d"):
                try:
                    nav_date = datetime.strptime(nav_date, fmt).strftime("%Y-%m-%d")
                    break
                except ValueError:
                    pass

            # ── Fix 2: infer category at parse time from scheme name ──
            category = infer_category(scheme_name)

            rows.append({
                "scheme_code": scheme_code,
                "amc":         current_amc,
                "isin_growth": isin_growth,
                "scheme_name": scheme_name,
                "category":    category,
                "nav":         nav,
                "nav_date":    nav_date,
            })
        except (ValueError, IndexError):
            continue

    df = pd.DataFrame(rows)
    log.info("Parsed %d NAV records from AMFI", len(df))
    return df


def infer_category(scheme_name: str) -> str:
    """
    Fix 2: Derive fund category from scheme name string.
    Called at AMFI parse time so category is stored in the DB,
    never inferred at display time.
    """
    name = scheme_name.lower()
    if "small cap" in name:                          return "Small Cap"
    if "mid cap" in name or "midcap" in name:        return "Mid Cap"
    if "large & mid" in name:                        return "Large & Mid Cap"
    if "large cap" in name or "largecap" in name or "bluechip" in name or "frontline" in name:
                                                     return "Large Cap"
    if "flexi" in name or "multi cap" in name or "multicap" in name:
                                                     return "Flexi Cap"
    if "elss" in name or "tax saver" in name or "tax saving" in name:
                                                     return "ELSS"
    if "index" in name or "nifty" in name or "sensex" in name or "bse" in name:
                                                     return "Index"
    if "hybrid" in name or "balanced" in name or "equity savings" in name:
                                                     return "Hybrid"
    if "sectoral" in name or "thematic" in name or "infrastructure" in name \
       or "banking" in name or "pharma" in name or "technology" in name \
       or "fmcg" in name or "psu" in name or "digital" in name:
                                                     return "Sectoral"
    if "international" in name or "global" in name or "overseas" in name:
                                                     return "International"
    if "debt" in name or "bond" in name or "gilt" in name \
       or "liquid" in name or "money market" in name or "overnight" in name:
                                                     return "Debt"
    if "fof" in name or "fund of fund" in name:      return "FoF"
    return "Equity"   # broad fallback — still accurate for equity funds


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
    """Insert new NAV records + fund_metadata; skip duplicates."""
    # Save NAV records
    records = df[["scheme_code", "scheme_name", "category", "nav", "nav_date"]].to_dict("records")
    inserted = 0
    for r in records:
        try:
            conn.execute(
                "INSERT OR IGNORE INTO nav_history "
                "(scheme_code, scheme_name, category, nav, nav_date) "
                "VALUES (:scheme_code, :scheme_name, :category, :nav, :nav_date)",
                r
            )
            inserted += conn.total_changes
        except sqlite3.Error:
            pass

    # Upsert fund_metadata with category
    meta_records = df[["scheme_code","scheme_name","amc","category"]].drop_duplicates(
        subset="scheme_code").to_dict("records")
    for m in meta_records:
        try:
            conn.execute(
                "INSERT OR REPLACE INTO fund_metadata (scheme_code, scheme_name, amc, category) "
                "VALUES (:scheme_code, :scheme_name, :amc, :category)",
                m
            )
        except sqlite3.Error:
            pass

    conn.commit()
    log.info("Saved %d new NAV records to DB", inserted)
    return inserted


# ── Historical NAV via MFApi — with retry & rate limiting (Fix 1) ─────────────

def fetch_historical_nav(scheme_code: str,
                          conn: sqlite3.Connection,
                          retries: int = 3,
                          skip_if_sufficient: bool = True) -> pd.DataFrame:
    """
    Fetch full NAV history for a single fund from mfapi.in.

    Fix 1 — Rate limiting & retry:
      - Polite delay (0.35s + jitter) between every request
      - Exponential backoff on failure (1s, 2s, 4s)
      - Returns empty DataFrame after all retries exhausted (never crashes pipeline)

    Fix 1 — skip_if_sufficient:
      - If DB already has MIN_HISTORY_RECORDS for this fund, skip the API call entirely
      - This is the key speed optimisation for daily runs
    """
    # Check cache first
    if skip_if_sufficient:
        cached = get_nav_history(scheme_code, conn)
        if len(cached) >= MIN_HISTORY_RECORDS:
            log.debug("Skipping API — %d cached records for %s", len(cached), scheme_code)
            return cached

    url = f"{MFAPI_BASE}/{scheme_code}"
    log.info("Fetching history for scheme %s", scheme_code)

    last_error = None
    for attempt in range(retries):
        try:
            # Polite delay — never hammer mfapi.in
            sleep_time = REQUEST_DELAY + random.uniform(0, REQUEST_DELAY_JITTER)
            if attempt > 0:
                backoff = (2 ** attempt)   # 1s, 2s, 4s
                log.warning("Retry %d/%d for %s — waiting %ds",
                            attempt + 1, retries, scheme_code, backoff)
                time.sleep(backoff)
            else:
                time.sleep(sleep_time)

            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            # Parse
            name    = data.get("meta", {}).get("scheme_name", "")
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
                log.warning("Empty data returned for scheme %s", scheme_code)
                _log_discovery(conn, scheme_code, name, 0, "empty")
                return pd.DataFrame()

            # Validate minimum data threshold
            if len(records) < MIN_HISTORY_RECORDS:
                log.warning("Only %d records for %s — below minimum threshold of %d",
                            len(records), scheme_code, MIN_HISTORY_RECORDS)
                _log_discovery(conn, scheme_code, name, len(records), "insufficient")
                # Still save what we have — might be a newer fund
                df = _save_records(records, conn)
                return df

            df = _save_records(records, conn)
            _log_discovery(conn, scheme_code, name, len(records), "ok")
            log.info("Saved %d historical records for %s", len(records), scheme_code)
            return df

        except requests.exceptions.Timeout:
            last_error = f"Timeout on attempt {attempt+1}"
            log.warning("Timeout fetching %s (attempt %d/%d)", scheme_code, attempt+1, retries)
        except requests.exceptions.HTTPError as e:
            last_error = str(e)
            if e.response is not None and e.response.status_code == 404:
                log.warning("Scheme %s not found (404) — skipping", scheme_code)
                _log_discovery(conn, scheme_code, "", 0, "not_found")
                return pd.DataFrame()   # no point retrying a 404
            log.warning("HTTP error for %s: %s", scheme_code, e)
        except Exception as e:
            last_error = str(e)
            log.warning("Error fetching %s: %s", scheme_code, e)

    log.error("All %d retries exhausted for scheme %s. Last error: %s",
              retries, scheme_code, last_error)
    _log_discovery(conn, scheme_code, "", 0, "failed")
    return pd.DataFrame()


def _save_records(records: list, conn: sqlite3.Connection) -> pd.DataFrame:
    """Bulk insert NAV records, return as sorted DataFrame."""
    conn.executemany(
        "INSERT OR IGNORE INTO nav_history (scheme_code, scheme_name, nav, nav_date) "
        "VALUES (:scheme_code, :scheme_name, :nav, :nav_date)",
        records
    )
    conn.commit()
    df = pd.DataFrame(records).sort_values("nav_date").reset_index(drop=True)
    df["nav_date"] = pd.to_datetime(df["nav_date"])
    return df


def _log_discovery(conn, scheme_code, scheme_name, records_saved, status):
    """Record discovery attempt outcome in discovery_log table."""
    try:
        conn.execute(
            "INSERT OR REPLACE INTO discovery_log "
            "(scheme_code, scheme_name, records_saved, status, attempted_at) "
            "VALUES (?, ?, ?, ?, datetime('now'))",
            (str(scheme_code), scheme_name, records_saved, status)
        )
        conn.commit()
    except sqlite3.Error:
        pass


def get_nav_history(scheme_code: str, conn: sqlite3.Connection,
                    from_date: str = None) -> pd.DataFrame:
    """Load NAV history from DB for a given fund."""
    query  = "SELECT nav_date, nav FROM nav_history WHERE scheme_code = ?"
    params = [str(scheme_code)]
    if from_date:
        query += " AND nav_date >= ?"
        params.append(from_date)
    query += " ORDER BY nav_date ASC"
    df = pd.read_sql_query(query, conn, params=params)
    if df.empty:
        return df
    df["nav_date"] = pd.to_datetime(df["nav_date"])
    return df


def get_fund_category(scheme_code: str, conn: sqlite3.Connection) -> str:
    """
    Fix 2: Retrieve stored category from fund_metadata table.
    Falls back to infer_category from scheme name in nav_history if not found.
    """
    row = conn.execute(
        "SELECT category FROM fund_metadata WHERE scheme_code = ?", (str(scheme_code),)
    ).fetchone()
    if row and row[0]:
        return row[0]

    # Fallback: get name from nav_history and infer
    row2 = conn.execute(
        "SELECT scheme_name FROM nav_history WHERE scheme_code = ? LIMIT 1",
        (str(scheme_code),)
    ).fetchone()
    if row2 and row2[0]:
        return infer_category(row2[0])

    return "Equity"


def get_already_discovered(conn: sqlite3.Connection) -> set:
    """Return set of scheme codes already attempted in discovery."""
    rows = conn.execute(
        "SELECT scheme_code FROM discovery_log WHERE status IN ('ok','insufficient','not_found')"
    ).fetchall()
    return {r[0] for r in rows}


def get_discovery_stats(conn: sqlite3.Connection) -> dict:
    """Summary of discovery log."""
    rows = conn.execute(
        "SELECT status, COUNT(*) FROM discovery_log GROUP BY status"
    ).fetchall()
    return {r[0]: r[1] for r in rows}