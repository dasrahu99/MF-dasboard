"""
pipeline.py — Automated daily pipeline runner

Usage:
  python pipeline.py               # run once — processes WATCHLIST funds only (fast, <2 min)
  python pipeline.py --schedule    # run daily at 20:30 IST
  python pipeline.py --discovery   # one-time: fetch history for ALL 400+ equity funds (slow)

Architecture:
  Daily run  → uses WATCHLIST (10 curated funds) → always fast
  Discovery  → scans full AMFI dump, fetches history for every equity fund found
               run once overnight; daily runs then rank from the larger DB pool
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    import schedule
    HAS_SCHEDULE = True
except ImportError:
    HAS_SCHEDULE = False

from fetcher    import (init_db, fetch_todays_nav, save_nav_to_db,
                        fetch_historical_nav, get_nav_history,
                        get_fund_category, get_already_discovered,
                        get_discovery_stats, infer_category, DB_PATH)
from engine     import fund_report, wealth_projection
from momentum   import enrich_latest_report
from portfolio  import build_portfolio_report
from contrarian import run_contrarian_analysis

Path("data").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("data/pipeline.log"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

REPORTS_DIR = Path("data/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Curated watchlist — always processed on every daily run ──────────────────
WATCHLIST = {
    "120503": "Quant Small Cap Fund - Growth",
    "118778": "Nippon India Small Cap Fund - Growth",
    "118989": "HDFC Mid-Cap Opportunities Fund - Growth",
    "122639": "Parag Parikh Flexi Cap Fund - Growth",
    "120586": "Mirae Asset Large Cap Fund - Growth",
    "119598": "SBI Bluechip Fund - Growth",
    "120505": "Axis Long Term Equity (ELSS) - Growth",
    "100356": "UTI Nifty 50 Index Fund - Growth",
    "118701": "ICICI Pru Bluechip Fund - Growth",
    "119270": "Kotak Emerging Equity Fund - Growth",
}

# ── Investment parameters ─────────────────────────────────────────────────────
SIP_AMOUNT = 10_000     # ₹ monthly SIP (base for contrarian multiplier)
LUMP_SUM   = 1_00_000   # ₹ lump sum reference

# ── Dynamic watchlist keywords ────────────────────────────────────────────────
EQUITY_INCLUDE = [
    "equity", "flexi", "large cap", "largecap", "mid cap", "midcap",
    "small cap", "smallcap", "elss", "tax saver", "tax saving",
    "bluechip", "frontline", "multi cap", "multicap",
]
EQUITY_EXCLUDE = [
    "idcw", "dividend", "etf", "fof", "fund of fund",
    "liquid", "overnight", "money market", "debt", "bond",
    "gilt", "credit", "arbitrage", "international", "global",
    "overseas", "interval", "hybrid", "balanced",
]


# ── Dynamic watchlist builder ─────────────────────────────────────────────────

def get_dynamic_watchlist(nav_df: pd.DataFrame) -> dict:
    """
    Scan the full AMFI NAV DataFrame and extract all Direct Growth equity funds.
    Returns {scheme_code: scheme_name} for all candidates.

    Fix 1 + 2: Category is already stored at parse time (infer_category in fetcher.py)
    so we filter using the parsed category column, not raw name inference here.
    """
    if nav_df.empty:
        log.warning("NAV DataFrame empty — cannot build dynamic watchlist")
        return WATCHLIST

    # Only Direct Growth plans
    mask = nav_df["scheme_name"].str.lower().str.contains("direct", na=False) & \
           nav_df["scheme_name"].str.lower().str.contains("growth", na=False)

    df = nav_df[mask].copy()

    # Keyword include filter
    include_mask = df["scheme_name"].str.lower().apply(
        lambda n: any(kw in n for kw in EQUITY_INCLUDE)
    )
    # Keyword exclude filter
    exclude_mask = df["scheme_name"].str.lower().apply(
        lambda n: any(kw in n for kw in EQUITY_EXCLUDE)
    )

    df = df[include_mask & ~exclude_mask]

    # Build dict — scheme_code → scheme_name
    result = dict(zip(df["scheme_code"].astype(str), df["scheme_name"]))
    log.info("Dynamic watchlist: %d equity funds identified from AMFI", len(result))
    return result


def weighted_rank(reports: dict, top_n: int = 10) -> list:
    """
    Fix: Rank funds by weighted composite, not raw CAGR.
    Formula: (5Y CAGR × 40%) + (Sharpe × 30%) + (Momentum score × 30%)

    Normalises each metric to 0–1 before weighting so different scales
    don't distort the ranking (a Sharpe of 1.4 shouldn't outweigh a CAGR of 30%).
    Returns list of (code, report) tuples, best first, limited to top_n.
    """
    candidates = []
    for code, r in reports.items():
        if "error" in r:
            continue
        cagr5   = r.get("cagr", {}).get("5y")
        sharpe  = r.get("risk", {}).get("sharpe_3y")
        mom     = r.get("momentum", {}).get("composite_score")
        if cagr5 is None:
            continue
        candidates.append({
            "code":   code,
            "report": r,
            "cagr5":  cagr5 or 0,
            "sharpe": sharpe or 0,
            "mom":    mom or 0,
        })

    if not candidates:
        return []

    # Normalise each metric to 0–1 range across the candidate pool
    def normalise(vals):
        mn, mx = min(vals), max(vals)
        if mx == mn:
            return [0.5] * len(vals)
        return [(v - mn) / (mx - mn) for v in vals]

    cagr_n   = normalise([c["cagr5"]  for c in candidates])
    sharpe_n = normalise([c["sharpe"] for c in candidates])
    mom_n    = normalise([c["mom"]    for c in candidates])

    for i, c in enumerate(candidates):
        c["weighted_score"] = (
            cagr_n[i]   * 0.40 +
            sharpe_n[i] * 0.30 +
            mom_n[i]    * 0.30
        )
        c["report"]["weighted_rank_score"] = round(c["weighted_score"] * 100, 1)

    candidates.sort(key=lambda x: x["weighted_score"], reverse=True)
    return [(c["code"], c["report"]) for c in candidates[:top_n]]


# ── Daily pipeline ────────────────────────────────────────────────────────────

def run_daily_pipeline():
    """
    Main pipeline: fetch NAVs → compute → save reports.
    Always runs on the curated WATCHLIST — fast (<2 min after first run).
    """
    log.info("=" * 60)
    log.info("Pipeline started at %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    conn = init_db(DB_PATH)

    # Step 1 — Fetch today's bulk NAV from AMFI
    nav_df = pd.DataFrame()
    try:
        nav_df = fetch_todays_nav()
        saved  = save_nav_to_db(nav_df, conn)
        log.info("Today's NAVs: %d records saved", saved)
    except Exception as e:
        log.error("Bulk NAV fetch failed: %s", e)

    # Step 2 — Process watchlist funds
    all_reports = {}
    for code, name in WATCHLIST.items():
        log.info("Processing: %s [%s]", name, code)
        try:
            hist = get_nav_history(code, conn, from_date="2015-01-01")
            if len(hist) < 100:
                log.info("Fetching history for %s...", code)
                hist = fetch_historical_nav(code, conn)
                if hist.empty:
                    log.warning("No history found for %s", code)
                    continue
            else:
                log.info("Using %d cached records for %s", len(hist), code)

            # Attach stored category (Fix 2)
            category = get_fund_category(code, conn)

            report = fund_report(
                scheme_code=code,
                df=hist,
                sip_amount=SIP_AMOUNT,
                lump_sum=LUMP_SUM,
            )
            report["fund_name"] = name
            report["category"]  = category
            all_reports[code]   = report

        except Exception as e:
            log.error("Failed to process %s: %s", code, e)
            continue

    # Step 3 — Save raw report
    report_path = REPORTS_DIR / f"report_{datetime.now().strftime('%Y%m%d')}.json"
    with open(report_path, "w") as f:
        json.dump(all_reports, f, indent=2, default=str)

    # Step 4 — Save latest.json
    with open(REPORTS_DIR / "latest.json", "w") as f:
        json.dump(all_reports, f, indent=2, default=str)
    log.info("Reports saved → %s", REPORTS_DIR)

    # Step 5 — Momentum scoring
    log.info("Running momentum scoring...")
    try:
        enrich_latest_report(WATCHLIST, conn)
    except Exception as e:
        log.error("Momentum scoring failed: %s", e)

    # Step 6 — Contrarian analysis
    log.info("Running contrarian analysis...")
    try:
        run_contrarian_analysis(WATCHLIST, conn, SIP_AMOUNT)
    except Exception as e:
        log.error("Contrarian analysis failed: %s", e)

    # Step 7 — Reload latest.json (now enriched) and apply weighted ranking
    try:
        with open(REPORTS_DIR / "latest.json") as f:
            enriched = json.load(f)
        top10 = weighted_rank(enriched, top_n=10)
        top10_report = {code: rep for code, rep in top10}
        with open(REPORTS_DIR / "top10.json", "w") as f:
            json.dump(top10_report, f, indent=2, default=str)
        log.info("Top 10 by weighted rank saved → top10.json")
    except Exception as e:
        log.error("Weighted ranking failed: %s", e)

    # Step 8 — Portfolio report
    log.info("Running portfolio simulation...")
    try:
        build_portfolio_report(conn=conn)
    except Exception as e:
        log.error("Portfolio report failed: %s", e)

    # Step 9 — Wealth projection
    try:
        projection = wealth_projection(
            initial_lump_sum=5_00_000,
            monthly_sip=SIP_AMOUNT,
            annual_sip_step_up=0.10,
            expected_cagr=0.15,
            years=20,
            inflation_rate=0.06,
        )
        projection.to_csv(REPORTS_DIR / "wealth_projection.csv", index=False)
        log.info("Wealth projection saved")
    except Exception as e:
        log.error("Wealth projection failed: %s", e)

    # Step 10 — Print ranked summary
    _print_summary(all_reports)
    conn.close()
    log.info("Pipeline completed ✓")


# ── Discovery pipeline ────────────────────────────────────────────────────────

def run_discovery_pipeline():
    """
    One-time discovery run: scans all 400+ equity funds from AMFI,
    fetches historical NAV for each, stores in DB.

    This is intentionally separate from the daily run to keep daily
    processing fast. Run once overnight — subsequent daily runs are instant
    because they use the cached DB.

    Rate limiting: ~0.5s per fund = ~3-4 hours for 400 funds.
    Progress is saved after each fund — safe to interrupt and resume.
    """
    log.info("=" * 60)
    log.info("DISCOVERY MODE — scanning full AMFI fund universe")
    log.info("This will take 3-4 hours for ~400 funds. Safe to interrupt.")
    log.info("=" * 60)

    conn = init_db(DB_PATH)

    # Fetch today's AMFI dump
    try:
        nav_df = fetch_todays_nav()
        save_nav_to_db(nav_df, conn)
    except Exception as e:
        log.error("AMFI fetch failed: %s", e)
        conn.close()
        return

    # Build dynamic watchlist from AMFI data
    dynamic_watchlist = get_dynamic_watchlist(nav_df)
    log.info("Discovered %d equity funds to process", len(dynamic_watchlist))

    # Skip funds already successfully fetched
    already_done = get_already_discovered(conn)
    remaining    = {k: v for k, v in dynamic_watchlist.items() if k not in already_done}
    log.info("%d already in DB, %d remaining to fetch", len(already_done), len(remaining))

    # Fetch history for each — with rate limiting (Fix 1)
    success = failed = skipped = 0
    total   = len(remaining)

    for i, (code, name) in enumerate(remaining.items(), 1):
        log.info("[%d/%d] Fetching: %s [%s]", i, total, name[:50], code)
        try:
            df = fetch_historical_nav(code, conn, retries=3, skip_if_sufficient=False)
            if df.empty:
                failed += 1
            else:
                success += 1
        except Exception as e:
            log.error("Unexpected error for %s: %s", code, e)
            failed += 1

        # Progress checkpoint every 50 funds
        if i % 50 == 0:
            stats = get_discovery_stats(conn)
            log.info("Progress: %d/%d | Stats: %s", i, total, stats)

    stats = get_discovery_stats(conn)
    log.info("=" * 60)
    log.info("Discovery complete: %d success, %d failed", success, failed)
    log.info("Final DB stats: %s", stats)
    log.info("Run `python pipeline.py` now to compute metrics on all discovered funds.")
    log.info("=" * 60)

    conn.close()


# ── Summary printer ───────────────────────────────────────────────────────────

def _print_summary(reports: dict):
    rows = []
    for code, r in reports.items():
        if "error" in r:
            continue
        m = r.get("momentum", {})
        c = r.get("contrarian", {})
        rows.append({
            "Fund":        r.get("fund_name", r.get("scheme_name",""))[:32],
            "Category":    r.get("category",""),
            "1Y%":         r.get("cagr",{}).get("1y"),
            "3Y%":         r.get("cagr",{}).get("3y"),
            "5Y%":         r.get("cagr",{}).get("5y"),
            "Sharpe":      r.get("risk",{}).get("sharpe_3y"),
            "Momentum":    m.get("composite_score"),
            "Rank Score":  r.get("weighted_rank_score"),
            "Contrarian":  c.get("signal","—"),
            "Quality":     c.get("quality_grade","—"),
        })
    if not rows:
        log.warning("No reports generated.")
        return
    df = pd.DataFrame(rows).sort_values("Rank Score", ascending=False, na_position="last")
    print("\n" + "=" * 90)
    print("  PIPELINE SUMMARY — ranked by weighted score (5Y CAGR 40% + Sharpe 30% + Momentum 30%)")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90 + "\n")


# ── Scheduler ─────────────────────────────────────────────────────────────────

def run_scheduled():
    if not HAS_SCHEDULE:
        log.error("Install 'schedule': pip install schedule")
        sys.exit(1)
    log.info("Scheduler started — pipeline runs daily at 20:30 IST")
    schedule.every().day.at("20:30").do(run_daily_pipeline)
    run_daily_pipeline()
    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MF Intelligence Pipeline")
    parser.add_argument("--schedule",  action="store_true",
                        help="Run daily at 20:30 IST")
    parser.add_argument("--discovery", action="store_true",
                        help="One-time: fetch history for all 400+ equity funds (run overnight)")
    args = parser.parse_args()

    if args.discovery:
        run_discovery_pipeline()
    elif args.schedule:
        run_scheduled()
    else:
        run_daily_pipeline()