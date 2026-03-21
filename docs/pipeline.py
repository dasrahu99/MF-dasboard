"""
pipeline.py — Automated daily pipeline runner
Schedules AMFI NAV fetch, computes metrics, and exports reports.
Run: python pipeline.py           (runs once immediately)
     python pipeline.py --schedule (runs daily at 20:30 IST)
"""

import argparse
import json
import logging
import sqlite3
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

from fetcher   import init_db, fetch_todays_nav, save_nav_to_db, \
                      fetch_historical_nav, get_nav_history, DB_PATH
from engine    import fund_report, wealth_projection
from momentum  import enrich_latest_report
from benchmark import run_benchmark_analysis
from drawdown  import run_drawdown_analysis
from tax       import enrich_with_tax
from portfolio import build_portfolio_report, DEFAULT_PORTFOLIO
from stepup    import save_stepup_report

Path("data").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True,
    handlers=[
        logging.FileHandler("data/pipeline.log"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

REPORTS_DIR = Path("data/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
SIP_AMOUNT   = 10_000     # ₹ monthly SIP per fund
LUMP_SUM     = 1_00_000   # ₹ lump sum reference amount
MAX_FUNDS    = 200        # safety cap — increase if you want even more coverage

# Keywords that indicate NON-equity or unwanted schemes (case-insensitive)
EXCLUDE_KEYWORDS = [
    "direct", "dividend", "idcw", "bonus", "institutional", "segregated",
    "liquid", "overnight", "money market", "gilt", "bond", "debt",
    "fixed", "float", "income", "credit risk", "banking & psu",
    "corporate bond", "fund of fund", "fof", "interval", "maturity",
    "savings", "ultra short", "low duration", "short duration",
    "medium duration", "long duration", "dynamic bond", "arbitrage",
    "children", "retirement", "target maturity", "treasury",
    "mmf", "g-sec", "gsec", "hybrid", "balanced", "conservative",
    "aggressive hybrid", "equity savings", "discontinued",
    "nifty 1d rate", "crisil", "constant", "banking and psu",
]

# Equity-related keywords — fund name must match at least one
EQUITY_KEYWORDS = [
    "large cap", "largecap", "mid cap", "midcap", "small cap", "smallcap",
    "flexi cap", "flexicap", "multi cap", "multicap", "large & mid",
    "elss", "tax saver", "equity", "value", "contra", "focused",
    "index", "nifty", "sensex", "opportunities", "bluechip",
    "emerging", "growth fund", "thematic", "sector", "infra",
    "pharma", "banking", "technology", "digital", "consumption",
    "manufacturing", "innovation", "esg", "quant", "momentum",
    "dividend yield",
]


def discover_equity_funds(nav_df: pd.DataFrame) -> dict:
    """
    Auto-discover open-ended equity growth (regular plan) mutual funds
    from today's bulk AMFI NAV download.

    Returns: dict of {scheme_code: scheme_name}
    """
    if nav_df.empty:
        log.warning("Empty NAV DataFrame — cannot discover funds")
        return {}

    df = nav_df.copy()
    df["name_lower"] = df["scheme_name"].str.lower()

    # Must contain "growth" (eliminates dividend/IDCW variants)
    df = df[df["name_lower"].str.contains("growth", na=False)]

    # Exclude unwanted keywords
    for kw in EXCLUDE_KEYWORDS:
        df = df[~df["name_lower"].str.contains(kw, na=False)]

    # Must match at least one equity-related keyword
    equity_mask = df["name_lower"].apply(
        lambda n: any(ek in n for ek in EQUITY_KEYWORDS)
    )
    df = df[equity_mask]

    # De-duplicate by scheme_code (take first occurrence)
    df = df.drop_duplicates(subset="scheme_code", keep="first")

    # Sort by NAV descending (higher NAV = older = more established fund)
    df = df.sort_values("nav", ascending=False)

    # Apply safety cap
    df = df.head(MAX_FUNDS)

    watchlist = dict(zip(df["scheme_code"].astype(str), df["scheme_name"]))
    log.info("Auto-discovered %d equity growth funds from %d total AMFI schemes",
             len(watchlist), len(nav_df))
    return watchlist


def run_daily_pipeline():
    """Main pipeline: fetch NAVs → compute → save reports."""
    log.info("=" * 60)
    log.info("Pipeline started at %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    conn = init_db(DB_PATH)

    # 1. Fetch today's bulk NAV
    nav_df = pd.DataFrame()
    try:
        nav_df = fetch_todays_nav()
        saved  = save_nav_to_db(nav_df, conn)
        log.info("Today's NAVs: %d records saved", saved)
    except Exception as e:
        log.error("Bulk NAV fetch failed: %s", e)

    # 2. Auto-discover equity growth funds (replaces hardcoded WATCHLIST)
    WATCHLIST = discover_equity_funds(nav_df)
    if not WATCHLIST:
        log.error("No funds discovered — aborting pipeline")
        conn.close()
        return
    log.info("Processing %d funds...", len(WATCHLIST))

    # 3. Fetch history & compute reports for each fund
    all_reports = {}
    for i, (code, name) in enumerate(WATCHLIST.items(), 1):
        log.info("[%d/%d] Processing: %s [%s]", i, len(WATCHLIST), name[:40], code)
        try:
            # Check if we have history; if not, fetch it
            hist = get_nav_history(code, conn, from_date="2015-01-01")
            if len(hist) < 100:
                log.info("  ↳ Fetching full history from mfapi.in...")
                hist = fetch_historical_nav(code, conn)
                if hist.empty:
                    log.warning("  ↳ No history found — skipping")
                    continue
                # Rate-limit mfapi.in calls (0.3s between requests)
                time.sleep(0.3)
            else:
                log.info("  ↳ Using %d cached records", len(hist))

            # 4. Compute full report
            report = fund_report(
                scheme_code=code,
                df=hist,
                sip_amount=SIP_AMOUNT,
                lump_sum=LUMP_SUM,
            )
            report["fund_name"] = name
            all_reports[code] = report

        except Exception as e:
            log.error("  ↳ Failed: %s", e)
            continue

    # 4. Save JSON report
    report_path = REPORTS_DIR / f"report_{datetime.now().strftime('%Y%m%d')}.json"
    with open(report_path, "w") as f:
        json.dump(all_reports, f, indent=2, default=str)
    log.info("Report saved → %s", report_path)

    # 5. Save latest.json (always points to most recent)
    with open(REPORTS_DIR / "latest.json", "w") as f:
        json.dump(all_reports, f, indent=2, default=str)

    # 6. Enrich with momentum scores
    log.info("Running momentum scoring engine...")
    try:
        enrich_latest_report(WATCHLIST, conn)
    except Exception as e:
        log.error("Momentum scoring failed: %s", e)

    # 6.1 Benchmark Analysis
    log.info("Running benchmark analysis...")
    try:
        run_benchmark_analysis(WATCHLIST, conn)
    except Exception as e:
        log.error("Benchmark analysis failed: %s", e)

    # 6.2 Drawdown Tracking
    log.info("Running drawdown tracking...")
    try:
        run_drawdown_analysis(WATCHLIST, conn)
    except Exception as e:
        log.error("Drawdown tracking failed: %s", e)

    # 6.3 Tax-Adjusted Returns
    log.info("Running tax-adjusted return calculations...")
    try:
        enrich_with_tax()
    except Exception as e:
        log.error("Tax calculations failed: %s", e)

    # 6.4 Portfolio Simulator
    log.info("Running portfolio simulator...")
    try:
        report = build_portfolio_report(DEFAULT_PORTFOLIO, conn)
        out = REPORTS_DIR / "portfolio_report.json"
        with open(out, "w") as f:
            json.dump(report, f, indent=2, default=str)
    except Exception as e:
        log.error("Portfolio simulator failed: %s", e)

    # 6.5 SIP Step-Up Planner
    log.info("Running SIP step-up planner...")
    try:
        save_stepup_report()
    except Exception as e:
        log.error("SIP step-up planner failed: %s", e)

    # 7. Generate wealth projection for user's goals
    projection = wealth_projection(
        initial_lump_sum=5_00_000,
        monthly_sip=SIP_AMOUNT,
        annual_sip_step_up=0.10,
        expected_cagr=0.15,       # conservative equity estimate
        years=20,
        inflation_rate=0.06,
    )
    proj_path = REPORTS_DIR / "wealth_projection.csv"
    projection.to_csv(proj_path, index=False)
    log.info("Wealth projection saved → %s", proj_path)

    # 7. Print summary to console
    _print_summary(all_reports)
    conn.close()
    log.info("Pipeline completed ✓")


def _print_summary(reports: dict):
    """Print a ranked summary table to stdout."""
    rows = []
    for code, r in reports.items():
        if "error" in r:
            continue
        rows.append({
            "Fund":          r.get("fund_name", r.get("scheme_name",""))[:35],
            "Latest NAV":    r.get("latest_nav"),
            "1Y CAGR%":      r.get("cagr",{}).get("1y"),
            "3Y CAGR%":      r.get("cagr",{}).get("3y"),
            "5Y CAGR%":      r.get("cagr",{}).get("5y"),
            "Sharpe":        r.get("risk",{}).get("sharpe_3y"),
            "Max DD%":       r.get("risk",{}).get("max_drawdown"),
            "SIP XIRR%":     r.get("sip_simulation",{}).get("xirr_pct"),
            "Signal":        r.get("lump_sum_signal",{}).get("signal"),
            "Entry Score":   r.get("lump_sum_signal",{}).get("entry_score"),
        })

    if not rows:
        log.warning("No reports generated.")
        return

    df = pd.DataFrame(rows).sort_values("5Y CAGR%", ascending=False)
    print("\n" + "=" * 80)
    print("  MUTUAL FUND PIPELINE SUMMARY")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80 + "\n")


# ── Scheduler ─────────────────────────────────────────────────────────────────

def run_scheduled():
    """Run pipeline every day at 20:30 IST (AMFI publishes NAV by ~8pm)."""
    if not HAS_SCHEDULE:
        log.error("Install 'schedule' package: pip install schedule")
        sys.exit(1)

    log.info("Scheduler started — pipeline will run daily at 20:30 IST")
    schedule.every().day.at("20:30").do(run_daily_pipeline)

    # Run immediately on startup too
    run_daily_pipeline()

    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MF Data Pipeline")
    parser.add_argument("--schedule", action="store_true",
                        help="Run on a daily schedule (20:30 IST)")
    args = parser.parse_args()

    if args.schedule:
        run_scheduled()
    else:
        run_daily_pipeline()