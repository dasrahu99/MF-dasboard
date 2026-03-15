"""
benchmark.py — Benchmark Comparison Engine (Module 4)
Compares every fund against Nifty 50 and Nifty 500 across multiple horizons.
Computes alpha, information ratio, up/down capture, and active return consistency.
Ruthlessly flags funds where index investing would have been better.
"""

import numpy as np
import pandas as pd
import sqlite3
import json
import logging
from pathlib import Path
from typing import Optional

from fetcher  import get_nav_history, fetch_historical_nav, DB_PATH, init_db
from engine   import compute_cagr_for_horizon, volatility, _rsi

log = logging.getLogger(__name__)

REPORTS_DIR = Path("data/reports")

# Benchmarks — add Nifty 500 index fund for broader comparison
BENCHMARKS = {
    "100356": "Nifty 50",     # UTI Nifty 50 Index
    "120716": "Nifty 500",    # Motilal Oswal Nifty 500 Index
}


# ── Core alpha computation ────────────────────────────────────────────────────

def compute_alpha(fund_df: pd.DataFrame,
                  bench_df: pd.DataFrame,
                  years: float) -> Optional[float]:
    """CAGR alpha = fund CAGR minus benchmark CAGR over same period."""
    fund_cagr  = compute_cagr_for_horizon(fund_df,  years)
    bench_cagr = compute_cagr_for_horizon(bench_df, years)
    if fund_cagr is None or bench_cagr is None:
        return None
    return fund_cagr - bench_cagr


def information_ratio(fund_df: pd.DataFrame,
                       bench_df: pd.DataFrame,
                       years: float = 3.0) -> Optional[float]:
    """
    Information Ratio = active return / tracking error.
    Measures consistency of outperformance. Higher = better active manager.
    """
    fund_df  = fund_df.sort_values("nav_date").set_index("nav_date")
    bench_df = bench_df.sort_values("nav_date").set_index("nav_date")

    cutoff = fund_df.index.max() - pd.DateOffset(years=int(years))
    fund_r  = fund_df[fund_df.index >= cutoff]["nav"].pct_change().dropna()
    bench_r = bench_df[bench_df.index >= cutoff]["nav"].pct_change().dropna()

    # Align on common dates
    combined = pd.concat([fund_r, bench_r], axis=1, join="inner")
    combined.columns = ["fund", "bench"]
    if len(combined) < 60:
        return None

    active_return   = combined["fund"] - combined["bench"]
    tracking_error  = active_return.std() * np.sqrt(252)
    avg_active_ann  = active_return.mean() * 252

    if tracking_error == 0:
        return None
    return round(avg_active_ann / tracking_error, 3)


def up_down_capture(fund_df: pd.DataFrame,
                     bench_df: pd.DataFrame,
                     years: float = 3.0) -> dict:
    """
    Up-capture: how much of benchmark's up-months the fund captures.
    Down-capture: how much of benchmark's down-months the fund suffers.
    Ideal: up-capture > 100%, down-capture < 100%.
    """
    fund_df  = fund_df.sort_values("nav_date").set_index("nav_date")
    bench_df = bench_df.sort_values("nav_date").set_index("nav_date")

    cutoff = fund_df.index.max() - pd.DateOffset(years=int(years))

    fund_monthly  = fund_df[fund_df.index >= cutoff]["nav"].resample("ME").last().pct_change().dropna()
    bench_monthly = bench_df[bench_df.index >= cutoff]["nav"].resample("ME").last().pct_change().dropna()

    combined = pd.concat([fund_monthly, bench_monthly], axis=1, join="inner")
    combined.columns = ["fund", "bench"]
    if len(combined) < 12:
        return {}

    up_periods   = combined[combined["bench"] > 0]
    down_periods = combined[combined["bench"] < 0]

    up_capture   = (up_periods["fund"].mean()   / up_periods["bench"].mean()   * 100) \
                    if not up_periods.empty and up_periods["bench"].mean() != 0 else None
    down_capture = (down_periods["fund"].mean() / down_periods["bench"].mean() * 100) \
                    if not down_periods.empty and down_periods["bench"].mean() != 0 else None

    capture_ratio = (up_capture / down_capture) if (up_capture and down_capture and down_capture != 0) else None

    return {
        "up_capture_pct":    round(up_capture,   1) if up_capture   else None,
        "down_capture_pct":  round(down_capture, 1) if down_capture else None,
        "capture_ratio":     round(capture_ratio, 3) if capture_ratio else None,
        "verdict": (
            "Excellent — captures upside, limits downside"
            if capture_ratio and capture_ratio > 1.2 else
            "Good" if capture_ratio and capture_ratio > 1.0 else
            "Poor — consider switching to index fund"
        ),
    }


def active_return_consistency(fund_df: pd.DataFrame,
                               bench_df: pd.DataFrame,
                               years: float = 5.0) -> Optional[float]:
    """
    % of rolling 12-month periods where fund beat benchmark.
    100% = always beats; 0% = never beats.
    """
    fund_df  = fund_df.sort_values("nav_date").set_index("nav_date")
    bench_df = bench_df.sort_values("nav_date").set_index("nav_date")

    cutoff = fund_df.index.max() - pd.DateOffset(years=int(years))
    fund_m  = fund_df[fund_df.index >= cutoff]["nav"].resample("ME").last()
    bench_m = bench_df[bench_df.index >= cutoff]["nav"].resample("ME").last()

    combined = pd.concat([fund_m, bench_m], axis=1, join="inner")
    combined.columns = ["fund", "bench"]
    if len(combined) < 13:
        return None

    beats = 0
    total = 0
    for i in range(12, len(combined)):
        fund_ret  = combined["fund"].iloc[i]  / combined["fund"].iloc[i-12]  - 1
        bench_ret = combined["bench"].iloc[i] / combined["bench"].iloc[i-12] - 1
        total += 1
        if fund_ret > bench_ret:
            beats += 1

    return round(beats / total * 100, 1) if total > 0 else None


# ── Full fund vs benchmark report ─────────────────────────────────────────────

def benchmark_report(scheme_code: str,
                      fund_df: pd.DataFrame,
                      bench_dfs: dict,
                      scheme_name: str = "") -> dict:
    """
    Full benchmark comparison for one fund against all benchmarks.
    Returns alpha table, IR, capture ratios, and a clear verdict.
    """
    result = {
        "scheme_code": scheme_code,
        "scheme_name": scheme_name,
        "benchmarks":  {},
    }

    for bench_code, bench_name in BENCHMARKS.items():
        bench_df = bench_dfs.get(bench_code, pd.DataFrame())
        if bench_df.empty:
            continue

        alpha = {
            "1y":  _pct(compute_alpha(fund_df, bench_df, 1)),
            "3y":  _pct(compute_alpha(fund_df, bench_df, 3)),
            "5y":  _pct(compute_alpha(fund_df, bench_df, 5)),
            "10y": _pct(compute_alpha(fund_df, bench_df, 10)),
        }
        ir        = information_ratio(fund_df, bench_df, 3)
        capture   = up_down_capture(fund_df, bench_df, 3)
        consist   = active_return_consistency(fund_df, bench_df, 5)

        # Verdict logic
        positive_alphas = sum(1 for v in alpha.values() if v and v > 0)
        verdict = _verdict(alpha, ir, capture, consist, positive_alphas)

        result["benchmarks"][bench_name] = {
            "alpha_pct":             alpha,
            "information_ratio":     round(ir, 3) if ir else None,
            "capture":               capture,
            "beat_consistency_pct":  consist,
            "verdict":               verdict,
        }

    # Overall recommendation
    result["recommendation"] = _overall_recommendation(result["benchmarks"])
    return result


def _verdict(alpha, ir, capture, consist, positive_alphas):
    score = 0
    if positive_alphas >= 3:     score += 2
    elif positive_alphas >= 2:   score += 1
    if ir and ir > 0.5:          score += 2
    elif ir and ir > 0:          score += 1
    cr = capture.get("capture_ratio")
    if cr and cr > 1.2:          score += 2
    elif cr and cr > 1.0:        score += 1
    if consist and consist > 60: score += 2
    elif consist and consist > 50: score += 1

    if score >= 6: return "KEEP — strong active management value"
    if score >= 4: return "KEEP — adds some alpha over benchmark"
    if score >= 2: return "REVIEW — marginal outperformance"
    return "REPLACE — index fund would likely perform better"


def _overall_recommendation(benchmarks: dict) -> str:
    verdicts = [v.get("verdict", "") for v in benchmarks.values()]
    keeps    = sum(1 for v in verdicts if "KEEP" in v)
    if keeps == len(verdicts): return "KEEP"
    if keeps > 0:              return "REVIEW"
    return "REPLACE WITH INDEX FUND"


def _pct(val):
    return round(val * 100, 2) if val is not None else None


# ── Batch benchmark all watchlist funds ───────────────────────────────────────

def run_benchmark_analysis(watchlist: dict,
                             conn: sqlite3.Connection) -> dict:
    """Score all watchlist funds vs benchmarks. Enriches latest.json."""
    log.info("Loading benchmark data...")
    bench_dfs = {}
    for code, name in BENCHMARKS.items():
        df = get_nav_history(code, conn)
        if df.empty:
            df = fetch_historical_nav(code, conn)
        bench_dfs[code] = df
        log.info("Benchmark %s: %d records", name, len(df))

    results = {}
    for code, name in watchlist.items():
        log.info("Benchmarking: %s", name)
        fund_df = get_nav_history(code, conn)
        if fund_df.empty:
            fund_df = fetch_historical_nav(code, conn)
        if fund_df.empty:
            continue
        results[code] = benchmark_report(code, fund_df, bench_dfs, name)

    # Save
    out = Path("data/reports/benchmark_analysis.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Benchmark analysis saved → %s", out)

    # Merge into latest.json
    latest_path = Path("data/reports/latest.json")
    if latest_path.exists():
        with open(latest_path) as f:
            report = json.load(f)
        for code, bench_data in results.items():
            if code in report:
                report[code]["benchmark"] = bench_data
        with open(latest_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

    _print_benchmark_table(results)
    return results


def _print_benchmark_table(results: dict):
    rows = []
    for code, r in results.items():
        nifty = r.get("benchmarks", {}).get("Nifty 50", {})
        rows.append({
            "Fund":          r.get("scheme_name", "")[:30],
            "Alpha 1Y%":     nifty.get("alpha_pct", {}).get("1y"),
            "Alpha 3Y%":     nifty.get("alpha_pct", {}).get("3y"),
            "Alpha 5Y%":     nifty.get("alpha_pct", {}).get("5y"),
            "Info Ratio":    nifty.get("information_ratio"),
            "Up Capture%":   nifty.get("capture", {}).get("up_capture_pct"),
            "Dn Capture%":   nifty.get("capture", {}).get("down_capture_pct"),
            "Beat %":        nifty.get("beat_consistency_pct"),
            "Recommendation":r.get("recommendation"),
        })
    df = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print("  BENCHMARK ANALYSIS vs NIFTY 50")
    print("=" * 100)
    print(df.to_string(index=False))
    print("=" * 100 + "\n")


if __name__ == "__main__":
    import sys
    from pipeline import WATCHLIST
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    conn = init_db(DB_PATH)
    run_benchmark_analysis(WATCHLIST, conn)
    conn.close()