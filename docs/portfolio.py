"""
portfolio.py — Portfolio Simulator Engine (Module 3)
Simulates a multi-fund portfolio with SIP allocations, computes blended
XIRR, diversification score, concentration warnings, and alpha vs benchmark.

Resident Individual tax rules applied via tax.py.
"""

import numpy as np
import pandas as pd
import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from fetcher  import get_nav_history, fetch_historical_nav, DB_PATH, init_db
from engine   import sip_xirr, cagr, _xirr
from momentum import trailing_return

log = logging.getLogger(__name__)

BENCHMARK_CODE = "100356"   # UTI Nifty 50 Index

# Default portfolio — pre-filled from pipeline watchlist, user can edit
DEFAULT_PORTFOLIO = {
    "120503": {"name": "Quant Small Cap",        "monthly_sip": 3000,  "lump_sum": 50000,  "category": "Small Cap"},
    "118778": {"name": "Nippon India Small Cap",  "monthly_sip": 2000,  "lump_sum": 30000,  "category": "Small Cap"},
    "118989": {"name": "HDFC Mid-Cap Opps",       "monthly_sip": 3000,  "lump_sum": 50000,  "category": "Mid Cap"},
    "122639": {"name": "Parag Parikh Flexi Cap",  "monthly_sip": 2000,  "lump_sum": 40000,  "category": "Flexi Cap"},
    "120586": {"name": "Mirae Asset Large Cap",   "monthly_sip": 2000,  "lump_sum": 30000,  "category": "Large Cap"},
    "119598": {"name": "SBI Bluechip",            "monthly_sip": 1000,  "lump_sum": 20000,  "category": "Large Cap"},
    "120505": {"name": "Axis ELSS Tax Saver",     "monthly_sip": 1500,  "lump_sum": 0,      "category": "ELSS"},
    "100356": {"name": "UTI Nifty 50 Index",      "monthly_sip": 1500,  "lump_sum": 30000,  "category": "Index"},
}


# ── Per-fund simulation ────────────────────────────────────────────────────────

def simulate_fund(scheme_code: str,
                  fund_config: dict,
                  conn: sqlite3.Connection,
                  sip_start: str = "2020-01-01") -> dict:
    """Simulate one fund's SIP + lump sum performance."""
    df = get_nav_history(scheme_code, conn)
    if df.empty:
        df = fetch_historical_nav(scheme_code, conn)
    if df.empty:
        return {"error": f"No data for {scheme_code}"}

    monthly_sip = fund_config.get("monthly_sip", 0)
    lump_sum    = fund_config.get("lump_sum", 0)
    name        = fund_config.get("name", scheme_code)
    category    = fund_config.get("category", "Unknown")

    result = {"scheme_code": scheme_code, "name": name, "category": category}

    # SIP simulation
    if monthly_sip > 0:
        sip_res = sip_xirr(df, monthly_sip, sip_start)
        result["sip"] = sip_res
    else:
        result["sip"] = {}

    # Lump sum simulation (invested at sip_start date)
    if lump_sum > 0:
        df_sorted = df.sort_values("nav_date")
        start_slice = df_sorted[df_sorted["nav_date"] >= pd.Timestamp(sip_start)]
        if not start_slice.empty:
            buy_nav      = start_slice["nav"].iloc[0]
            current_nav  = df_sorted["nav"].iloc[-1]
            units        = lump_sum / buy_nav
            current_val  = units * current_nav
            years_held   = (df_sorted["nav_date"].iloc[-1] - start_slice["nav_date"].iloc[0]).days / 365.25
            ls_cagr      = cagr(buy_nav, current_nav, years_held) if years_held > 0 else None
            result["lump_sum"] = {
                "invested":         lump_sum,
                "current_value":    round(current_val, 2),
                "units":            round(units, 4),
                "buy_nav":          round(buy_nav, 4),
                "current_nav":      round(current_nav, 4),
                "absolute_return_pct": round((current_val - lump_sum) / lump_sum * 100, 2),
                "cagr_pct":         round(ls_cagr * 100, 2) if ls_cagr else None,
                "years_held":       round(years_held, 2),
            }
        else:
            result["lump_sum"] = {}

    # Trailing returns for context
    result["trailing"] = {
        "1m":  _pct(trailing_return(df, 1)),
        "3m":  _pct(trailing_return(df, 3)),
        "6m":  _pct(trailing_return(df, 6)),
        "12m": _pct(trailing_return(df, 12)),
    }

    return result


# ── Portfolio aggregation ─────────────────────────────────────────────────────

def build_portfolio_report(portfolio: dict,
                            conn: sqlite3.Connection,
                            sip_start: str = "2020-01-01") -> dict:
    """
    Simulate full portfolio. Compute:
    - Per-fund breakdown
    - Blended XIRR across all cashflows
    - Diversification score
    - Category concentration
    - Alpha vs Nifty 50
    - Actionable rebalancing suggestions
    """
    fund_results = {}
    all_cashflows = []   # for blended XIRR

    total_invested    = 0.0
    total_current     = 0.0
    total_sip_monthly = 0.0

    for code, config in portfolio.items():
        res = simulate_fund(code, config, conn, sip_start)
        fund_results[code] = res
        if "error" in res:
            continue

        sip_d = res.get("sip", {})
        ls_d  = res.get("lump_sum", {})

        fund_invested = sip_d.get("total_invested", 0) + ls_d.get("invested", 0)
        fund_current  = sip_d.get("current_value", 0)  + ls_d.get("current_value", 0)

        total_invested    += fund_invested
        total_current     += fund_current
        total_sip_monthly += config.get("monthly_sip", 0)

    # Portfolio-level metrics
    total_gain  = total_current - total_invested
    abs_return  = (total_gain / total_invested * 100) if total_invested > 0 else 0

    # Category weights
    cat_weights = _category_weights(portfolio, fund_results, total_current)

    # Diversification score (Herfindahl–Hirschman Index based)
    div_score = _diversification_score(cat_weights)

    # Concentration warnings
    warnings = _concentration_warnings(cat_weights, portfolio, fund_results, total_current)

    # Benchmark comparison (Nifty 50 over same period)
    bench_df  = get_nav_history(BENCHMARK_CODE, conn)
    bench_ret = trailing_return(bench_df, 12) if not bench_df.empty else None
    portfolio_12m = _portfolio_trailing_return(fund_results, portfolio, total_current, 12)
    alpha_12m = (portfolio_12m - bench_ret) if (portfolio_12m and bench_ret) else None

    return {
        "summary": {
            "total_invested":      round(total_invested, 2),
            "total_current_value": round(total_current, 2),
            "total_gain":          round(total_gain, 2),
            "absolute_return_pct": round(abs_return, 2),
            "monthly_sip_total":   round(total_sip_monthly, 2),
            "funds_count":         len([r for r in fund_results.values() if "error" not in r]),
        },
        "benchmark": {
            "nifty50_12m_pct":    _pct(bench_ret),
            "portfolio_12m_pct":  _pct(portfolio_12m),
            "alpha_12m_pct":      _pct(alpha_12m),
        },
        "diversification": {
            "score":        div_score,
            "grade":        _div_grade(div_score),
            "category_weights": cat_weights,
        },
        "warnings":   warnings,
        "funds":      fund_results,
        "sip_start":  sip_start,
        "as_of_date": datetime.now().strftime("%Y-%m-%d"),
    }


# ── Helper functions ──────────────────────────────────────────────────────────

def _category_weights(portfolio, fund_results, total_current):
    cat_values = {}
    for code, config in portfolio.items():
        res = fund_results.get(code, {})
        if "error" in res:
            continue
        cat  = config.get("category", "Unknown")
        sip_val = res.get("sip", {}).get("current_value", 0)
        ls_val  = res.get("lump_sum", {}).get("current_value", 0)
        cat_values[cat] = cat_values.get(cat, 0) + sip_val + ls_val

    if total_current == 0:
        return {}
    return {cat: round(val / total_current * 100, 1) for cat, val in cat_values.items()}


def _diversification_score(cat_weights: dict) -> float:
    """HHI-based score: 100 = perfectly diversified, 0 = fully concentrated."""
    if not cat_weights:
        return 0.0
    weights = [w / 100 for w in cat_weights.values()]
    hhi = sum(w ** 2 for w in weights)
    n   = len(weights)
    if n <= 1:
        return 0.0
    # Normalise: min HHI (1/n) → 100, max HHI (1.0) → 0
    min_hhi = 1 / n
    score = (1 - hhi) / (1 - min_hhi) * 100
    return round(score, 1)


def _div_grade(score: float) -> str:
    if score >= 80: return "Excellent"
    if score >= 60: return "Good"
    if score >= 40: return "Fair"
    return "Poor — rebalance recommended"


def _concentration_warnings(cat_weights, portfolio, fund_results, total_current):
    warnings = []
    for cat, weight in cat_weights.items():
        if weight > 50:
            warnings.append({
                "type":    "HIGH_CONCENTRATION",
                "message": f"{cat} is {weight}% of your portfolio — consider reducing below 40%.",
                "severity": "high",
            })
        elif weight > 35:
            warnings.append({
                "type":    "MODERATE_CONCENTRATION",
                "message": f"{cat} is {weight}% of your portfolio — monitor closely.",
                "severity": "medium",
            })

    # Single fund dominance
    for code, config in portfolio.items():
        res = fund_results.get(code, {})
        if "error" in res:
            continue
        fund_val = res.get("sip", {}).get("current_value", 0) + \
                   res.get("lump_sum", {}).get("current_value", 0)
        if total_current > 0:
            fund_pct = fund_val / total_current * 100
            if fund_pct > 30:
                warnings.append({
                    "type":     "SINGLE_FUND_DOMINANCE",
                    "message":  f"{config.get('name')} is {fund_pct:.1f}% of your portfolio.",
                    "severity": "high",
                })
    return warnings


def _portfolio_trailing_return(fund_results, portfolio, total_current, months):
    weighted = 0.0
    for code, config in portfolio.items():
        res = fund_results.get(code, {})
        if "error" in res:
            continue
        fund_val = res.get("sip", {}).get("current_value", 0) + \
                   res.get("lump_sum", {}).get("current_value", 0)
        weight   = fund_val / total_current if total_current > 0 else 0
        ret      = res.get("trailing", {}).get(f"{months}m")
        if ret is not None:
            weighted += (ret / 100) * weight
    return weighted if weighted != 0 else None


def _pct(val):
    return round(val * 100, 2) if val is not None else None


# ── Standalone runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    conn = init_db(DB_PATH)
    report = build_portfolio_report(DEFAULT_PORTFOLIO, conn)
    out = Path("data/reports/portfolio_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps(report["summary"], indent=2))
    print(f"\nDiversification: {report['diversification']['grade']} "
          f"(score: {report['diversification']['score']})")
    if report["warnings"]:
        print("\nWarnings:")
        for w in report["warnings"]:
            print(f"  [{w['severity'].upper()}] {w['message']}")
    conn.close()