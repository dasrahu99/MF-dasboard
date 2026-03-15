"""
drawdown.py — Drawdown Recovery Tracking Engine (Module 5)
Tracks every significant drawdown event, recovery time, and SIP benefit during crashes.
Tells you exactly how long to hold through a crash and when SIP investing helps most.
"""

import numpy as np
import pandas as pd
import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from fetcher import get_nav_history, fetch_historical_nav, DB_PATH, init_db

log = logging.getLogger(__name__)
REPORTS_DIR = Path("data/reports")


# ── Drawdown event detection ───────────────────────────────────────────────────

def find_drawdown_events(df: pd.DataFrame,
                          threshold: float = 0.10) -> list:
    """
    Identify all drawdown events where NAV fell >= threshold from a peak.
    Returns list of events with: peak, trough, recovery, depth, duration, recovery_days.

    threshold: minimum drawdown to register as an event (0.10 = 10%)
    """
    df = df.sort_values("nav_date").reset_index(drop=True)
    navs  = df["nav"].values
    dates = df["nav_date"].values

    events = []
    i = 0
    n = len(navs)

    while i < n:
        # Find local peak
        peak_val  = navs[i]
        peak_idx  = i

        j = i + 1
        while j < n and navs[j] >= peak_val:
            peak_val = navs[j]
            peak_idx = j
            j += 1

        if j >= n:
            break

        # Find trough from peak
        trough_val = navs[j]
        trough_idx = j
        k = j + 1
        while k < n and navs[k] <= trough_val:
            trough_val = navs[k]
            trough_idx = k
            k += 1

        depth = (trough_val - peak_val) / peak_val   # negative number

        if abs(depth) >= threshold:
            # Find recovery: first day NAV >= peak_val after trough
            recovery_idx  = None
            recovery_days = None
            for m in range(trough_idx + 1, n):
                if navs[m] >= peak_val:
                    recovery_idx  = m
                    recovery_days = (pd.Timestamp(dates[m]) -
                                     pd.Timestamp(dates[trough_idx])).days
                    break

            event = {
                "peak_date":       str(pd.Timestamp(dates[peak_idx]).date()),
                "peak_nav":        round(float(peak_val), 4),
                "trough_date":     str(pd.Timestamp(dates[trough_idx]).date()),
                "trough_nav":      round(float(trough_val), 4),
                "drawdown_pct":    round(depth * 100, 2),
                "drawdown_days":   int((pd.Timestamp(dates[trough_idx]) -
                                        pd.Timestamp(dates[peak_idx])).days),
                "recovered":       recovery_idx is not None,
                "recovery_date":   str(pd.Timestamp(dates[recovery_idx]).date())
                                   if recovery_idx else "Not yet recovered",
                "recovery_days":   recovery_days,
                "total_event_days":int((pd.Timestamp(dates[recovery_idx]) -
                                        pd.Timestamp(dates[peak_idx])).days)
                                   if recovery_idx else None,
            }

            # SIP benefit during drawdown: units accumulated below peak
            event["sip_benefit"] = _sip_benefit_during_drawdown(
                df, peak_idx, trough_idx,
                recovery_idx if recovery_idx else min(trough_idx + 365, n - 1)
            )

            events.append(event)

        i = trough_idx + 1 if k > trough_idx else k

    return sorted(events, key=lambda e: e["drawdown_pct"])   # worst first


def _sip_benefit_during_drawdown(df: pd.DataFrame,
                                   peak_idx: int,
                                   trough_idx: int,
                                   recovery_idx: int,
                                   monthly_sip: float = 10000) -> dict:
    """
    Simulate SIP of ₹10k/month during drawdown vs lump sum at peak.
    Shows rupee cost averaging benefit.
    """
    navs  = df["nav"].values
    dates = df["nav_date"].values

    # Lump sum at peak
    lump_units = monthly_sip * 12 / navs[peak_idx]   # 1 year of SIP as lump sum
    lump_value_at_recovery = lump_units * navs[recovery_idx]

    # SIP during drawdown (monthly)
    sip_units = 0.0
    total_invested = 0.0
    peak_date  = pd.Timestamp(dates[peak_idx])
    recovery_date = pd.Timestamp(dates[recovery_idx])
    months = int((recovery_date - peak_date).days / 30) + 1

    for m in range(months):
        sip_date = peak_date + pd.DateOffset(months=m)
        closest  = df[df["nav_date"] <= sip_date]
        if closest.empty:
            continue
        sip_nav     = closest["nav"].iloc[-1]
        units       = monthly_sip / sip_nav
        sip_units  += units
        total_invested += monthly_sip

    sip_value_at_recovery = sip_units * navs[recovery_idx]
    lump_invested = monthly_sip * months

    return {
        "sip_total_invested":   round(total_invested, 2),
        "sip_value_at_recovery":round(sip_value_at_recovery, 2),
        "sip_return_pct":       round((sip_value_at_recovery - total_invested) /
                                       total_invested * 100, 2) if total_invested > 0 else None,
        "lump_sum_invested":    round(lump_invested, 2),
        "lump_value_at_recovery":round(lump_value_at_recovery, 2),
        "lump_return_pct":      round((lump_value_at_recovery - lump_invested) /
                                       lump_invested * 100, 2) if lump_invested > 0 else None,
        "sip_advantage_pct":    round(
            ((sip_value_at_recovery - total_invested) / total_invested -
             (lump_value_at_recovery - lump_invested) / lump_invested) * 100, 2
        ) if total_invested > 0 and lump_invested > 0 else None,
    }


# ── Summary statistics ────────────────────────────────────────────────────────

def drawdown_summary(events: list) -> dict:
    """Aggregate statistics across all drawdown events."""
    if not events:
        return {}

    recovered = [e for e in events if e["recovered"]]
    depths     = [abs(e["drawdown_pct"]) for e in events]
    rec_days   = [e["recovery_days"] for e in recovered if e["recovery_days"]]

    return {
        "total_events":           len(events),
        "worst_drawdown_pct":     round(max(depths), 2),
        "avg_drawdown_pct":       round(sum(depths) / len(depths), 2),
        "events_recovered":       len(recovered),
        "avg_recovery_days":      round(sum(rec_days) / len(rec_days), 0) if rec_days else None,
        "avg_recovery_months":    round(sum(rec_days) / len(rec_days) / 30, 1) if rec_days else None,
        "max_recovery_days":      max(rec_days) if rec_days else None,
        "currently_in_drawdown":  not events[0]["recovered"] if events else False,
        "sip_always_wins":        all(
            (e["sip_benefit"].get("sip_advantage_pct") or 0) > 0
            for e in recovered if e.get("sip_benefit")
        ),
    }


# ── Full drawdown report ──────────────────────────────────────────────────────

def full_drawdown_report(scheme_code: str,
                          fund_df: pd.DataFrame,
                          scheme_name: str = "",
                          threshold: float = 0.10) -> dict:
    """Complete drawdown analysis for one fund."""
    events  = find_drawdown_events(fund_df, threshold)
    summary = drawdown_summary(events)

    # Current drawdown status
    df_sorted   = fund_df.sort_values("nav_date")
    current_nav = df_sorted["nav"].iloc[-1]
    ath         = df_sorted["nav"].max()
    current_dd  = (current_nav - ath) / ath * 100

    return {
        "scheme_code":     scheme_code,
        "scheme_name":     scheme_name,
        "current_nav":     round(current_nav, 4),
        "all_time_high":   round(ath, 4),
        "current_drawdown_pct": round(current_dd, 2),
        "recovery_needed_pct":  round((ath / current_nav - 1) * 100, 2)
                                 if current_nav < ath else 0,
        "summary":  summary,
        "events":   events[:10],   # top 10 worst drawdowns
    }


# ── Batch run ─────────────────────────────────────────────────────────────────

def run_drawdown_analysis(watchlist: dict,
                           conn: sqlite3.Connection) -> dict:
    results = {}
    for code, name in watchlist.items():
        log.info("Drawdown analysis: %s", name)
        df = get_nav_history(code, conn)
        if df.empty:
            df = fetch_historical_nav(code, conn)
        if df.empty:
            continue
        results[code] = full_drawdown_report(code, df, name)

    out = REPORTS_DIR / "drawdown_analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Merge into latest.json
    latest_path = REPORTS_DIR / "latest.json"
    if latest_path.exists():
        with open(latest_path) as f:
            report = json.load(f)
        for code, dd_data in results.items():
            if code in report:
                report[code]["drawdown"] = dd_data
        with open(latest_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

    log.info("Drawdown analysis saved → %s", out)
    _print_drawdown_table(results)
    return results


def _print_drawdown_table(results: dict):
    rows = []
    for code, r in results.items():
        s = r.get("summary", {})
        rows.append({
            "Fund":              r.get("scheme_name", "")[:30],
            "Current DD%":       r.get("current_drawdown_pct"),
            "Worst DD%":         s.get("worst_drawdown_pct"),
            "Avg Recovery (mo)": s.get("avg_recovery_months"),
            "Max Recovery (days)":s.get("max_recovery_days"),
            "Events":            s.get("total_events"),
            "SIP Always Wins":   s.get("sip_always_wins"),
        })
    df = pd.DataFrame(rows)
    print("\n" + "=" * 90)
    print("  DRAWDOWN RECOVERY ANALYSIS")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90 + "\n")


if __name__ == "__main__":
    from pipeline import WATCHLIST
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    conn = init_db(DB_PATH)
    run_drawdown_analysis(WATCHLIST, conn)
    conn.close()