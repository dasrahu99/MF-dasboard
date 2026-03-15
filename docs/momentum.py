"""
momentum.py — Momentum Scoring Engine
Computes multi-factor momentum scores for Indian mutual funds.
Plugs directly into the existing pipeline (reads from mf_navs.db, outputs to latest.json).

Factors scored:
  1. Price momentum        — 1M, 3M, 6M, 12M trailing returns
  2. Momentum acceleration — is momentum speeding up or slowing down?
  3. Trend strength        — NAV above/below 50d/200d MA, slope angle
  4. Relative momentum     — fund vs Nifty 50 benchmark
  5. Consistency score     — % of rolling 3M periods with positive returns
  6. Momentum quality      — reward/risk ratio of up-months vs down-months

Final score: 0–100 composite (weighted blend of all factors)
Signal:  STRONG BUY (≥75) | BUY (≥60) | NEUTRAL (≥40) | REDUCE (<40)
"""

import numpy as np
import pandas as pd
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fetcher import init_db, fetch_historical_nav, get_nav_history, DB_PATH
from engine  import _rsi, volatility

log = logging.getLogger(__name__)

REPORTS_DIR = Path("data/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Nifty 50 index fund as benchmark (UTI Nifty 50 Index — Growth)
BENCHMARK_CODE = "100356"

# Factor weights — tune these to bias toward your style
WEIGHTS = {
    "momentum_3m":       0.20,   # recent 3M return (most predictive short-term)
    "momentum_6m":       0.20,   # 6M return
    "momentum_12m":      0.15,   # 12M return
    "acceleration":      0.15,   # is momentum accelerating?
    "trend_strength":    0.10,   # MA alignment + slope
    "relative_momentum": 0.10,   # alpha vs benchmark
    "consistency":       0.05,   # % positive rolling periods
    "momentum_quality":  0.05,   # up capture vs down capture ratio
}


# ── Individual factor computations ────────────────────────────────────────────

def trailing_return(df: pd.DataFrame, months: int) -> Optional[float]:
    """Return over trailing N months (annualised for >3M, raw for ≤1M)."""
    df = df.sort_values("nav_date")
    end_nav  = df["nav"].iloc[-1]
    end_date = df["nav_date"].max()
    start_target = end_date - pd.DateOffset(months=months)
    past = df[df["nav_date"] <= start_target]
    if past.empty:
        return None
    start_nav = past["nav"].iloc[-1]
    raw_return = (end_nav - start_nav) / start_nav
    # Annualise for periods > 3 months
    if months > 3:
        years = (end_date - past["nav_date"].iloc[-1]).days / 365.25
        if years <= 0:
            return None
        return (1 + raw_return) ** (1 / years) - 1
    return raw_return


def momentum_acceleration(df: pd.DataFrame) -> Optional[float]:
    """
    Measures if momentum is accelerating.
    Compares recent 3M return vs prior 3M return.
    Positive = accelerating, negative = decelerating.
    """
    df = df.sort_values("nav_date")
    end_date = df["nav_date"].max()

    # Recent 3M
    t3 = end_date - pd.DateOffset(months=3)
    t6 = end_date - pd.DateOffset(months=6)

    recent = df[df["nav_date"] >= t3]["nav"]
    prior_slice = df[(df["nav_date"] >= t6) & (df["nav_date"] < t3)]["nav"]

    if recent.empty or prior_slice.empty or recent.iloc[0] == 0:
        return None

    recent_return = (recent.iloc[-1] - recent.iloc[0]) / recent.iloc[0]
    prior_return  = (prior_slice.iloc[-1] - prior_slice.iloc[0]) / prior_slice.iloc[0] \
                    if prior_slice.iloc[0] != 0 else 0

    return recent_return - prior_return


def trend_strength(df: pd.DataFrame) -> dict:
    """
    MA alignment score + trend slope.
    Returns score 0–1 and component details.
    """
    df = df.sort_values("nav_date").copy()
    navs = df["nav"]
    current = navs.iloc[-1]

    ma20  = navs.rolling(20,  min_periods=10).mean().iloc[-1]
    ma50  = navs.rolling(50,  min_periods=30).mean().iloc[-1]
    ma200 = navs.rolling(200, min_periods=100).mean().iloc[-1]

    score = 0.5   # neutral baseline

    # Price vs MAs
    if not np.isnan(ma20)  and current > ma20:   score += 0.10
    if not np.isnan(ma50)  and current > ma50:   score += 0.15
    if not np.isnan(ma200) and current > ma200:  score += 0.20

    # MA stack order (bullish alignment: ma20 > ma50 > ma200)
    if not any(np.isnan([ma20, ma50, ma200])):
        if ma20 > ma50 > ma200:
            score += 0.15
        elif ma20 < ma50 < ma200:
            score -= 0.20

    # 50d slope — rate of change
    if len(navs) >= 50:
        slope = (navs.iloc[-1] - navs.iloc[-50]) / navs.iloc[-50]
        if slope > 0.05:    score += 0.10
        elif slope < -0.05: score -= 0.10

    return {
        "score": max(0.0, min(1.0, score)),
        "ma20":  round(float(ma20),  4) if not np.isnan(ma20)  else None,
        "ma50":  round(float(ma50),  4) if not np.isnan(ma50)  else None,
        "ma200": round(float(ma200), 4) if not np.isnan(ma200) else None,
        "price_above_ma50":  bool(current > ma50)  if not np.isnan(ma50)  else None,
        "price_above_ma200": bool(current > ma200) if not np.isnan(ma200) else None,
        "ma_bullish_stack":  bool(ma20 > ma50 > ma200)
                             if not any(np.isnan([ma20, ma50, ma200])) else None,
    }


def relative_momentum(fund_df: pd.DataFrame,
                       bench_df: pd.DataFrame,
                       months: int = 6) -> Optional[float]:
    """
    Fund return minus benchmark return over N months.
    Positive = outperforming benchmark (alpha).
    """
    fund_ret  = trailing_return(fund_df,  months)
    bench_ret = trailing_return(bench_df, months)
    if fund_ret is None or bench_ret is None:
        return None
    return fund_ret - bench_ret


def consistency_score(df: pd.DataFrame,
                       window_months: int = 3,
                       lookback_months: int = 36) -> Optional[float]:
    """
    % of rolling `window_months` periods (over last `lookback_months`)
    that had positive returns. Measures reliability.
    """
    df = df.sort_values("nav_date")
    cutoff = df["nav_date"].max() - pd.DateOffset(months=lookback_months)
    df = df[df["nav_date"] >= cutoff].copy()

    if len(df) < 30:
        return None

    # Monthly resampled NAV
    df_monthly = df.set_index("nav_date")["nav"].resample("ME").last().dropna()
    if len(df_monthly) < window_months + 1:
        return None

    positive = 0
    total = 0
    for i in range(window_months, len(df_monthly)):
        start_nav = df_monthly.iloc[i - window_months]
        end_nav   = df_monthly.iloc[i]
        if start_nav > 0:
            total += 1
            if end_nav > start_nav:
                positive += 1

    return positive / total if total > 0 else None


def momentum_quality(df: pd.DataFrame,
                      lookback_months: int = 12) -> Optional[float]:
    """
    Up-capture / down-capture ratio using monthly returns.
    >1.0 = fund captures more upside than downside (quality momentum).
    Normalised to 0–1 for scoring.
    """
    df = df.sort_values("nav_date")
    cutoff = df["nav_date"].max() - pd.DateOffset(months=lookback_months)
    df = df[df["nav_date"] >= cutoff].copy()

    df_monthly = df.set_index("nav_date")["nav"].resample("ME").last().dropna()
    monthly_returns = df_monthly.pct_change().dropna()

    if len(monthly_returns) < 6:
        return None

    up_months   = monthly_returns[monthly_returns > 0]
    down_months = monthly_returns[monthly_returns < 0]

    if down_months.empty or up_months.empty:
        return 0.8 if up_months.empty else 1.0

    avg_up   = up_months.mean()
    avg_down = abs(down_months.mean())

    ratio = avg_up / avg_down if avg_down > 0 else 2.0
    # Normalise: ratio of 1 → 0.5, ratio of 2 → 1.0, ratio of 0.5 → 0.0
    return max(0.0, min(1.0, (ratio - 0.5) / 1.5))


# ── Score normalisation helpers ───────────────────────────────────────────────

def _normalise_return(ret: Optional[float],
                      low: float = -0.20,
                      high: float = 0.60) -> float:
    """Map a return value to 0–1 range."""
    if ret is None:
        return 0.5
    return max(0.0, min(1.0, (ret - low) / (high - low)))


def _normalise_accel(accel: Optional[float],
                     low: float = -0.15,
                     high: float = 0.15) -> float:
    if accel is None:
        return 0.5
    return max(0.0, min(1.0, (accel - low) / (high - low)))


def _normalise_relative(rel: Optional[float],
                         low: float = -0.10,
                         high: float = 0.20) -> float:
    if rel is None:
        return 0.5
    return max(0.0, min(1.0, (rel - low) / (high - low)))


# ── Master momentum score ─────────────────────────────────────────────────────

def compute_momentum_score(fund_df: pd.DataFrame,
                            bench_df: pd.DataFrame,
                            scheme_code: str = "",
                            scheme_name: str = "") -> dict:
    """
    Compute the full multi-factor momentum score for one fund.
    Returns a rich dict with all factor values and composite score.
    """
    if fund_df.empty:
        return {"error": "No data", "scheme_code": scheme_code}

    # ── Raw factor values ──
    ret_1m  = trailing_return(fund_df, 1)
    ret_3m  = trailing_return(fund_df, 3)
    ret_6m  = trailing_return(fund_df, 6)
    ret_12m = trailing_return(fund_df, 12)
    accel   = momentum_acceleration(fund_df)
    trend   = trend_strength(fund_df)
    rel_6m  = relative_momentum(fund_df, bench_df, 6)
    consist = consistency_score(fund_df)
    quality = momentum_quality(fund_df)
    rsi     = _rsi(fund_df.sort_values("nav_date")["nav"], 14)

    # ── Normalise each factor to 0–1 ──
    factors = {
        "momentum_3m":       _normalise_return(ret_3m,  -0.10, 0.30),
        "momentum_6m":       _normalise_return(ret_6m,  -0.15, 0.50),
        "momentum_12m":      _normalise_return(ret_12m, -0.20, 0.60),
        "acceleration":      _normalise_accel(accel),
        "trend_strength":    trend["score"],
        "relative_momentum": _normalise_relative(rel_6m),
        "consistency":       consist if consist is not None else 0.5,
        "momentum_quality":  quality if quality is not None else 0.5,
    }

    # ── Weighted composite score (0–100) ──
    composite = sum(factors[k] * WEIGHTS[k] for k in WEIGHTS) * 100

    # RSI override: cap score if severely overbought/oversold
    if rsi:
        if rsi > 80:   composite = min(composite, 55)   # overbought — cap
        if rsi < 25:   composite = max(composite, 30)   # oversold  — floor

    composite = round(composite, 1)

    # ── Signal ──
    if composite >= 75:
        signal    = "STRONG BUY"
        action    = "High momentum with acceleration. Ideal for lump sum top-up."
        color     = "green"
    elif composite >= 60:
        signal    = "BUY"
        action    = "Positive momentum trend. Good for SIP increase or small lump sum."
        color     = "green"
    elif composite >= 40:
        signal    = "NEUTRAL"
        action    = "Mixed signals. Maintain existing SIP, avoid fresh lump sum."
        color     = "yellow"
    else:
        signal    = "REDUCE"
        action    = "Momentum deteriorating. Consider pausing SIP top-up, review allocation."
        color     = "red"

    return {
        "scheme_code":   scheme_code,
        "scheme_name":   scheme_name,
        "as_of_date":    str(fund_df["nav_date"].max().date()),
        "composite_score": composite,
        "signal":        signal,
        "action":        action,
        "color":         color,
        "factors": {
            "trailing_returns": {
                "1m_pct":  _pct(ret_1m),
                "3m_pct":  _pct(ret_3m),
                "6m_pct":  _pct(ret_6m),
                "12m_pct": _pct(ret_12m),
            },
            "acceleration_pct":      _pct(accel),
            "trend":                 trend,
            "relative_6m_alpha_pct": _pct(rel_6m),
            "consistency_pct":       _pct(consist),
            "momentum_quality_ratio":round(quality, 3) if quality else None,
            "rsi_14":                round(rsi, 1) if rsi else None,
        },
        "factor_scores": {k: round(v, 3) for k, v in factors.items()},
    }


def _pct(val):
    return round(val * 100, 2) if val is not None else None


# ── Rank all watchlist funds ───────────────────────────────────────────────────

def rank_funds_by_momentum(watchlist: dict,
                            conn: sqlite3.Connection) -> list:
    """
    Score all funds in watchlist, rank by composite momentum score.
    Returns list of dicts sorted highest → lowest score.
    """
    log.info("Loading benchmark NAV history...")
    bench_df = get_nav_history(BENCHMARK_CODE, conn)
    if bench_df.empty:
        log.info("Fetching benchmark history from mfapi...")
        bench_df = fetch_historical_nav(BENCHMARK_CODE, conn)

    results = []
    for code, name in watchlist.items():
        log.info("Scoring momentum: %s", name)
        fund_df = get_nav_history(code, conn)
        if fund_df.empty:
            fund_df = fetch_historical_nav(code, conn)
        if fund_df.empty:
            log.warning("No data for %s — skipping", name)
            continue
        score = compute_momentum_score(fund_df, bench_df,
                                        scheme_code=code,
                                        scheme_name=name)
        results.append(score)

    results.sort(key=lambda x: x.get("composite_score", 0), reverse=True)
    log.info("Momentum scoring complete — %d funds ranked", len(results))
    return results


def enrich_latest_report(watchlist: dict,
                          conn: sqlite3.Connection):
    """
    Load latest.json, add momentum scores to each fund entry,
    save back to latest.json and momentum_scores.json.
    """
    latest_path = REPORTS_DIR / "latest.json"
    if not latest_path.exists():
        log.warning("latest.json not found — run pipeline.py first")
        return

    with open(latest_path) as f:
        report = json.load(f)

    rankings = rank_funds_by_momentum(watchlist, conn)
    rankings_by_code = {r["scheme_code"]: r for r in rankings}

    # Merge momentum into existing report
    for code, fund_data in report.items():
        if code in rankings_by_code:
            fund_data["momentum"] = rankings_by_code[code]

    # Save enriched latest.json
    with open(latest_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Save standalone momentum rankings
    momentum_path = REPORTS_DIR / "momentum_scores.json"
    with open(momentum_path, "w") as f:
        json.dump(rankings, f, indent=2, default=str)

    log.info("Saved enriched latest.json and momentum_scores.json")
    _print_momentum_table(rankings)


def _print_momentum_table(rankings: list):
    """Print ranked momentum table to console."""
    rows = []
    for r in rankings:
        f = r.get("factors", {})
        tr = f.get("trailing_returns", {})
        rows.append({
            "Fund":          r.get("scheme_name","")[:30],
            "Score":         r.get("composite_score"),
            "Signal":        r.get("signal"),
            "1M%":           tr.get("1m_pct"),
            "3M%":           tr.get("3m_pct"),
            "6M%":           tr.get("6m_pct"),
            "12M%":          tr.get("12m_pct"),
            "Accel%":        f.get("acceleration_pct"),
            "Alpha6M%":      f.get("relative_6m_alpha_pct"),
            "RSI":           f.get("rsi_14"),
            "Consistency%":  f.get("consistency_pct"),
        })
    df = pd.DataFrame(rows)
    print("\n" + "=" * 90)
    print("  MOMENTUM RANKINGS — highest score = strongest momentum")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90 + "\n")


# ── Standalone runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from pipeline import WATCHLIST

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("data/momentum.log"),
            logging.StreamHandler(),
        ]
    )

    conn = init_db(DB_PATH)
    enrich_latest_report(WATCHLIST, conn)
    conn.close()