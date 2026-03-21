"""
contrarian.py — Contrarian Intelligence Engine
Identifies HIGH-CONVICTION BUY opportunities specifically when the market is down.

Core thesis: The best time to invest is when:
  1. Market is in drawdown (fear is high)
  2. Fund fundamentals are still strong (quality hasn't changed)
  3. Fund has historically recovered faster than peers
  4. Valuations are attractive relative to historical NAV range

Signals produced:
  CONTRARIAN BUY    — market down, fund strong, high conviction entry
  ACCUMULATE        — market weak, fund holding up well, add to SIP
  VALUE ZONE        — fund significantly below historical fair value
  AVOID             — fund weak AND market weak (double risk)
  WAIT FOR BOTTOM   — still falling, not yet stabilised

This engine INVERTS the momentum logic intentionally.
High momentum = expensive. Low momentum + strong fundamentals = opportunity.
"""

import json
import logging
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fetcher  import get_nav_history, fetch_historical_nav, DB_PATH, init_db
from engine   import compute_cagr_for_horizon, sharpe_ratio, max_drawdown, volatility
from momentum import trailing_return, trend_strength, _rsi

log = logging.getLogger(__name__)
REPORTS_DIR = Path("data/reports")
BENCHMARK_CODE = "100356"   # Nifty 50 index


# ── Market regime detection ───────────────────────────────────────────────────

def detect_market_regime(bench_df: pd.DataFrame) -> dict:
    """
    Classify the current market as:
      BEAR       — Nifty down >20% from recent high
      CORRECTION — Nifty down 10-20%
      PULLBACK   — Nifty down 5-10%
      SIDEWAYS   — Nifty flat ±5%
      BULL       — Nifty in uptrend, near highs

    Also computes:
      - % from 52-week high
      - % from all-time high
      - Market trend (MA50 vs MA200)
      - Panic indicator (RSI < 35 = fear zone)
    """
    if bench_df.empty:
        return {"regime": "UNKNOWN", "error": "No benchmark data"}

    bench_df = bench_df.sort_values("nav_date")
    navs     = bench_df["nav"]
    current  = navs.iloc[-1]
    ath      = navs.max()

    one_year_ago = bench_df["nav_date"].max() - pd.DateOffset(years=1)
    year_navs    = bench_df[bench_df["nav_date"] >= one_year_ago]["nav"]
    high_52w     = year_navs.max() if not year_navs.empty else ath

    pct_from_ath  = (current - ath)    / ath    * 100
    pct_from_52wh = (current - high_52w) / high_52w * 100

    ma50  = navs.rolling(50,  min_periods=30).mean().iloc[-1]
    ma200 = navs.rolling(200, min_periods=100).mean().iloc[-1]
    rsi   = _rsi(navs, 14)

    # 3-month return
    ret_3m = trailing_return(bench_df, 3)
    ret_1m = trailing_return(bench_df, 1)
    ret_6m = trailing_return(bench_df, 6)

    # Regime classification
    if pct_from_ath <= -20:
        regime = "BEAR"
        regime_desc = "Bear market — Nifty down >20% from peak. Maximum opportunity zone."
        opportunity = "HIGH"
    elif pct_from_ath <= -10:
        regime = "CORRECTION"
        regime_desc = "Market correction — Nifty down 10-20%. Excellent accumulation zone."
        opportunity = "HIGH"
    elif pct_from_ath <= -5:
        regime = "PULLBACK"
        regime_desc = "Market pullback — Nifty down 5-10%. Good entry for quality funds."
        opportunity = "MODERATE"
    elif pct_from_ath <= 5:
        regime = "SIDEWAYS"
        regime_desc = "Sideways market. Mixed signals — be selective."
        opportunity = "MODERATE"
    else:
        regime = "BULL"
        regime_desc = "Bull market — Nifty near highs. Be cautious with lump sums."
        opportunity = "LOW"

    # Panic zone override
    panic = rsi is not None and rsi < 35
    if panic:
        regime_desc += " RSI below 35 — PANIC ZONE. Historically the best time to buy."
        opportunity = "VERY HIGH"

    # Trend
    if not any(np.isnan([ma50, ma200])):
        if ma50 > ma200:   trend = "BULLISH"
        elif ma50 < ma200: trend = "BEARISH"
        else:              trend = "NEUTRAL"
    else:
        trend = "NEUTRAL"

    return {
        "regime":           regime,
        "regime_desc":      regime_desc,
        "opportunity_level":opportunity,
        "panic_zone":       panic,
        "current_nav":      round(float(current), 2),
        "ath":              round(float(ath), 2),
        "52w_high":         round(float(high_52w), 2),
        "pct_from_ath":     round(pct_from_ath, 2),
        "pct_from_52w_high":round(pct_from_52wh, 2),
        "ma50":             round(float(ma50), 2) if not np.isnan(ma50) else None,
        "ma200":            round(float(ma200), 2) if not np.isnan(ma200) else None,
        "trend":            trend,
        "rsi_14":           round(rsi, 1) if rsi else None,
        "returns": {
            "1m_pct": _pct(ret_1m),
            "3m_pct": _pct(ret_3m),
            "6m_pct": _pct(ret_6m),
        }
    }


# ── Fund quality score (regime-independent) ───────────────────────────────────

def fund_quality_score(df: pd.DataFrame) -> dict:
    """
    Measures INTRINSIC fund quality — independent of current market direction.
    This is what separates funds that RECOVER vs funds that STAY DOWN.

    Factors:
      1. Long-term CAGR consistency    — 3Y, 5Y, 10Y all strong?
      2. Risk-adjusted return (Sharpe) — quality of returns, not just quantity
      3. Recovery speed                — how fast did it recover from past crashes?
      4. Drawdown severity             — does it fall less than peers?
      5. Manager alpha consistency     — persistent outperformance?
      6. Low-market resilience         — returns during market down-months?
    """
    if df.empty:
        return {"quality_score": 0, "error": "No data"}

    score = 0.0
    details = {}

    # 1. CAGR consistency (0-25 pts)
    cagr3  = compute_cagr_for_horizon(df, 3)
    cagr5  = compute_cagr_for_horizon(df, 5)
    cagr10 = compute_cagr_for_horizon(df, 10)

    cagr_pts = 0
    if cagr3  and cagr3  > 0.12: cagr_pts += 8
    if cagr5  and cagr5  > 0.14: cagr_pts += 9
    if cagr10 and cagr10 > 0.14: cagr_pts += 8
    score += cagr_pts
    details["cagr_consistency"] = {
        "points": cagr_pts, "max": 25,
        "3y_pct": _pct(cagr3), "5y_pct": _pct(cagr5), "10y_pct": _pct(cagr10)
    }

    # 2. Sharpe ratio (0-20 pts)
    sharpe = sharpe_ratio(df, 3)
    sharpe_pts = 0
    if sharpe:
        if sharpe > 1.4:   sharpe_pts = 20
        elif sharpe > 1.1: sharpe_pts = 15
        elif sharpe > 0.8: sharpe_pts = 10
        elif sharpe > 0.5: sharpe_pts = 5
    score += sharpe_pts
    details["sharpe"] = {"points": sharpe_pts, "max": 20, "value": round(sharpe, 3) if sharpe else None}

    # 3. Max drawdown severity (0-20 pts) — LOWER drawdown = HIGHER score
    mdd = max_drawdown(df)
    dd_pts = 0
    if mdd < 0.20:   dd_pts = 20
    elif mdd < 0.30: dd_pts = 15
    elif mdd < 0.40: dd_pts = 10
    elif mdd < 0.50: dd_pts = 5
    score += dd_pts
    details["drawdown_resilience"] = {"points": dd_pts, "max": 20, "max_dd_pct": round(mdd*100,1)}

    # 4. Low-market resilience (0-20 pts)
    # % of down-market months where fund lost LESS than 5%
    df_m = df.sort_values("nav_date").set_index("nav_date")["nav"].resample("ME").last()
    monthly_ret = df_m.pct_change().dropna()
    down_months = monthly_ret[monthly_ret < -0.01]
    if len(down_months) >= 6:
        resilient = (down_months > -0.05).sum() / len(down_months)
        resilience_pts = round(resilient * 20)
        details["low_market_resilience"] = {
            "points": resilience_pts, "max": 20,
            "resilience_pct": round(resilient*100, 1),
            "down_months_analysed": len(down_months),
        }
    else:
        resilience_pts = 10   # neutral if insufficient data
        details["low_market_resilience"] = {"points": 10, "max": 20, "note": "Insufficient data"}
    score += resilience_pts

    # 5. Recovery speed from past drawdowns (0-15 pts)
    recovery_pts = _recovery_speed_score(df)
    score += recovery_pts
    details["recovery_speed"] = {"points": recovery_pts, "max": 15}

    quality_score = round(min(score, 100), 1)

    if quality_score >= 75:   grade = "ELITE"
    elif quality_score >= 60: grade = "STRONG"
    elif quality_score >= 45: grade = "AVERAGE"
    elif quality_score >= 30: grade = "WEAK"
    else:                     grade = "POOR"

    return {
        "quality_score": quality_score,
        "grade":         grade,
        "details":       details,
    }


def _recovery_speed_score(df: pd.DataFrame) -> int:
    """Score based on average recovery time from past 10%+ drawdowns."""
    df = df.sort_values("nav_date").reset_index(drop=True)
    navs  = df["nav"].values
    dates = df["nav_date"].values
    recovery_times = []

    i = 0
    while i < len(navs) - 10:
        peak_val = navs[i]
        peak_i   = i
        j = i + 1
        # Find trough
        while j < len(navs) and navs[j] < peak_val:
            peak_val = max(peak_val, navs[j-1] if j > 0 else navs[j])
            if navs[j] < peak_val * 0.90:   # 10%+ drawdown
                trough_val = navs[j]
                trough_i   = j
                # Find recovery
                k = j + 1
                while k < len(navs) and navs[k] < peak_val:
                    k += 1
                if k < len(navs):
                    rec_days = (pd.Timestamp(dates[k]) -
                                pd.Timestamp(dates[trough_i])).days
                    recovery_times.append(rec_days)
                i = k
                break
            j += 1
        else:
            i = j

    if not recovery_times:
        return 8   # neutral

    avg_rec = sum(recovery_times) / len(recovery_times)
    if avg_rec < 90:    return 15   # recovers in < 3 months
    elif avg_rec < 180: return 12   # < 6 months
    elif avg_rec < 365: return 8    # < 1 year
    elif avg_rec < 730: return 4    # < 2 years
    else:               return 0


# ── Valuation zone ────────────────────────────────────────────────────────────

def valuation_zone(df: pd.DataFrame) -> dict:
    """
    Where is the fund's current NAV relative to its historical distribution?
    Uses percentile ranking of current NAV within 3-year rolling window.

    Percentile < 20 = historically cheap (VALUE ZONE)
    Percentile > 80 = historically expensive (AVOID lump sum)
    """
    df = df.sort_values("nav_date")
    three_yr = df["nav_date"].max() - pd.DateOffset(years=3)
    window   = df[df["nav_date"] >= three_yr]["nav"]

    if len(window) < 60:
        return {"percentile": 50, "zone": "INSUFFICIENT_DATA"}

    current    = window.iloc[-1]
    percentile = (window < current).sum() / len(window) * 100

    one_yr  = df["nav_date"].max() - pd.DateOffset(years=1)
    yr_navs = df[df["nav_date"] >= one_yr]["nav"]
    fair_value_est = yr_navs.mean() if not yr_navs.empty else current

    discount_to_fair = (current - fair_value_est) / fair_value_est * 100

    if percentile < 15:
        zone = "DEEP VALUE"
        desc = "NAV in bottom 15% of 3Y range — historically excellent entry."
    elif percentile < 30:
        zone = "VALUE ZONE"
        desc = "NAV in bottom 30% of 3Y range — attractive entry point."
    elif percentile < 50:
        zone = "FAIR VALUE"
        desc = "NAV near 3Y median — reasonable entry."
    elif percentile < 75:
        zone = "SLIGHTLY EXPENSIVE"
        desc = "NAV in upper half of 3Y range — prefer SIP over lump sum."
    else:
        zone = "EXPENSIVE"
        desc = "NAV in top 25% of 3Y range — wait for pullback."

    return {
        "percentile":         round(percentile, 1),
        "zone":               zone,
        "description":        desc,
        "current_nav":        round(float(current), 4),
        "3y_nav_min":         round(float(window.min()), 4),
        "3y_nav_max":         round(float(window.max()), 4),
        "3y_nav_median":      round(float(window.median()), 4),
        "fair_value_est":     round(float(fair_value_est), 4),
        "discount_to_fair_pct": round(discount_to_fair, 2),
    }


# ── Contrarian signal engine ──────────────────────────────────────────────────

def contrarian_signal(
    fund_df:      pd.DataFrame,
    market:       dict,
    quality:      dict,
    valuation:    dict,
    scheme_code:  str = "",
    scheme_name:  str = "",
    category:     str = "",
) -> dict:
    """
    Combines market regime + fund quality + valuation into a single
    contrarian investment recommendation.

    The key insight: a HIGH QUALITY fund at LOW VALUATION during a DOWN MARKET
    is the single best investment opportunity. This function finds exactly that.
    """
    regime      = market.get("regime", "UNKNOWN")
    opportunity = market.get("opportunity_level", "LOW")
    q_score     = quality.get("quality_score", 0)
    q_grade     = quality.get("grade", "WEAK")
    val_zone    = valuation.get("zone", "FAIR VALUE")
    val_pct     = valuation.get("percentile", 50)
    pct_from_ath= market.get("pct_from_ath", 0)
    panic       = market.get("panic_zone", False)

    # ── Composite contrarian score (0–100) ──
    score = 0.0

    # Market component (35 pts) — worse market = higher score
    market_pts = {
        "BEAR":       35,
        "CORRECTION": 28,
        "PULLBACK":   20,
        "SIDEWAYS":   12,
        "BULL":       5,
        "UNKNOWN":    10,
    }.get(regime, 10)
    if panic: market_pts = min(35, market_pts + 8)
    score += market_pts

    # Quality component (40 pts)
    quality_pts = round(q_score * 0.40)
    score += quality_pts

    # Valuation component (25 pts) — cheaper = higher score
    val_pts = max(0, round((100 - val_pct) * 0.25))
    score += val_pts

    score = round(min(score, 100), 1)

    # ── Signal determination ──
    is_market_down  = regime in ("BEAR", "CORRECTION", "PULLBACK")
    is_quality_good = q_grade in ("ELITE", "STRONG")
    is_cheap        = val_zone in ("DEEP VALUE", "VALUE ZONE", "FAIR VALUE")
    is_expensive    = val_zone in ("EXPENSIVE",)

    if is_market_down and is_quality_good and is_cheap:
        signal   = "CONTRARIAN BUY"
        action   = "Market down + fund strong + valuation attractive. Ideal lump sum entry."
        strategy = "Deploy 2-3x your normal monthly SIP as lump sum. Continue regular SIP."
        color    = "cyan"
        priority = 1

    elif is_market_down and is_quality_good and is_expensive:
        signal   = "ACCUMULATE"
        action   = "Market down, fund holding strong. Slightly expensive but quality justifies entry."
        strategy = "Increase SIP by 50%. Avoid large lump sum — wait for deeper pullback."
        color    = "cyan"
        priority = 2

    elif is_market_down and not is_quality_good and is_cheap:
        signal   = "WAIT FOR BOTTOM"
        action   = "Market down and fund quality is weak. Cheap for a reason — wait for stabilisation."
        strategy = "Maintain existing SIP only. Watch for quality improvement signals."
        color    = "amber"
        priority = 3

    elif is_market_down and not is_quality_good and not is_cheap:
        signal   = "AVOID"
        action   = "Market down AND fund weak AND not cheap. High risk, low reward."
        strategy = "Pause SIP top-ups. Redirect capital to stronger funds."
        color    = "red"
        priority = 4

    elif not is_market_down and is_quality_good and is_cheap:
        signal   = "VALUE ZONE"
        action   = "Market not in correction but fund is trading cheap. Selective opportunity."
        strategy = "Small lump sum entry. Continue SIP. Set alert for further dip."
        color    = "green"
        priority = 2

    elif not is_market_down and is_quality_good:
        signal   = "SIP ONLY"
        action   = "Good fund but market is not offering a discount. Continue SIP, skip lump sum."
        strategy = "Maintain SIP. Do not add lump sum at current levels."
        color    = "neutral"
        priority = 3

    else:
        signal   = "REVIEW"
        action   = "Mixed signals. Review fund fundamentals before investing."
        strategy = "Maintain existing SIP only."
        color    = "neutral"
        priority = 4

    # Specific lump sum sizing recommendation
    lump_sum_multiplier = _lump_sum_multiplier(regime, q_score, val_pct, panic)

    return {
        "scheme_code":          scheme_code,
        "scheme_name":          scheme_name,
        "category":             category,
        "contrarian_score":     score,
        "signal":               signal,
        "action":               action,
        "strategy":             strategy,
        "color":                color,
        "priority":             priority,
        "lump_sum_multiplier":  lump_sum_multiplier,
        "lump_sum_advice":      f"Deploy {lump_sum_multiplier}x your normal monthly SIP as one-time investment."
                                 if lump_sum_multiplier > 1 else "Stick to regular SIP only.",
        "market_regime":        regime,
        "opportunity_level":    opportunity,
        "quality_grade":        q_grade,
        "quality_score":        q_score,
        "valuation_zone":       val_zone,
        "valuation_percentile": val_pct,
        "pct_from_market_peak": pct_from_ath,
        "panic_zone":           panic,
    }


def _lump_sum_multiplier(regime, quality_score, val_percentile, panic):
    """How many times your normal monthly SIP should you invest as lump sum?"""
    base = 1.0
    if regime == "BEAR":        base = 3.0
    elif regime == "CORRECTION":base = 2.0
    elif regime == "PULLBACK":  base = 1.5
    else:                       return 1   # no lump sum in bull market

    if quality_score >= 75:      base *= 1.3
    elif quality_score < 45:     base *= 0.5

    if val_percentile < 20:      base *= 1.2
    elif val_percentile > 70:    base *= 0.7

    if panic:                    base *= 1.2

    return round(min(base, 5.0), 1)   # cap at 5x


# ── SIP strategy during down markets ─────────────────────────────────────────

def down_market_sip_strategy(
    market:      dict,
    fund_list:   list,   # list of contrarian signal dicts
    monthly_sip: float = 10000,
) -> dict:
    """
    Given current market regime and fund rankings, produce a complete
    SIP + lump sum allocation strategy for deploying capital right now.
    """
    regime     = market.get("regime", "UNKNOWN")
    is_down    = regime in ("BEAR", "CORRECTION", "PULLBACK")
    buy_funds  = [f for f in fund_list if f["signal"] in ("CONTRARIAN BUY", "ACCUMULATE", "VALUE ZONE")]
    buy_funds.sort(key=lambda x: x["contrarian_score"], reverse=True)

    if not is_down:
        return {
            "market_assessment": f"Market is in {regime} — stick to regular SIP. No lump sum advised.",
            "total_deploy":      monthly_sip,
            "strategy_type":     "REGULAR SIP",
            "allocations":       [],
        }

    # Total capital to deploy = monthly SIP × average multiplier of top funds
    top3   = buy_funds[:3]
    if not top3:
        return {
            "market_assessment": "No high-quality funds found at attractive levels right now.",
            "total_deploy":      monthly_sip,
            "strategy_type":     "HOLD",
            "allocations":       [],
        }

    avg_mult    = sum(f["lump_sum_multiplier"] for f in top3) / len(top3)
    total_deploy = round(monthly_sip * avg_mult)

    # Allocate proportional to contrarian score
    total_score  = sum(f["contrarian_score"] for f in top3)
    allocations  = []
    for f in top3:
        alloc_pct = f["contrarian_score"] / total_score if total_score > 0 else 1/len(top3)
        alloc_amt = round(total_deploy * alloc_pct / 100) * 100
        allocations.append({
            "fund":              f["scheme_name"],
            "signal":            f["signal"],
            "contrarian_score":  f["contrarian_score"],
            "quality_grade":     f["quality_grade"],
            "valuation_zone":    f["valuation_zone"],
            "allocation_pct":    round(alloc_pct * 100, 1),
            "deploy_amount":     alloc_amt,
            "strategy":          f["strategy"],
        })

    market_desc = {
        "BEAR":       "BEAR MARKET — Maximum opportunity. Deploy aggressively into quality.",
        "CORRECTION": "CORRECTION — Strong buying zone. Accumulate quality funds.",
        "PULLBACK":   "PULLBACK — Good entry. Focus on top-quality funds only.",
    }.get(regime, "")

    return {
        "market_assessment": market_desc,
        "regime":            regime,
        "panic_zone":        market.get("panic_zone", False),
        "strategy_type":     "CONTRARIAN DEPLOYMENT",
        "monthly_sip":       monthly_sip,
        "total_deploy":      total_deploy,
        "deploy_multiplier": round(avg_mult, 1),
        "allocations":       allocations,
        "historical_note":   (
            "Historically, deploying 2-3x SIP during Nifty corrections of 10%+ "
            "has generated 25-40% returns within 18 months (2020 COVID crash, "
            "2022 rate hike selloff, 2018 IL&FS crisis)."
        ),
    }


# ── Full contrarian analysis runner ──────────────────────────────────────────

def run_contrarian_analysis(watchlist: dict,
                             conn: sqlite3.Connection,
                             monthly_sip: float = 10000) -> dict:
    """
    Run full contrarian analysis for all watchlist funds.
    Saves to contrarian_analysis.json and enriches latest.json.
    """
    log.info("Running contrarian intelligence engine...")

    # Benchmark / market regime
    bench_df = get_nav_history(BENCHMARK_CODE, conn)
    if bench_df.empty:
        bench_df = fetch_historical_nav(BENCHMARK_CODE, conn)

    market = detect_market_regime(bench_df)
    log.info("Market regime: %s (opportunity: %s)",
             market["regime"], market["opportunity_level"])

    fund_signals = []
    for code, name in watchlist.items():
        log.info("Contrarian analysis: %s", name)
        df = get_nav_history(code, conn)
        if df.empty:
            df = fetch_historical_nav(code, conn)
        if df.empty:
            continue

        # Get category from latest.json if available
        latest_path = REPORTS_DIR / "latest.json"
        category = "Equity"
        if latest_path.exists():
            with open(latest_path) as f:
                rep = json.load(f)
            category = rep.get(code, {}).get("category", "Equity")

        quality   = fund_quality_score(df)
        valuation = valuation_zone(df)
        signal    = contrarian_signal(
            df, market, quality, valuation,
            scheme_code=code, scheme_name=name, category=category
        )
        signal["quality_details"]   = quality.get("details", {})
        signal["valuation_details"] = valuation
        fund_signals.append(signal)

    fund_signals.sort(key=lambda x: x["contrarian_score"], reverse=True)

    # Down-market deployment strategy
    strategy = down_market_sip_strategy(market, fund_signals, monthly_sip)

    result = {
        "generated_at":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "market_regime": market,
        "fund_signals":  fund_signals,
        "deployment_strategy": strategy,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "contrarian_analysis.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    log.info("Contrarian analysis saved → %s", out)

    # Enrich latest.json
    latest_path = REPORTS_DIR / "latest.json"
    if latest_path.exists():
        with open(latest_path) as f:
            rep = json.load(f)
        for sig in fund_signals:
            code = sig.get("scheme_code","")
            if code in rep:
                rep[code]["contrarian"] = sig
        with open(latest_path, "w") as f:
            json.dump(rep, f, indent=2, default=str)

    _print_contrarian_table(market, fund_signals, strategy)
    return result


def _print_contrarian_table(market, signals, strategy):
    print("\n" + "═"*90)
    print(f"  CONTRARIAN INTELLIGENCE  |  Market: {market['regime']}  "
          f"|  Opportunity: {market['opportunity_level']}")
    print("═"*90)
    rows = []
    for s in signals:
        rows.append({
            "Fund":          s.get("scheme_name","")[:28],
            "Signal":        s.get("signal",""),
            "Score":         s.get("contrarian_score"),
            "Quality":       s.get("quality_grade",""),
            "Valuation":     s.get("valuation_zone",""),
            "Multiplier":    f"{s.get('lump_sum_multiplier',1)}x",
            "Deploy":        f"₹{strategy.get('total_deploy',0):,.0f}",
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print("\nDeployment Strategy:", strategy.get("market_assessment",""))
    if strategy.get("allocations"):
        for a in strategy["allocations"]:
            print(f"  ₹{a['deploy_amount']:,.0f} → {a['fund'][:30]}  ({a['signal']})")
    print("═"*90 + "\n")


def _pct(val):
    return round(val * 100, 2) if val is not None else None


if __name__ == "__main__":
    import sys
    import json
    from pipeline import SIP_AMOUNT
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler("data/contrarian.log"),
                                  logging.StreamHandler()])
    conn = init_db(DB_PATH)

    # Load watchlist dynamically from latest.json to ensure we only analyze processed funds
    try:
        with open("data/reports/latest.json") as f:
            rep = json.load(f)
            dynamic_watchlist = {code: data.get("fund_name", data.get("scheme_name", "")) for code, data in rep.items()}
    except FileNotFoundError:
        log.warning("latest.json not found, falling back to dynamic extraction.")
        from pipeline import get_dynamic_watchlist
        from fetcher import fetch_todays_nav
        dynamic_watchlist = get_dynamic_watchlist(fetch_todays_nav())

    run_contrarian_analysis(dynamic_watchlist, conn, SIP_AMOUNT)
    conn.close()