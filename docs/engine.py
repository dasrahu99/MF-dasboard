"""
engine.py — CAGR & returns computation engine
Computes CAGR, rolling returns, Sharpe ratio, drawdown, SIP returns,
lump-sum timing signals, and long-term wealth projections.
"""

import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from typing import Optional

RISK_FREE_RATE = 0.065   # 6.5% — India 10Y G-Sec approximate


# ── Core return calculations ───────────────────────────────────────────────────

def cagr(nav_start: float, nav_end: float, years: float) -> Optional[float]:
    """Compound Annual Growth Rate."""
    if years <= 0 or nav_start <= 0:
        return None
    return (nav_end / nav_start) ** (1 / years) - 1


def compute_cagr_for_horizon(df: pd.DataFrame, years: float) -> Optional[float]:
    """
    Given a NAV DataFrame (columns: nav_date, nav), compute CAGR
    over the last `years` years from the most recent NAV.
    """
    if df.empty:
        return None
    df = df.sort_values("nav_date")
    end_date  = df["nav_date"].max()
    start_date = end_date - pd.DateOffset(years=int(years),
                                          months=int((years % 1) * 12))
    # Find closest available date
    past = df[df["nav_date"] <= start_date]
    if past.empty:
        return None
    nav_start = past.iloc[-1]["nav"]
    nav_end   = df.iloc[-1]["nav"]
    actual_years = (end_date - past.iloc[-1]["nav_date"]).days / 365.25
    if actual_years < years * 0.8:   # need at least 80% of requested period
        return None
    return cagr(nav_start, nav_end, actual_years)


def rolling_returns(df: pd.DataFrame, window_years: float = 1.0) -> pd.Series:
    """
    Compute rolling annualised returns over a moving window.
    Returns a Series indexed by nav_date.
    """
    df = df.sort_values("nav_date").set_index("nav_date")
    window_days = int(window_years * 365)
    results = {}
    navs = df["nav"]
    dates = navs.index.tolist()

    for i, end_dt in enumerate(dates):
        start_target = end_dt - pd.Timedelta(days=window_days)
        past = navs[navs.index <= start_target]
        if past.empty:
            continue
        start_nav = past.iloc[-1]
        end_nav   = navs.iloc[i]
        actual_yrs = (end_dt - past.index[-1]).days / 365.25
        if actual_yrs < window_years * 0.8:
            continue
        results[end_dt] = cagr(start_nav, end_nav, actual_yrs)

    return pd.Series(results, name=f"rolling_{window_years}y_return")


def sharpe_ratio(df: pd.DataFrame, window_years: float = 3.0) -> Optional[float]:
    """
    Annualised Sharpe ratio using daily NAV returns.
    Uses last `window_years` of data.
    """
    df = df.sort_values("nav_date")
    if len(df) < 60:
        return None
    cutoff = df["nav_date"].max() - pd.DateOffset(years=int(window_years))
    df = df[df["nav_date"] >= cutoff]
    daily_returns = df["nav"].pct_change().dropna()
    if daily_returns.std() == 0 or len(daily_returns) < 30:
        return None
    daily_rf = RISK_FREE_RATE / 252
    excess   = daily_returns - daily_rf
    return float((excess.mean() / excess.std()) * np.sqrt(252))


def max_drawdown(df: pd.DataFrame) -> float:
    """Maximum peak-to-trough drawdown (as a positive percentage)."""
    df = df.sort_values("nav_date")
    navs = df["nav"].values
    peak = np.maximum.accumulate(navs)
    drawdowns = (navs - peak) / peak
    return float(abs(drawdowns.min()))


def volatility(df: pd.DataFrame, window_years: float = 3.0) -> Optional[float]:
    """Annualised standard deviation of daily returns."""
    df = df.sort_values("nav_date")
    cutoff = df["nav_date"].max() - pd.DateOffset(years=int(window_years))
    df = df[df["nav_date"] >= cutoff]
    daily_returns = df["nav"].pct_change().dropna()
    if len(daily_returns) < 30:
        return None
    return float(daily_returns.std() * np.sqrt(252))


# ── SIP return engine ──────────────────────────────────────────────────────────

def sip_xirr(df: pd.DataFrame,
             monthly_amount: float,
             start_date: str,
             end_date: str = None) -> dict:
    """
    Simulate SIP (monthly investment on 1st of each month).
    Returns: total_invested, current_value, absolute_return%, XIRR approximation,
             units_held, unit_cost_avg.

    Uses Newton-Raphson to compute XIRR from cashflows.
    """
    df = df.sort_values("nav_date").set_index("nav_date")
    start = pd.Timestamp(start_date)
    end   = pd.Timestamp(end_date) if end_date else df.index.max()

    # Generate monthly SIP dates
    sip_dates = pd.date_range(start=start, end=end, freq="MS")  # Month Start
    cashflows = []   # list of (date, amount)  — negative = invested
    units_held = 0.0

    for sip_date in sip_dates:
        # Find closest NAV on or after sip_date
        available = df[df.index >= sip_date]
        if available.empty:
            break
        purchase_date = available.index[0]
        nav_price     = available["nav"].iloc[0]
        units         = monthly_amount / nav_price
        units_held   += units
        cashflows.append((purchase_date, -monthly_amount))

    if not cashflows or units_held == 0:
        return {}

    # Current value at last NAV
    current_nav   = df["nav"].iloc[-1]
    current_value = units_held * current_nav
    total_invested = abs(sum(c[1] for c in cashflows))
    cashflows.append((end, current_value))   # redemption

    # XIRR via Newton-Raphson
    xirr_val = _xirr(cashflows)

    # Average cost per unit (rupee cost averaging)
    avg_cost = total_invested / units_held

    return {
        "total_invested":   round(total_invested, 2),
        "current_value":    round(current_value, 2),
        "units_held":       round(units_held, 4),
        "avg_cost_per_unit":round(avg_cost, 2),
        "current_nav":      round(current_nav, 2),
        "absolute_return_pct": round((current_value - total_invested) / total_invested * 100, 2),
        "xirr_pct":         round(xirr_val * 100, 2) if xirr_val else None,
        "months_invested":  len(cashflows) - 1,
    }


def _xirr(cashflows: list, guess: float = 0.1, max_iter: int = 100) -> Optional[float]:
    """
    Newton-Raphson XIRR computation.
    cashflows: list of (date, amount) — negative = outflow, positive = inflow.
    """
    if not cashflows:
        return None
    dates   = [cf[0] for cf in cashflows]
    amounts = [cf[1] for cf in cashflows]
    t0      = dates[0]
    times   = [(d - t0).days / 365.25 for d in dates]

    rate = guess
    for _ in range(max_iter):
        npv  = sum(a / (1 + rate) ** t for a, t in zip(amounts, times))
        dnpv = sum(-t * a / (1 + rate) ** (t + 1) for a, t in zip(amounts, times))
        if dnpv == 0:
            break
        new_rate = rate - npv / dnpv
        if abs(new_rate - rate) < 1e-7:
            return new_rate
        rate = new_rate
    return rate if -1 < rate < 100 else None


# ── Lump sum timing signals ────────────────────────────────────────────────────

def lump_sum_signals(df: pd.DataFrame) -> dict:
    """
    Generate lump-sum timing signals based on technical indicators:
    - 52-week high/low proximity
    - 50-day / 200-day moving average crossover
    - RSI (14-day)
    - % off ATH (All-Time High)

    Returns a dict with signal values and a buy/hold/wait recommendation.
    """
    df = df.sort_values("nav_date").copy()
    navs = df["nav"]
    current_nav = navs.iloc[-1]

    # 52-week range
    one_year_ago = df["nav_date"].max() - pd.Timedelta(days=365)
    year_data    = df[df["nav_date"] >= one_year_ago]["nav"]
    high_52w = year_data.max()
    low_52w  = year_data.min()
    pct_from_52w_high = (current_nav - high_52w) / high_52w * 100

    # Moving averages
    ma50  = navs.rolling(50,  min_periods=30).mean().iloc[-1]
    ma200 = navs.rolling(200, min_periods=100).mean().iloc[-1]
    golden_cross = bool(ma50 > ma200) if not (np.isnan(ma50) or np.isnan(ma200)) else None

    # ATH
    ath = navs.max()
    pct_from_ath = (current_nav - ath) / ath * 100

    # RSI (14-day)
    rsi = _rsi(navs, 14)

    # Valuation score: 0–100 (higher = better entry point)
    score = 50.0
    if pct_from_52w_high < -15:   score += 15   # below 52w high — discount
    if pct_from_52w_high < -30:   score += 10   # deep discount
    if golden_cross is False:      score += 10   # bearish MA → opportunity
    if golden_cross is True:       score -= 5
    if rsi and rsi < 40:           score += 15   # oversold
    if rsi and rsi > 70:           score -= 20   # overbought
    if pct_from_ath < -20:         score += 10
    score = max(0, min(100, score))

    if score >= 65:
        signal = "BUY"
        rationale = "Fund is trading at a discount with favourable technical setup."
    elif score >= 45:
        signal = "HOLD / SIP"
        rationale = "Neutral conditions — continue SIP, avoid large lump sum."
    else:
        signal = "WAIT"
        rationale = "Fund near highs / overbought — wait for better entry."

    return {
        "current_nav":       round(current_nav, 4),
        "52w_high":          round(high_52w, 4),
        "52w_low":           round(low_52w, 4),
        "pct_from_52w_high": round(pct_from_52w_high, 2),
        "pct_from_ath":      round(pct_from_ath, 2),
        "ma50":              round(float(ma50), 4) if not np.isnan(ma50) else None,
        "ma200":             round(float(ma200), 4) if not np.isnan(ma200) else None,
        "golden_cross":      golden_cross,
        "rsi_14":            round(rsi, 1) if rsi else None,
        "entry_score":       round(score, 1),
        "signal":            signal,
        "rationale":         rationale,
    }


def _rsi(prices: pd.Series, period: int = 14) -> Optional[float]:
    """Relative Strength Index."""
    if len(prices) < period + 1:
        return None
    delta = prices.diff().dropna()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss
    rsi   = 100 - (100 / (1 + rs))
    val   = rsi.iloc[-1]
    return float(val) if not np.isnan(val) else None


# ── Long-term wealth projection ────────────────────────────────────────────────

def wealth_projection(
    initial_lump_sum: float = 0,
    monthly_sip: float = 0,
    annual_sip_step_up: float = 0.10,   # 10% increase in SIP every year
    expected_cagr: float = 0.15,
    years: int = 20,
    inflation_rate: float = 0.06,
) -> pd.DataFrame:
    """
    Project wealth accumulation combining lump sum + SIP with step-up.
    Returns yearly snapshot DataFrame.
    Adjusts for inflation to show real (today's money) value.
    """
    rows = []
    corpus     = initial_lump_sum
    monthly_r  = (1 + expected_cagr) ** (1/12) - 1
    sip        = monthly_sip
    total_invested = initial_lump_sum

    for yr in range(1, years + 1):
        # 12 months of SIP growth in this year
        for _ in range(12):
            corpus = corpus * (1 + monthly_r) + sip
            total_invested += sip

        # Step up SIP annually
        sip *= (1 + annual_sip_step_up)

        # Inflation-adjusted real value
        real_value = corpus / ((1 + inflation_rate) ** yr)

        rows.append({
            "year":           yr,
            "total_invested": round(total_invested, 2),
            "corpus":         round(corpus, 2),
            "real_value":     round(real_value, 2),
            "wealth_multiple": round(corpus / max(total_invested, 1), 2),
        })

    return pd.DataFrame(rows)


# ── Full fund report ───────────────────────────────────────────────────────────

def fund_report(scheme_code: str, df: pd.DataFrame,
                sip_amount: float = 10000,
                lump_sum: float = 100000) -> dict:
    """
    Generate a comprehensive report for a single fund.
    df: NAV history DataFrame with columns nav_date, nav.
    """
    if df.empty:
        return {"error": "No NAV data available"}

    report = {
        "scheme_code": scheme_code,
        "scheme_name": df.iloc[-1].get("scheme_name", ""),
        "latest_nav":  round(df.sort_values("nav_date")["nav"].iloc[-1], 4),
        "as_of_date":  str(df["nav_date"].max().date()),
        "cagr": {
            "1y":  _pct(compute_cagr_for_horizon(df, 1)),
            "3y":  _pct(compute_cagr_for_horizon(df, 3)),
            "5y":  _pct(compute_cagr_for_horizon(df, 5)),
            "10y": _pct(compute_cagr_for_horizon(df, 10)),
        },
        "risk": {
            "sharpe_3y":    round(sharpe_ratio(df, 3), 3) if sharpe_ratio(df, 3) else None,
            "max_drawdown": round(max_drawdown(df) * 100, 2),
            "volatility_3y_ann": _pct(volatility(df, 3)),
        },
        "sip_simulation": sip_xirr(
            df,
            monthly_amount=sip_amount,
            start_date=str((df["nav_date"].max() - pd.DateOffset(years=5)).date()),
        ),
        "lump_sum_signal": lump_sum_signals(df),
        "data_points": len(df),
    }
    return report


def _pct(val):
    return round(val * 100, 2) if val is not None else None