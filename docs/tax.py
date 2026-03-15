"""
tax.py — Tax-Adjusted Returns Engine (Module 7)
Computes post-tax XIRR and returns for Resident Individual investors.

Indian tax rules applied (FY 2024-25):
  Equity funds (held > 12 months):  LTCG @ 12.5% on gains above ₹1.25L/year (no indexation)
  Equity funds (held ≤ 12 months):  STCG @ 20%
  Debt funds (all holding periods):  Slab rate (assumed 30% for high-income bracket)
  ELSS:                              LTCG rules apply (3yr lock-in counts as LTCG)
  Surcharge + Cess:                  4% health & education cess on tax

LTCG exemption: ₹1,25,000 per financial year (updated Budget 2024)
"""

import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)
REPORTS_DIR = Path("data/reports")

# ── Tax constants ─────────────────────────────────────────────────────────────
LTCG_RATE           = 0.125     # 12.5% (Budget 2024)
STCG_RATE           = 0.20      # 20%
DEBT_SLAB_RATE      = 0.30      # 30% slab (assumed)
CESS_RATE           = 0.04      # 4% health & education cess
LTCG_EXEMPTION      = 125000    # ₹1,25,000 per year (Budget 2024)
LTCG_HOLDING_DAYS   = 365       # 12 months for equity LTCG

EQUITY_CATEGORIES = {
    "Large Cap", "Mid Cap", "Small Cap", "Flexi Cap",
    "ELSS", "Index", "Sectoral", "Hybrid"
}


# ── Tax calculation helpers ───────────────────────────────────────────────────

def effective_tax_rate(rate: float, cess: float = CESS_RATE) -> float:
    """Apply cess on top of base tax rate."""
    return rate * (1 + cess)


EFFECTIVE_LTCG = effective_tax_rate(LTCG_RATE)    # 13.0%
EFFECTIVE_STCG = effective_tax_rate(STCG_RATE)    # 20.8%
EFFECTIVE_DEBT = effective_tax_rate(DEBT_SLAB_RATE)# 31.2%


def tax_on_gain(gain: float,
                holding_days: int,
                category: str,
                ltcg_used_this_year: float = 0) -> dict:
    """
    Compute tax on a redemption.

    gain: profit amount (₹)
    holding_days: days the investment was held
    category: fund category string
    ltcg_used_this_year: LTCG exemption already consumed this FY

    Returns: tax_amount, effective_rate, net_gain, regime
    """
    is_equity = category in EQUITY_CATEGORIES

    if not is_equity:
        # Debt — always slab rate regardless of holding
        tax   = gain * EFFECTIVE_DEBT
        return {
            "regime":         "Debt — slab rate (30% + cess)",
            "gross_gain":     round(gain, 2),
            "taxable_gain":   round(gain, 2),
            "tax_amount":     round(tax, 2),
            "effective_rate": round(EFFECTIVE_DEBT * 100, 2),
            "net_gain":       round(gain - tax, 2),
        }

    if holding_days <= LTCG_HOLDING_DAYS:
        # STCG — 20% + cess
        tax = gain * EFFECTIVE_STCG
        return {
            "regime":         "Equity STCG (≤12 months) — 20% + cess",
            "gross_gain":     round(gain, 2),
            "taxable_gain":   round(gain, 2),
            "tax_amount":     round(tax, 2),
            "effective_rate": round(EFFECTIVE_STCG * 100, 2),
            "net_gain":       round(gain - tax, 2),
        }

    # LTCG — 12.5% + cess, first ₹1.25L exempt
    remaining_exemption = max(0, LTCG_EXEMPTION - ltcg_used_this_year)
    taxable_gain = max(0, gain - remaining_exemption)
    tax = taxable_gain * EFFECTIVE_LTCG
    eff_rate = (tax / gain * 100) if gain > 0 else 0

    return {
        "regime":                 "Equity LTCG (>12 months) — 12.5% + cess",
        "gross_gain":             round(gain, 2),
        "exemption_applied":      round(min(gain, remaining_exemption), 2),
        "taxable_gain":           round(taxable_gain, 2),
        "tax_amount":             round(tax, 2),
        "effective_rate":         round(eff_rate, 2),
        "net_gain":               round(gain - tax, 2),
        "ltcg_exemption_used":    round(min(gain, remaining_exemption), 2),
    }


# ── Post-tax SIP returns ──────────────────────────────────────────────────────

def post_tax_sip_return(sip_result: dict,
                         category: str,
                         holding_years: float) -> dict:
    """
    Given a pre-tax SIP result (from engine.sip_xirr), compute post-tax returns.
    Assumes all units redeemed at once after holding_years.
    """
    invested      = sip_result.get("total_invested", 0)
    current_value = sip_result.get("current_value", 0)
    xirr_pct      = sip_result.get("xirr_pct")
    gain          = current_value - invested

    if gain <= 0:
        return {
            "pre_tax_xirr_pct":   xirr_pct,
            "post_tax_xirr_pct":  xirr_pct,
            "tax_amount":         0,
            "net_gain":           round(gain, 2),
            "note":               "No gain — no tax applicable",
        }

    holding_days = int(holding_years * 365)
    tax_calc     = tax_on_gain(gain, holding_days, category)

    net_value    = invested + tax_calc["net_gain"]

    # Approximate post-tax XIRR: scale down by tax drag
    tax_drag     = tax_calc["tax_amount"] / current_value if current_value > 0 else 0
    post_tax_xirr = round(xirr_pct * (1 - tax_drag), 2) if xirr_pct else None

    return {
        "pre_tax": {
            "total_invested":  invested,
            "current_value":   current_value,
            "gross_gain":      round(gain, 2),
            "xirr_pct":        xirr_pct,
        },
        "tax": tax_calc,
        "post_tax": {
            "net_value":       round(net_value, 2),
            "net_gain":        tax_calc["net_gain"],
            "post_tax_xirr_pct": post_tax_xirr,
            "tax_drag_pct":    round(tax_drag * 100, 2),
        },
    }


# ── Post-tax CAGR ─────────────────────────────────────────────────────────────

def post_tax_cagr(pre_tax_cagr_pct: float,
                   holding_years: float,
                   invested: float,
                   category: str) -> dict:
    """
    Given a pre-tax CAGR%, compute the post-tax CAGR for a lump sum scenario.
    """
    if pre_tax_cagr_pct is None:
        return {}

    cagr_decimal = pre_tax_cagr_pct / 100
    final_value  = invested * ((1 + cagr_decimal) ** holding_years)
    gain         = final_value - invested
    holding_days = int(holding_years * 365)
    tax_calc     = tax_on_gain(gain, holding_days, category)

    net_value    = invested + tax_calc["net_gain"]
    post_cagr    = ((net_value / invested) ** (1 / holding_years) - 1) * 100 \
                    if holding_years > 0 and invested > 0 else None

    return {
        "pre_tax_cagr_pct":  round(pre_tax_cagr_pct, 2),
        "post_tax_cagr_pct": round(post_cagr, 2) if post_cagr else None,
        "tax_drag_pct":      round(pre_tax_cagr_pct - post_cagr, 2) if post_cagr else None,
        "tax_amount":        tax_calc["tax_amount"],
        "regime":            tax_calc["regime"],
        "holding_years":     holding_years,
        "invested":          invested,
        "gross_final_value": round(final_value, 2),
        "net_final_value":   round(net_value, 2),
    }


# ── ELSS vs regular equity comparison ────────────────────────────────────────

def elss_vs_equity_comparison(monthly_sip: float = 10000,
                                years: int = 10,
                                cagr_pct: float = 15.0) -> dict:
    """
    Compare ELSS (80C deduction + LTCG) vs regular equity fund (LTCG only).
    Shows true post-tax advantage of ELSS for a 30% slab taxpayer.

    ELSS benefit: ₹1.5L/year 80C deduction @ 30% slab = ₹45,000 tax saved/year
    """
    annual_sip       = monthly_sip * 12
    cagr_d           = cagr_pct / 100
    slab_rate        = 0.30

    # Tax saved from 80C (up to ₹1.5L limit)
    annual_80c_eligible  = min(annual_sip, 150000)
    annual_tax_saving    = annual_80c_eligible * slab_rate * (1 + CESS_RATE)
    total_tax_saving     = annual_tax_saving * min(years, 20)  # assume 20yr benefit cap

    # Corpus for both (same CAGR, same SIP)
    months   = years * 12
    monthly_r = (1 + cagr_d) ** (1/12) - 1
    corpus   = sum(monthly_sip * (1 + monthly_r) ** (months - i) for i in range(months))
    gain     = corpus - (monthly_sip * months)

    # ELSS: LTCG on corpus
    elss_tax = tax_on_gain(gain, 1095, "ELSS")   # 3yr lockup = LTCG
    elss_net = corpus - elss_tax["tax_amount"] + total_tax_saving

    # Regular equity: LTCG
    equity_tax = tax_on_gain(gain, 365 * years, "Large Cap")
    equity_net = corpus - equity_tax["tax_amount"]

    elss_advantage = elss_net - equity_net

    return {
        "monthly_sip":         monthly_sip,
        "years":               years,
        "expected_cagr_pct":   cagr_pct,
        "gross_corpus":        round(corpus, 2),
        "elss": {
            "net_corpus":        round(elss_net, 2),
            "tax_on_gains":      elss_tax["tax_amount"],
            "80c_tax_saved":     round(total_tax_saving, 2),
            "effective_return":  round((elss_net / (monthly_sip * months) - 1) * 100, 2),
        },
        "regular_equity": {
            "net_corpus":        round(equity_net, 2),
            "tax_on_gains":      equity_tax["tax_amount"],
            "80c_tax_saved":     0,
            "effective_return":  round((equity_net / (monthly_sip * months) - 1) * 100, 2),
        },
        "elss_advantage_rs":   round(elss_advantage, 2),
        "verdict":             "ELSS is better" if elss_advantage > 0 else "Regular equity is better",
    }


# ── Enrich latest.json with post-tax returns ──────────────────────────────────

def enrich_with_tax(latest_path: Path = REPORTS_DIR / "latest.json",
                     default_invested: float = 500000,
                     holding_years: float = 5.0):
    """Add post-tax CAGR to every fund in latest.json."""
    if not latest_path.exists():
        log.warning("latest.json not found")
        return

    with open(latest_path) as f:
        report = json.load(f)

    for code, fund in report.items():
        cat  = fund.get("category", "Large Cap")
        cagr = fund.get("cagr", {})
        tax_cagrs = {}
        for horizon, val in cagr.items():
            if val:
                yrs = {"1y": 1, "3y": 3, "5y": 5, "10y": 10}.get(horizon, 5)
                tax_cagrs[horizon] = post_tax_cagr(val, yrs, default_invested, cat)
        fund["post_tax_cagr"] = tax_cagrs

        # Post-tax SIP
        sip_sim = fund.get("sip_simulation", {})
        if sip_sim:
            fund["post_tax_sip"] = post_tax_sip_return(sip_sim, cat, holding_years)

    with open(latest_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # ELSS comparison
    elss_comp = elss_vs_equity_comparison()
    out = REPORTS_DIR / "tax_analysis.json"
    with open(out, "w") as f:
        json.dump({"elss_vs_equity": elss_comp}, f, indent=2, default=str)

    log.info("Tax analysis complete → %s", out)
    _print_tax_table(report)


def _print_tax_table(report: dict):
    rows = []
    for code, r in report.items():
        pt = r.get("post_tax_cagr", {})
        y5 = pt.get("5y", {})
        rows.append({
            "Fund":               r.get("fund_name", r.get("scheme_name", ""))[:30],
            "Pre-tax 5Y CAGR%":   r.get("cagr", {}).get("5y"),
            "Post-tax 5Y CAGR%":  y5.get("post_tax_cagr_pct"),
            "Tax Drag%":          y5.get("tax_drag_pct"),
            "Regime":             y5.get("regime", "")[:30] if y5 else "",
        })
    df = pd.DataFrame(rows).dropna(subset=["Pre-tax 5Y CAGR%"])
    print("\n" + "=" * 90)
    print("  POST-TAX CAGR ANALYSIS (Resident Individual)")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    enrich_with_tax()

    # Demo ELSS comparison
    comp = elss_vs_equity_comparison(monthly_sip=12500, years=10, cagr_pct=15)
    print(f"\nELSS vs Regular Equity (₹12,500/mo, 10yr, 15% CAGR):")
    print(f"  ELSS net corpus:   ₹{comp['elss']['net_corpus']:,.0f}")
    print(f"  Equity net corpus: ₹{comp['regular_equity']['net_corpus']:,.0f}")
    print(f"  ELSS advantage:    ₹{comp['elss_advantage_rs']:,.0f}")
    print(f"  Verdict:           {comp['verdict']}")