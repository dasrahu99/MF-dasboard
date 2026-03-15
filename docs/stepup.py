"""
stepup.py — SIP Step-Up Calculator Engine (Module 6)
Models wealth accumulation with annual SIP increases.
Compares flat SIP vs step-up SIP, shows break-even points,
goal-based reverse planning, and inflation-adjusted real wealth.
"""

import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)
REPORTS_DIR = Path("data/reports")


# ── Core step-up projection ───────────────────────────────────────────────────

def stepup_projection(
    initial_sip:       float,
    annual_stepup_pct: float,
    expected_cagr:     float,
    years:             int,
    initial_lump_sum:  float = 0,
    inflation_rate:    float = 0.06,
) -> pd.DataFrame:
    """
    Month-by-month SIP step-up simulation.
    Returns yearly DataFrame with corpus, invested, real value, wealth multiple.
    """
    rows = []
    corpus    = initial_lump_sum
    sip       = initial_sip
    monthly_r = (1 + expected_cagr) ** (1 / 12) - 1
    total_inv = initial_lump_sum

    for yr in range(1, years + 1):
        for _ in range(12):
            corpus    = corpus * (1 + monthly_r) + sip
            total_inv += sip
        sip *= (1 + annual_stepup_pct)

        real_value = corpus / ((1 + inflation_rate) ** yr)
        rows.append({
            "year":            yr,
            "monthly_sip":     round(initial_sip * ((1 + annual_stepup_pct) ** (yr - 1)), 2),
            "total_invested":  round(total_inv, 2),
            "corpus":          round(corpus, 2),
            "real_value":      round(real_value, 2),
            "wealth_multiple": round(corpus / max(total_inv, 1), 2),
            "gain":            round(corpus - total_inv, 2),
        })

    return pd.DataFrame(rows)


def compare_flat_vs_stepup(
    initial_sip:       float,
    annual_stepup_pct: float,
    expected_cagr:     float,
    years:             int,
    initial_lump_sum:  float = 0,
    inflation_rate:    float = 0.06,
) -> dict:
    """
    Compare flat SIP (no step-up) vs step-up SIP.
    Returns both projections + extra wealth from step-up.
    """
    flat = stepup_projection(
        initial_sip, 0.0, expected_cagr, years,
        initial_lump_sum, inflation_rate
    )
    stepup = stepup_projection(
        initial_sip, annual_stepup_pct, expected_cagr, years,
        initial_lump_sum, inflation_rate
    )

    final_flat   = flat.iloc[-1]
    final_stepup = stepup.iloc[-1]

    extra_corpus   = final_stepup["corpus"]   - final_flat["corpus"]
    extra_invested = final_stepup["total_invested"] - final_flat["total_invested"]
    extra_gain     = extra_corpus - extra_invested

    return {
        "flat_sip": {
            "monthly_sip_constant": initial_sip,
            "total_invested":       round(float(final_flat["total_invested"]), 2),
            "final_corpus":         round(float(final_flat["corpus"]), 2),
            "real_value":           round(float(final_flat["real_value"]), 2),
            "wealth_multiple":      round(float(final_flat["wealth_multiple"]), 2),
            "yearly":               flat.to_dict("records"),
        },
        "stepup_sip": {
            "initial_sip":          initial_sip,
            "final_monthly_sip":    round(float(stepup.iloc[-1]["monthly_sip"]), 2),
            "annual_stepup_pct":    round(annual_stepup_pct * 100, 1),
            "total_invested":       round(float(final_stepup["total_invested"]), 2),
            "final_corpus":         round(float(final_stepup["corpus"]), 2),
            "real_value":           round(float(final_stepup["real_value"]), 2),
            "wealth_multiple":      round(float(final_stepup["wealth_multiple"]), 2),
            "yearly":               stepup.to_dict("records"),
        },
        "step_up_advantage": {
            "extra_corpus":         round(extra_corpus, 2),
            "extra_invested":       round(extra_invested, 2),
            "extra_gain_from_stepup": round(extra_gain, 2),
            "corpus_boost_pct":     round(extra_corpus / float(final_flat["corpus"]) * 100, 1),
        },
        "params": {
            "years":            years,
            "expected_cagr_pct":round(expected_cagr * 100, 1),
            "inflation_pct":    round(inflation_rate * 100, 1),
            "initial_lump_sum": initial_lump_sum,
        },
    }


# ── Goal-based reverse planning ───────────────────────────────────────────────

def reverse_plan(
    target_corpus:     float,
    years:             int,
    expected_cagr:     float,
    annual_stepup_pct: float = 0.10,
    initial_lump_sum:  float = 0,
) -> dict:
    """
    Given a target corpus, compute required starting monthly SIP.
    Uses binary search to find the right initial SIP.
    """
    def simulate_corpus(sip):
        df = stepup_projection(sip, annual_stepup_pct, expected_cagr, years, initial_lump_sum)
        return df.iloc[-1]["corpus"]

    # Binary search
    low, high = 100, 500000
    for _ in range(50):
        mid = (low + high) / 2
        result = simulate_corpus(mid)
        if result < target_corpus:
            low = mid
        else:
            high = mid
        if abs(high - low) < 1:
            break

    required_sip = round((low + high) / 2, 0)
    final_df     = stepup_projection(required_sip, annual_stepup_pct,
                                      expected_cagr, years, initial_lump_sum)

    return {
        "target_corpus":        target_corpus,
        "required_initial_sip": required_sip,
        "final_monthly_sip":    round(float(final_df.iloc[-1]["monthly_sip"]), 2),
        "total_invested":       round(float(final_df.iloc[-1]["total_invested"]), 2),
        "projected_corpus":     round(float(final_df.iloc[-1]["corpus"]), 2),
        "years":                years,
        "annual_stepup_pct":    round(annual_stepup_pct * 100, 1),
        "expected_cagr_pct":    round(expected_cagr * 100, 1),
        "milestones":           _milestones(final_df, target_corpus),
    }


def _milestones(df: pd.DataFrame, target: float) -> list:
    """Years at which corpus crosses 25%, 50%, 75%, 100% of target."""
    milestones = []
    for pct in [0.25, 0.50, 0.75, 1.0]:
        threshold = target * pct
        hit = df[df["corpus"] >= threshold]
        if not hit.empty:
            milestones.append({
                "target_pct":   int(pct * 100),
                "amount":       round(threshold, 2),
                "reached_year": int(hit.iloc[0]["year"]),
            })
    return milestones


# ── Preset goals ─────────────────────────────────────────────────────────────

PRESET_GOALS = {
    "retirement_2cr":     {"target": 2_00_00_000, "years": 25, "label": "Retirement ₹2 Cr"},
    "retirement_5cr":     {"target": 5_00_00_000, "years": 25, "label": "Retirement ₹5 Cr"},
    "child_education_1cr":{"target": 1_00_00_000, "years": 15, "label": "Child Education ₹1 Cr"},
    "house_downpayment":  {"target": 50_00_000,   "years": 7,  "label": "House Down Payment ₹50L"},
    "emergency_corpus":   {"target": 20_00_000,   "years": 5,  "label": "Emergency Fund ₹20L"},
}


def run_all_goals(expected_cagr: float = 0.15,
                   annual_stepup: float = 0.10,
                   initial_lump_sum: float = 0) -> dict:
    """Run reverse planning for all preset goals."""
    results = {}
    for key, goal in PRESET_GOALS.items():
        results[key] = {
            "label": goal["label"],
            "plan":  reverse_plan(
                target_corpus     = goal["target"],
                years             = goal["years"],
                expected_cagr     = expected_cagr,
                annual_stepup_pct = annual_stepup,
                initial_lump_sum  = initial_lump_sum,
            )
        }
    return results


def save_stepup_report(initial_sip: float = 10000,
                        stepup_pct: float = 0.10,
                        cagr: float = 0.15,
                        years: int = 20,
                        lump_sum: float = 500000):
    """Generate and save full step-up report."""
    comparison = compare_flat_vs_stepup(initial_sip, stepup_pct, cagr, years, lump_sum)
    goals      = run_all_goals(cagr, stepup_pct, lump_sum)

    report = {"comparison": comparison, "goals": goals}
    out = REPORTS_DIR / "stepup_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("Step-up report saved → %s", out)

    # Print summary
    c = comparison
    print("\n" + "=" * 70)
    print("  SIP STEP-UP COMPARISON")
    print("=" * 70)
    print(f"  Flat SIP  ₹{initial_sip:,.0f}/mo  → ₹{c['flat_sip']['final_corpus']:,.0f}  "
          f"({c['flat_sip']['wealth_multiple']}x)")
    print(f"  Step-up   ₹{initial_sip:,.0f}/mo +{stepup_pct*100:.0f}%/yr → "
          f"₹{c['stepup_sip']['final_corpus']:,.0f}  ({c['stepup_sip']['wealth_multiple']}x)")
    print(f"  Step-up advantage: +₹{c['step_up_advantage']['extra_corpus']:,.0f}  "
          f"(+{c['step_up_advantage']['corpus_boost_pct']}%)")
    print("=" * 70 + "\n")
    print("  GOAL PLANNING:")
    for key, g in goals.items():
        p = g["plan"]
        print(f"  {g['label']:<35} → Start SIP: ₹{p['required_initial_sip']:,.0f}/mo")
    print("=" * 70 + "\n")

    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    save_stepup_report()