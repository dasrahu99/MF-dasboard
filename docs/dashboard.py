"""
dashboard.py — Full Fund Intelligence Dashboard
Streamlit UI for all 7 pipeline modules.
Run: streamlit run dashboard.py
"""

import json
import sqlite3
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MF Intelligence Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
REPORTS_DIR  = Path("data/reports")
LATEST_JSON  = REPORTS_DIR / "latest.json"
STEPUP_JSON  = REPORTS_DIR / "stepup_report.json"
MOMENTUM_JSON= REPORTS_DIR / "momentum_scores.json"
BENCH_JSON   = REPORTS_DIR / "benchmark_analysis.json"
DD_JSON      = REPORTS_DIR / "drawdown_analysis.json"
PORTFOLIO_JSON = REPORTS_DIR / "portfolio_report.json"
TAX_JSON     = REPORTS_DIR / "tax_analysis.json"

# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_json(path):
    if not Path(path).exists():
        return {}
    with open(path) as f:
        return json.load(f)


def fmt_inr(val):
    if val is None: return "—"
    if val >= 1e7:  return f"₹{val/1e7:.2f} Cr"
    if val >= 1e5:  return f"₹{val/1e5:.2f} L"
    return f"₹{val:,.0f}"


def pct(val, decimals=1):
    if val is None: return "—"
    return f"{val:+.{decimals}f}%"


def color_pct(val):
    if val is None: return "—"
    color = "green" if val > 0 else "red"
    return f":{color}[{val:+.1f}%]"


SIGNAL_COLORS = {
    "STRONG BUY": "🟢", "BUY": "🟢",
    "NEUTRAL": "🟡", "HOLD / SIP": "🟡",
    "REDUCE": "🔴", "WAIT": "🔴",
    "REPLACE WITH INDEX FUND": "🔴",
    "REVIEW": "🟡", "KEEP": "🟢",
}

# ── Sidebar navigation ────────────────────────────────────────────────────────

st.sidebar.title("📈 MF Intelligence")
st.sidebar.caption(f"Last updated: {datetime.now().strftime('%d %b %Y %H:%M')}")

pages = [
    "🏠 Overview",
    "💼 My Portfolio",
    "📊 Benchmark Analyser",
    "📉 Drawdown Tracker",
    "📈 SIP Step-Up Planner",
    "🧾 Tax-Adjusted Returns",
    "⚡ Momentum Signals",
]
page = st.sidebar.radio("Navigate", pages)

if st.sidebar.button("🔄 Refresh data"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.divider()
st.sidebar.caption("Pipeline modules active:")
for m in ["fetcher ✓", "engine ✓", "momentum ✓", "portfolio ✓",
          "benchmark ✓", "drawdown ✓", "stepup ✓", "tax ✓"]:
    st.sidebar.caption(f"  {m}")


# ═══════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("Fund Intelligence Dashboard")
    st.caption("Real-time Indian mutual fund analytics — AMFI data")

    report = load_json(LATEST_JSON)
    if not report:
        st.warning("Run `python pipeline.py` first to generate data.")
        st.stop()

    funds = [v for v in report.values() if "error" not in v]

    # KPI row
    col1, col2, col3, col4, col5 = st.columns(5)
    avg_cagr_5y = [f.get("cagr", {}).get("5y") for f in funds if f.get("cagr", {}).get("5y")]
    buy_signals = sum(1 for f in funds
                      if f.get("momentum", {}).get("signal") in ("STRONG BUY", "BUY"))
    best_sharpe_fund = max(funds, key=lambda f: f.get("risk", {}).get("sharpe_3y") or 0)
    avg5 = round(sum(avg_cagr_5y) / len(avg_cagr_5y), 1) if avg_cagr_5y else None

    col1.metric("Funds tracked",    len(funds))
    col2.metric("Avg 5Y CAGR",      f"{avg5}%" if avg5 else "—")
    col3.metric("BUY signals",       buy_signals)
    col4.metric("Best Sharpe",
                best_sharpe_fund.get("fund_name", "")[:20],
                f"{best_sharpe_fund.get('risk',{}).get('sharpe_3y','—')}")
    col5.metric("vs Nifty 50",       "Benchmark: ~14%")

    st.divider()

    # Fund rankings table
    st.subheader("Fund Rankings")
    rows = []
    for code, f in report.items():
        if "error" in f: continue
        m = f.get("momentum", {})
        rows.append({
            "Fund":         f.get("fund_name", f.get("scheme_name", ""))[:35],
            "1Y%":          f.get("cagr", {}).get("1y"),
            "3Y%":          f.get("cagr", {}).get("3y"),
            "5Y%":          f.get("cagr", {}).get("5y"),
            "Sharpe":       f.get("risk", {}).get("sharpe_3y"),
            "Max DD%":      f.get("risk", {}).get("max_drawdown"),
            "SIP XIRR%":    f.get("sip_simulation", {}).get("xirr_pct"),
            "Momentum":     f"{SIGNAL_COLORS.get(m.get('signal',''), '⚪')} {m.get('signal','—')}",
            "Score":        m.get("composite_score"),
        })
    df = pd.DataFrame(rows).sort_values("5Y%", ascending=False)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()

    # ── Top 3 Closest to BUY ─────────────────────────────────────
    st.subheader("🎯 Watchlist — Closest to BUY signal")
    st.caption(
        "NEUTRAL funds ranked by how close they are to flipping BUY (score ≥ 60). "
        "These are your next entry opportunities — watch these first."
    )

    neutral_funds = []
    for code, f in report.items():
        if "error" in f:
            continue
        m     = f.get("momentum", {})
        score = m.get("composite_score")
        sig   = m.get("signal", "")
        if score is None:
            continue
        # Include NEUTRAL (40-59) and REDUCE above 35; skip BUY / STRONG BUY
        if sig in ("NEUTRAL", "REDUCE") and score >= 35:
            neutral_funds.append({
                "code":   code,
                "fund":   f,
                "score":  score,
                "signal": sig,
                "m":      m,
            })

    neutral_funds.sort(key=lambda x: x["score"], reverse=True)
    top3 = neutral_funds[:3]

    if not top3:
        st.info(
            "No NEUTRAL funds found — either all funds are BUY/STRONG BUY already, "
            "or momentum scores haven't been generated yet. "
            "Run `python momentum.py` to refresh."
        )
    else:
        card_cols = st.columns(3)
        for col, entry in zip(card_cols, top3):
            f     = entry["fund"]
            m     = entry["m"]
            score = entry["score"]
            fa    = m.get("factors", {})
            tr    = fa.get("trailing_returns", {})

            gap_to_buy = round(60 - score, 1)
            name       = f.get("fund_name", f.get("scheme_name", ""))[:28]
            cagr5      = f.get("cagr", {}).get("5y")
            sharpe     = f.get("risk", {}).get("sharpe_3y")

            # Progress 35→60 maps to 0→1
            progress = max(0.0, min(1.0, (score - 35) / 25))

            with col:
                with st.container(border=True):
                    st.markdown(f"**{name}**")
                    if cagr5:
                        st.caption(f"{f.get('category', '')} · 5Y CAGR: {cagr5:.1f}%")
                    else:
                        st.caption(f"{f.get('category', '')}")

                    sc1, sc2 = st.columns(2)
                    sc1.metric("Score", f"{score:.0f} / 100")
                    sc2.metric("Gap to BUY", f"{gap_to_buy:.1f} pts", delta="Need ≥ 60",
                               delta_color="off")

                    st.progress(progress, text=f"Score {score:.0f} → BUY threshold: 60")

                    t1, t2 = st.columns(2)
                    r1m = tr.get("1m_pct")
                    r3m = tr.get("3m_pct")
                    t1.metric("1M", f"{r1m:+.1f}%" if r1m is not None else "—")
                    t2.metric("3M", f"{r3m:+.1f}%" if r3m is not None else "—")

                    accel = fa.get("acceleration_pct")
                    if accel is not None:
                        arrow = "▲" if accel > 0 else "▼"
                        msg   = f"{arrow} {abs(accel):.1f}% momentum acceleration"
                        if accel > 0:
                            st.success(msg)
                        else:
                            st.warning(msg)

                    fs = m.get("factor_scores", {})
                    if fs:
                        weak = sorted(fs.items(), key=lambda x: x[1])[:2]
                        st.caption(
                            "Weakest: " + ", ".join(f"{k} ({v*100:.0f})" for k, v in weak)
                        )

                    if sharpe:
                        st.caption(f"Sharpe (3Y): {sharpe:.2f}")


# ═══════════════════════════════════════════════════════════════════
# PAGE 2 — MY PORTFOLIO (Module 3)
# ═══════════════════════════════════════════════════════════════════
elif page == "💼 My Portfolio":
    st.title("Portfolio Simulator")
    st.caption("Your personalised multi-fund portfolio — SIP + lump sum analysis")

    report = load_json(LATEST_JSON)
    portfolio_report = load_json(PORTFOLIO_JSON)

    if not portfolio_report:
        st.info("Run `python portfolio.py` to generate your portfolio report.")
        st.stop()

    s = portfolio_report.get("summary", {})
    b = portfolio_report.get("benchmark", {})
    d = portfolio_report.get("diversification", {})

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Invested",    fmt_inr(s.get("total_invested")))
    c2.metric("Current Value",     fmt_inr(s.get("total_current_value")),
              delta=f"{s.get('absolute_return_pct',0):+.1f}%")
    c3.metric("Total Gain",        fmt_inr(s.get("total_gain")))
    c4.metric("Monthly SIP",       fmt_inr(s.get("monthly_sip_total")))
    c5.metric("Diversification",   d.get("grade", "—"),
              delta=f"Score: {d.get('score', 0)}")

    st.divider()

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Fund Breakdown")
        fund_rows = []
        for code, f in portfolio_report.get("funds", {}).items():
            if "error" in f: continue
            sip = f.get("sip", {})
            ls  = f.get("lump_sum", {})
            fund_rows.append({
                "Fund":       f.get("name", "")[:30],
                "Category":   f.get("category", ""),
                "SIP Value":  fmt_inr(sip.get("current_value")),
                "SIP XIRR%":  sip.get("xirr_pct"),
                "LS Value":   fmt_inr(ls.get("current_value")),
                "LS CAGR%":   ls.get("cagr_pct"),
                "1M%":        f.get("trailing", {}).get("1m"),
                "3M%":        f.get("trailing", {}).get("3m"),
            })
        st.dataframe(pd.DataFrame(fund_rows), use_container_width=True, hide_index=True)

    with col_right:
        st.subheader("Category Allocation")
        cat_w = d.get("category_weights", {})
        if cat_w:
            import plotly.express as px
            fig = px.pie(
                values=list(cat_w.values()),
                names=list(cat_w.keys()),
                hole=0.5,
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_layout(
                showlegend=True, height=280,
                margin=dict(t=0, b=0, l=0, r=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Portfolio vs Nifty 50")
    bc1, bc2, bc3 = st.columns(3)
    bc1.metric("Portfolio 12M",   pct(b.get("portfolio_12m_pct")))
    bc2.metric("Nifty 50 12M",    pct(b.get("nifty50_12m_pct")))
    bc3.metric("Your Alpha",      pct(b.get("alpha_12m_pct")), delta="vs benchmark")

    warnings = portfolio_report.get("warnings", [])
    if warnings:
        st.subheader("⚠️ Portfolio Warnings")
        for w in warnings:
            level = st.warning if w["severity"] == "high" else st.info
            level(w["message"])

    with st.expander("✏️ Edit portfolio allocation"):
        st.caption("Modify your fund allocations below and re-run `python portfolio.py`")
        from portfolio import DEFAULT_PORTFOLIO
        edited_rows = []
        for code, cfg in DEFAULT_PORTFOLIO.items():
            edited_rows.append({
                "Scheme Code": code,
                "Fund Name":   cfg["name"],
                "Category":    cfg["category"],
                "Monthly SIP (₹)": cfg["monthly_sip"],
                "Lump Sum (₹)":    cfg["lump_sum"],
            })
        st.data_editor(pd.DataFrame(edited_rows), use_container_width=True,
                        num_rows="dynamic", hide_index=True, key="portfolio_editor")
        st.caption("After editing, update DEFAULT_PORTFOLIO in portfolio.py and re-run.")


# ═══════════════════════════════════════════════════════════════════
# PAGE 3 — BENCHMARK ANALYSER (Module 4)
# ═══════════════════════════════════════════════════════════════════
elif page == "📊 Benchmark Analyser":
    st.title("Benchmark Analyser")
    st.caption("Fund alpha vs Nifty 50 & Nifty 500 — KEEP, REVIEW, or REPLACE?")

    bench = load_json(BENCH_JSON)
    if not bench:
        st.info("Run `python benchmark.py` to generate benchmark data.")
        st.stop()

    recs = [r.get("recommendation", "") for r in bench.values()]
    c1, c2, c3 = st.columns(3)
    c1.metric("KEEP",    sum(1 for r in recs if r == "KEEP"),    help="Consistently beats benchmark")
    c2.metric("REVIEW",  sum(1 for r in recs if r == "REVIEW"),  help="Marginal outperformance")
    c3.metric("REPLACE", sum(1 for r in recs if "REPLACE" in r), help="Index fund is better")

    st.divider()

    st.subheader("Alpha vs Nifty 50")
    alpha_rows = []
    for code, r in bench.items():
        nifty = r.get("benchmarks", {}).get("Nifty 50", {})
        alpha = nifty.get("alpha_pct", {})
        cap   = nifty.get("capture", {})
        alpha_rows.append({
            "Fund":           r.get("scheme_name", "")[:30],
            "Alpha 1Y%":      alpha.get("1y"),
            "Alpha 3Y%":      alpha.get("3y"),
            "Alpha 5Y%":      alpha.get("5y"),
            "Info Ratio":     nifty.get("information_ratio"),
            "Up Capture%":    cap.get("up_capture_pct"),
            "Down Capture%":  cap.get("down_capture_pct"),
            "Beat Bench%":    nifty.get("beat_consistency_pct"),
            "Recommendation": f"{SIGNAL_COLORS.get(r.get('recommendation',''), '⚪')} "
                               f"{r.get('recommendation','')}",
        })
    st.dataframe(pd.DataFrame(alpha_rows), use_container_width=True, hide_index=True)

    st.subheader("Fund Deep Dive")
    selected = st.selectbox("Select fund", [r.get("scheme_name") for r in bench.values()])
    sel_data = next((r for r in bench.values() if r.get("scheme_name") == selected), {})
    if sel_data:
        for bname, bdata in sel_data.get("benchmarks", {}).items():
            with st.expander(f"vs {bname}", expanded=True):
                a1, a2, a3, a4 = st.columns(4)
                a1.metric("Info Ratio",       bdata.get("information_ratio", "—"))
                a2.metric("Beat Consistency", f"{bdata.get('beat_consistency_pct','—')}%")
                cap = bdata.get("capture", {})
                a3.metric("Up Capture",       f"{cap.get('up_capture_pct','—')}%")
                a4.metric("Down Capture",     f"{cap.get('down_capture_pct','—')}%")
                st.info(f"**Capture verdict:** {cap.get('verdict','—')}")
                st.success(f"**Recommendation:** {bdata.get('verdict','—')}")


# ═══════════════════════════════════════════════════════════════════
# PAGE 4 — DRAWDOWN TRACKER (Module 5)
# ═══════════════════════════════════════════════════════════════════
elif page == "📉 Drawdown Tracker":
    st.title("Drawdown Recovery Tracker")
    st.caption("Every crash analysed — how deep, how long to recover, SIP benefit during crashes")

    dd = load_json(DD_JSON)
    if not dd:
        st.info("Run `python drawdown.py` to generate drawdown data.")
        st.stop()

    st.subheader("Current drawdown status")
    dd_rows = []
    for code, r in dd.items():
        s = r.get("summary", {})
        dd_rows.append({
            "Fund":               r.get("scheme_name", "")[:30],
            "Current DD%":        r.get("current_drawdown_pct"),
            "Recovery needed%":   r.get("recovery_needed_pct"),
            "Worst DD%":          s.get("worst_drawdown_pct"),
            "Avg recovery (mo)":  s.get("avg_recovery_months"),
            "Max recovery (days)":s.get("max_recovery_days"),
            "SIP always wins":    "✅" if s.get("sip_always_wins") else "❌",
        })
    df = pd.DataFrame(dd_rows).sort_values("Current DD%")
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()

    selected = st.selectbox("Deep dive into fund", [r.get("scheme_name") for r in dd.values()])
    sel = next((r for r in dd.values() if r.get("scheme_name") == selected), {})
    if sel:
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Current NAV",     f"₹{sel.get('current_nav', 0):.2f}")
        d2.metric("All-Time High",   f"₹{sel.get('all_time_high', 0):.2f}")
        d3.metric("From ATH",        pct(sel.get("current_drawdown_pct")))
        d4.metric("Recovery needed", pct(sel.get("recovery_needed_pct")))

        events = sel.get("events", [])
        if events:
            st.subheader("Historical drawdown events")
            ev_rows = []
            for e in events[:8]:
                sip_b = e.get("sip_benefit", {})
                ev_rows.append({
                    "Peak date":      e.get("peak_date"),
                    "Trough date":    e.get("trough_date"),
                    "Drawdown%":      e.get("drawdown_pct"),
                    "Fall (days)":    e.get("drawdown_days"),
                    "Recovery (days)":e.get("recovery_days", "Not recovered"),
                    "Total (days)":   e.get("total_event_days", "—"),
                    "SIP return%":    sip_b.get("sip_return_pct"),
                    "Lump return%":   sip_b.get("lump_return_pct"),
                    "SIP advantage%": sip_b.get("sip_advantage_pct"),
                })
            st.dataframe(pd.DataFrame(ev_rows), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE 5 — SIP STEP-UP PLANNER (Module 6)
# ═══════════════════════════════════════════════════════════════════
elif page == "📈 SIP Step-Up Planner":
    st.title("SIP Step-Up Planner")
    st.caption("Drag the sliders — see your 20-year wealth update live")

    import plotly.graph_objects as go

    col_ctrl, col_chart = st.columns([1, 2])

    with col_ctrl:
        initial_sip    = st.slider("Starting monthly SIP (₹)", 1000, 100000, 10000, 500)
        stepup_pct     = st.slider("Annual SIP step-up %", 0, 30, 10, 1) / 100
        expected_cagr  = st.slider("Expected CAGR %", 8, 25, 15, 1) / 100
        years          = st.slider("Investment horizon (years)", 5, 35, 20, 1)
        lump_sum       = st.number_input("One-time lump sum (₹)", 0, 10000000, 500000, 50000)
        inflation      = st.slider("Inflation rate %", 4, 10, 6, 1) / 100

        from stepup import compare_flat_vs_stepup, run_all_goals
        result = compare_flat_vs_stepup(
            initial_sip, stepup_pct, expected_cagr, years, lump_sum, inflation
        )

        flat_f   = result["flat_sip"]
        stepup_f = result["stepup_sip"]
        adv      = result["step_up_advantage"]

        st.divider()
        st.metric("Flat SIP corpus",   fmt_inr(flat_f["final_corpus"]),
                   f"{flat_f['wealth_multiple']}x")
        st.metric("Step-up corpus",    fmt_inr(stepup_f["final_corpus"]),
                   f"+{adv['corpus_boost_pct']}% vs flat", delta_color="normal")
        st.metric("Extra wealth",      fmt_inr(adv["extra_corpus"]))
        st.metric("Final monthly SIP", fmt_inr(stepup_f["final_monthly_sip"]))

    with col_chart:
        flat_df   = pd.DataFrame(flat_f["yearly"])
        stepup_df = pd.DataFrame(stepup_f["yearly"])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=stepup_df["year"], y=stepup_df["corpus"],
            name="Step-up corpus", line=dict(color="#00C896", width=2.5)
        ))
        fig.add_trace(go.Scatter(
            x=flat_df["year"], y=flat_df["corpus"],
            name="Flat SIP corpus", line=dict(color="#6B8CFF", width=2, dash="dash")
        ))
        fig.add_trace(go.Scatter(
            x=stepup_df["year"], y=stepup_df["real_value"],
            name="Real value (inflation-adj)", line=dict(color="#FF8C42", width=1.5, dash="dot")
        ))
        fig.add_trace(go.Scatter(
            x=stepup_df["year"], y=stepup_df["total_invested"],
            name="Total invested", fill="tozeroy",
            fillcolor="rgba(107,140,255,0.08)",
            line=dict(color="rgba(107,140,255,0.3)", width=1)
        ))
        fig.update_layout(
            title="20-Year Wealth Projection",
            xaxis_title="Years", yaxis_title="₹ Corpus",
            height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(tickformat=",.0f"),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Goal-Based Reverse Planning")
    st.caption("How much SIP do you need to start today to hit these goals?")

    goals = run_all_goals(expected_cagr, stepup_pct, lump_sum)
    g_rows = []
    for key, g in goals.items():
        p = g["plan"]
        g_rows.append({
            "Goal":             g["label"],
            "Target":           fmt_inr(p["target_corpus"]),
            "Years":            p["years"],
            "Start SIP (₹/mo)": f"₹{p['required_initial_sip']:,.0f}",
            "Final SIP (₹/mo)": f"₹{p['final_monthly_sip']:,.0f}",
            "Total Invested":   fmt_inr(p["total_invested"]),
            "Projected Corpus": fmt_inr(p["projected_corpus"]),
        })
    st.dataframe(pd.DataFrame(g_rows), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE 6 — TAX-ADJUSTED RETURNS (Module 7)
# ═══════════════════════════════════════════════════════════════════
elif page == "🧾 Tax-Adjusted Returns":
    st.title("Tax-Adjusted Returns")
    st.caption("Post-tax CAGR & XIRR — Resident Individual | LTCG 12.5% · STCG 20% · Cess 4%")

    report = load_json(LATEST_JSON)
    if not report:
        st.warning("Run pipeline.py first.")
        st.stop()

    with st.expander("📋 Tax rules applied (FY 2024-25)", expanded=False):
        st.markdown("""
| Type | Holding | Rate | Exemption |
|------|---------|------|-----------|
| Equity LTCG | > 12 months | 12.5% + 4% cess | ₹1,25,000/year |
| Equity STCG | ≤ 12 months | 20% + 4% cess | None |
| Debt funds  | Any         | Slab rate (30%) + cess | None |
| ELSS        | > 36 months | 12.5% + 4% cess | ₹1,25,000/year |
        """)

    st.subheader("Pre-tax vs Post-tax CAGR (5-year)")
    tax_rows = []
    for code, f in report.items():
        if "error" in f: continue
        pre5  = f.get("cagr", {}).get("5y")
        pt5   = f.get("post_tax_cagr", {}).get("5y", {})
        post5 = pt5.get("post_tax_cagr_pct")
        drag  = pt5.get("tax_drag_pct")
        tax_rows.append({
            "Fund":              f.get("fund_name", f.get("scheme_name",""))[:30],
            "Pre-tax 5Y CAGR%":  pre5,
            "Post-tax 5Y CAGR%": post5,
            "Tax drag%":         drag,
            "Regime":            pt5.get("regime", "")[:35] if pt5 else "",
        })
    df = pd.DataFrame(tax_rows).dropna(subset=["Pre-tax 5Y CAGR%"]).sort_values(
        "Post-tax 5Y CAGR%", ascending=False
    )
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("Interactive Tax Calculator")
    tcol1, tcol2 = st.columns(2)
    with tcol1:
        inv_amount   = st.number_input("Invested amount (₹)", 10000, 10000000, 500000, 10000)
        pre_tax_cagr = st.slider("Pre-tax CAGR %", 5.0, 35.0, 15.0, 0.5)
        holding_yrs  = st.slider("Holding period (years)", 1, 20, 5)
        fund_cat     = st.selectbox("Fund category",
                                     ["Large Cap","Mid Cap","Small Cap","Flexi Cap",
                                      "ELSS","Index","Debt"])
    with tcol2:
        from tax import post_tax_cagr as calc_post_tax
        result = calc_post_tax(pre_tax_cagr, holding_yrs, inv_amount, fund_cat)
        if result:
            st.metric("Gross final value",  fmt_inr(result.get("gross_final_value")))
            st.metric("Tax amount",         fmt_inr(result.get("tax_amount")))
            st.metric("Net final value",    fmt_inr(result.get("net_final_value")))
            st.metric("Post-tax CAGR",      f"{result.get('post_tax_cagr_pct','—')}%",
                       delta=f"Tax drag: -{result.get('tax_drag_pct','—')}%")
            st.info(f"**Tax regime:** {result.get('regime','')}")

    st.divider()

    st.subheader("ELSS vs Regular Equity — Which wins after tax?")
    ec1, ec2, ec3 = st.columns(3)
    elss_sip  = ec1.number_input("Monthly SIP (₹)", 1000, 100000, 12500, 500, key="elss_sip")
    elss_yrs  = ec2.slider("Years", 3, 20, 10, key="elss_yrs")
    elss_cagr = ec3.slider("Expected CAGR %", 8, 25, 15, key="elss_cagr")

    from tax import elss_vs_equity_comparison
    comp = elss_vs_equity_comparison(elss_sip, elss_yrs, float(elss_cagr))

    r1, r2, r3 = st.columns(3)
    r1.metric("ELSS net corpus",  fmt_inr(comp["elss"]["net_corpus"]),
               f"80C saved: {fmt_inr(comp['elss']['80c_tax_saved'])}")
    r2.metric("Regular equity",   fmt_inr(comp["regular_equity"]["net_corpus"]))
    r3.metric("ELSS advantage",   fmt_inr(comp["elss_advantage_rs"]),
               comp["verdict"], delta_color="normal")


# ═══════════════════════════════════════════════════════════════════
# PAGE 7 — MOMENTUM SIGNALS
# ═══════════════════════════════════════════════════════════════════
elif page == "⚡ Momentum Signals":
    st.title("Momentum Signals")
    st.caption("Multi-factor momentum scoring — 8 factors, 0–100 composite score")

    momentum = load_json(MOMENTUM_JSON)
    if not momentum:
        st.info("Run `python momentum.py` to generate momentum scores.")
        st.stop()

    import plotly.graph_objects as go

    names  = [m.get("scheme_name", "")[:25] for m in momentum]
    scores = [m.get("composite_score", 0) for m in momentum]
    colors = ["#00C896" if s >= 60 else "#FFB830" if s >= 40 else "#FF4B4B" for s in scores]

    fig = go.Figure(go.Bar(
        x=scores, y=names, orientation="h",
        marker_color=colors,
        text=[f"{s:.0f}" for s in scores],
        textposition="outside",
    ))
    fig.update_layout(
        height=max(300, len(names) * 36),
        xaxis=dict(range=[0, 110], title="Momentum Score"),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=60, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Factor Breakdown")
    selected = st.selectbox("Inspect fund", names)
    sel = next((m for m in momentum if m.get("scheme_name", "")[:25] == selected), {})
    if sel:
        fa = sel.get("factors", {})
        tr = fa.get("trailing_returns", {})

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Composite Score", sel.get("composite_score"))
        mc2.metric("Signal",          sel.get("signal"))
        mc3.metric("RSI (14)",        fa.get("rsi_14", "—"))

        st.info(f"**Action:** {sel.get('action', '—')}")

        t1, t2, t3, t4 = st.columns(4)
        t1.metric("1M return",  pct(tr.get("1m_pct")))
        t2.metric("3M return",  pct(tr.get("3m_pct")))
        t3.metric("6M return",  pct(tr.get("6m_pct")))
        t4.metric("12M return", pct(tr.get("12m_pct")))

        fs = sel.get("factor_scores", {})
        if fs:
            categories    = list(fs.keys())
            values        = [fs[c] * 100 for c in categories]
            values_closed = values + [values[0]]
            cats_closed   = categories + [categories[0]]

            fig2 = go.Figure(go.Scatterpolar(
                r=values_closed, theta=cats_closed,
                fill="toself",
                line_color="#00C896",
                fillcolor="rgba(0,200,150,0.15)",
            ))
            fig2.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                height=320,
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig2, use_container_width=True)

        trend = fa.get("trend", {})
        if trend:
            tc1, tc2, tc3 = st.columns(3)
            tc1.metric("MA50",  f"₹{trend.get('ma50','—')}")
            tc2.metric("MA200", f"₹{trend.get('ma200','—')}")
            tc3.metric("Bullish MA stack",
                        "✅ Yes" if trend.get("ma_bullish_stack") else "❌ No")