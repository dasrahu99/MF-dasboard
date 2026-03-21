"""
dashboard.py — Quantum Dark Fund Intelligence Dashboard
Futuristic UI with animated components, live ticker, glow effects.
Run: streamlit run dashboard.py
"""

import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="MF Intelligence",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject theme ──────────────────────────────────────────────────
def load_css():
    css_path = Path(__file__).parent / "theme.css"
    if css_path.exists():
        with open(css_path) as f:
            raw = f.read()
        # wrap in <style> if not already
        if not raw.strip().startswith("<style"):
            raw = f"<style>{raw}</style>"
        st.markdown(raw, unsafe_allow_html=True)

load_css()

# ── Plotly dark theme ─────────────────────────────────────────────
PLOTLY = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Mono, monospace", color="#7a9bb5", size=11),
    xaxis=dict(gridcolor="rgba(0,229,255,0.06)", zerolinecolor="rgba(0,229,255,0.1)",
               tickfont=dict(size=10)),
    yaxis=dict(gridcolor="rgba(0,229,255,0.06)", zerolinecolor="rgba(0,229,255,0.1)",
               tickfont=dict(size=10)),
    margin=dict(l=8, r=8, t=32, b=8),
    hoverlabel=dict(bgcolor="#0d1520", bordercolor="#00e5ff",
                    font=dict(family="DM Mono", size=11, color="#e8f4f8")),
)

# ── Paths ─────────────────────────────────────────────────────────
REPORTS = Path("data/reports")
LATEST_JSON   = REPORTS / "latest.json"
MOMENTUM_JSON = REPORTS / "momentum_scores.json"
BENCH_JSON    = REPORTS / "benchmark_analysis.json"
DD_JSON       = REPORTS / "drawdown_analysis.json"
PORT_JSON     = REPORTS / "portfolio_report.json"
TAX_JSON      = REPORTS / "tax_analysis.json"

# ── Helpers ───────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load(path):
    p = Path(path)
    if not p.exists(): return {}
    with open(p) as f: return json.load(f)

def inr(v):
    if v is None: return "—"
    if abs(v) >= 1e7: return f"₹{v/1e7:.2f}Cr"
    if abs(v) >= 1e5: return f"₹{v/1e5:.1f}L"
    return f"₹{v:,.0f}"

def pp(v, d=1):
    if v is None: return "—"
    return f"{v:+.{d}f}%"

def signal_html(sig):
    cls = {"STRONG BUY":"signal-strong","BUY":"signal-buy",
           "NEUTRAL":"signal-neutral","REDUCE":"signal-reduce",
           "WAIT":"signal-reduce","HOLD / SIP":"signal-neutral"}.get(sig,"signal-neutral")
    return f'<span class="signal-badge {cls}">{sig}</span>'

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 1.5rem;">
      <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;
                  letter-spacing:0.15em;text-transform:uppercase;
                  background:linear-gradient(135deg,#e8f4f8,#00e5ff);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        ⬡ MF INTEL
      </div>
      <div style="font-family:'DM Mono',monospace;font-size:9px;color:#3d5a72;
                  letter-spacing:0.15em;text-transform:uppercase;margin-top:4px;">
        <span class="live-dot"></span>LIVE · AMFI DATA
      </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("", [
        "⬡  OVERVIEW",
        "◈  MOMENTUM",
        "◉  PORTFOLIO",
        "⊕  BENCHMARK",
        "▽  DRAWDOWN",
        "◎  STEP-UP",
        "⊛  TAX",
    ], label_visibility="collapsed")

    st.divider()

    if st.button("⟳  REFRESH", use_container_width=True):
        with st.spinner("Running pipeline to fetch latest data..."):
            import subprocess
            import sys
            import os
            
            script_dir = Path(__file__).parent
            pipeline_path = script_dir / "pipeline.py"
            
            try:
                result = subprocess.run(
                    [sys.executable, str(pipeline_path)],
                    capture_output=True, text=True, check=True
                )
                st.cache_data.clear()
                st.rerun()
            except subprocess.CalledProcessError as e:
                st.error(f"Pipeline failed! Error:\n{e.stderr}\n\nOutput:\n{e.stdout}")

    st.markdown("""
    <div style="margin-top:1rem;">
    """, unsafe_allow_html=True)

    report_check = load(LATEST_JSON)
    modules = [
        ("FETCHER",   bool(report_check)),
        ("ENGINE",    bool(report_check)),
        ("MOMENTUM",  Path(MOMENTUM_JSON).exists()),
        ("PORTFOLIO", Path(PORT_JSON).exists()),
        ("BENCHMARK", Path(BENCH_JSON).exists()),
        ("DRAWDOWN",  Path(DD_JSON).exists()),
    ]
    for name, ok in modules:
        dot = "#00e5ff" if ok else "#3d5a72"
        st.markdown(
            f'<div style="font-family:DM Mono,monospace;font-size:9px;'
            f'color:#3d5a72;letter-spacing:0.1em;padding:2px 0;">'
            f'<span style="color:{dot}">●</span> {name}</div>',
            unsafe_allow_html=True
        )


# ════════════════════════════════════════════════════════════════════
# OVERVIEW
# ════════════════════════════════════════════════════════════════════
if "OVERVIEW" in page:

    report = load(LATEST_JSON)
    momentum = load(MOMENTUM_JSON)
    if not report:
        st.info("Data not found. Running the pipeline automatically for the first time... (This may take 2-3 minutes as it downloads 5-10 years of historical NAV data).")
        with st.spinner("Fetching data from AMFI..."):
            import subprocess
            import sys
            script_dir = Path(__file__).parent
            try:
                subprocess.run(
                    [sys.executable, str(script_dir / "pipeline.py")],
                    capture_output=True, text=True, check=True
                )
                st.rerun()
            except subprocess.CalledProcessError as e:
                st.error(f"Pipeline failed on startup! Error:\n{e.stderr}")
                st.stop()

    funds = [v for v in report.values() if "error" not in v]
    mom_by_code = {m.get("scheme_code"): m for m in (momentum or [])}

    # ── Ticker tape ───────────────────────────────────────────────
    ticker_items = ""
    for code, f in list(report.items())[:10]:
        name  = f.get("fund_name", f.get("scheme_name",""))[:18]
        cagr1 = f.get("cagr",{}).get("1y")
        if cagr1 is None: continue
        cls   = "up" if cagr1 > 0 else "down"
        ticker_items += (
            f'<div class="ticker-item">'
            f'<span class="name">{name}</span>'
            f'<span class="{cls}">{cagr1:+.1f}%</span>'
            f'</div>'
        )
    # duplicate for seamless loop
    st.markdown(
        f'<div class="ticker-wrap"><div class="ticker-inner">'
        f'{ticker_items}{ticker_items}'
        f'</div></div>',
        unsafe_allow_html=True
    )

    # ── Page header ───────────────────────────────────────────────
    st.markdown("""
    <div style="margin-bottom:2rem;">
      <h1 style="margin:0;">Fund Intelligence</h1>
      <p style="font-family:'DM Mono',monospace;font-size:11px;color:#3d5a72;
                letter-spacing:0.1em;text-transform:uppercase;margin-top:6px;">
        Indian equity market · real-time AMFI data · 
        """ + datetime.now().strftime("%d %b %Y %H:%M") + """
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI cards ─────────────────────────────────────────────────
    avg5 = [f.get("cagr",{}).get("5y") for f in funds if f.get("cagr",{}).get("5y")]
    avg5 = round(sum(avg5)/len(avg5),1) if avg5 else None
    buys = sum(1 for m in (momentum or []) if m.get("signal") in ("BUY","STRONG BUY"))
    best = max(funds, key=lambda f: f.get("risk",{}).get("sharpe_3y") or 0)
    neutrals = [m for m in (momentum or []) if m.get("signal") in ("NEUTRAL","REDUCE")
                and (m.get("composite_score") or 0) >= 20]
    neutrals.sort(key=lambda x: x.get("composite_score",0), reverse=True)

    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Funds tracked",  len(funds))
    k2.metric("Avg 5Y CAGR",    f"{avg5}%" if avg5 else "—",    delta="vs Nifty ~14%")
    k3.metric("BUY signals",    buys,                            delta="active now")
    k4.metric("Best Sharpe",    best.get("fund_name","")[:16],   delta=f"{best.get('risk',{}).get('sharpe_3y','—')}")
    k5.metric("Watchlist",      len(neutrals),                   delta="near BUY")

    st.divider()

    # ── Main layout ───────────────────────────────────────────────
    left, right = st.columns([3,2], gap="large")

    with left:
        st.markdown("### Fund Rankings")
        rows = []
        for code, f in report.items():
            if "error" in f: continue
            m = mom_by_code.get(code, {})
            rows.append({
                "Fund":      f.get("fund_name", f.get("scheme_name",""))[:32],
                "1Y%":       f.get("cagr",{}).get("1y"),
                "3Y%":       f.get("cagr",{}).get("3y"),
                "5Y%":       f.get("cagr",{}).get("5y"),
                "Sharpe":    f.get("risk",{}).get("sharpe_3y"),
                "DD%":       f.get("risk",{}).get("max_drawdown"),
                "XIRR%":     f.get("sip_simulation",{}).get("xirr_pct"),
                "Score":     m.get("composite_score"),
                "Signal":    m.get("signal","—"),
            })
        df = pd.DataFrame(rows).sort_values("5Y%", ascending=False)
        st.dataframe(df, use_container_width=True, hide_index=True,
                     column_config={
                        "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f"),
                        "5Y%":   st.column_config.NumberColumn("5Y%", format="%.1f%%"),
                        "3Y%":   st.column_config.NumberColumn("3Y%", format="%.1f%%"),
                        "1Y%":   st.column_config.NumberColumn("1Y%", format="%.1f%%"),
                     })

    with right:
        st.markdown("### Category CAGR")
        categories_map = {"Large Cap":"Large Cap", "Mid Cap":"Mid Cap", "Small Cap":"Small Cap", "Flexi Cap":"Flexi", "ELSS":"ELSS", "Index":"Index"}
        cats = list(categories_map.keys())
        cat_avgs = []
        for cat, keyword in categories_map.items():
            vals = [f.get("cagr",{}).get("5y") for f in funds
                    if (cat == f.get("category","") or keyword in f.get("fund_name", f.get("scheme_name", "")))
                    and f.get("cagr",{}).get("5y")]
            cat_avgs.append(round(sum(vals)/len(vals),1) if vals else 0)

        fig = go.Figure(go.Bar(
            x=cat_avgs, y=cats, orientation="h",
            marker=dict(
                color=cat_avgs,
                colorscale=[[0,"#162338"],[0.5,"#00b8cc"],[1,"#00e5ff"]],
                line=dict(color="rgba(0,229,255,0.3)", width=1),
            ),
            text=[f"{v:.1f}%" for v in cat_avgs],
            textposition="outside",
            textfont=dict(family="DM Mono", size=10, color="#00e5ff"),
        ))
        fig.update_layout(**{k:v for k,v in PLOTLY.items() if k not in ("xaxis","yaxis")}, height=240,
                          yaxis=dict(autorange="reversed",
                                     **PLOTLY["yaxis"]),
                          xaxis=dict(title="5Y CAGR %", **PLOTLY["xaxis"]))
        st.plotly_chart(fig, use_container_width=True)

        # Risk vs Return bubble
        st.markdown("### Risk · Return")
        scatter_pts = []
        for code, f in report.items():
            if "error" in f: continue
            cagr5 = f.get("cagr",{}).get("5y")
            vol   = f.get("risk",{}).get("volatility_3y_ann")
            aum   = f.get("aum", 20000)
            if cagr5 and vol:
                scatter_pts.append(dict(
                    x=vol, y=cagr5,
                    name=f.get("fund_name", f.get("scheme_name",""))[:20],
                    size=max(8, min(20, (aum or 20000)/5000)),
                ))

        if scatter_pts:
            fig2 = go.Figure()
            for pt in scatter_pts:
                fig2.add_trace(go.Scatter(
                    x=[pt["x"]], y=[pt["y"]],
                    mode="markers",
                    marker=dict(size=pt["size"], color="#00e5ff",
                                line=dict(color="#162338", width=1),
                                opacity=0.8),
                    showlegend=False,
                    hovertemplate=f"<b>{pt['name']}</b><br>CAGR: {pt['y']:.1f}%<br>Vol: {pt['x']:.1f}%<extra></extra>",
                ))
            fig2.update_layout(**{k:v for k,v in PLOTLY.items() if k not in ("xaxis","yaxis")}, height=220,
                               xaxis=dict(title="Volatility %", **PLOTLY["xaxis"]),
                               yaxis=dict(title="5Y CAGR %",    **PLOTLY["yaxis"]))
            st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── Watchlist: Top 3 closest to BUY ──────────────────────────
    st.markdown("### 🎯 Watchlist — Closest to BUY")
    st.markdown(
        '<p style="font-family:DM Mono,monospace;font-size:10px;color:#3d5a72;'
        'text-transform:uppercase;letter-spacing:0.1em;">'
        'NEUTRAL funds ranked by proximity to BUY threshold (score ≥ 60)</p>',
        unsafe_allow_html=True
    )

    top3 = neutrals[:3]
    if not top3:
        st.info("No NEUTRAL funds found — run `python momentum.py` to refresh.")
    else:
        # Render each card in its own st.columns slot to avoid HTML size limits
        wl_cols = st.columns(len(top3))
        for wl_col, m in zip(wl_cols, top3):
            score  = m.get("composite_score", 0)
            fa     = m.get("factors", {})
            tr     = fa.get("trailing_returns", {})
            accel  = fa.get("acceleration_pct")
            gap    = round(60 - score, 1)
            pct    = max(0, min(100, int((score / 60) * 100)))

            code   = m.get("scheme_code","")
            f      = report.get(code, {})
            name   = f.get("fund_name", m.get("scheme_name",""))[:28]
            cagr5  = f.get("cagr",{}).get("5y")
            sharpe = f.get("risk",{}).get("sharpe_3y")

            r1m = tr.get("1m_pct")
            r3m = tr.get("3m_pct")
            r1m_s = f"{r1m:+.1f}%" if r1m is not None else "—"
            r3m_s = f"{r3m:+.1f}%" if r3m is not None else "—"
            r1m_col = "#00e5ff" if r1m and r1m > 0 else "#ff4060" if r1m and r1m < 0 else "#7a9bb5"
            r3m_col = "#00e5ff" if r3m and r3m > 0 else "#ff4060" if r3m and r3m < 0 else "#7a9bb5"

            arc_col = "#00e5ff" if gap <= 5 else "#ffb300" if gap <= 15 else "#ff4060"

            # Acceleration line
            accel_line = ""
            if accel is not None:
                a_icon = "▲" if accel > 0 else "▼"
                a_col  = "#00ff88" if accel > 0 else "#ff4060"
                accel_line = f'<div style="margin-top:10px;padding:4px 8px;border-radius:6px;background:rgba(0,0,0,0.3);font-size:10px;color:{a_col};text-align:center;">{a_icon} {abs(accel):.1f}% momentum</div>'

            # Weakest factors line
            fs = m.get("factor_scores",{})
            weak_line = ""
            if fs:
                weak = sorted(fs.items(), key=lambda x: x[1])[:2]
                weak_line = '<div style="margin-top:6px;font-size:9px;color:#3d5a72;font-family:DM Mono,monospace;">' + " · ".join(f"{k.replace('_',' ').title()} {v*100:.0f}" for k,v in weak) + '</div>'

            # Sharpe line
            sharpe_line = ""
            if sharpe and str(sharpe).lower() != "nan":
                sharpe_line = f'<div style="font-size:9px;color:#3d5a72;margin-top:2px;font-family:DM Mono,monospace;">Sharpe 3Y: {sharpe:.2f}</div>'

            # SVG ring — compact
            ring = (
                f'<svg width="64" height="64" viewBox="0 0 64 64">'
                f'<circle cx="32" cy="32" r="27" fill="none" stroke="#162338" stroke-width="4"/>'
                f'<circle cx="32" cy="32" r="27" fill="none" stroke="{arc_col}" stroke-width="4" '
                f'stroke-dasharray="{pct * 1.696} {169.6 - pct * 1.696}" '
                f'stroke-dashoffset="42.4" stroke-linecap="round"/>'
                f'<text x="32" y="30" text-anchor="middle" fill="#e8f4f8" '
                f'font-family="Syne,sans-serif" font-size="16" font-weight="800">{score:.0f}</text>'
                f'<text x="32" y="42" text-anchor="middle" fill="#3d5a72" '
                f'font-family="DM Mono,monospace" font-size="7">/60</text>'
                f'</svg>'
            )

            card = f'''<div style="background:linear-gradient(145deg,#0a1628,#0d1b2a);border:1px solid rgba(0,229,255,0.08);border-radius:12px;padding:16px;position:relative;">
<div style="position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,{arc_col},transparent);"></div>
<div style="font-family:Syne,sans-serif;font-weight:700;font-size:14px;color:#e8f4f8;margin-bottom:2px;">{name}</div>
<div style="font-family:DM Mono,monospace;font-size:9px;color:#3d5a72;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:12px;">{"5Y: "+str(cagr5)+"%" if cagr5 else "—"}</div>
<div style="display:flex;align-items:center;gap:14px;">
<div>{ring}</div>
<div style="flex:1;">
<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px;">
<span style="font-family:DM Mono,monospace;font-size:9px;color:#7a9bb5;">GAP</span>
<span style="font-family:Syne,sans-serif;font-weight:700;font-size:13px;color:{arc_col};">{gap:.1f} pts</span>
</div>
<div style="width:100%;height:3px;background:#162338;border-radius:2px;overflow:hidden;margin-bottom:8px;">
<div style="width:{pct}%;height:100%;background:{arc_col};border-radius:2px;"></div>
</div>
<div style="display:flex;gap:14px;">
<div><div style="font-size:8px;color:#3d5a72;font-family:DM Mono,monospace;">1M</div><div style="font-size:13px;font-weight:700;color:{r1m_col};font-family:Syne,sans-serif;">{r1m_s}</div></div>
<div><div style="font-size:8px;color:#3d5a72;font-family:DM Mono,monospace;">3M</div><div style="font-size:13px;font-weight:700;color:{r3m_col};font-family:Syne,sans-serif;">{r3m_s}</div></div>
</div>
</div>
</div>
{accel_line}{weak_line}{sharpe_line}
</div>'''

            with wl_col:
                st.markdown(card, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# MOMENTUM
# ════════════════════════════════════════════════════════════════════
elif "MOMENTUM" in page:
    st.markdown("<h1>Momentum Signals</h1>", unsafe_allow_html=True)
    st.markdown('<p style="font-family:DM Mono,monospace;font-size:10px;color:#3d5a72;'
                'text-transform:uppercase;letter-spacing:0.1em;">'
                '8-factor composite scoring · 0–100 scale · updated daily</p>',
                unsafe_allow_html=True)

    momentum = load(MOMENTUM_JSON)
    if not momentum:
        st.info("Run `python momentum.py` to generate momentum scores.")
        st.stop()

    names  = [m.get("scheme_name","")[:28] for m in momentum]
    scores = [m.get("composite_score",0) for m in momentum]

    # Colour gradient: red→amber→cyan
    bar_colors = []
    for s in scores:
        if s >= 60:   bar_colors.append("#00e5ff")
        elif s >= 40: bar_colors.append("#ffb300")
        else:         bar_colors.append("#ff4060")

    fig = go.Figure()
    fig.add_vline(x=60, line=dict(color="#00e5ff", width=1, dash="dot"),
                  annotation_text="BUY", annotation_font=dict(color="#00e5ff", size=9))
    fig.add_vline(x=40, line=dict(color="#ffb300", width=1, dash="dot"),
                  annotation_text="NEUTRAL", annotation_font=dict(color="#ffb300", size=9))
    fig.add_trace(go.Bar(
        x=scores, y=names, orientation="h",
        marker=dict(color=bar_colors,
                    line=dict(color="rgba(0,0,0,0.3)", width=1)),
        text=[f"{s:.0f}" for s in scores],
        textposition="outside",
        textfont=dict(family="DM Mono", size=10, color="#7a9bb5"),
    ))
    fig.update_layout(**{k:v for k,v in PLOTLY.items() if k not in ("xaxis","yaxis")},
                      height=max(320, len(names)*38),
                      xaxis=dict(range=[0,115], title="Composite Score", **PLOTLY["xaxis"]),
                      yaxis=dict(autorange="reversed", **PLOTLY["yaxis"]))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Factor breakdown ──────────────────────────────────────────
    left, right = st.columns([1,1], gap="large")

    with left:
        st.markdown("### Factor Radar")
        sel_name = st.selectbox("Select fund", names, label_visibility="collapsed")
        sel = next((m for m in momentum if m.get("scheme_name","")[:28] == sel_name), {})

    with right:
        if sel:
            fa = sel.get("factors",{})
            tr = fa.get("trailing_returns",{})
            mc1,mc2,mc3 = st.columns(3)
            mc1.metric("Score",    sel.get("composite_score"))
            mc2.metric("Signal",   sel.get("signal","—"))
            mc3.metric("RSI(14)",  fa.get("rsi_14","—"))
            st.info(f"**Action:** {sel.get('action','—')}")

            t1,t2,t3,t4 = st.columns(4)
            t1.metric("1M",  pp(tr.get("1m_pct")))
            t2.metric("3M",  pp(tr.get("3m_pct")))
            t3.metric("6M",  pp(tr.get("6m_pct")))
            t4.metric("12M", pp(tr.get("12m_pct")))

    if sel:
        fs = sel.get("factor_scores",{})
        if fs:
            cats   = list(fs.keys())
            vals   = [fs[c]*100 for c in cats]
            cats_c = cats + [cats[0]]
            vals_c = vals + [vals[0]]

            fig2 = go.Figure(go.Scatterpolar(
                r=vals_c, theta=cats_c,
                fill="toself",
                fillcolor="rgba(0,229,255,0.07)",
                line=dict(color="#00e5ff", width=2),
                marker=dict(color="#00e5ff", size=5),
            ))
            fig2.update_layout(
                **{k:v for k,v in PLOTLY.items() if k not in ("xaxis","yaxis","margin")},
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0,100],
                                   gridcolor="rgba(0,229,255,0.1)",
                                   tickfont=dict(size=8,color="#3d5a72")),
                    angularaxis=dict(tickfont=dict(size=9,color="#7a9bb5"),
                                     gridcolor="rgba(0,229,255,0.08)"),
                ),
                height=340,
                margin=dict(l=40,r=40,t=40,b=40),
            )
            st.plotly_chart(fig2, use_container_width=True)

        trend = fa.get("trend",{})
        if trend:
            tc1,tc2,tc3 = st.columns(3)
            tc1.metric("MA 50",  f"₹{trend.get('ma50','—')}")
            tc2.metric("MA 200", f"₹{trend.get('ma200','—')}")
            tc3.metric("Bullish stack", "✅" if trend.get("ma_bullish_stack") else "❌")


# ════════════════════════════════════════════════════════════════════
# PORTFOLIO
# ════════════════════════════════════════════════════════════════════
elif "PORTFOLIO" in page:
    st.markdown("<h1>Portfolio Simulator</h1>", unsafe_allow_html=True)

    port = load(PORT_JSON)
    if not port:
        st.info("Run `python portfolio.py` to generate your portfolio report.")
        st.stop()

    s = port.get("summary",{})
    b = port.get("benchmark",{})
    d = port.get("diversification",{})

    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Invested",        inr(s.get("total_invested")))
    k2.metric("Current Value",   inr(s.get("total_current_value")),
              delta=f"{s.get('absolute_return_pct',0):+.1f}%")
    k3.metric("Total Gain",      inr(s.get("total_gain")))
    k4.metric("Monthly SIP",     inr(s.get("monthly_sip_total")))
    k5.metric("Diversification", d.get("grade","—"),
              delta=f"Score: {d.get('score',0)}")

    st.divider()

    left, right = st.columns([3,2], gap="large")

    with left:
        st.markdown("### Fund Breakdown")
        fund_rows = []
        for code, f in port.get("funds",{}).items():
            if "error" in f: continue
            sip = f.get("sip",{})
            ls  = f.get("lump_sum",{})
            fund_rows.append({
                "Fund":     f.get("name","")[:28],
                "Category": f.get("category",""),
                "SIP Val":  inr(sip.get("current_value")),
                "SIP XIRR":sip.get("xirr_pct"),
                "LS Val":   inr(ls.get("current_value")),
                "LS CAGR":  ls.get("cagr_pct"),
                "1M%":      f.get("trailing",{}).get("1m"),
                "3M%":      f.get("trailing",{}).get("3m"),
            })
        st.dataframe(pd.DataFrame(fund_rows), use_container_width=True, hide_index=True)

        st.markdown("### vs Nifty 50")
        bc1,bc2,bc3 = st.columns(3)
        bc1.metric("Portfolio 12M", pp(b.get("portfolio_12m_pct")))
        bc2.metric("Nifty 50 12M",  pp(b.get("nifty50_12m_pct")))
        bc3.metric("Your Alpha",    pp(b.get("alpha_12m_pct")), delta="vs benchmark")

    with right:
        st.markdown("### Allocation")
        cat_w = d.get("category_weights",{})
        if cat_w:
            colors = ["#00e5ff","#ffb300","#00ff88","#ff4060","#7a9bb5","#b388ff","#ff8a65"]
            fig = go.Figure(go.Pie(
                values=list(cat_w.values()),
                labels=list(cat_w.keys()),
                hole=0.62,
                marker=dict(colors=colors[:len(cat_w)],
                            line=dict(color="#030508", width=3)),
                textfont=dict(family="DM Mono", size=10),
                textinfo="label+percent",
            ))
            fig.update_layout(**{k:v for k,v in PLOTLY.items() if k not in ("xaxis","yaxis","margin")},
                              height=280, margin=dict(l=0,r=0,t=8,b=0),
                              showlegend=False)
            fig.add_annotation(text=f"<b>{len(cat_w)}</b><br>categories",
                               x=0.5, y=0.5, showarrow=False,
                               font=dict(family="Syne", size=13, color="#7a9bb5"))
            st.plotly_chart(fig, use_container_width=True)

        warnings = port.get("warnings",[])
        if warnings:
            st.markdown("### ⚠ Warnings")
            for w in warnings:
                if w["severity"] == "high": st.warning(w["message"])
                else: st.info(w["message"])

    with st.expander("✏  Edit portfolio — portfolio_config.json"):
        st.caption("Edit data/portfolio_config.json directly, then re-run python portfolio.py")
        cfg_path = Path("data/portfolio_config.json")
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg_text = f.read()
            edited = st.text_area("portfolio_config.json", cfg_text, height=300,
                                   label_visibility="collapsed")
            if st.button("💾  Save config"):
                try:
                    json.loads(edited)   # validate JSON
                    with open(cfg_path,"w") as f: f.write(edited)
                    st.success("Saved — run python portfolio.py to regenerate report.")
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON: {e}")


# ════════════════════════════════════════════════════════════════════
# BENCHMARK
# ════════════════════════════════════════════════════════════════════
elif "BENCHMARK" in page:
    st.markdown("<h1>Benchmark Analyser</h1>", unsafe_allow_html=True)
    st.markdown('<p style="font-family:DM Mono,monospace;font-size:10px;color:#3d5a72;'
                'text-transform:uppercase;letter-spacing:0.1em;">'
                'Alpha vs Nifty 50 · information ratio · up/down capture · KEEP or REPLACE</p>',
                unsafe_allow_html=True)

    bench = load(BENCH_JSON)
    if not bench:
        st.info("Run `python benchmark.py` to generate benchmark data.")
        st.stop()

    recs = [r.get("recommendation","") for r in bench.values()]
    k1,k2,k3 = st.columns(3)
    k1.metric("KEEP",    sum(1 for r in recs if r=="KEEP"),            delta="beats benchmark")
    k2.metric("REVIEW",  sum(1 for r in recs if r=="REVIEW"),          delta="marginal alpha")
    k3.metric("REPLACE", sum(1 for r in recs if "REPLACE" in r),       delta="use index fund")

    st.divider()

    alpha_rows = []
    for code, r in bench.items():
        nifty = r.get("benchmarks",{}).get("Nifty 50",{})
        alpha = nifty.get("alpha_pct",{})
        cap   = nifty.get("capture",{})
        alpha_rows.append({
            "Fund":          r.get("scheme_name","")[:28],
            "Alpha 1Y%":     alpha.get("1y"),
            "Alpha 3Y%":     alpha.get("3y"),
            "Alpha 5Y%":     alpha.get("5y"),
            "Info Ratio":    nifty.get("information_ratio"),
            "Up Cap%":       cap.get("up_capture_pct"),
            "Dn Cap%":       cap.get("down_capture_pct"),
            "Beat%":         nifty.get("beat_consistency_pct"),
            "Verdict":       r.get("recommendation",""),
        })
    st.dataframe(pd.DataFrame(alpha_rows), use_container_width=True, hide_index=True)

    st.markdown("### Deep Dive")
    selected = st.selectbox("Fund", [r.get("scheme_name") for r in bench.values()],
                             label_visibility="collapsed")
    sel = next((r for r in bench.values() if r.get("scheme_name")==selected), {})
    if sel:
        for bname, bdata in sel.get("benchmarks",{}).items():
            with st.expander(f"vs {bname}", expanded=True):
                a1,a2,a3,a4 = st.columns(4)
                a1.metric("Info Ratio",    bdata.get("information_ratio","—"))
                a2.metric("Beat %",        f"{bdata.get('beat_consistency_pct','—')}%")
                cap = bdata.get("capture",{})
                a3.metric("Up Capture",    f"{cap.get('up_capture_pct','—')}%")
                a4.metric("Down Capture",  f"{cap.get('down_capture_pct','—')}%")
                st.info(f"Capture: {cap.get('verdict','—')}")
                rec = bdata.get("verdict","—")
                if "KEEP" in rec:    st.success(f"Recommendation: {rec}")
                elif "REVIEW" in rec:st.warning(f"Recommendation: {rec}")
                else:                st.error(f"Recommendation: {rec}")


# ════════════════════════════════════════════════════════════════════
# DRAWDOWN
# ════════════════════════════════════════════════════════════════════
elif "DRAWDOWN" in page:
    st.markdown("<h1>Drawdown Tracker</h1>", unsafe_allow_html=True)

    dd = load(DD_JSON)
    if not dd:
        st.info("Run `python drawdown.py` to generate drawdown data.")
        st.stop()

    dd_rows = []
    for code, r in dd.items():
        s = r.get("summary",{})
        dd_rows.append({
            "Fund":            r.get("scheme_name","")[:28],
            "Current DD%":     r.get("current_drawdown_pct"),
            "Recovery Needed%":r.get("recovery_needed_pct"),
            "Worst DD%":       s.get("worst_drawdown_pct"),
            "Avg Recov (mo)":  s.get("avg_recovery_months"),
            "Max Recov (d)":   s.get("max_recovery_days"),
            "SIP Wins":        "✅" if s.get("sip_always_wins") else "❌",
        })
    df = pd.DataFrame(dd_rows).sort_values("Current DD%")
    st.dataframe(df, use_container_width=True, hide_index=True,
                 column_config={
                    "Current DD%":  st.column_config.NumberColumn("Current DD%", format="%.1f%%"),
                    "Worst DD%":    st.column_config.NumberColumn("Worst DD%",   format="%.1f%%"),
                 })

    st.divider()

    selected = st.selectbox("Deep dive", [r.get("scheme_name") for r in dd.values()],
                             label_visibility="collapsed")
    sel = next((r for r in dd.values() if r.get("scheme_name")==selected), {})
    if sel:
        d1,d2,d3,d4 = st.columns(4)
        d1.metric("Current NAV",     f"₹{sel.get('current_nav',0):.2f}")
        d2.metric("All-Time High",   f"₹{sel.get('all_time_high',0):.2f}")
        d3.metric("From ATH",        pp(sel.get("current_drawdown_pct")))
        d4.metric("Recovery Needed", pp(sel.get("recovery_needed_pct")))

        events = sel.get("events",[])
        if events:
            st.markdown("### Historical Drawdown Events")
            ev_rows = []
            for e in events[:8]:
                sb = e.get("sip_benefit",{})
                ev_rows.append({
                    "Peak":         e.get("peak_date"),
                    "Trough":       e.get("trough_date"),
                    "DD%":          e.get("drawdown_pct"),
                    "Fall(d)":      e.get("drawdown_days"),
                    "Recov(d)":     e.get("recovery_days","Not recovered"),
                    "SIP ret%":     sb.get("sip_return_pct"),
                    "LS ret%":      sb.get("lump_return_pct"),
                    "SIP edge%":    sb.get("sip_advantage_pct"),
                })
            st.dataframe(pd.DataFrame(ev_rows), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════
# STEP-UP
# ════════════════════════════════════════════════════════════════════
elif "STEP-UP" in page:
    st.markdown("<h1>SIP Step-Up Planner</h1>", unsafe_allow_html=True)

    ctrl, chart = st.columns([1,2], gap="large")

    with ctrl:
        initial_sip   = st.slider("Starting SIP (₹/mo)", 1000, 100000, 10000, 500)
        stepup_pct    = st.slider("Annual step-up %", 0, 30, 10, 1) / 100
        expected_cagr = st.slider("Expected CAGR %", 8, 25, 15, 1) / 100
        years         = st.slider("Horizon (years)", 5, 35, 20, 1)
        lump_sum      = st.number_input("Lump sum (₹)", 0, 10000000, 500000, 50000)
        inflation     = st.slider("Inflation %", 4, 10, 6, 1) / 100

        from stepup import compare_flat_vs_stepup, run_all_goals
        result = compare_flat_vs_stepup(initial_sip, stepup_pct, expected_cagr,
                                         years, lump_sum, inflation)
        flat_f   = result["flat_sip"]
        stepup_f = result["stepup_sip"]
        adv      = result["step_up_advantage"]

        st.divider()
        st.metric("Flat SIP corpus",   inr(flat_f["final_corpus"]),
                   f"{flat_f['wealth_multiple']}x invested")
        st.metric("Step-up corpus",    inr(stepup_f["final_corpus"]),
                   f"+{adv['corpus_boost_pct']}% vs flat")
        st.metric("Extra wealth",      inr(adv["extra_corpus"]))
        st.metric("Final monthly SIP", inr(stepup_f["final_monthly_sip"]))

    with chart:
        flat_df   = pd.DataFrame(flat_f["yearly"])
        stepup_df = pd.DataFrame(stepup_f["yearly"])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=stepup_df["year"], y=stepup_df["corpus"],
            name="Step-up", line=dict(color="#00e5ff", width=2.5),
            fill="tozeroy", fillcolor="rgba(0,229,255,0.04)",
        ))
        fig.add_trace(go.Scatter(
            x=flat_df["year"], y=flat_df["corpus"],
            name="Flat SIP", line=dict(color="#7a9bb5", width=1.5, dash="dash"),
        ))
        fig.add_trace(go.Scatter(
            x=stepup_df["year"], y=stepup_df["real_value"],
            name="Real (inflation-adj)", line=dict(color="#ffb300", width=1.5, dash="dot"),
        ))
        fig.add_trace(go.Scatter(
            x=stepup_df["year"], y=stepup_df["total_invested"],
            name="Invested", line=dict(color="#3d5a72", width=1),
            fill="tozeroy", fillcolor="rgba(61,90,114,0.08)",
        ))
        fig.update_layout(**{k:v for k,v in PLOTLY.items() if k not in ("xaxis","yaxis")}, height=360,
                          legend=dict(orientation="h", y=1.05,
                                      font=dict(size=9)),
                          yaxis=dict(tickformat=",.0f", title="₹ Corpus",
                                     **PLOTLY["yaxis"]),
                          xaxis=dict(title="Years", **PLOTLY["xaxis"]))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Goal Planning")
    goals = run_all_goals(expected_cagr, stepup_pct, lump_sum)
    g_rows = []
    for key, g in goals.items():
        p = g["plan"]
        g_rows.append({
            "Goal":           g["label"],
            "Target":         inr(p["target_corpus"]),
            "Years":          p["years"],
            "Start SIP/mo":   f"₹{p['required_initial_sip']:,.0f}",
            "Final SIP/mo":   f"₹{p['final_monthly_sip']:,.0f}",
            "Total Invested": inr(p["total_invested"]),
            "Projected":      inr(p["projected_corpus"]),
        })
    st.dataframe(pd.DataFrame(g_rows), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════
# TAX
# ════════════════════════════════════════════════════════════════════
elif "TAX" in page:
    st.markdown("<h1>Tax-Adjusted Returns</h1>", unsafe_allow_html=True)
    st.markdown('<p style="font-family:DM Mono,monospace;font-size:10px;color:#3d5a72;'
                'text-transform:uppercase;letter-spacing:0.1em;">'
                'Resident Individual · LTCG 12.5% · STCG 20% · 4% cess · FY 2024-25</p>',
                unsafe_allow_html=True)

    report = load(LATEST_JSON)
    if not report:
        st.warning("Run pipeline.py first.")
        st.stop()

    with st.expander("Tax rules — FY 2024-25"):
        st.markdown("""
| Type | Holding | Rate | Exemption |
|---|---|---|---|
| Equity LTCG | > 12 months | 12.5% + 4% cess | ₹1,25,000/yr |
| Equity STCG | ≤ 12 months | 20% + 4% cess | None |
| Debt | Any | Slab 30% + cess | None |
| ELSS | > 36 months | 12.5% + 4% cess | ₹1,25,000/yr |
        """)

    tax_rows = []
    for code, f in report.items():
        if "error" in f: continue
        pre5 = f.get("cagr",{}).get("5y")
        pt5  = f.get("post_tax_cagr",{}).get("5y",{})
        tax_rows.append({
            "Fund":           f.get("fund_name", f.get("scheme_name",""))[:28],
            "Pre-tax 5Y%":    pre5,
            "Post-tax 5Y%":   pt5.get("post_tax_cagr_pct"),
            "Tax drag%":      pt5.get("tax_drag_pct"),
            "Regime":         pt5.get("regime","")[:32] if pt5 else "",
        })
    df = pd.DataFrame(tax_rows).dropna(subset=["Pre-tax 5Y%"]).sort_values(
        "Post-tax 5Y%", ascending=False)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### Interactive Tax Calculator")
    tc1, tc2 = st.columns(2)
    with tc1:
        inv    = st.number_input("Invested (₹)", 10000, 10000000, 500000, 10000)
        pcagr  = st.slider("Pre-tax CAGR %", 5.0, 35.0, 15.0, 0.5)
        hyrs   = st.slider("Holding (years)", 1, 20, 5)
        fcat   = st.selectbox("Category",
                               ["Large Cap","Mid Cap","Small Cap","Flexi Cap","ELSS","Index","Debt"])
    with tc2:
        from tax import post_tax_cagr as ptc
        res = ptc(pcagr, hyrs, inv, fcat)
        if res:
            st.metric("Gross final",   inr(res.get("gross_final_value")))
            st.metric("Tax owed",      inr(res.get("tax_amount")))
            st.metric("Net final",     inr(res.get("net_final_value")))
            st.metric("Post-tax CAGR", f"{res.get('post_tax_cagr_pct','—')}%",
                       delta=f"Drag: -{res.get('tax_drag_pct','—')}%")
            st.info(f"Regime: {res.get('regime','')}")

    st.divider()
    st.markdown("### ELSS vs Regular Equity")
    e1,e2,e3 = st.columns(3)
    esip  = e1.number_input("SIP/mo (₹)", 1000, 100000, 12500, 500, key="etax_sip")
    eyrs  = e2.slider("Years", 3, 20, 10, key="etax_yrs")
    ecagr = e3.slider("CAGR %", 8, 25, 15, key="etax_cagr")
    from tax import elss_vs_equity_comparison
    comp = elss_vs_equity_comparison(esip, eyrs, float(ecagr))
    r1,r2,r3 = st.columns(3)
    r1.metric("ELSS net",    inr(comp["elss"]["net_corpus"]),
               f"80C saved: {inr(comp['elss']['80c_tax_saved'])}")
    r2.metric("Regular EQ",  inr(comp["regular_equity"]["net_corpus"]))
    r3.metric("ELSS edge",   inr(comp["elss_advantage_rs"]), comp["verdict"])