import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objects as go
import plotly.io as pio
from datetime import datetime

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Cardboard Compass", layout="wide")

# ============================================================
# THEME (OPTION B — distinct from blue/green)
# ============================================================
THEME = {
    "primary": "#6D5EF7",     # violet
    "secondary": "#8B7CFF",   # lighter violet
    "accent_red": "#FF4D4D",
    "bg": "#F6F7FB",
    "card": "#FFFFFF",
    "text": "#111827",
    "muted": "#6B7280",
    "border": "#E5E7EB",
    "grid": "#EEF2F7",
}

APP_TITLE = "CARDBOARD COMPASS"
APP_SUBTITLE = "eBay market-index insights — built by Pancake Analytics"

# ============================================================
# GLOBAL CSS (includes slide/print + animations)
# ============================================================
st.markdown(
    f"""
    <style>
      body {{ background-color: {THEME['bg']}; }}
      .block-container {{ padding-top: 1.4rem; padding-bottom: 2rem; }}
      h1,h2,h3,h4 {{ color: {THEME['text']}; }}
      .muted {{ color: {THEME['muted']}; font-size: 0.90rem; }}

      /* Cards */
      .pa-card {{
        background: {THEME['card']};
        border: 1px solid {THEME['border']};
        border-radius: 16px;
        padding: 16px 18px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
      }}

      /* Header bar (prevents cutoff) */
      .pa-header {{
        border-radius: 18px;
        overflow: hidden;
        border: 1px solid {THEME['border']};
        box-shadow: 0 10px 20px rgba(0,0,0,0.05);
      }}
      .pa-header-inner {{
        display: grid;
        grid-template-columns: 2.5fr 1fr;
        align-items: stretch;
      }}
      .pa-left {{
        padding: 22px 24px;
        background: {THEME['primary']};
        color: white;
      }}
      .pa-right {{
        padding: 22px 24px;
        background: {THEME['secondary']};
        color: white;
        text-align: right;
      }}
      .pa-title {{
        font-weight: 900;
        letter-spacing: 0.6px;
        font-size: 28px;
        line-height: 1.1;
        margin: 0;
      }}
      .pa-sub {{
        margin-top: 8px;
        font-size: 13px;
        opacity: 0.92;
      }}
      .pa-asof {{
        font-size: 13px;
        opacity: 0.95;
        margin: 0;
      }}
      .pa-asof b {{
        display: block;
        margin-top: 6px;
        font-size: 22px;
        letter-spacing: 0.3px;
      }}

      /* Fade-in animation for section transitions */
      .fade-in {{
        animation: fadeIn 0.45s ease-in-out;
      }}
      @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(6px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
      }}

      /* Remove iframe rounding artifacts */
      iframe {{ border-radius: 12px; }}

      /* STREAMLIT TABLE SCROLL FIXES:
         - st.dataframe is inherently scrollable; use st.table for "no scrolling" requirements */
      div[data-testid="stDataFrame"] > div {{
        overflow: auto;
      }}

      /* Slide mode container */
      .slide-wrap {{
        max-width: 1180px;
        margin: 0 auto;
      }}

      /* Print rules — "export-ready PDF layout" */
      @media print {{
        header, footer, [data-testid="stSidebar"], [data-testid="stToolbar"] {{
          display: none !important;
        }}
        .block-container {{
          padding: 0 !important;
        }}
        .pa-pagebreak {{
          page-break-after: always;
          break-after: page;
        }}
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# DATA LOADING
# ============================================================
DATA_URL = "https://pancakebreakfaststats.com/wp-content/uploads/2025/12/data_file_012.xlsx"

@st.cache_data
def load_data(url: str) -> pd.DataFrame:
    return pd.read_excel(url)

df_raw = load_data(DATA_URL)
df_raw["Month_Year"] = pd.to_datetime(df_raw["Month"] + " " + df_raw["Year"].astype(str))

CATEGORIES = [
    "Fortnite", "Marvel", "Pokemon", "Star Wars", "Magic the Gathering",
    "Baseball", "Basketball", "Football", "Hockey", "Soccer"
]

# ============================================================
# HELPERS
# ============================================================
def preprocess(df: pd.DataFrame, cat: str) -> pd.DataFrame:
    d = df[df["Category"] == cat].copy()
    d["Month_Year"] = pd.to_datetime(d["Month"] + " " + d["Year"].astype(str))
    return d.groupby("Month_Year")["market_value"].mean().reset_index()

def forecast(df: pd.DataFrame, horizon=12, seasonal_periods=12, trend="add", seasonal="add", ci_level=0.95):
    y = df.market_value.astype(float)
    model = ExponentialSmoothing(y, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods).fit()
    fc = model.forecast(horizon)

    z_table = {0.80:1.2816, 0.90:1.6449, 0.95:1.9600, 0.98:2.3263, 0.99:2.5758}
    z = z_table.get(round(ci_level, 2), 1.96)
    ci = z * np.std(model.resid)

    future = pd.date_range(df.Month_Year.iloc[-1] + pd.DateOffset(months=1), periods=horizon, freq="MS")
    fc_df = pd.DataFrame({"Date": future, "Forecast": fc, "Upper": fc + ci, "Lower": fc - ci})
    hist_df = pd.DataFrame({"Date": df.Month_Year, "Historical": df.market_value.values})
    return hist_df, fc_df

def macd(df: pd.DataFrame):
    s = df.market_value.ewm(span=12, adjust=False).mean()
    l = df.market_value.ewm(span=26, adjust=False).mean()
    m = s - l
    sig = m.ewm(span=9, adjust=False).mean()
    bucket = pd.cut(
        m - sig,
        [-np.inf, -1.5, -0.5, 0, 0.5, 1.5, np.inf],
        labels=["High Down", "Med Down", "Low Down", "Low Up", "Med Up", "High Up"]
    )
    return m, sig, bucket

def yoy_3mo(series: pd.Series, latest: pd.Timestamp):
    now = series.get(latest, np.nan)
    yr  = series.get(latest - pd.DateOffset(years=1), np.nan)
    m3  = series.get(latest - pd.DateOffset(months=3), np.nan)
    yoy = np.nan if np.isnan(now) or np.isnan(yr) else (now - yr) / yr * 100
    r3  = np.nan if np.isnan(now) or np.isnan(m3) else (now - m3) / m3 * 100
    return yoy, r3

def apply_fig_theme(fig: go.Figure, height: int, slide_mode: bool):
    # Animation: smooth chart transitions
    fig.update_layout(
        transition=dict(duration=450, easing="cubic-in-out"),
        paper_bgcolor=THEME["card"] if not slide_mode else "#FFFFFF",
        plot_bgcolor=THEME["card"] if not slide_mode else "#FFFFFF",
        font=dict(color=THEME["text"]),
        margin=dict(l=16, r=16, t=64, b=18),
        height=height,
        title=dict(x=0.02, xanchor="left", y=0.98),
    )
    fig.update_xaxes(gridcolor=THEME["grid"], zerolinecolor=THEME["grid"])
    fig.update_yaxes(gridcolor=THEME["grid"], zerolinecolor=THEME["grid"])
    return fig

def kpi_card(label: str, value: str, sub: str | None = None):
    sub_html = f"<div class='muted'>{sub}</div>" if sub else ""
    st.markdown(
        f"""
        <div class="pa-card fade-in">
          <div class="muted">{label}</div>
          <div style="font-size:32px; font-weight:900; margin-top:4px;">{value}</div>
          {sub_html}
        </div>
        """,
        unsafe_allow_html=True
    )

def section_card(title: str, body_html: str):
    st.markdown(
        f"""
        <div class="pa-card fade-in">
          <div style="font-weight:900; font-size:16px; margin-bottom:8px;">{title}</div>
          {body_html}
        </div>
        """,
        unsafe_allow_html=True
    )

def build_market_summary(df: pd.DataFrame, cats: list[str]):
    wide = (
        df[df["Category"].isin(cats)]
        .pivot_table(values="market_value", index="Month_Year", columns="Category", aggfunc="mean")
        .sort_index()
    )

    last_row = wide.index.max()
    y_ago = last_row - pd.DateOffset(years=1)
    m_3  = last_row - pd.DateOffset(months=3)

    def pct(series, t0, t1):
        v0, v1 = series.get(t0, np.nan), series.get(t1, np.nan)
        return np.nan if np.isnan(v0) or np.isnan(v1) else (v1 - v0) / v0 * 100

    rows = []
    for c in cats:
        s = wide[c]
        rows.append({
            "Category": c,
            "YoY %": pct(s, y_ago, last_row),
            "3-Mo %": pct(s, m_3, last_row)
        })

    summary = pd.DataFrame(rows).set_index("Category").sort_index()
    composite = wide[cats].mean(axis=1)
    comp_yoy = pct(composite, y_ago, last_row)
    comp_3mo = pct(composite, m_3, last_row)
    breadth = float(np.mean(summary["3-Mo %"] > 0) * 100)

    return summary, last_row, comp_yoy, comp_3mo, breadth

def download_print_ready_html(html: str, filename: str):
    st.download_button(
        "⬇️ Download print-ready HTML (save as PDF from browser)",
        data=html.encode("utf-8"),
        file_name=filename,
        mime="text/html"
    )

# ============================================================
# SIDEBAR NAV (kept)
# ============================================================
PAGES = [
    "Pancake Analytics Trading Card Market Report",
    "Category Analysis",
    "Market HeatMap",
    "State of Market",
    "Custom Index Builder",
    "Seasonality HeatMap",
    "Rolling Volatility",
    "Correlation Matrix",
    "Flip Forecast",
]

with st.sidebar:
    st.markdown(f"### {APP_TITLE}")
    st.markdown(f"<div class='muted'>{APP_SUBTITLE}</div>", unsafe_allow_html=True)
    st.markdown("---")

    # Slide mode toggle
    slide_mode = st.toggle("📄 Slide Mode", value=False, help="Collector print/PDF-friendly layout")

    page = st.selectbox(
        "Choose an analysis",
        PAGES,
        index=PAGES.index("Pancake Analytics Trading Card Market Report")
    )

    st.markdown("---")
    cat1 = st.selectbox("Primary category", CATEGORIES, index=CATEGORIES.index("Pokemon"))
    cat2 = st.selectbox("Compare against", ["None"] + [c for c in CATEGORIES if c != cat1])

# Optional slide wrapper
if slide_mode:
    st.markdown("<div class='slide-wrap'>", unsafe_allow_html=True)

# ============================================================
# MARKET REPORT (AUTO LOAD, EXEC FORMAT)
# ============================================================
if page == "Pancake Analytics Trading Card Market Report":
    summary, last_row, comp_yoy, comp_3mo, breadth = build_market_summary(df_raw, CATEGORIES)

    # Header (no logo)
    st.markdown(
        f"""
        <div class="pa-header fade-in">
          <div class="pa-header-inner">
            <div class="pa-left">
              <div class="muted" style="color:rgba(255,255,255,0.85)">@pancake_analytics</div>
              <p class="pa-title">TRADING CARD<br/>MARKET REPORT</p>
              <div class="pa-sub">Collector snapshot of YoY + 3-Mo momentum</div>
            </div>
            <div class="pa-right">
              <p class="pa-asof">AS OF<b>{last_row:%b %Y}</b></p>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # KPI row
    c1, c2, c3 = st.columns(3)
    with c1: kpi_card("Composite YoY", f"{comp_yoy:0.1f}%")
    with c2: kpi_card("Composite 3-Mo", f"{comp_3mo:0.1f}%")
    with c3: kpi_card("Breadth (3-Mo > 0)", f"{breadth:0.0f}%")

    st.markdown("")

    # Leaders / Laggards
    top_3mo = summary.sort_values("3-Mo %", ascending=False).head(3)
    top_yoy = summary.sort_values("YoY %", ascending=False).head(3)
    lag_3mo = summary.sort_values("3-Mo %", ascending=True).head(2)

    leaders_text = (
        f"<div class='muted'><b>What the Data Says:</b> "
        f"3-Mo leaders: {', '.join(top_3mo.index)}. "
        f"YoY leaders: {', '.join(top_yoy.index)}. "
        f"Recent laggards: {', '.join(lag_3mo.index)}.</div>"
        f"<div style='margin-top:10px;'><b>What It Means:</b> "
        f"The last 3 months show where momentum is concentrating. Laggards can be a buyer’s window — "
        f"especially if you’re building long-term.</div>"
    )

    # ==== ONE ROW: Momentum map | Donut | Top YoY chart (no blank space) ====
    left, mid, right = st.columns([2.05, 1.35, 1.35])

    # Momentum map
    fig_sc = go.Figure()
    fig_sc.add_trace(go.Scatter(
        x=summary["3-Mo %"],
        y=summary["YoY %"],
        mode="markers+text",
        text=summary.index,
        textposition="top center",
        marker=dict(size=10, color=THEME["primary"]),
        name="",
        showlegend=False
    ))
    fig_sc.add_hline(y=0, line_dash="dash", opacity=0.5)
    fig_sc.add_vline(x=0, line_dash="dash", opacity=0.5)
    fig_sc.update_layout(
        title=f"Momentum Map — YoY vs 3-Mo (through {last_row:%b %Y})",
        xaxis_title="3-Month % change",
        yaxis_title="YoY % change",
    )
    apply_fig_theme(fig_sc, height=360, slide_mode=slide_mode)

    # Donut: share of 3-Mo momentum (Top-3 only, normalized)
    top3 = top_3mo.copy()
    top3_sum = float(top3["3-Mo %"].sum()) if not top3["3-Mo %"].isna().all() else 0.0
    shares = (top3["3-Mo %"] / top3_sum * 100) if top3_sum != 0 else pd.Series([0, 0, 0], index=top3.index)

    fig_dn = go.Figure(go.Pie(
        labels=top3.index,
        values=shares,
        hole=0.62,
        textinfo="label+percent",
        sort=False,
        marker=dict(colors=[THEME["primary"], THEME["secondary"], THEME["accent_red"]]),
        showlegend=True
    ))
    fig_dn.update_layout(
        title="Top-3 movers — share of 3-Mo momentum",
        annotations=[dict(
            text="Normalized<br>Top-3 only",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(color=THEME["muted"], size=12)
        )],
        legend=dict(orientation="v", x=1.02, y=0.95)
    )
    apply_fig_theme(fig_dn, height=360, slide_mode=slide_mode)
    # Prevent title truncation explicitly:
    fig_dn.update_layout(margin=dict(l=16, r=80, t=74, b=18))

    # Top YoY bar (fix "undefined" by ensuring name is empty + legends off)
    top5_yoy = summary.sort_values("YoY %", ascending=False).head(5)
    fig_y = go.Figure()
    fig_y.add_trace(go.Bar(
        x=top5_yoy["YoY %"].values,
        y=top5_yoy.index.tolist(),
        orientation="h",
        marker=dict(color=THEME["primary"]),
        name="",
        showlegend=False
    ))
    fig_y.update_layout(
        title="Top YoY %",
        xaxis_title="YoY %",
        yaxis_title="",
        showlegend=False
    )
    fig_y.update_yaxes(autorange="reversed")
    apply_fig_theme(fig_y, height=360, slide_mode=slide_mode)
    fig_y.update_layout(margin=dict(l=16, r=16, t=74, b=18))  # avoid truncation

    # Render row
    with left:
        st.plotly_chart(fig_sc, use_container_width=True, theme="streamlit")
    with mid:
        st.plotly_chart(fig_dn, use_container_width=True, theme="streamlit")
    with right:
        st.plotly_chart(fig_y, use_container_width=True, theme="streamlit")

    st.markdown("")

    # Leaders + Top Category YoY + exec meaning row (no weird gaps)
    r1, r2 = st.columns([1.15, 0.85])
    with r1:
        section_card("This Month’s Leaders", leaders_text)

    with r2:
        best_cat = summary["YoY %"].idxmax()
        best_val = float(summary.loc[best_cat, "YoY %"])
        body = (
            f"<div class='muted'>Top category YoY</div>"
            f"<div style='font-size:42px; font-weight:900; margin-top:4px;'>{best_val:0.1f}%</div>"
            f"<div class='muted' style='margin-top:6px;'>{best_cat}</div>"
        )
        section_card("Top Category YoY", body)

    st.markdown("")

    # Bottom full table (NO SCROLL): use st.table
    st.markdown("### Full Category Table (YoY + 3-Mo)")
    bottom_tbl = summary[["YoY %", "3-Mo %"]].round(2).loc[sorted(CATEGORIES)]
    st.table(bottom_tbl)

    # Print-ready HTML download (optional convenience)
    st.markdown("---")
    st.markdown("#### Export")
    st.markdown(
        "<div class='muted'>Use Slide Mode → then browser Print → Save as PDF for a deck-ready export.</div>",
        unsafe_allow_html=True
    )

    # Simple print-ready HTML (layout + table only, charts remain in-app)
    html_snapshot = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>Trading Card Market Report</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 24px; }}
  h1 {{ margin: 0; }}
  .kpis {{ display:flex; gap:14px; margin-top:16px; }}
  .card {{ border:1px solid #ddd; border-radius:12px; padding:14px 16px; flex:1; }}
  .muted {{ color:#666; font-size:13px; }}
  table {{ border-collapse: collapse; width:100%; margin-top:14px; }}
  th, td {{ border:1px solid #ddd; padding:8px 10px; text-align:left; }}
  th {{ background:#f5f5f5; }}
</style>
</head>
<body>
  <div>
    <div class="muted">@pancake_analytics</div>
    <h1>Trading Card Market Report</h1>
    <div class="muted">As of {last_row:%B %Y}</div>
  </div>

  <div class="kpis">
    <div class="card"><div class="muted">Composite YoY</div><div style="font-size:28px;font-weight:800;">{comp_yoy:0.1f}%</div></div>
    <div class="card"><div class="muted">Composite 3-Mo</div><div style="font-size:28px;font-weight:800;">{comp_3mo:0.1f}%</div></div>
    <div class="card"><div class="muted">Breadth (3-Mo &gt; 0)</div><div style="font-size:28px;font-weight:800;">{breadth:0.0f}%</div></div>
  </div>

  <h3 style="margin-top:20px;">Full Category Table (YoY + 3-Mo)</h3>
  {bottom_tbl.to_html()}
</body>
</html>"""
    download_print_ready_html(
        html_snapshot,
        filename=f"cardboard_compass_market_report_{last_row:%Y_%m}.html"
    )

# ============================================================
# CATEGORY ANALYSIS
# ============================================================
elif page == "Category Analysis":
    st.markdown(f"<div class='pa-card fade-in'><h3>Category Analysis</h3><div class='muted'>Forecast + MACD + seasonality</div></div>", unsafe_allow_html=True)
    st.markdown("")

    def show_category(cat: str):
        d = preprocess(df_raw, cat)

        with st.expander("Forecast settings", expanded=False):
            horizon = st.slider("Horizon (months)", 6, 24, 12, step=1, key=f"h_{cat}")
            ci = st.select_slider("Confidence interval", options=[0.80, 0.90, 0.95, 0.98, 0.99],
                                  value=0.95, key=f"ci_{cat}")
            hw_trend = st.selectbox("Trend", ["add", "mul"], index=0, key=f"t_{cat}")
            hw_seasonal = st.selectbox("Seasonal", ["add", "mul"], index=0, key=f"s_{cat}")
            sp = st.number_input("Seasonal periods", min_value=4, max_value=24, value=12, step=1, key=f"sp_{cat}")

        hist_df, fc_df = forecast(d, horizon=horizon, seasonal_periods=sp, trend=hw_trend, seasonal=hw_seasonal, ci_level=ci)

        pct_change = (fc_df.Forecast.iloc[-1] - d.market_value.iloc[-1]) / d.market_value.iloc[-1] * 100

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_df["Date"], y=hist_df["Historical"], mode="lines", name="Historical"))
        fig.add_trace(go.Scatter(x=fc_df["Date"], y=fc_df["Forecast"], mode="lines", name="Forecast", line=dict(dash="dash")))
        fig.add_trace(go.Scatter(
            x=pd.concat([fc_df["Date"], fc_df["Date"][::-1]]),
            y=pd.concat([fc_df["Upper"], fc_df["Lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(109, 94, 247, 0.15)",
            line=dict(color="rgba(109, 94, 247, 0)"),
            name=f"{int(ci*100)}% interval",
            hoverinfo="skip",
            showlegend=True
        ))
        fig.update_layout(title=f"{cat} — {horizon}-Month Holt-Winters Forecast", xaxis_title="Month", yaxis_title="Market Value")
        apply_fig_theme(fig, height=420, slide_mode=slide_mode)
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")

        section_card(
            "Forecast Read",
            f"<div><b>What the Data Says:</b> Next {horizon} months project <b>{pct_change:+.1f}%</b> vs last observed.</div>"
            f"<div style='margin-top:8px;'><b>What It Means:</b> Use this as directionally helpful — not an exact card price predictor.</div>"
        )

        # MACD
        m, sig, bucket = macd(d)
        macd_df = pd.DataFrame({"Date": d.Month_Year, "MACD": m.values, "Signal": sig.values})

        fig_m = go.Figure()
        fig_m.add_trace(go.Scatter(x=macd_df["Date"], y=macd_df["MACD"], mode="lines", name="MACD"))
        fig_m.add_trace(go.Scatter(x=macd_df["Date"], y=macd_df["Signal"], mode="lines", name="Signal"))
        fig_m.add_hline(y=0, line_dash="dash", opacity=0.6)
        fig_m.update_layout(title=f"{cat} — MACD Trend (most recent: {bucket.iloc[-1]})", xaxis_title="Month", yaxis_title="MACD")
        apply_fig_theme(fig_m, height=340, slide_mode=slide_mode)
        st.plotly_chart(fig_m, use_container_width=True, theme="streamlit")

        # Seasonality
        dd = d.copy()
        dd["Month"] = dd.Month_Year.dt.month_name()
        month_order = ["January","February","March","April","May","June","July","August","September","October","November","December"]
        month_avg = dd.groupby("Month").market_value.mean().reindex(month_order)

        fig_s = go.Figure(go.Bar(x=month_avg.index, y=month_avg.values, marker=dict(color=THEME["primary"]), name=""))
        fig_s.update_layout(title=f"{cat} — Seasonality (Avg by Month)", xaxis_title="Month", yaxis_title="Avg Value", showlegend=False)
        apply_fig_theme(fig_s, height=320, slide_mode=slide_mode)
        st.plotly_chart(fig_s, use_container_width=True, theme="streamlit")

    if cat2 == "None":
        show_category(cat1)
    else:
        a, b = st.columns(2)
        with a: show_category(cat1)
        with b: show_category(cat2)

# ============================================================
# MARKET HEATMAP
# ============================================================
elif page == "Market HeatMap":
    st.markdown(f"<div class='pa-card fade-in'><h3>Market HeatMap</h3><div class='muted'>MACD bucket snapshot by category</div></div>", unsafe_allow_html=True)
    st.markdown("")

    rows = []
    for c in CATEGORIES:
        d = preprocess(df_raw, c)
        bucket = macd(d)[2].iloc[-1]
        rows.append({"Category": c, "MACD Bucket": str(bucket)})

    heat = pd.DataFrame(rows).sort_values("Category")
    st.table(heat)

# ============================================================
# STATE OF MARKET
# ============================================================
elif page == "State of Market":
    st.markdown(f"<div class='pa-card fade-in'><h3>State of Market</h3><div class='muted'>YoY vs 3-Mo momentum by category</div></div>", unsafe_allow_html=True)
    st.markdown("")

    latest = df_raw["Month_Year"].max()
    yoy_vals, mo3_vals = [], []
    for c in CATEGORIES:
        s = df_raw[df_raw["Category"] == c].groupby("Month_Year")["market_value"].mean()
        y, r = yoy_3mo(s, latest)
        yoy_vals.append(y)
        mo3_vals.append(r)

    mkt = pd.DataFrame({"Category": CATEGORIES, "YoY %": yoy_vals, "3-Mo %": mo3_vals}).round(2)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=mkt["Category"], y=mkt["YoY %"], name="YoY %", marker=dict(color=THEME["primary"])))
    fig.add_trace(go.Bar(x=mkt["Category"], y=mkt["3-Mo %"], name="3-Mo %", marker=dict(color=THEME["secondary"])))
    fig.add_hline(y=0, line_dash="dash", opacity=0.6)
    fig.update_layout(barmode="group", title=f"YoY vs 3-Mo Momentum (through {latest:%b %Y})", xaxis_title="Category", yaxis_title="% change")
    apply_fig_theme(fig, height=420, slide_mode=slide_mode)
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")

    st.table(mkt.sort_values("YoY %", ascending=False))

# ============================================================
# CUSTOM INDEX BUILDER
# ============================================================
elif page == "Custom Index Builder":
    st.markdown(f"<div class='pa-card fade-in'><h3>Custom Index Builder</h3><div class='muted'>Blend categories into a custom market index</div></div>", unsafe_allow_html=True)
    st.markdown("")

    sel = st.multiselect("Categories", CATEGORIES, default=["Pokemon", "Magic the Gathering"])
    if not sel:
        st.warning("Pick at least one category.")
        st.stop()

    raw_w = {c: st.slider(f"{c} weight (%)", 0, 100, 20, 5) for c in sel}
    if sum(raw_w.values()) == 0:
        st.warning("Weights above 0 required.")
        st.stop()

    weights = pd.Series(raw_w, dtype=float) / sum(raw_w.values())
    pivot = (
        df_raw.pivot_table(values="market_value", index="Month_Year", columns="Category", aggfunc="mean")
        .reindex(columns=CATEGORIES)
        .sort_index()
    )
    custom = (pivot[sel] * weights).sum(axis=1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=custom.index, y=custom.values, mode="lines", name="My Index", line=dict(width=3, color=THEME["primary"])))
    fig.update_layout(title="Custom Index", xaxis_title="Month", yaxis_title="Value")
    apply_fig_theme(fig, height=420, slide_mode=slide_mode)
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")

    st.markdown("#### Weights")
    st.table(weights.mul(100).round(1).rename("Weight %"))

# ============================================================
# SEASONALITY HEATMAP
# ============================================================
elif page == "Seasonality HeatMap":
    st.markdown(f"<div class='pa-card fade-in'><h3>Seasonality HeatMap</h3><div class='muted'>Average MoM % change by month</div></div>", unsafe_allow_html=True)
    st.markdown("")

    df_tmp = df_raw.copy()
    df_tmp["Month_Num"] = df_tmp["Month_Year"].dt.month

    wide = (
        df_tmp.pivot_table(values="market_value", index="Category", columns="Month_Num", aggfunc="mean")
        .reindex(index=CATEGORIES)
    )
    pct = (wide.pct_change(axis=1) * 100).round(2)
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    fig = go.Figure(data=go.Heatmap(
        z=pct.values,
        x=month_labels,
        y=pct.index.tolist(),
        colorscale="RdYlGn",
        zmin=-20, zmax=20
    ))
    fig.update_layout(title="Seasonality — Avg MoM % Change", xaxis_title="Month", yaxis_title="Category")
    apply_fig_theme(fig, height=420, slide_mode=slide_mode)
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")

    st.table(pct.fillna("—"))

# ============================================================
# ROLLING VOLATILITY
# ============================================================
elif page == "Rolling Volatility":
    st.markdown(f"<div class='pa-card fade-in'><h3>Rolling Volatility</h3><div class='muted'>Coefficient of variation over time</div></div>", unsafe_allow_html=True)
    st.markdown("")

    pick = st.selectbox("Category", CATEGORIES, index=CATEGORIES.index(cat1))
    d = preprocess(df_raw, pick).set_index("Month_Year").sort_index()

    w1 = st.slider("Short window (months)", 3, 18, 6, step=1)
    w2 = st.slider("Long window (months)", 6, 36, 12, step=1)

    cv1 = (d.market_value.rolling(w1).std() / d.market_value.rolling(w1).mean() * 100).rename(f"{w1}-Mo")
    cv2 = (d.market_value.rolling(w2).std() / d.market_value.rolling(w2).mean() * 100).rename(f"{w2}-Mo")
    cv_df = pd.concat([cv1, cv2], axis=1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cv_df.index, y=cv_df.iloc[:,0], mode="lines", name=cv_df.columns[0], line=dict(color=THEME["primary"])))
    fig.add_trace(go.Scatter(x=cv_df.index, y=cv_df.iloc[:,1], mode="lines", name=cv_df.columns[1], line=dict(color=THEME["secondary"])))
    fig.update_layout(title=f"{pick} — Rolling Volatility", xaxis_title="Month", yaxis_title="CoV (%)")
    apply_fig_theme(fig, height=420, slide_mode=slide_mode)
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")

    st.table(cv_df.round(2).tail(24))

# ============================================================
# CORRELATION MATRIX
# ============================================================
elif page == "Correlation Matrix":
    st.markdown(f"<div class='pa-card fade-in'><h3>Correlation Matrix</h3><div class='muted'>Category co-movement (returns or levels)</div></div>", unsafe_allow_html=True)
    st.markdown("")

    wide = (
        df_raw.pivot_table(values="market_value", index="Month_Year", columns="Category", aggfunc="mean")
        .sort_index()[CATEGORIES]
    )

    basis = st.radio("Correlation basis", ["Monthly returns (pct_change)", "Levels (raw index)"], index=0, horizontal=True)
    mat = wide.pct_change().dropna() if basis.startswith("Monthly") else wide.dropna()
    corr = mat.corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        zmin=-1, zmax=1,
        colorscale="RdYlGn"
    ))
    fig.update_layout(title="Correlation Heatmap", xaxis_title="Category", yaxis_title="Category")
    apply_fig_theme(fig, height=520, slide_mode=slide_mode)
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")

    st.table(corr.round(2))

# ============================================================
# FLIP FORECAST (Monte Carlo)
# ============================================================
elif page == "Flip Forecast":
    st.markdown(f"<div class='pa-card fade-in'><h3>Flip Forecast</h3><div class='muted'>Monte Carlo projection based on category history</div></div>", unsafe_allow_html=True)
    st.markdown("")

    sim_category = st.selectbox("Choose Category for Forecast", CATEGORIES, index=CATEGORIES.index(cat1))
    d = preprocess(df_raw, sim_category).sort_values("Month_Year")
    d["pct_change"] = d["market_value"].pct_change()
    d = d.dropna()

    expected_return = d["pct_change"].mean()
    monthly_volatility = min(max(d["pct_change"].std(), 0.01), 0.30)

    asking_price = st.number_input("Your Asking Price ($)", min_value=0.0, value=100.0, step=1.0)
    purchase_price = st.number_input("Your Purchase Price ($)", min_value=0.0, value=75.0, step=1.0)

    num_months = st.slider("Horizon (months)", 6, 36, 12, step=1)
    num_simulations = st.slider("Number of Simulations", 200, 20000, 2000, step=200)

    rng = np.random.default_rng()
    results = np.empty((num_simulations, num_months + 1), dtype=float)
    results[:, 0] = float(d["market_value"].tail(3).mean())

    for i in range(num_simulations):
        for m in range(num_months):
            rand_return = rng.normal(expected_return, monthly_volatility)
            results[i, m+1] = results[i, m] * (1 + rand_return)

    months_ahead = np.arange(num_months + 1)
    p10 = np.percentile(results, 10, axis=0)
    p50 = np.percentile(results, 50, axis=0)
    p90 = np.percentile(results, 90, axis=0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=months_ahead, y=p50, mode="lines", name="Median", line=dict(color=THEME["primary"], dash="dash", width=3)))
    fig.add_trace(go.Scatter(
        x=np.concatenate([months_ahead, months_ahead[::-1]]),
        y=np.concatenate([p90, p10[::-1]]),
        fill="toself",
        fillcolor="rgba(109, 94, 247, 0.16)",
        line=dict(color="rgba(0,0,0,0)"),
        name="10–90% band",
        hoverinfo="skip"
    ))
    fig.add_hline(y=asking_price, line_dash="dot", opacity=0.7)
    fig.update_layout(title=f"Flip Forecast — {sim_category}", xaxis_title="Months Ahead", yaxis_title="Simulated Price ($)")
    apply_fig_theme(fig, height=420, slide_mode=slide_mode)
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")

    final_prices = results[:, -1]
    prob_hit = float(np.mean(final_prices >= asking_price) * 100)
    st.table(pd.DataFrame({
        "Metric": ["Expected Return", "Monthly Volatility (capped)", "Probability Asking Price Hit"],
        "Value": [f"{expected_return:.2%}", f"{monthly_volatility:.2%}", f"{prob_hit:.1f}%"]
    }))

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
    ---
    **Cardboard Compass** — built by Pancake Analytics LLC  
    *Analytics read, not financial advice.*
    """
)

if slide_mode:
    st.markdown("</div>", unsafe_allow_html=True)
