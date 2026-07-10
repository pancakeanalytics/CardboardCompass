import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Cardboard Compass", layout="wide")

THEME = {
    "primary": "#6D5EF7",
    "secondary": "#8B7CFF",
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

st.markdown(
    f"""
    <style>
      body {{ background-color: {THEME['bg']}; }}
      .block-container {{ padding-top: 1.4rem; padding-bottom: 2rem; }}
      h1,h2,h3,h4 {{ color: {THEME['text']}; }}
      .muted {{ color: {THEME['muted']}; font-size: 0.90rem; }}
      .pa-card {{
        background: {THEME['card']};
        border: 1px solid {THEME['border']};
        border-radius: 16px;
        padding: 16px 18px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
      }}
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
      .pa-sub {{ margin-top: 8px; font-size: 13px; opacity: 0.92; }}
      .pa-asof {{ font-size: 13px; opacity: 0.95; margin: 0; }}
      .pa-asof b {{ display: block; margin-top: 6px; font-size: 22px; letter-spacing: 0.3px; }}
      .fade-in {{ animation: fadeIn 0.45s ease-in-out; }}
      @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(6px); }} to {{ opacity: 1; transform: translateY(0); }} }}
      iframe {{ border-radius: 12px; }}
      div[data-testid="stDataFrame"] > div {{ overflow: auto; }}
      .slide-wrap {{ max-width: 1180px; margin: 0 auto; }}
      .allocator-note {{ color: {THEME['muted']}; font-size: 0.92rem; line-height: 1.45; }}
      @media print {{
        header, footer, [data-testid="stSidebar"], [data-testid="stToolbar"] {{ display: none !important; }}
        .block-container {{ padding: 0 !important; }}
        .pa-pagebreak {{ page-break-after: always; break-after: page; }}
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_URL = "https://pancakebreakfaststats.com/wp-content/uploads/2026/07/data_file_019.xlsx"

@st.cache_data(show_spinner=False)
def load_data(url: str) -> pd.DataFrame:
    df = pd.read_excel(url).copy()
    df.columns = [str(c).strip() for c in df.columns]
    df["Category"] = df["Category"].astype(str).str.strip()
    df["Month"] = df["Month"].astype(str).str.strip()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["market_value"] = pd.to_numeric(df["market_value"], errors="coerce")
    df["Month_Year"] = pd.to_datetime(df["Month"] + " " + df["Year"].astype(str), format="%B %Y", errors="coerce")
    df = df.dropna(subset=["Category", "Month_Year", "market_value"]).copy()
    return df

df_raw = load_data(DATA_URL)

CATEGORIES = [
    "Fortnite", "Marvel", "Pokemon", "Star Wars", "Magic the Gathering",
    "Baseball", "Basketball", "Football", "Hockey", "Soccer"
]

DEFAULT_BUCKETS = [
    {"bucket": "Vintage / pre-1980", "risk": 2.0, "base_return": 7.0, "liquidity": 7.0, "min_pct": 10, "max_pct": 60},
    {"bucket": "GOATs / blue-chip stars", "risk": 3.0, "base_return": 9.0, "liquidity": 8.0, "min_pct": 10, "max_pct": 55},
    {"bucket": "Established modern stars", "risk": 5.0, "base_return": 12.0, "liquidity": 7.0, "min_pct": 5, "max_pct": 40},
    {"bucket": "Prospects / breakout bets", "risk": 9.0, "base_return": 22.0, "liquidity": 4.0, "min_pct": 0, "max_pct": 30},
    {"bucket": "Sealed wax", "risk": 6.0, "base_return": 11.0, "liquidity": 5.0, "min_pct": 0, "max_pct": 30},
    {"bucket": "Cash / opportunistic reserve", "risk": 1.0, "base_return": 3.0, "liquidity": 10.0, "min_pct": 5, "max_pct": 35},
]

SPORT_TILTS = {
    'Balanced multi-sport': {'Vintage / pre-1980': 0, 'GOATs / blue-chip stars': 0, 'Established modern stars': 0, 'Prospects / breakout bets': 0, 'Sealed wax': 0, 'Cash / opportunistic reserve': 0},
    'Baseball': {'Vintage / pre-1980': 1.2, 'GOATs / blue-chip stars': 0.8, 'Established modern stars': 0.4, 'Prospects / breakout bets': -0.2, 'Sealed wax': 0.2, 'Cash / opportunistic reserve': 0},
    'Basketball': {'Vintage / pre-1980': -0.2, 'GOATs / blue-chip stars': 0.5, 'Established modern stars': 0.8, 'Prospects / breakout bets': 0.8, 'Sealed wax': 0.1, 'Cash / opportunistic reserve': 0},
    'Football': {'Vintage / pre-1980': -0.3, 'GOATs / blue-chip stars': 0.2, 'Established modern stars': 0.7, 'Prospects / breakout bets': 1.1, 'Sealed wax': 0.4, 'Cash / opportunistic reserve': 0},
    'Soccer': {'Vintage / pre-1980': -0.3, 'GOATs / blue-chip stars': 0.6, 'Established modern stars': 0.3, 'Prospects / breakout bets': 1.0, 'Sealed wax': 0.3, 'Cash / opportunistic reserve': 0},
    'Pokemon / TCG': {'Vintage / pre-1980': -0.6, 'GOATs / blue-chip stars': 0.4, 'Established modern stars': 0.2, 'Prospects / breakout bets': 0.5, 'Sealed wax': 1.4, 'Cash / opportunistic reserve': 0},
}

BUCKET_CATEGORY_MAP = {
    "Vintage / pre-1980": ["Baseball", "Hockey", "Basketball"],
    "GOATs / blue-chip stars": ["Baseball", "Basketball", "Football", "Soccer", "Pokemon"],
    "Established modern stars": ["Basketball", "Football", "Soccer", "Pokemon", "Magic the Gathering"],
    "Prospects / breakout bets": ["Baseball", "Basketball", "Football", "Soccer"],
    "Sealed wax": ["Pokemon", "Magic the Gathering", "Marvel", "Star Wars", "Fortnite"],
    "Cash / opportunistic reserve": []
}

def preprocess(df: pd.DataFrame, cat: str) -> pd.DataFrame:
    d = df[df["Category"] == cat].copy()
    return d.groupby("Month_Year", as_index=False)["market_value"].mean().sort_values("Month_Year")

def pct_change_between(series: pd.Series, start_date: pd.Timestamp, end_date: pd.Timestamp) -> float:
    if start_date not in series.index or end_date not in series.index:
        return np.nan
    v0 = series.loc[start_date]
    v1 = series.loc[end_date]
    if pd.isna(v0) or pd.isna(v1) or v0 == 0:
        return np.nan
    return (v1 - v0) / v0 * 100

def yoy_3mo(series: pd.Series, latest: pd.Timestamp):
    now = series.get(latest, np.nan)
    yr = series.get(latest - pd.DateOffset(years=1), np.nan)
    m3 = series.get(latest - pd.DateOffset(months=3), np.nan)
    yoy = np.nan if pd.isna(now) or pd.isna(yr) or yr == 0 else (now - yr) / yr * 100
    r3 = np.nan if pd.isna(now) or pd.isna(m3) or m3 == 0 else (now - m3) / m3 * 100
    return yoy, r3

def fmt_pct(x: float, decimals: int = 1) -> str:
    return "—" if pd.isna(x) else f"{x:.{decimals}f}%"

def forecast(df: pd.DataFrame, horizon=12, seasonal_periods=12, trend="add", seasonal="add", ci_level=0.95):
    y = df["market_value"].astype(float)
    model = ExponentialSmoothing(y, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods).fit()
    fc = model.forecast(horizon)
    z_table = {0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600, 0.98: 2.3263, 0.99: 2.5758}
    z = z_table.get(round(ci_level, 2), 1.96)
    ci = z * np.std(model.resid)
    future = pd.date_range(df["Month_Year"].iloc[-1] + pd.DateOffset(months=1), periods=horizon, freq="MS")
    fc_df = pd.DataFrame({"Date": future, "Forecast": fc.values, "Upper": fc.values + ci, "Lower": fc.values - ci})
    hist_df = pd.DataFrame({"Date": df["Month_Year"].values, "Historical": df["market_value"].values})
    return hist_df, fc_df

def macd(df: pd.DataFrame):
    s = df["market_value"].ewm(span=12, adjust=False).mean()
    l = df["market_value"].ewm(span=26, adjust=False).mean()
    m = s - l
    sig = m.ewm(span=9, adjust=False).mean()
    bucket = pd.cut(m - sig, [-np.inf, -1.5, -0.5, 0, 0.5, 1.5, np.inf], labels=["High Down", "Med Down", "Low Down", "Low Up", "Med Up", "High Up"])
    return m, sig, bucket

def apply_fig_theme(fig: go.Figure, height: int, slide_mode: bool):
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
    st.markdown(f"""<div class="pa-card fade-in"><div class="muted">{label}</div><div style="font-size:32px; font-weight:900; margin-top:4px;">{value}</div>{sub_html}</div>""", unsafe_allow_html=True)

def section_card(title: str, body_html: str):
    st.markdown(f"""<div class="pa-card fade-in"><div style="font-weight:900; font-size:16px; margin-bottom:8px;">{title}</div>{body_html}</div>""", unsafe_allow_html=True)

def build_market_summary(df: pd.DataFrame, cats: list[str]):
    wide = df[df["Category"].isin(cats)].pivot_table(values="market_value", index="Month_Year", columns="Category", aggfunc="mean").reindex(columns=cats).sort_index().apply(pd.to_numeric, errors="coerce")
    last_row = wide.index.max()
    y_ago = last_row - pd.DateOffset(years=1)
    m_3 = last_row - pd.DateOffset(months=3)
    rows = []
    for c in cats:
        s = wide[c]
        rows.append({"Category": c, "YoY %": pct_change_between(s, y_ago, last_row), "3-Mo %": pct_change_between(s, m_3, last_row)})
    summary = pd.DataFrame(rows).set_index("Category").sort_index()
    comp_yoy = summary["YoY %"].mean(skipna=True)
    comp_3mo = summary["3-Mo %"].mean(skipna=True)
    breadth = float(summary["3-Mo %"].gt(0).mean() * 100)
    return summary, last_row, comp_yoy, comp_3mo, breadth

def download_print_ready_html(html: str, filename: str):
    st.download_button("⬇️ Download print-ready HTML (save as PDF from browser)", data=html.encode("utf-8"), file_name=filename, mime="text/html")

def compute_category_signal_table(df_raw: pd.DataFrame) -> pd.DataFrame:
    latest = df_raw["Month_Year"].max()
    pivot = df_raw.pivot_table(values="market_value", index="Month_Year", columns="Category", aggfunc="mean").sort_index().reindex(columns=CATEGORIES)
    corr = pivot.pct_change().dropna().corr()
    rows = []
    for c in CATEGORIES:
        d = preprocess(df_raw, c).set_index("Month_Year").sort_index()
        s = d["market_value"]
        yoy, mo3 = yoy_3mo(s, latest)
        rolling_cv = s.rolling(6).std() / s.rolling(6).mean() * 100
        volatility = float(rolling_cv.dropna().iloc[-1]) if not rolling_cv.dropna().empty else np.nan
        avg_corr = float(corr[c].drop(labels=[c]).mean()) if c in corr.columns else np.nan
        rows.append({
            "Category": c,
            "YoY %": yoy,
            "3-Mo %": mo3,
            "6-Mo CoV %": volatility,
            "Avg Corr": avg_corr,
        })
    out = pd.DataFrame(rows)
    out["Momentum Score"] = out[["YoY %", "3-Mo %"]].mean(axis=1)
    return out

def compute_bucket_signal_table(signal_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bucket, cats in BUCKET_CATEGORY_MAP.items():
        if len(cats) == 0:
            rows.append({"bucket": bucket, "Mapped Categories": "—", "Momentum Score": 0.0, "6-Mo CoV %": 0.0, "Avg Corr": 0.0})
        else:
            sub = signal_df[signal_df["Category"].isin(cats)]
            rows.append({
                "bucket": bucket,
                "Mapped Categories": ", ".join(cats),
                "Momentum Score": float(sub["Momentum Score"].mean(skipna=True)),
                "6-Mo CoV %": float(sub["6-Mo CoV %"].mean(skipna=True)),
                "Avg Corr": float(sub["Avg Corr"].mean(skipna=True)),
            })
    return pd.DataFrame(rows)

def normalize_series(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    if s.nunique(dropna=True) <= 1:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) != 0 else 1)

def build_signal_adjusted_buckets(base_df: pd.DataFrame, bucket_signal_df: pd.DataFrame, use_signals: bool) -> pd.DataFrame:
    out = base_df.copy()
    merged = out.merge(bucket_signal_df, on="bucket", how="left")
    merged["Momentum z"] = normalize_series(merged["Momentum Score"]).fillna(0)
    merged["Volatility z"] = normalize_series(merged["6-Mo CoV %"]).fillna(0)
    merged["Corr z"] = normalize_series(merged["Avg Corr"]).fillna(0)
    if use_signals:
        merged["adj_return"] = (merged["base_return"] + merged["Momentum z"] * 2.0).clip(0, 40)
        merged["adj_risk"] = (merged["risk"] + merged["Volatility z"] * 1.3 + merged["Corr z"] * 0.5).clip(1, 10)
        merged["adj_liquidity"] = (merged["liquidity"] - merged["Volatility z"] * 0.7 - merged["Corr z"] * 0.3).clip(1, 10)
    else:
        merged["adj_return"] = merged["base_return"]
        merged["adj_risk"] = merged["risk"]
        merged["adj_liquidity"] = merged["liquidity"]
    return merged

def allocate_portfolio(df, bankroll, risk_tolerance, horizon, liquidity_need, sport):
    out = df.copy()
    horizon_bonus = np.interp(horizon, [1, 5], [-2.0, 2.0])
    risk_pref = np.interp(risk_tolerance, [1, 10], [9.5, 1.5])
    liq_pref = np.interp(liquidity_need, [1, 10], [1.5, 9.5])
    scores = []
    for _, row in out.iterrows():
        tilt = SPORT_TILTS[sport].get(row['bucket'], 0)
        score = (
            (12 - abs(row['adj_risk'] - risk_pref)) * 2.2
            + row['adj_return'] * (1.0 + horizon_bonus / 10)
            + row['adj_liquidity'] * (liq_pref / 4)
            + tilt * 3
        )
        if row['bucket'] == 'Cash / opportunistic reserve' and risk_tolerance >= 8:
            score -= 8
        if row['bucket'] == 'Prospects / breakout bets' and horizon <= 2:
            score -= 5
        scores.append(max(score, 0.1))
    out['score'] = scores
    raw_pct = out['score'] / out['score'].sum() * 100
    out['target_pct'] = raw_pct
    mins = out['min_pct'].to_numpy(dtype=float)
    maxs = out['max_pct'].to_numpy(dtype=float)
    pct = np.clip(out['target_pct'].to_numpy(dtype=float), mins, maxs)
    for _ in range(1000):
        total = pct.sum()
        if abs(total - 100) < 1e-6:
            break
        if total < 100:
            room = np.maximum(maxs - pct, 0)
            if room.sum() == 0:
                break
            pct += room / room.sum() * (100 - total)
        else:
            excess = np.maximum(pct - mins, 0)
            if excess.sum() == 0:
                break
            pct -= excess / excess.sum() * (total - 100)
    out['target_pct'] = pct
    out['allocation_usd'] = np.round(bankroll * out['target_pct'] / 100, 2)
    out['expected_return_pct'] = out['adj_return']
    out['weighted_return'] = out['target_pct'] * out['expected_return_pct'] / 100
    out['weighted_risk'] = out['target_pct'] * out['adj_risk'] / 100
    out['weighted_liquidity'] = out['target_pct'] * out['adj_liquidity'] / 100
    return out.sort_values('target_pct', ascending=False).reset_index(drop=True)


def render_raw_vs_grade_engine():
    st.markdown(f"<div class='pa-card fade-in'><h3>Raw vs Grade Decision Engine</h3><div class='muted'>Estimate whether buying raw, grading, or buying slabbed looks best under your assumptions</div></div>", unsafe_allow_html=True)
    st.markdown("")

    a1, a2, a3, a4 = st.columns(4)
    with a1:
        raw_price = st.number_input("Raw purchase price ($)", min_value=0.0, value=120.0, step=1.0, key='rvg_raw_price')
    with a2:
        grading_fee = st.number_input("Total grading cost ($)", min_value=0.0, value=30.0, step=1.0, key='rvg_grading_fee')
    with a3:
        shipping_misc = st.number_input("Shipping / misc ($)", min_value=0.0, value=8.0, step=1.0, key='rvg_shipping')
    with a4:
        sell_fee_pct = st.slider("Selling fees (%)", 0.0, 20.0, 13.0, 0.5, key='rvg_sell_fee')

    b1, b2, b3 = st.columns(3)
    with b1:
        target_grade = st.selectbox("Primary grade lens", ['PSA 8','PSA 9','PSA 10','BGS 9.5','SGC 10'], index=2, key='rvg_target_grade')
    with b2:
        hold_months = st.slider("Planned hold (months)", 0, 24, 3, 1, key='rvg_hold_months')
    with b3:
        annual_carry_pct = st.slider("Annual capital cost (%)", 0.0, 20.0, 8.0, 0.5, key='rvg_carry_pct')

    st.markdown("#### Grade Outcome Assumptions")
    grade_df = pd.DataFrame([
        {"grade": "PSA 8", "probability": 0.15, "market_price": 170.0},
        {"grade": "PSA 9", "probability": 0.45, "market_price": 300.0},
        {"grade": "PSA 10", "probability": 0.25, "market_price": 680.0},
        {"grade": "BGS 9.5", "probability": 0.10, "market_price": 520.0},
        {"grade": "SGC 10", "probability": 0.05, "market_price": 430.0},
    ])
    grade_edit = st.data_editor(
        grade_df,
        use_container_width=True,
        hide_index=True,
        num_rows='fixed',
        column_config={
            'grade': st.column_config.TextColumn('Grade', disabled=True),
            'probability': st.column_config.NumberColumn('Probability', min_value=0.0, max_value=1.0, step=0.01, format='%.2f'),
            'market_price': st.column_config.NumberColumn('Expected sale price ($)', min_value=0.0, step=1.0, format='%.2f'),
        },
        key='rvg_grade_editor'
    )

    prob_sum = float(grade_edit['probability'].sum())
    if prob_sum <= 0:
        st.error('Total grade probability must be above 0.')
        return
    grade_edit['probability_norm'] = grade_edit['probability'] / prob_sum

    st.markdown("#### Slab Purchase Benchmark")
    c1, c2, c3 = st.columns(3)
    with c1:
        slab_buy_price = st.number_input("Comparable slab buy price ($)", min_value=0.0, value=620.0, step=1.0, key='rvg_slab_buy')
    with c2:
        slab_expected_sale = st.number_input("Expected slab resale ($)", min_value=0.0, value=690.0, step=1.0, key='rvg_slab_sale')
    with c3:
        slab_grade_options = grade_edit['grade'].tolist()
        slab_grade = st.selectbox("Comparable slab grade", slab_grade_options, index=slab_grade_options.index(target_grade) if target_grade in slab_grade_options else 0, key='rvg_slab_grade')

    total_raw_basis = raw_price + grading_fee + shipping_misc
    hold_cost = total_raw_basis * (annual_carry_pct / 100) * (hold_months / 12)
    exp_gross_sale = float((grade_edit['probability_norm'] * grade_edit['market_price']).sum())
    exp_net_sale = exp_gross_sale * (1 - sell_fee_pct / 100)
    exp_profit_grade = exp_net_sale - total_raw_basis - hold_cost
    exp_roi_grade = 0 if total_raw_basis == 0 else exp_profit_grade / total_raw_basis * 100

    if target_grade in grade_edit['grade'].values:
        raw_expected_sale = float(grade_edit.loc[grade_edit['grade'] == target_grade, 'market_price'].iloc[0] * 0.45)
    else:
        raw_expected_sale = raw_price * 1.1
    raw_net_sale = raw_expected_sale * (1 - sell_fee_pct / 100)
    raw_basis = raw_price + shipping_misc
    raw_hold_cost = raw_basis * (annual_carry_pct / 100) * (hold_months / 12)
    raw_profit = raw_net_sale - raw_basis - raw_hold_cost
    raw_roi = 0 if raw_basis == 0 else raw_profit / raw_basis * 100

    slab_hold_cost = slab_buy_price * (annual_carry_pct / 100) * (hold_months / 12)
    slab_net_sale = slab_expected_sale * (1 - sell_fee_pct / 100)
    slab_profit = slab_net_sale - slab_buy_price - slab_hold_cost
    slab_roi = 0 if slab_buy_price == 0 else slab_profit / slab_buy_price * 100

    outcomes = pd.DataFrame([
        {"Path": "Buy raw and sell raw", "Basis $": raw_basis, "Expected Net Sale $": raw_net_sale, "Expected Profit $": raw_profit, "ROI %": raw_roi},
        {"Path": "Buy raw and grade", "Basis $": total_raw_basis, "Expected Net Sale $": exp_net_sale, "Expected Profit $": exp_profit_grade, "ROI %": exp_roi_grade},
        {"Path": f"Buy existing {slab_grade}", "Basis $": slab_buy_price, "Expected Net Sale $": slab_net_sale, "Expected Profit $": slab_profit, "ROI %": slab_roi},
    ]).sort_values('ROI %', ascending=False).reset_index(drop=True)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card('Best path', outcomes.iloc[0]['Path'])
    with k2:
        kpi_card('Top expected ROI', f"{outcomes.iloc[0]['ROI %']:.1f}%")
    with k3:
        kpi_card('Expected grading ROI', f"{exp_roi_grade:.1f}%")
    with k4:
        kpi_card('Probabilities total', f"{prob_sum:.2f}", 'Normalized automatically in model')

    st.markdown('#### Decision Table')
    st.dataframe(outcomes.round(2), use_container_width=True, hide_index=True)

    lcol, rcol = st.columns([1.1, 1])
    with lcol:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=outcomes['ROI %'], y=outcomes['Path'], orientation='h', marker=dict(color=[THEME['primary'], THEME['secondary'], THEME['accent_red']]), showlegend=False))
        fig.update_layout(title='Expected ROI by Path', xaxis_title='ROI %', yaxis_title='')
        fig.update_yaxes(autorange='reversed')
        apply_fig_theme(fig, height=380, slide_mode=False)
        st.plotly_chart(fig, use_container_width=True, theme='streamlit')
    with rcol:
        grade_view = grade_edit[['grade', 'probability_norm', 'market_price']].copy()
        grade_view.columns = ['Grade', 'Normalized Prob.', 'Sale Price $']
        st.markdown('#### Grade Distribution Used')
        st.dataframe(grade_view.round(3), use_container_width=True, hide_index=True)

    st.markdown('#### What the engine is doing')
    notes = [
        f"Raw-and-grade basis is ${total_raw_basis:,.2f}, which includes raw card cost, grading, and misc costs.",
        f"Expected graded sale is probability-weighted across outcomes, producing ${exp_gross_sale:,.2f} gross and ${exp_net_sale:,.2f} net after selling fees.",
        f"Hold cost adds a time penalty of ${hold_cost:,.2f} based on your annual capital cost and hold period.",
        f"The current top path is {outcomes.iloc[0]['Path']} based on expected ROI, not guaranteed realized outcome.",
    ]
    for note in notes:
        st.write(f"- {note}")

    csv = outcomes.to_csv(index=False).encode('utf-8')
    st.download_button('Download decision table CSV', data=csv, file_name='raw_vs_grade_decision.csv', mime='text/csv', key='rvg_csv_download')


def render_liquidity_exit_monitor():
    st.markdown(f"<div class='pa-card fade-in'><h3>Liquidity + Exit Risk Monitor</h3><div class='muted'>Measure how easily you may be able to exit a position based on turnover, volatility, spread, and downside pressure</div></div>", unsafe_allow_html=True)
    st.markdown("")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        monitor_category = st.selectbox('Category', CATEGORIES, index=CATEGORIES.index(cat1) if cat1 in CATEGORIES else 0, key='liq_category')
    with c2:
        lookback_months = st.slider('Lookback window (months)', 6, 36, 12, 1, key='liq_lookback')
    with c3:
        target_sale_price = st.number_input('Target exit price ($)', min_value=1.0, value=300.0, step=5.0, key='liq_target_exit')
    with c4:
        risk_mode = st.selectbox('Exit profile', ['Conservative', 'Balanced', 'Aggressive'], index=1, key='liq_risk_mode')

    d = preprocess(df_raw, monitor_category).sort_values('Month_Year').copy()
    d['pct_change'] = d['market_value'].pct_change()
    if len(d) < 6:
        st.warning('Not enough history to compute a reliable liquidity monitor for this category.')
        return

    recent = d.tail(lookback_months).copy()
    recent_changes = recent['pct_change'].dropna()
    if recent_changes.empty:
        recent_changes = d['pct_change'].dropna()

    latest_price = float(recent['market_value'].iloc[-1])
    sale_frequency = len(recent)
    avg_monthly_turnover_proxy = sale_frequency / max(lookback_months, 1)
    volatility_pct = float(recent_changes.std() * 100) if len(recent_changes) > 1 else 0.0
    mean_return_pct = float(recent_changes.mean() * 100) if len(recent_changes) > 0 else 0.0
    drawdown_pct = float(((recent['market_value'] / recent['market_value'].cummax()) - 1).min() * 100)
    spread_pct = float((recent['market_value'].max() - recent['market_value'].min()) / recent['market_value'].mean() * 100) if recent['market_value'].mean() else 0.0
    target_gap_pct = float((target_sale_price - latest_price) / latest_price * 100) if latest_price else 0.0

    downside_prob = float((recent_changes < 0).mean() * 100) if len(recent_changes) else 0.0
    hit_target_prob = float((recent['market_value'] >= target_sale_price).mean() * 100)
    months_above_target = int((recent['market_value'] >= target_sale_price).sum())

    mode_adj = {'Conservative': 1.25, 'Balanced': 1.0, 'Aggressive': 0.8}[risk_mode]
    liquidity_score = 10 - min(10, (volatility_pct / 6.5) * mode_adj + (spread_pct / 18) * mode_adj + max(target_gap_pct, 0) / 12 + downside_prob / 20)
    liquidity_score = float(np.clip(liquidity_score, 1, 10))
    exit_risk_score = 10 - liquidity_score

    status = 'Healthy' if liquidity_score >= 7.5 else 'Watchlist' if liquidity_score >= 5.0 else 'High Exit Risk'

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card('Liquidity score', f"{liquidity_score:.1f} / 10", status)
    with k2:
        kpi_card('Exit risk score', f"{exit_risk_score:.1f} / 10")
    with k3:
        kpi_card('Downside months', f"{downside_prob:.0f}%")
    with k4:
        kpi_card('Target hit frequency', f"{hit_target_prob:.0f}%")

    summary_df = pd.DataFrame([
        {'Metric': 'Latest price', 'Value': latest_price},
        {'Metric': 'Target exit price', 'Value': target_sale_price},
        {'Metric': 'Target gap %', 'Value': target_gap_pct},
        {'Metric': 'Avg monthly turnover proxy', 'Value': avg_monthly_turnover_proxy},
        {'Metric': 'Volatility %', 'Value': volatility_pct},
        {'Metric': 'Mean monthly return %', 'Value': mean_return_pct},
        {'Metric': 'Max drawdown %', 'Value': drawdown_pct},
        {'Metric': 'Price spread %', 'Value': spread_pct},
        {'Metric': 'Months at/above target', 'Value': months_above_target},
    ])

    left, right = st.columns([1.2, 1])
    with left:
        st.markdown('#### Exit Risk Table')
        st.dataframe(summary_df.round(2), use_container_width=True, hide_index=True)
    with right:
        fig_g = go.Figure()
        fig_g.add_trace(go.Bar(x=['Liquidity', 'Exit Risk'], y=[liquidity_score, exit_risk_score], marker=dict(color=[THEME['primary'], THEME['accent_red']]), showlegend=False))
        fig_g.update_layout(title='Liquidity vs Exit Risk', yaxis_title='Score', xaxis_title='')
        fig_g.update_yaxes(range=[0, 10])
        apply_fig_theme(fig_g, height=360, slide_mode=False)
        st.plotly_chart(fig_g, use_container_width=True, theme='streamlit')

    t1, t2 = st.columns(2)
    with t1:
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(x=recent['Month_Year'], y=recent['market_value'], mode='lines+markers', name='Market value', line=dict(color=THEME['primary'], width=3)))
        fig_price.add_hline(y=target_sale_price, line_dash='dash', line_color=THEME['accent_red'])
        fig_price.update_layout(title=f'{monitor_category} price path vs target', xaxis_title='Month', yaxis_title='Value ($)')
        apply_fig_theme(fig_price, height=360, slide_mode=False)
        st.plotly_chart(fig_price, use_container_width=True, theme='streamlit')
    with t2:
        rolling_vol = recent['pct_change'].rolling(3).std() * 100
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(x=recent['Month_Year'], y=rolling_vol, mode='lines+markers', name='3-mo rolling vol', line=dict(color=THEME['secondary'], width=3)))
        fig_vol.update_layout(title='Short-term volatility trend', xaxis_title='Month', yaxis_title='Volatility %')
        apply_fig_theme(fig_vol, height=360, slide_mode=False)
        st.plotly_chart(fig_vol, use_container_width=True, theme='streamlit')

    st.markdown('#### Exit Interpretation')
    notes = []
    if liquidity_score >= 7.5:
        notes.append('This category currently screens as relatively liquid, with lower modeled exit friction.')
    elif liquidity_score >= 5.0:
        notes.append('This category is tradable, but exit conditions are mixed and need active monitoring.')
    else:
        notes.append('This category currently carries elevated exit risk, so sizing and entry discipline matter more.')
    if target_gap_pct > 15:
        notes.append('Your target is materially above the latest market level, which raises execution risk.')
    if downside_prob > 50:
        notes.append('More than half of the recent months were negative, which weakens exit confidence.')
    if abs(drawdown_pct) > 20:
        notes.append('Recent drawdown depth has been significant, which can make forced exits painful.')
    if spread_pct > 25:
        notes.append('The recent price range is wide relative to the average level, signaling unstable exit pricing.')
    for note in notes:
        st.write(f'- {note}')

    export_df = summary_df.copy()
    export_df['Category'] = monitor_category
    csv = export_df.to_csv(index=False).encode('utf-8')
    st.download_button('Download liquidity monitor CSV', data=csv, file_name='liquidity_exit_monitor.csv', mime='text/csv', key='liq_csv_download')


def render_episode_companion_dashboard():
    st.markdown(f"<div class='pa-card fade-in'><h3>Episode Companion Dashboard</h3><div class='muted'>Turn each podcast episode into a live data page with thesis, charts, watchlist, and action points</div></div>", unsafe_allow_html=True)
    st.markdown("")

    c1, c2, c3 = st.columns([1.1, 1.2, 0.9])
    with c1:
        episode_title = st.text_input('Episode title', value='Why Pokemon Liquidity Still Matters in 2026', key='ep_title')
    with c2:
        episode_hook = st.text_input('Core thesis / hook', value='Collectors are paying for liquidity and confidence, not just scarcity.', key='ep_hook')
    with c3:
        episode_date = st.date_input('Episode date', value=pd.Timestamp.today(), key='ep_date')

    d1, d2, d3 = st.columns(3)
    with d1:
        focus_categories = st.multiselect('Focus categories', CATEGORIES, default=['Pokemon', 'Basketball'], key='ep_cats')
    with d2:
        stance = st.selectbox('Episode stance', ['Bullish', 'Neutral', 'Bearish', 'Mixed'], index=3, key='ep_stance')
    with d3:
        call_to_action = st.text_input('Listener CTA', value='Audit your top five cards for liquidity, not just headline comp value.', key='ep_cta')

    if not focus_categories:
        st.warning('Pick at least one focus category for the episode dashboard.')
        return

    summary, last_row, comp_yoy, comp_3mo, breadth = build_market_summary(df_raw, focus_categories)
    signal_df = compute_category_signal_table(df_raw)
    episode_signals = signal_df[signal_df['Category'].isin(focus_categories)].copy()

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card('Episode stance', stance)
    with k2:
        kpi_card('Focus breadth', f"{len(focus_categories)} cats")
    with k3:
        kpi_card('Avg YoY', fmt_pct(summary['YoY %'].mean(skipna=True)))
    with k4:
        kpi_card('Avg 3-Mo', fmt_pct(summary['3-Mo %'].mean(skipna=True)))

    st.markdown('#### Episode Thesis Card')
    thesis_html = f"""
    <div class='pa-card fade-in'>
      <div class='muted'>Episode thesis</div>
      <div style='font-size:30px; font-weight:900; margin-top:6px; line-height:1.15;'>{episode_title}</div>
      <div style='margin-top:10px; font-size:16px; line-height:1.5;'>{episode_hook}</div>
      <div class='muted' style='margin-top:12px;'>Recorded for {pd.to_datetime(episode_date):%B %d, %Y} • Data through {last_row:%b %Y}</div>
    </div>
    """
    st.markdown(thesis_html, unsafe_allow_html=True)

    l1, l2 = st.columns([1.2, 1])
    with l1:
        fig = go.Figure()
        for cat in focus_categories:
            d = preprocess(df_raw, cat)
            fig.add_trace(go.Scatter(x=d['Month_Year'], y=d['market_value'], mode='lines', name=cat))
        fig.update_layout(title='Focus Category Trendlines', xaxis_title='Month', yaxis_title='Market Value')
        apply_fig_theme(fig, height=380, slide_mode=False)
        st.plotly_chart(fig, use_container_width=True, theme='streamlit')
    with l2:
        bar_df = summary.reset_index().sort_values('3-Mo %', ascending=False)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=bar_df['Category'], y=bar_df['3-Mo %'], marker=dict(color=THEME['primary']), showlegend=False))
        fig2.add_hline(y=0, line_dash='dash', opacity=0.6)
        fig2.update_layout(title='3-Month Momentum Snapshot', xaxis_title='Category', yaxis_title='3-Mo %')
        apply_fig_theme(fig2, height=380, slide_mode=False)
        st.plotly_chart(fig2, use_container_width=True, theme='streamlit')

    st.markdown('#### Episode Signal Table')
    signal_view = episode_signals[['Category', 'YoY %', '3-Mo %', '6-Mo CoV %', 'Avg Corr', 'Momentum Score']].copy().sort_values('Momentum Score', ascending=False)
    st.dataframe(signal_view.round(2), use_container_width=True, hide_index=True)

    st.markdown('#### Listener Watchlist')
    watchlist_default = pd.DataFrame([
        {'Item': 'High-liquidity blue chip', 'Why it matters': 'Easy to exit if episode thesis is wrong', 'Priority': 'High'},
        {'Item': 'Momentum leader', 'Why it matters': 'Confirms whether trend is broadening or fading', 'Priority': 'High'},
        {'Item': 'Lagging category', 'Why it matters': 'Potential rotation candidate if thesis is right', 'Priority': 'Medium'},
        {'Item': 'Sealed or alternative sleeve', 'Why it matters': 'Useful for comparing collector vs investor demand', 'Priority': 'Low'},
    ])
    watchlist = st.data_editor(
        watchlist_default,
        use_container_width=True,
        hide_index=True,
        num_rows='dynamic',
        column_config={
            'Item': st.column_config.TextColumn('Item'),
            'Why it matters': st.column_config.TextColumn('Why it matters'),
            'Priority': st.column_config.SelectboxColumn('Priority', options=['High', 'Medium', 'Low'])
        },
        key='ep_watchlist'
    )

    e1, e2 = st.columns([1.1, 1])
    with e1:
        st.markdown('#### Episode Run of Show')
        run_of_show = pd.DataFrame([
            {'Segment': 'Opening thesis', 'Talking point': episode_hook},
            {'Segment': 'What the data says', 'Talking point': f"Average YoY is {fmt_pct(summary['YoY %'].mean(skipna=True))} across selected categories."},
            {'Segment': 'What I am watching', 'Talking point': call_to_action},
            {'Segment': 'Risk check', 'Talking point': 'Watch liquidity, volatility, and whether the current leader can sustain bid depth.'},
        ])
        st.dataframe(run_of_show, use_container_width=True, hide_index=True)
    with e2:
        stance_colors = {'Bullish': THEME['primary'], 'Neutral': THEME['secondary'], 'Bearish': THEME['accent_red'], 'Mixed': '#F59E0B'}
        pie_values = [max(1, float(summary['3-Mo %'].gt(0).sum())), max(1, float(summary['3-Mo %'].le(0).sum()))]
        fig3 = go.Figure(go.Pie(labels=['Positive momentum', 'Flat / negative momentum'], values=pie_values, hole=0.62, marker=dict(colors=[stance_colors.get(stance, THEME['primary']), THEME['border']])))
        fig3.update_layout(title='Breadth Check')
        apply_fig_theme(fig3, height=360, slide_mode=False)
        st.plotly_chart(fig3, use_container_width=True, theme='streamlit')

    st.markdown('#### Publishing Notes')
    notes = [
        f"Episode title: {episode_title}",
        f"Core thesis: {episode_hook}",
        f"Current stance: {stance}",
        f"Listener action: {call_to_action}",
        f"Top momentum category in this episode set: {signal_view.iloc[0]['Category'] if not signal_view.empty else 'N/A'}",
    ]
    for note in notes:
        st.write(f'- {note}')

    export_df = signal_view.copy()
    export_df['Episode Title'] = episode_title
    csv = export_df.to_csv(index=False).encode('utf-8')
    st.download_button('Download episode dashboard CSV', data=csv, file_name='episode_companion_dashboard.csv', mime='text/csv', key='ep_csv_download')


def render_topic_generator_dashboard():
    st.markdown(f"<div class='pa-card fade-in'><h3>Weekly Topic Generator + Episode Dashboard</h3><div class='muted'>Pressure test a headlines-only workflow by scraping recent sports card and TCG articles, weighting repeated themes, and using Cardboard Compass data to build an episode dashboard</div></div>", unsafe_allow_html=True)
    st.markdown('')

    c1, c2, c3 = st.columns(3)
    with c1:
        universe = st.selectbox('Universe', ['Sports cards', 'TCG', 'Both'], index=2, key='wt_universe')
    with c2:
        lookback_days = st.selectbox('Lookback window', [7, 14, 30], index=1, key='wt_lookback')
    with c3:
        min_keyword_hits = st.selectbox('Min repeated keyword hits', [1, 2, 3], index=1, key='wt_min_hits')

    st.markdown('#### 1. Scrape recent article headlines')

    import requests, re
    from bs4 import BeautifulSoup
    from collections import Counter

    source_map = {
        'Sports cards': [
            ('Beckett News', 'https://www.beckett.com/news/'),
            ('Sports Collectors Daily', 'https://www.sportscollectorsdaily.com/'),
            ('Sports Card Portal', 'https://sportscardportal.com/news'),
            ('Collectibles on SI', 'https://www.si.com/collectibles/industry-news'),
        ],
        'TCG': [
            ('Double Holo', 'https://doubleholo.com/articles'),
            ('Rippr Blog', 'https://rippr.app/blog'),
            ('Monster Card Corner', 'https://monstercardcorner.co.uk/blogs/news'),
            ('Guardian TCG', 'https://guardiantcg.app/market/movers'),
        ],
        'Both': [
            ('Beckett News', 'https://www.beckett.com/news/'),
            ('Sports Collectors Daily', 'https://www.sportscollectorsdaily.com/'),
            ('Sports Card Portal', 'https://sportscardportal.com/news'),
            ('Collectibles on SI', 'https://www.si.com/collectibles/industry-news'),
            ('Double Holo', 'https://doubleholo.com/articles'),
            ('Rippr Blog', 'https://rippr.app/blog'),
            ('Monster Card Corner', 'https://monstercardcorner.co.uk/blogs/news'),
            ('Guardian TCG', 'https://guardiantcg.app/market/movers'),
        ],
    }

    stopwords = {
        'the','and','for','with','from','this','that','into','your','what','week','july','2026','card','cards','tcg','sports',
        'news','blog','latest','market','price','movers','daily','update','top','new','guide','dates','release','calendar'
    }

    def scrape_headlines(source_name, url):
        try:
            r = requests.get(url, timeout=20, headers={'User-Agent': 'Mozilla/5.0'})
            r.raise_for_status()
            soup = BeautifulSoup(r.text, 'html.parser')
            seen = set()
            rows = []
            for tag in soup.find_all(['h1','h2','h3','a'])[:250]:
                txt = ' '.join(tag.get_text(' ', strip=True).split())
                if not txt or len(txt) < 25 or len(txt) > 180:
                    continue
                low = txt.lower()
                if txt in seen:
                    continue
                if any(x in low for x in ['cookie policy','privacy policy','sign up','subscribe','advertisement']):
                    continue
                seen.add(txt)
                rows.append({'Source': source_name, 'Title': txt, 'URL': url})
            return rows[:25]
        except Exception:
            return []

    @st.cache_data(show_spinner=False, ttl=3600)
    def collect_articles(selected_universe):
        rows = []
        for src_name, src_url in source_map[selected_universe]:
            rows.extend(scrape_headlines(src_name, src_url))
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.drop_duplicates(subset=['Title']).reset_index(drop=True)

    articles = collect_articles(universe)
    if articles.empty:
        st.error('No article headlines could be scraped from the selected sources.')
        return

    st.dataframe(articles, use_container_width=True, hide_index=True)

    st.markdown('#### 2. Weight repeated themes from headlines')

    def tokenize(text):
        toks = re.findall(r"[A-Za-z0-9\-']+", str(text).lower())
        cleaned = []
        for t in toks:
            t = t.strip("-'")
            if len(t) < 3 or t in stopwords or t.isdigit():
                continue
            cleaned.append(t)
        return cleaned

    token_counts = Counter()
    bigram_counts = Counter()
    for title in articles['Title']:
        toks = tokenize(title)
        token_counts.update(set(toks))
        bigram_counts.update(set(zip(toks, toks[1:])))

    rows = []
    for k, v in token_counts.items():
        if v >= min_keyword_hits:
            rows.append({'Theme': k, 'Hits': v, 'Type': 'keyword'})
    for (a, b), v in bigram_counts.items():
        if v >= min_keyword_hits:
            rows.append({'Theme': f'{a} {b}', 'Hits': v, 'Type': 'phrase'})
    token_df = pd.DataFrame(rows)
    if token_df.empty:
        st.warning('Not enough repeated headline patterns to derive a topic. Lower the min hit threshold.')
        return

    token_df = token_df.sort_values(['Hits', 'Type'], ascending=[False, True]).reset_index(drop=True)
    token_df['Weight'] = token_df['Hits'] / token_df['Hits'].sum()
    st.dataframe(token_df.head(20), use_container_width=True, hide_index=True)

    best_theme = token_df.iloc[0]['Theme']
    related = articles[articles['Title'].str.contains(str(best_theme), case=False, na=False)].copy()
    if related.empty:
        related = articles.head(5).copy()

    if universe == 'Sports cards':
        default_focus = ['Baseball', 'Basketball', 'Football']
    elif universe == 'TCG':
        default_focus = ['Pokemon', 'Magic the Gathering']
    else:
        default_focus = ['Baseball', 'Basketball', 'Pokemon']

    guessed_stance = 'Mixed'
    bearish_words = ['drop', 'down', 'cold', 'losers', 'backlog', 'pressure', 'risk']
    bullish_words = ['hot', 'gainers', 'surge', 'jump', 'boom', 'record']
    title_blob = ' '.join(related['Title'].astype(str).tolist()).lower()
    if any(w in title_blob for w in bullish_words):
        guessed_stance = 'Bullish'
    if any(w in title_blob for w in bearish_words):
        guessed_stance = 'Bearish' if guessed_stance == 'Mixed' else 'Mixed'

    st.markdown('#### 3. Hot article topic and dashboard shell')
    suggested_title = f'Why {best_theme.title()} Is a Weekly Hobby Story Right Now'
    suggested_hook = f'Recent article headlines repeatedly point to {best_theme} as a cross-source theme worth pressure testing against card market data.'

    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.markdown(f"<div class='pa-card fade-in'><div class='muted'>Derived hot topic from scraped headlines</div><div style='font-size:24px;font-weight:900;margin-top:4px;'>{best_theme.title()}</div><div style='margin-top:8px;'>{suggested_hook}</div><div class='muted' style='margin-top:10px;'>Universe: {universe} · Matched headlines: {len(related)}</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown('Supporting articles')
        for _, row in related.head(5).iterrows():
            st.markdown(f"- [{row['Title']}]({row['URL']})")

    ep_title = st.text_input('Episode title', value=suggested_title, key='wt_ep_title')
    ep_hook = st.text_input('Core thesis / hook', value=suggested_hook, key='wt_ep_hook')
    ep_stance = st.selectbox('Episode stance', ['Bullish', 'Neutral', 'Bearish', 'Mixed'], index=['Bullish','Neutral','Bearish','Mixed'].index(guessed_stance), key='wt_ep_stance')
    focus_categories = st.multiselect('Focus categories', CATEGORIES, default=[c for c in default_focus if c in CATEGORIES], key='wt_ep_cats')

    if not focus_categories:
        st.warning('Pick at least one focus category to render the episode dashboard.')
        return

    summary, last_row, comp_yoy, comp_3mo, breadth = build_market_summary(df_raw, focus_categories)
    signal_df = compute_category_signal_table(df_raw)
    episode_signals = signal_df[signal_df['Category'].isin(focus_categories)].copy()

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card('Episode stance', ep_stance)
    with k2:
        kpi_card('Article matches', f"{len(related)}")
    with k3:
        kpi_card('Avg YoY', fmt_pct(summary['YoY %'].mean(skipna=True)))
    with k4:
        kpi_card('Avg 3-Mo', fmt_pct(summary['3-Mo %'].mean(skipna=True)))

    thesis_html = f"""
    <div class='pa-card fade-in'>
      <div class='muted'>Episode thesis</div>
      <div style='font-size:30px; font-weight:900; margin-top:6px; line-height:1.15;'>{ep_title}</div>
      <div style='margin-top:10px; font-size:16px; line-height:1.5;'>{ep_hook}</div>
      <div class='muted' style='margin-top:12px;'>Data through {last_row:%b %Y}</div>
    </div>
    """
    st.markdown(thesis_html, unsafe_allow_html=True)

    l1, l2 = st.columns([1.2, 1])
    with l1:
        fig = go.Figure()
        for cat in focus_categories:
            d = preprocess(df_raw, cat)
            fig.add_trace(go.Scatter(x=d['Month_Year'], y=d['market_value'], mode='lines', name=cat))
        fig.update_layout(title='Focus Category Trendlines', xaxis_title='Month', yaxis_title='Market Value')
        apply_fig_theme(fig, height=380, slide_mode=False)
        st.plotly_chart(fig, use_container_width=True, theme='streamlit')
    with l2:
        bar_df = summary.reset_index().sort_values('3-Mo %', ascending=False)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=bar_df['Category'], y=bar_df['3-Mo %'], marker=dict(color=THEME['primary']), showlegend=False))
        fig2.add_hline(y=0, line_dash='dash', opacity=0.6)
        fig2.update_layout(title='3-Month Momentum Snapshot', xaxis_title='Category', yaxis_title='3-Mo %')
        apply_fig_theme(fig2, height=380, slide_mode=False)
        st.plotly_chart(fig2, use_container_width=True, theme='streamlit')

    st.markdown('#### Episode Signal Table')
    signal_view = episode_signals[['Category', 'YoY %', '3-Mo %', '6-Mo CoV %', 'Avg Corr', 'Momentum Score']].copy().sort_values('Momentum Score', ascending=False)
    st.dataframe(signal_view.round(2), use_container_width=True, hide_index=True)

    st.markdown('#### Source headlines used')
    st.dataframe(related[['Source', 'Title', 'URL']].head(10), use_container_width=True, hide_index=True)

    export_df = signal_view.copy()
    export_df['Episode Title'] = ep_title
    export_df['Derived Topic'] = best_theme
    csv = export_df.to_csv(index=False).encode('utf-8')
    st.download_button('Download headline-topic CSV', data=csv, file_name='headline_topic_episode_dashboard.csv', mime='text/csv', key='wt_csv_download')

def render_portfolio_allocator():
    st.markdown(f"<div class='pa-card fade-in'><h3>Portfolio Allocator</h3><div class='muted'>Build a rules-based sports card allocation across collection buckets</div></div>", unsafe_allow_html=True)
    st.markdown("")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        bankroll = st.number_input("Bankroll ($)", min_value=500, max_value=500000, value=10000, step=500, key="alloc_bankroll")
    with c2:
        risk_tolerance = st.slider("Risk tolerance", 1, 10, 6, key="alloc_risk")
    with c3:
        horizon = st.slider("Hold horizon (years)", 1, 5, 3, key="alloc_horizon")
    with c4:
        liquidity_need = st.slider("Need for liquidity", 1, 10, 6, key="alloc_liquidity")

    sport = st.selectbox("Primary focus", list(SPORT_TILTS.keys()), index=0, key="alloc_sport")
    use_signals = st.toggle("Use live Cardboard Compass signals", value=True, key="alloc_use_signals")
    st.markdown("<div class='allocator-note'>This allocator visibly adjusts bucket return, risk, and liquidity assumptions using the same momentum, volatility, and correlation signals used elsewhere in Cardboard Compass.</div>", unsafe_allow_html=True)
    st.markdown("")

    signal_df = compute_category_signal_table(df_raw)
    bucket_signal_df = compute_bucket_signal_table(signal_df)
    base_df = pd.DataFrame(DEFAULT_BUCKETS)
    model_df = build_signal_adjusted_buckets(base_df, bucket_signal_df, use_signals)

    st.markdown("#### Signal Mapping")
    signal_view = model_df[["bucket", "Mapped Categories", "Momentum Score", "6-Mo CoV %", "Avg Corr", "base_return", "adj_return", "risk", "adj_risk", "liquidity", "adj_liquidity"]].copy()
    signal_view.columns = ["Bucket", "Mapped Categories", "Momentum Score", "6-Mo CoV %", "Avg Corr", "Base Return", "Adj Return", "Base Risk", "Adj Risk", "Base Liquidity", "Adj Liquidity"]
    st.dataframe(signal_view.round(2), use_container_width=True, hide_index=True)

    with st.expander("See category-level signals feeding the allocator", expanded=False):
        st.dataframe(signal_df.round(2), use_container_width=True, hide_index=True)

    editable = model_df[["bucket", "adj_risk", "adj_return", "adj_liquidity", "min_pct", "max_pct"]].copy()
    editable.columns = ["bucket", "risk", "base_return", "liquidity", "min_pct", "max_pct"]

    st.markdown("#### Editable Bucket Assumptions")
    edited = st.data_editor(
        editable,
        use_container_width=True,
        num_rows='fixed',
        column_config={
            'bucket': st.column_config.TextColumn('Bucket', disabled=True),
            'risk': st.column_config.NumberColumn('Risk (1-10)', min_value=1.0, max_value=10.0, step=0.1),
            'base_return': st.column_config.NumberColumn('Exp. return %', min_value=0.0, max_value=40.0, step=0.1),
            'liquidity': st.column_config.NumberColumn('Liquidity (1-10)', min_value=1.0, max_value=10.0, step=0.1),
            'min_pct': st.column_config.NumberColumn('Min %', min_value=0, max_value=100, step=1),
            'max_pct': st.column_config.NumberColumn('Max %', min_value=0, max_value=100, step=1),
        },
        hide_index=True,
        key="alloc_editor_visible"
    )

    if (edited['min_pct'] > edited['max_pct']).any():
        st.error('Each bucket must have min % less than or equal to max %.')
        return
    if edited['min_pct'].sum() > 100:
        st.error('Minimum allocations sum to more than 100%. Lower one or more minimums.')
        return
    if edited['max_pct'].sum() < 100:
        st.error('Maximum allocations sum to less than 100%. Raise one or more maximums.')
        return

    alloc_input = edited.copy()
    alloc_input.rename(
        columns={
            "risk": "adj_risk",
            "base_return": "adj_return",
            "liquidity": "adj_liquidity",
        },
        inplace=True,
    )

    alloc = allocate_portfolio(alloc_input, bankroll, risk_tolerance, horizon, liquidity_need, sport)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card("Expected portfolio return", f"{alloc['weighted_return'].sum():.1f}%")
    with k2:
        kpi_card("Weighted risk score", f"{alloc['weighted_risk'].sum():.1f} / 10")
    with k3:
        kpi_card("Weighted liquidity", f"{alloc['weighted_liquidity'].sum():.1f} / 10")
    with k4:
        cash_reserve = alloc.loc[alloc['bucket'].eq('Cash / opportunistic reserve'), 'allocation_usd'].sum()
        kpi_card("Opportunity reserve", f"${cash_reserve:,.0f}")

    lcol, rcol = st.columns([1.25, 1])
    with lcol:
        st.markdown("#### Recommended Allocation")
        show_df = alloc[['bucket', 'target_pct', 'allocation_usd', 'expected_return_pct', 'adj_risk', 'adj_liquidity']].copy()
        show_df.columns = ['Bucket', 'Target %', 'Allocation $', 'Exp. return %', 'Risk', 'Liquidity']
        st.dataframe(show_df.round(2), use_container_width=True, hide_index=True)
    with rcol:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=alloc['target_pct'].values, y=alloc['bucket'].values, orientation='h', marker=dict(color=THEME['primary']), name='', showlegend=False))
        fig.update_layout(title='Allocation Mix', xaxis_title='Target %', yaxis_title='')
        fig.update_yaxes(autorange='reversed')
        apply_fig_theme(fig, height=420, slide_mode=False)
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")

    st.markdown("#### Rebalance Notes")
    notes = []
    if use_signals:
        hottest = bucket_signal_df.sort_values("Momentum Score", ascending=False).iloc[0]["bucket"]
        riskiest = bucket_signal_df.sort_values("6-Mo CoV %", ascending=False).iloc[0]["bucket"]
        notes.append(f"Signal model is currently most constructive on {hottest} based on mapped category momentum.")
        notes.append(f"Highest recent volatility is flowing through {riskiest}, so position sizing matters more there.")
    if risk_tolerance <= 4:
        notes.append('Keep prospects capped and lean harder into vintage, blue-chip stars, and dry powder.')
    elif risk_tolerance >= 8:
        notes.append('You can push more exposure into prospects and sealed wax, but set hard position limits.')
    if liquidity_need >= 8:
        notes.append('Favor buckets with frequent comp activity and avoid overweighting illiquid niche slabs.')
    if horizon <= 2:
        notes.append('Shorter horizons usually work better with liquid stars and event-driven flips than long holds.')
    if sport == 'Pokemon / TCG':
        notes.append('Sealed can be a core allocation here, but watch reprint risk and grading submission waves.')
    if sport == 'Baseball':
        notes.append('Baseball usually supports more vintage and all-time great exposure than other sports.')
    for note in notes:
        st.write(f"- {note}")

    csv = alloc[['bucket', 'target_pct', 'allocation_usd', 'expected_return_pct', 'adj_risk', 'adj_liquidity']].to_csv(index=False).encode('utf-8')
    st.download_button('Download allocation CSV', data=csv, file_name='card_portfolio_allocation.csv', mime='text/csv', key='alloc_csv_download_visible')

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
    "Raw vs Grade Decision Engine",
    "Liquidity + Exit Risk Monitor",
    "Hot Topics of the Week",
    "Portfolio Allocator",
]

with st.sidebar:
    st.markdown(f"### {APP_TITLE}")
    st.markdown(f"<div class='muted'>{APP_SUBTITLE}</div>", unsafe_allow_html=True)
    st.markdown("---")
    slide_mode = st.toggle("📄 Slide Mode", value=False, help="Collector print/PDF-friendly layout")
    page = st.selectbox("Choose an analysis", PAGES, index=PAGES.index("Pancake Analytics Trading Card Market Report"))
    st.markdown("---")
    cat1 = st.selectbox("Primary category", CATEGORIES, index=CATEGORIES.index("Pokemon"))
    cat2 = st.selectbox("Compare against", ["None"] + [c for c in CATEGORIES if c != cat1])

if slide_mode:
    st.markdown("<div class='slide-wrap'>", unsafe_allow_html=True)

if page == "Pancake Analytics Trading Card Market Report":
    summary, last_row, comp_yoy, comp_3mo, breadth = build_market_summary(df_raw, CATEGORIES)
    st.markdown(f"""<div class="pa-header fade-in"><div class="pa-header-inner"><div class="pa-left"><div class="muted" style="color:rgba(255,255,255,0.85)">@pancake_analytics</div><p class="pa-title">TRADING CARD<br/>MARKET REPORT</p><div class="pa-sub">Collector snapshot of YoY + 3-Mo momentum</div></div><div class="pa-right"><p class="pa-asof">AS OF<b>{last_row:%b %Y}</b></p></div></div></div>""", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Composite YoY", fmt_pct(comp_yoy))
    with c2:
        kpi_card("Composite 3-Mo", fmt_pct(comp_3mo))
    with c3:
        kpi_card("Breadth (3-Mo > 0)", fmt_pct(breadth, 0))
    st.markdown("")
    top_3mo = summary.sort_values("3-Mo %", ascending=False).head(3)
    top_yoy = summary.sort_values("YoY %", ascending=False).head(3)
    lag_3mo = summary.sort_values("3-Mo %", ascending=True).head(2)
    leaders_text = (f"<div class='muted'><b>What the Data Says:</b> 3-Mo leaders: {', '.join(top_3mo.index)}. YoY leaders: {', '.join(top_yoy.index)}. Recent laggards: {', '.join(lag_3mo.index)}.</div>" f"<div style='margin-top:10px;'><b>What It Means:</b> The last 3 months show where momentum is concentrating. Laggards can be a buyer’s window — especially if you’re building long-term.</div>")
    left, mid, right = st.columns([2.05, 1.35, 1.35])
    fig_sc = go.Figure()
    fig_sc.add_trace(go.Scatter(x=summary["3-Mo %"], y=summary["YoY %"], mode="markers+text", text=summary.index, textposition="top center", marker=dict(size=10, color=THEME["primary"]), name="", showlegend=False))
    fig_sc.add_hline(y=0, line_dash="dash", opacity=0.5)
    fig_sc.add_vline(x=0, line_dash="dash", opacity=0.5)
    fig_sc.update_layout(title=f"Momentum Map — YoY vs 3-Mo (through {last_row:%b %Y})", xaxis_title="3-Month % change", yaxis_title="YoY % change")
    apply_fig_theme(fig_sc, height=360, slide_mode=slide_mode)
    top3 = top_3mo.copy()
    top3_sum = float(top3["3-Mo %"].sum()) if not top3["3-Mo %"].isna().all() else 0.0
    shares = (top3["3-Mo %"] / top3_sum * 100) if top3_sum != 0 else pd.Series([0, 0, 0], index=top3.index)
    fig_dn = go.Figure(go.Pie(labels=top3.index, values=shares, hole=0.62, textinfo="label+percent", sort=False, marker=dict(colors=[THEME["primary"], THEME["secondary"], THEME["accent_red"]]), showlegend=True))
    fig_dn.update_layout(title="Top-3 movers — share of 3-Mo momentum", annotations=[dict(text="Normalized<br>Top-3 only", x=0.5, y=0.5, showarrow=False, font=dict(color=THEME["muted"], size=12))], legend=dict(orientation="v", x=1.02, y=0.95))
    apply_fig_theme(fig_dn, height=360, slide_mode=slide_mode)
    fig_dn.update_layout(margin=dict(l=16, r=80, t=74, b=18))
    top5_yoy = summary.sort_values("YoY %", ascending=False).head(5)
    fig_y = go.Figure()
    fig_y.add_trace(go.Bar(x=top5_yoy["YoY %"].values, y=top5_yoy.index.tolist(), orientation="h", marker=dict(color=THEME["primary"]), name="", showlegend=False))
    fig_y.update_layout(title="Top YoY %", xaxis_title="YoY %", yaxis_title="", showlegend=False)
    fig_y.update_yaxes(autorange="reversed")
    apply_fig_theme(fig_y, height=360, slide_mode=slide_mode)
    fig_y.update_layout(margin=dict(l=16, r=16, t=74, b=18))
    with left:
        st.plotly_chart(fig_sc, use_container_width=True, theme="streamlit")
    with mid:
        st.plotly_chart(fig_dn, use_container_width=True, theme="streamlit")
    with right:
        st.plotly_chart(fig_y, use_container_width=True, theme="streamlit")
    st.markdown("")
    r1, r2 = st.columns([1.15, 0.85])
    with r1:
        section_card("This Month’s Leaders", leaders_text)
    with r2:
        best_cat = summary["YoY %"].idxmax()
        best_val = float(summary.loc[best_cat, "YoY %"])
        body = (f"<div class='muted'>Top category YoY</div>" f"<div style='font-size:42px; font-weight:900; margin-top:4px;'>{best_val:0.1f}%</div>" f"<div class='muted' style='margin-top:6px;'>{best_cat}</div>")
        section_card("Top Category YoY", body)
    st.markdown("")
    st.markdown("### Full Category Table (YoY + 3-Mo)")
    bottom_tbl = summary[["YoY %", "3-Mo %"]].round(2).loc[sorted(CATEGORIES)]
    st.table(bottom_tbl)
    st.markdown("---")
    st.markdown("#### Export")
    st.markdown("<div class='muted'>Use Slide Mode → then browser Print → Save as PDF for a deck-ready export.</div>", unsafe_allow_html=True)
    html_snapshot = f"""<!doctype html><html><head><meta charset="utf-8"/><title>Trading Card Market Report</title><style>body {{ font-family: Arial, sans-serif; margin: 24px; }} h1 {{ margin: 0; }} .kpis {{ display:flex; gap:14px; margin-top:16px; }} .card {{ border:1px solid #ddd; border-radius:12px; padding:14px 16px; flex:1; }} .muted {{ color:#666; font-size:13px; }} table {{ border-collapse: collapse; width:100%; margin-top:14px; }} th, td {{ border:1px solid #ddd; padding:8px 10px; text-align:left; }} th {{ background:#f5f5f5; }}</style></head><body><div><div class="muted">@pancake_analytics</div><h1>Trading Card Market Report</h1><div class="muted">As of {last_row:%B %Y}</div></div><div class="kpis"><div class="card"><div class="muted">Composite YoY</div><div style="font-size:28px;font-weight:800;">{fmt_pct(comp_yoy)}</div></div><div class="card"><div class="muted">Composite 3-Mo</div><div style="font-size:28px;font-weight:800;">{fmt_pct(comp_3mo)}</div></div><div class="card"><div class="muted">Breadth (3-Mo &gt; 0)</div><div style="font-size:28px;font-weight:800;">{fmt_pct(breadth, 0)}</div></div></div><h3 style="margin-top:20px;">Full Category Table (YoY + 3-Mo)</h3>{bottom_tbl.to_html()}</body></html>"""
    download_print_ready_html(html_snapshot, filename=f"cardboard_compass_market_report_{last_row:%Y_%m}.html")
elif page == "Category Analysis":
    st.markdown(f"<div class='pa-card fade-in'><h3>Category Analysis</h3><div class='muted'>Forecast + MACD + seasonality</div></div>", unsafe_allow_html=True)
    st.markdown("")

    def show_category(cat: str):
        d = preprocess(df_raw, cat)
        with st.expander("Forecast settings", expanded=False):
            horizon = st.slider("Horizon (months)", 6, 24, 12, step=1, key=f"h_{cat}")
            ci = st.select_slider("Confidence interval", options=[0.80, 0.90, 0.95, 0.98, 0.99], value=0.95, key=f"ci_{cat}")
            hw_trend = st.selectbox("Trend", ["add", "mul"], index=0, key=f"t_{cat}")
            hw_seasonal = st.selectbox("Seasonal", ["add", "mul"], index=0, key=f"s_{cat}")
            sp = st.number_input("Seasonal periods", min_value=4, max_value=24, value=12, step=1, key=f"sp_{cat}")
        hist_df, fc_df = forecast(d, horizon=horizon, seasonal_periods=sp, trend=hw_trend, seasonal=hw_seasonal, ci_level=ci)
        last_actual = d["market_value"].iloc[-1]
        last_forecast = fc_df["Forecast"].iloc[-1]
        pct_change = np.nan if last_actual == 0 else (last_forecast - last_actual) / last_actual * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_df["Date"], y=hist_df["Historical"], mode="lines", name="Historical"))
        fig.add_trace(go.Scatter(x=fc_df["Date"], y=fc_df["Forecast"], mode="lines", name="Forecast", line=dict(dash="dash")))
        fig.add_trace(go.Scatter(x=pd.concat([fc_df["Date"], fc_df["Date"][::-1]]), y=pd.concat([fc_df["Upper"], fc_df["Lower"][::-1]]), fill="toself", fillcolor="rgba(109, 94, 247, 0.15)", line=dict(color="rgba(109, 94, 247, 0)"), name=f"{int(ci*100)}% interval", hoverinfo="skip", showlegend=True))
        fig.update_layout(title=f"{cat} — {horizon}-Month Holt-Winters Forecast", xaxis_title="Month", yaxis_title="Market Value")
        apply_fig_theme(fig, height=420, slide_mode=slide_mode)
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")
        forecast_text = "—" if pd.isna(pct_change) else f"{pct_change:+.1f}%"
        section_card("Forecast Read", f"<div><b>What the Data Says:</b> Next {horizon} months project <b>{forecast_text}</b> vs last observed.</div><div style='margin-top:8px;'><b>What It Means:</b> Use this as directionally helpful — not an exact card price predictor.</div>")
        m, sig, bucket = macd(d)
        macd_df = pd.DataFrame({"Date": d["Month_Year"], "MACD": m.values, "Signal": sig.values})
        fig_m = go.Figure()
        fig_m.add_trace(go.Scatter(x=macd_df["Date"], y=macd_df["MACD"], mode="lines", name="MACD"))
        fig_m.add_trace(go.Scatter(x=macd_df["Date"], y=macd_df["Signal"], mode="lines", name="Signal"))
        fig_m.add_hline(y=0, line_dash="dash", opacity=0.6)
        fig_m.update_layout(title=f"{cat} — MACD Trend (most recent: {bucket.iloc[-1]})", xaxis_title="Month", yaxis_title="MACD")
        apply_fig_theme(fig_m, height=340, slide_mode=slide_mode)
        st.plotly_chart(fig_m, use_container_width=True, theme="streamlit")
        dd = d.copy()
        dd["Month"] = dd["Month_Year"].dt.month_name()
        month_order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        month_avg = dd.groupby("Month")["market_value"].mean().reindex(month_order)
        fig_s = go.Figure(go.Bar(x=month_avg.index, y=month_avg.values, marker=dict(color=THEME["primary"]), name=""))
        fig_s.update_layout(title=f"{cat} — Seasonality (Avg by Month)", xaxis_title="Month", yaxis_title="Avg Value", showlegend=False)
        apply_fig_theme(fig_s, height=320, slide_mode=slide_mode)
        st.plotly_chart(fig_s, use_container_width=True, theme="streamlit")

    if cat2 == "None":
        show_category(cat1)
    else:
        a, b = st.columns(2)
        with a:
            show_category(cat1)
        with b:
            show_category(cat2)
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
elif page == "State of Market":
    st.markdown(f"<div class='pa-card fade-in'><h3>State of Market</h3><div class='muted'>YoY vs 3-Mo momentum by category</div></div>", unsafe_allow_html=True)
    st.markdown("")
    latest = df_raw["Month_Year"].max()
    yoy_vals, mo3_vals = [], []
    for c in CATEGORIES:
        s = df_raw[df_raw["Category"] == c].groupby("Month_Year")["market_value"].mean().sort_index()
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
    pivot = df_raw.pivot_table(values="market_value", index="Month_Year", columns="Category", aggfunc="mean").reindex(columns=CATEGORIES).sort_index().apply(pd.to_numeric, errors="coerce")
    custom = (pivot[sel] * weights).sum(axis=1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=custom.index, y=custom.values, mode="lines", name="My Index", line=dict(width=3, color=THEME["primary"])))
    fig.update_layout(title="Custom Index", xaxis_title="Month", yaxis_title="Value")
    apply_fig_theme(fig, height=420, slide_mode=slide_mode)
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    st.markdown("#### Weights")
    st.table(weights.mul(100).round(1).rename("Weight %"))
elif page == "Seasonality HeatMap":
    st.markdown(f"<div class='pa-card fade-in'><h3>Seasonality HeatMap</h3><div class='muted'>Average MoM % change by month</div></div>", unsafe_allow_html=True)
    st.markdown("")
    df_tmp = df_raw.copy()
    df_tmp["Month_Num"] = df_tmp["Month_Year"].dt.month
    wide = df_tmp.pivot_table(values="market_value", index="Category", columns="Month_Num", aggfunc="mean").reindex(index=CATEGORIES)
    pct = (wide.pct_change(axis=1) * 100).round(2)
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig = go.Figure(data=go.Heatmap(z=pct.values, x=month_labels, y=pct.index.tolist(), colorscale="RdYlGn", zmin=-20, zmax=20))
    fig.update_layout(title="Seasonality — Avg MoM % Change", xaxis_title="Month", yaxis_title="Category")
    apply_fig_theme(fig, height=420, slide_mode=slide_mode)
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    st.table(pct.fillna("—"))
elif page == "Rolling Volatility":
    st.markdown(f"<div class='pa-card fade-in'><h3>Rolling Volatility</h3><div class='muted'>Coefficient of variation over time</div></div>", unsafe_allow_html=True)
    st.markdown("")
    pick = st.selectbox("Category", CATEGORIES, index=CATEGORIES.index(cat1))
    d = preprocess(df_raw, pick).set_index("Month_Year").sort_index()
    w1 = st.slider("Short window (months)", 3, 18, 6, step=1)
    w2 = st.slider("Long window (months)", 6, 36, 12, step=1)
    cv1 = (d["market_value"].rolling(w1).std() / d["market_value"].rolling(w1).mean() * 100).rename(f"{w1}-Mo")
    cv2 = (d["market_value"].rolling(w2).std() / d["market_value"].rolling(w2).mean() * 100).rename(f"{w2}-Mo")
    cv_df = pd.concat([cv1, cv2], axis=1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cv_df.index, y=cv_df.iloc[:, 0], mode="lines", name=cv_df.columns[0], line=dict(color=THEME["primary"])))
    fig.add_trace(go.Scatter(x=cv_df.index, y=cv_df.iloc[:, 1], mode="lines", name=cv_df.columns[1], line=dict(color=THEME["secondary"])))
    fig.update_layout(title=f"{pick} — Rolling Volatility", xaxis_title="Month", yaxis_title="CoV (%)")
    apply_fig_theme(fig, height=420, slide_mode=slide_mode)
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    st.table(cv_df.round(2).tail(24))
elif page == "Correlation Matrix":
    st.markdown(f"<div class='pa-card fade-in'><h3>Correlation Matrix</h3><div class='muted'>Category co-movement (returns or levels)</div></div>", unsafe_allow_html=True)
    st.markdown("")
    wide = df_raw.pivot_table(values="market_value", index="Month_Year", columns="Category", aggfunc="mean").sort_index()[CATEGORIES]
    basis = st.radio("Correlation basis", ["Monthly returns (pct_change)", "Levels (raw index)"], index=0, horizontal=True)
    mat = wide.pct_change().dropna() if basis.startswith("Monthly") else wide.dropna()
    corr = mat.corr()
    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(), zmin=-1, zmax=1, colorscale="RdYlGn"))
    fig.update_layout(title="Correlation Heatmap", xaxis_title="Category", yaxis_title="Category")
    apply_fig_theme(fig, height=520, slide_mode=slide_mode)
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    st.table(corr.round(2))
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
            results[i, m + 1] = results[i, m] * (1 + rand_return)
    months_ahead = np.arange(num_months + 1)
    p10 = np.percentile(results, 10, axis=0)
    p50 = np.percentile(results, 50, axis=0)
    p90 = np.percentile(results, 90, axis=0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=months_ahead, y=p50, mode="lines", name="Median", line=dict(color=THEME["primary"], dash="dash", width=3)))
    fig.add_trace(go.Scatter(x=np.concatenate([months_ahead, months_ahead[::-1]]), y=np.concatenate([p90, p10[::-1]]), fill="toself", fillcolor="rgba(109, 94, 247, 0.16)", line=dict(color="rgba(0,0,0,0)"), name="10–90% band", hoverinfo="skip"))
    fig.add_hline(y=asking_price, line_dash="dot", opacity=0.7)
    fig.update_layout(title=f"Flip Forecast — {sim_category}", xaxis_title="Months Ahead", yaxis_title="Simulated Price ($)")
    apply_fig_theme(fig, height=420, slide_mode=slide_mode)
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    final_prices = results[:, -1]
    prob_hit = float(np.mean(final_prices >= asking_price) * 100)
    st.table(pd.DataFrame({"Metric": ["Expected Return", "Monthly Volatility (capped)", "Probability Asking Price Hit"], "Value": [f"{expected_return:.2%}", f"{monthly_volatility:.2%}", f"{prob_hit:.1f}%"]}))
elif page == "Raw vs Grade Decision Engine":
    render_raw_vs_grade_engine()
elif page == "Liquidity + Exit Risk Monitor":
    render_liquidity_exit_monitor()
elif page == "Weekly Topic Generator":
    render_topic_generator_dashboard()
elif page == "Episode Companion Dashboard":
    render_episode_companion_dashboard()
elif page == "Portfolio Allocator":
    render_portfolio_allocator()

st.markdown("""
---
**Cardboard Compass** — built by Pancake Analytics LLC  
*Analytics read, not financial advice.*
""")

if slide_mode:
    st.markdown("</div>", unsafe_allow_html=True)
