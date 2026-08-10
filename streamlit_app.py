import streamlit as st
import pandas as pd
import numpy as np
import os
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Cardboard Compass", layout="wide")

THEME = {
    "primary": "#E4002B",      # signature red — accent words, primary series, highlights
    "secondary": "#1A1A1A",    # near-black — secondary series, contrast bars
    "accent_red": "#B3001B",   # deeper red — alerts, badges, negative call-outs
    "bg": "#FFFFFF",           # white app background
    "card": "#FFFFFF",         # white card background
    "text": "#111111",         # near-black primary text
    "muted": "#6B7280",        # grey muted text
    "border": "#E5E7EB",       # light grey borders / dividers
    "grid": "#EEF2F7",         # light grey gridlines
}

HEATMAP_SCALE = [[0.0, "#4B5563"], [0.5, "#FFFFFF"], [1.0, "#E4002B"]]

APP_TITLE = "CARDBOARD COMPASS"
APP_SUBTITLE = "eBay market-index insights — built by Pancake Analytics"

st.markdown(
    f"""
    <style>
      :root {{
        --background-color: {THEME['bg']};
        --secondary-background-color: #F7F7F8;
        --text-color: {THEME['text']};
        --primary-color: {THEME['primary']};
      }}
      html, body, .stApp,
      [data-testid="stAppViewContainer"],
      [data-testid="stMain"],
      [data-testid="stHeader"],
      [data-testid="stToolbar"],
      [data-testid="stBottomBlockContainer"] {{
        background-color: {THEME['bg']} !important;
      }}
      [data-testid="stHeader"] {{ background: transparent !important; }}
      .block-container {{ padding-top: 1.4rem; padding-bottom: 2rem; }}
      h1,h2,h3,h4,h5,h6,p,span,label,li {{ color: {THEME['text']}; }}
      .muted {{ color: {THEME['muted']}; font-size: 0.90rem; }}
      .pa-card {{
        background: {THEME['card']};
        border: 1px solid {THEME['border']};
        border-radius: 16px;
        padding: 16px 18px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
      }}
      .pa-header {{
        border-radius: 18px;
        overflow: hidden;
        border: 1px solid {THEME['border']};
        box-shadow: 0 10px 20px rgba(0,0,0,0.08);
      }}
      .pa-header-inner {{
        display: grid;
        grid-template-columns: 2.5fr 1fr;
        align-items: stretch;
      }}
      .pa-left {{
        padding: 22px 24px;
        background: {THEME['secondary']};
        color: #FFFFFF;
      }}
      .pa-right {{
        padding: 22px 24px;
        background: {THEME['primary']};
        color: #FFFFFF;
        text-align: right;
      }}
      .pa-title {{
        font-weight: 900;
        letter-spacing: 0.6px;
        font-size: 28px;
        line-height: 1.1;
        margin: 0;
        color: #FFFFFF;
      }}
      .pa-title em, .pa-title i {{ color: {THEME['primary']}; font-style: italic; }}
      .pa-sub {{ margin-top: 8px; font-size: 13px; opacity: 0.92; }}
      .pa-asof {{ font-size: 13px; opacity: 0.9; margin: 0; color: #FFFFFF; }}
      .pa-asof b {{ display: block; margin-top: 6px; font-size: 22px; letter-spacing: 0.3px; color: #FFFFFF; }}
      .fade-in {{ animation: fadeIn 0.45s ease-in-out; }}
      @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(6px); }} to {{ opacity: 1; transform: translateY(0); }} }}
      iframe {{ border-radius: 12px; }}
      div[data-testid="stDataFrame"] > div {{ overflow: auto; }}
      section[data-testid="stSidebar"], [data-testid="stSidebarContent"] {{ background-color: #FAFAFA !important; border-right: 1px solid {THEME['border']}; }}
      thead tr th {{ background-color: {THEME['secondary']} !important; color: #FFFFFF !important; border-color: {THEME['secondary']} !important; }}
      tbody tr td {{ background-color: {THEME['card']} !important; color: {THEME['text']} !important; border-color: {THEME['border']} !important; }}
      tbody tr:nth-child(even) td {{ background-color: #F7F7F8 !important; }}
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

DATA_URL = "https://pancakebreakfaststats.com/wp-content/uploads/2026/08/data_file_020.xlsx"

@st.cache_data(show_spinner=False, ttl=3600)  # refresh hourly — previously cached forever until app reboot
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
    "Fortnite", "Marvel", "Pokemon", "Star Wars", "Magic the Gathering", "Lorcana",
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
    "Sealed wax": ["Pokemon", "Magic the Gathering", "Lorcana", "Marvel", "Star Wars", "Fortnite"],
    "Cash / opportunistic reserve": []
}

def preprocess(df: pd.DataFrame, cat: str) -> pd.DataFrame:
    d = df[df["Category"] == cat].copy()
    return d.groupby("Month_Year", as_index=False)["market_value"].mean().sort_values("Month_Year")

def deseasonalize(df: pd.DataFrame, method: str = "ratio", window_months: int = 36) -> pd.DataFrame:
    """
    Adds a 'deseasonalized' column to a copy of df (expects Month_Year + market_value).
    method='ratio'  -> multiplicative seasonal-naive adjustment (value / seasonal index)
    method='diff'   -> additive seasonal-naive adjustment (value - seasonal component)

    The seasonal baseline is computed from a trailing window (default 36 months),
    not the full history — averaging in years with a structurally different price/
    volatility regime (e.g. a category's early history vs. its current state) can
    over- or under-correct for what "normal seasonal timing" looks like today.
    Falls back to full-history average for any calendar month not represented in
    the trailing window.
    """
    d = df.copy().sort_values("Month_Year")
    d["Month_Num"] = d["Month_Year"].dt.month

    recent = d.tail(window_months) if (window_months and len(d) > window_months) else d
    recent_overall_mean = recent["market_value"].mean()
    recent_month_avg = recent.groupby("Month_Num")["market_value"].mean()

    d["month_avg"] = d["Month_Num"].map(recent_month_avg)
    if d["month_avg"].isna().any():
        full_month_avg = d.groupby("Month_Num")["market_value"].transform("mean")
        d["month_avg"] = d["month_avg"].fillna(full_month_avg)

    if method == "ratio":
        seasonal_index = (d["month_avg"] / recent_overall_mean).replace(0, np.nan)
        d["deseasonalized"] = d["market_value"] / seasonal_index
    else:  # additive
        seasonal_component = d["month_avg"] - recent_overall_mean
        d["deseasonalized"] = d["market_value"] - seasonal_component
    d["deseasonalized"] = d["deseasonalized"].fillna(d["market_value"])
    return d.drop(columns=["month_avg"])

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

def macd(df: pd.DataFrame, value_col: str = "market_value", fast: int = 6, slow: int = 13, signal: int = 5):
    """
    12/26/9 is the standard convention for DAILY data. On monthly data that's a
    12-month vs 26-month (2+ year) comparison, which behaves more like a multi-year
    regime detector than a responsive momentum signal — so spans are rescaled here
    to roughly the same ratio, sized for monthly cadence instead.

    Bucketing is done on the MACD histogram as a % of the slow EMA (price level),
    not a fixed dollar amount — a $1.50 swing is nothing for a $190 Lorcana average
    and huge for a $28 Baseball average, so fixed absolute cutoffs would bias
    signals toward whichever categories happen to have higher price levels.
    """
    s = df[value_col].ewm(span=fast, adjust=False).mean()
    l = df[value_col].ewm(span=slow, adjust=False).mean()
    m = s - l
    sig = m.ewm(span=signal, adjust=False).mean()
    price_level = l.replace(0, np.nan)
    hist_pct = ((m - sig) / price_level * 100).fillna(0)
    bucket = pd.cut(hist_pct, [-np.inf, -3, -1, 0, 1, 3, np.inf], labels=["High Down", "Med Down", "Low Down", "Low Up", "Med Up", "High Up"])
    return m, sig, bucket

def bucket_to_score(bucket_label) -> int:
    mapping = {"High Down": -3, "Med Down": -2, "Low Down": -1, "Low Up": 1, "Med Up": 2, "High Up": 3}
    return mapping.get(str(bucket_label), 0)

def bucket_badge(bucket_label) -> str:
    # Red -> grey/black heat ramp: strongest up = deep red, strongest down = near-black.
    shades = {
        "High Up": ("#E4002B", "#FFFFFF"),
        "Med Up": ("#F0505F", "#FFFFFF"),
        "Low Up": ("#FBD6DA", "#111111"),
        "Low Down": ("#E5E7EB", "#111111"),
        "Med Down": ("#8E8E93", "#FFFFFF"),
        "High Down": ("#1A1A1A", "#FFFFFF"),
    }
    bg, fg = shades.get(str(bucket_label), (THEME["border"], THEME["text"]))
    return (f"<span style='display:inline-block; padding:3px 9px; border-radius:8px; "
            f"font-weight:700; font-size:12px; background:{bg} !important; color:{fg} !important;'>{bucket_label}</span>")

def signal_badge(label: str) -> str:
    # Matches the same red/black/grey badge convention used elsewhere (RISING=red, WATCH=black, FALLING=grey),
    # with outlined variants for the two conditional / forward-looking calls.
    styles = {
        "BUY": f"background:{THEME['primary']} !important; color:#FFFFFF !important; border:2px solid {THEME['primary']};",
        "BUY THE DIP": f"background:{THEME['primary']} !important; color:#FFFFFF !important; border:2px solid {THEME['primary']};",
        "SELL THE PEAK": f"background:#4B5563 !important; color:#FFFFFF !important; border:2px solid #4B5563;",
        "WATCH": f"background:{THEME['secondary']} !important; color:#FFFFFF !important; border:2px solid {THEME['secondary']};",
        "CAUTION": f"background:#FFFFFF !important; color:#4B5563 !important; border:2px solid #4B5563;",
        "LONG-TERM HOLD": f"background:#FFFFFF !important; color:{THEME['primary']} !important; border:2px solid {THEME['primary']};",
    }
    style = styles.get(label, f"background:{THEME['border']} !important; color:{THEME['text']} !important; border:2px solid {THEME['border']};")
    return (f"<span style='display:inline-block; padding:4px 12px; border-radius:999px; "
            f"font-weight:800; font-size:12px; letter-spacing:0.3px; {style}'>{label}</span>")

@st.cache_data(show_spinner=False, ttl=3600)
def compute_category_metrics(cat: str, hw_horizon: int, deseason_method: str, overextension_window: int = 6) -> dict:
    """
    The expensive part: fits raw + deseasonalized MACD, a Holt-Winters model, and an
    overextension read. Cached on (category, horizon, deseasonalizing method,
    overextension window) only — NOT on the buy/sell or overextension thresholds,
    since those are applied afterward in compute_signal_snapshot() and are cheap to
    recompute on every slider move without refitting anything.
    """
    df_raw = load_data(DATA_URL)
    d = preprocess(df_raw, cat)
    m, sig, bucket = macd(d)
    raw_bucket = str(bucket.iloc[-1])
    raw_score = bucket_to_score(raw_bucket)

    d_adj = deseasonalize(d, method=deseason_method)
    m_ds, sig_ds, bucket_ds = macd(d_adj, value_col="deseasonalized")
    ds_bucket = str(bucket_ds.iloc[-1])
    ds_score = bucket_to_score(ds_bucket)

    confirmed = (raw_score != 0) and (ds_score != 0) and (np.sign(raw_score) == np.sign(ds_score))

    hw_pct = np.nan
    if len(d) >= 24:
        try:
            model = ExponentialSmoothing(d["market_value"].astype(float), trend="add", seasonal="add", seasonal_periods=12).fit()
            fc = model.forecast(hw_horizon)
            last_actual = float(d["market_value"].iloc[-1])
            if last_actual:
                hw_pct = float((fc.iloc[-1] - last_actual) / last_actual * 100)
        except Exception:
            hw_pct = np.nan

    # Overextension: how far the current price sits above its OWN trailing average,
    # independent of MACD or the forecast. MACD measures whether the trend is
    # accelerating, not whether the price has already detached from its recent norm —
    # a card can show strong "High Up" momentum while also being deep into blow-off-top
    # territory (see: Umbreon ex, +55% Jan-Apr then -16% Apr-May in the carousel data).
    # The trailing average excludes the current month so the spike doesn't inflate its
    # own baseline.
    overextension_pct = np.nan
    if len(d) > overextension_window:
        trailing_avg = float(d["market_value"].iloc[-(overextension_window + 1):-1].mean())
        current_price = float(d["market_value"].iloc[-1])
        if trailing_avg:
            overextension_pct = float((current_price - trailing_avg) / trailing_avg * 100)

    return {
        "Category": cat,
        "Raw Bucket": raw_bucket,
        "Raw Score": raw_score,
        "Deseasonalized Bucket": ds_bucket,
        "Deseason Score": ds_score,
        "Confirmed": confirmed,
        "HW Pct": hw_pct,
        "Overextension %": overextension_pct,
    }

def compute_signal_snapshot(df_raw: pd.DataFrame, cat: str, hw_horizon: int = 12, hw_threshold: float = 8.0, deseason_method: str = "ratio", overextension_window: int = 6, overextension_threshold: float = 25.0) -> dict:
    """
    Blends four reads into a buy-low/sell-high call per category — this is deliberately
    a CONTRARIAN framework, not a trend-following one.
      - A confirmed mid-range decline does NOT by itself mean sell — it means WATCH unless
        price is still elevated near its own recent highs (overextended), in which case it's
        SELL THE PEAK. Chasing an ongoing decline lower, with no peak behind it, is the
        opposite of what a collector buying/selling at the right time wants.
      - Overextension ALONE triggers SELL THE PEAK, without waiting for momentum to visibly
        confirm-turn-down first — that confirmation is inherently lagging (by the time raw
        MACD shows a confirmed reversal, price has usually already dropped from the top), so
        requiring it would mean selling after the peak instead of at it.
      - Raw MACD: short-term momentum (may include seasonal noise)
      - Deseasonalized MACD: confirms whether that momentum survives outside normal seasonal timing
      - Overextension: price vs. its own trailing average — this is what identifies whether
        a momentum reading is happening AT a peak/trough or somewhere in the middle
      - Holt-Winters forecast: long-range direction, used for LONG-TERM HOLD (buy the dip on
        a longer view) even when price hasn't technically reached "oversold" yet
    The MACD/Holt-Winters/overextension fitting is cached separately (see
    compute_category_metrics) — this function only applies the thresholds, so it's
    cheap to call on every rerun.
    df_raw is kept as a parameter for call-site compatibility, but is not used directly:
    compute_category_metrics fetches its own (cached) copy via load_data() so its cache
    key stays limited to hashable primitives (category, horizon, method, window).
    """
    metrics = compute_category_metrics(cat, hw_horizon, deseason_method, overextension_window)
    raw_score = metrics["Raw Score"]
    confirmed = metrics["Confirmed"]
    hw_pct = metrics["HW Pct"]
    overext_pct = metrics["Overextension %"]

    long_term_up = (not pd.isna(hw_pct)) and hw_pct >= hw_threshold
    long_term_down = (not pd.isna(hw_pct)) and hw_pct <= -hw_threshold
    overextended = (not pd.isna(overext_pct)) and overext_pct >= overextension_threshold
    oversold = (not pd.isna(overext_pct)) and overext_pct <= -overextension_threshold

    # Overextension itself IS the sell-the-peak signal — it does NOT wait for momentum
    # to visibly confirm-turn-down first. Waiting for that confirmation means selling
    # only after the price has already started dropping from the top, which defeats
    # the point of selling AT the peak rather than after it.
    if overextended:
        signal = "SELL THE PEAK"
        if raw_score <= -1 and confirmed:
            why = (f"Price is still {overext_pct:.0f}% above its trailing {overextension_window}-mo average, and confirmed "
                   f"momentum has already turned down — the top is visibly rolling over. Sell before it falls further.")
        else:
            why = (f"Price is {overext_pct:.0f}% above its trailing {overextension_window}-mo average — deep into peak "
                   f"territory. Momentum hasn't visibly turned down yet, but waiting for that confirmation means selling "
                   f"after the drop has already started. This flags it now, while there's still strength to sell into.")
    elif oversold and raw_score >= -1:
        signal = "BUY THE DIP"
        why = (f"Price is {overext_pct:.0f}% below its trailing {overextension_window}-mo average, and downside "
               f"momentum has stopped accelerating — the profile of a bottom forming, not a falling knife. "
               f"(The long-term forecast can still lag a turn like this, so it isn't required to confirm here.)")
    elif raw_score <= -1 and long_term_up:
        signal = "LONG-TERM HOLD"
        why = f"Short-term is soft and not yet at a clean technical low, but the {hw_horizon}-mo Holt-Winters forecast projects a meaningful recovery."
    elif raw_score >= 2 and confirmed and not long_term_down:
        signal = "BUY"
        why = "Momentum holds up after removing seasonality, the long-term forecast isn't fighting it, and price isn't stretched relative to its own recent average — a healthy climb, not a spike."
    elif raw_score >= 1 and long_term_down:
        signal = "CAUTION"
        why = f"Short-term momentum is up, but the {hw_horizon}-mo forecast is turning down — don't chase this."
    elif raw_score != 0 and not confirmed:
        signal = "WATCH"
        why = "Raw MACD disagrees with the deseasonalized read — likely calendar timing, not real momentum."
    else:
        signal = "WATCH"
        why = "This isn't at either extreme — not stretched enough to call a peak, not cheap enough to call a low. Nothing actionable yet either way."

    return {
        "Category": cat,
        "Raw Bucket": metrics["Raw Bucket"],
        "Raw Score": raw_score,
        "Deseasonalized Bucket": metrics["Deseasonalized Bucket"],
        "Deseason Score": metrics["Deseason Score"],
        "Confirmed": confirmed,
        f"{hw_horizon}-Mo HW %": hw_pct,
        "Overextension %": overext_pct,
        "Signal": signal,
        "Why": why,
    }

SNAPSHOT_PATH = "signal_snapshots.csv"

def load_snapshot_history() -> pd.DataFrame:
    """
    Loads saved signal snapshots for week-over-week comparison.
    NOTE: this writes to the app's local filesystem, which persists for the life of
    the running container but is NOT guaranteed to survive a redeploy/reboot on
    Streamlit Community Cloud. For guaranteed long-term persistence across redeploys,
    point this at an external store instead (a Google Sheet via gspread, a small
    hosted Postgres/SQLite, etc.) — the save/load interface here is deliberately
    narrow (two functions, one CSV schema) so swapping the backend later is a
    localized change, not a rewrite.
    """
    if os.path.exists(SNAPSHOT_PATH):
        try:
            hist = pd.read_csv(SNAPSHOT_PATH, parse_dates=["snapshot_date"])
            return hist
        except Exception:
            return pd.DataFrame(columns=["snapshot_date", "Category", "Signal"])
    return pd.DataFrame(columns=["snapshot_date", "Category", "Signal"])

def save_snapshot(heat_df: pd.DataFrame) -> pd.DataFrame:
    snap = heat_df[["Category", "Signal"]].copy()
    snap["snapshot_date"] = pd.Timestamp.today().normalize()
    history = load_snapshot_history()
    history = pd.concat([history, snap], ignore_index=True)
    history.to_csv(SNAPSHOT_PATH, index=False)
    return history

def apply_fig_theme(fig: go.Figure, height: int, slide_mode: bool):
    bg = THEME["card"] if not slide_mode else "#FFFFFF"
    fig.update_layout(
        transition=dict(duration=450, easing="cubic-in-out"),
        paper_bgcolor=bg,
        plot_bgcolor=bg,
        font=dict(color=THEME["text"] if not slide_mode else "#111827"),
        margin=dict(l=16, r=16, t=64, b=18),
        height=height,
        title=dict(x=0.02, xanchor="left", y=0.98),
        colorway=[THEME["text"], THEME["primary"], THEME["muted"], THEME["secondary"], THEME["accent_red"]],
        legend=dict(font=dict(color=THEME["text"] if not slide_mode else "#111827")),
    )
    fig.update_xaxes(gridcolor=THEME["grid"], zerolinecolor=THEME["border"], linecolor=THEME["border"], color=THEME["text"] if not slide_mode else "#111827")
    fig.update_yaxes(gridcolor=THEME["grid"], zerolinecolor=THEME["border"], linecolor=THEME["border"], color=THEME["text"] if not slide_mode else "#111827")
    return fig

def kpi_card(label: str, value: str, sub: str | None = None):
    sub_html = f"<div class='muted'>{sub}</div>" if sub else ""
    st.markdown(f"""<div class="pa-card fade-in"><div class="muted">{label}</div><div style="font-size:32px; font-weight:900; margin-top:4px;">{value}</div>{sub_html}</div>""", unsafe_allow_html=True)

def section_card(title: str, body_html: str):
    st.markdown(f"""<div class="pa-card fade-in"><div style="font-weight:900; font-size:16px; margin-bottom:8px;">{title}</div>{body_html}</div>""", unsafe_allow_html=True)

def explainer_expander(page_label: str, intro_title: str, intro_html: str, examples: list[dict], expander_key: str = None):
    """
    Renders a collapsed 'how to read this page' explainer for collectors.
    examples: list of dicts with keys: badge_html (optional), example, meaning
    """
    with st.expander(f"📖 New here? How to read {page_label}", expanded=False):
        st.markdown(f"""
<div class='pa-card fade-in' style='margin-bottom:14px;'>
<div style='font-weight:900; font-size:16px; margin-bottom:6px;'>{intro_title}</div>
<div style='line-height:1.6;'>{intro_html}</div>
</div>
""", unsafe_allow_html=True)
        for ex in examples:
            badge = ex.get("badge_html", "")
            badge_block = f"<div style='margin-bottom:8px;'>{badge}</div>" if badge else ""
            st.markdown(f"""
<div class='pa-card fade-in' style='margin-bottom:10px;'>
{badge_block}<div style='margin-bottom:6px;'><b>Illustrative example:</b> {ex['example']}</div>
<div class='muted'><b>What it means for a collector:</b> {ex['meaning']}</div>
</div>
""", unsafe_allow_html=True)
        st.markdown("<div class='muted'>These are illustrative walkthroughs of the logic, not live calls on specific cards — check the numbers/table below for what's actually showing today.</div>", unsafe_allow_html=True)

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
    explainer_expander(
        "the Raw vs Grade Decision Engine",
        "What this is comparing",
        "You almost always have three ways to end up with a graded card: <b>buy it raw and sell it raw</b> (no grading risk, but you leave the grade premium on the table), "
        "<b>buy it raw and grade it yourself</b> (higher upside if it comes back a 10, but you're paying grading fees and carrying it for months not knowing what grade you'll get), "
        "or <b>just buy the slab someone else already graded</b> (you pay a premium, but there's no guesswork). "
        "This tool plugs in your costs and the probability of landing each grade, and tells you which path has the best expected return — not which one is guaranteed to win.",
        [
            {
                "example": "Say a raw card costs $120, grading runs $30, and you think there's a 25% shot at a PSA 10 (worth $680) but a 45% shot it comes back a 9 (worth $300). The engine blends every possible outcome by its probability to get one \"expected\" sale price, then subtracts your basis and fees.",
                "meaning": "\"Buy raw and grade\" looking best doesn't mean you're guaranteed a 10 — it means that across many similar cards graded under these odds, that path nets you more on average. A single card can still come back a 7 and lose money."
            },
            {
                "example": "If a comparable already-slabbed PSA 10 is selling for $620 and your probability-weighted expected sale after grading is only $520 net, buying the existing slab can actually beat the gamble of grading your own raw copy.",
                "meaning": "Grading isn't automatically the best move — sometimes paying the grade premium up front is cheaper than the risk-adjusted cost of rolling the dice yourself."
            },
        ],
    )
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
    explainer_expander(
        "the Liquidity + Exit Risk Monitor",
        "Why \"can I sell it\" matters as much as \"will it go up\"",
        "A category can be trending up and still be a pain to exit — wide price swings, a thin recent sale history, or a target price way above where it's actually trading all make it harder to get out at the number you want, when you want. "
        "This page scores that separately from momentum: a high <b>liquidity score</b> means recent activity has been steady and prices haven't been whipping around; a high <b>exit risk score</b> means the opposite — you may need to wait longer or accept a lower price to actually sell.",
        [
            {
                "example": "A card has been swinging 15%+ month to month with a wide gap between its recent high and low, and your target sale price sits well above where it's actually been trading lately.",
                "meaning": "This flags as high exit risk — even if the long-term trend looks fine, don't assume you can list it today at your target number and get a quick sale. Price it closer to recent reality or be patient."
            },
            {
                "example": "A card has been trading in a tight, consistent band for months with frequent recent sales at or near your target price.",
                "meaning": "This flags as healthy liquidity — a much safer bet if you need to convert this to cash on a specific timeline, versus a volatile card you might need to hold through a bad month to get your price."
            },
        ],
    )
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


def render_portfolio_allocator():
    st.markdown(f"<div class='pa-card fade-in'><h3>Portfolio Allocator</h3><div class='muted'>Build a rules-based sports card allocation across collection buckets</div></div>", unsafe_allow_html=True)
    st.markdown("")
    explainer_expander(
        "the Portfolio Allocator",
        "Treating your collection like a portfolio, not a pile of cards",
        "Instead of thinking card-by-card, this groups everything into buckets — vintage, blue-chip stars, prospects, sealed wax, cash reserve, etc. — each with its own typical risk, return, and liquidity profile. "
        "You tell it your bankroll, how much risk you're comfortable with, how long you plan to hold, and how much liquidity you need, and it proposes a target mix — then live-adjusts each bucket's assumptions using the same momentum/volatility signals from the rest of the app.",
        [
            {
                "example": "Someone with high risk tolerance, a 5-year horizon, and low near-term liquidity needs gets steered toward more prospects and sealed wax — the higher-variance, higher-upside buckets — since they can ride out the swings.",
                "meaning": "The allocation isn't a universal \"best\" mix — it's shaped by your specific risk tolerance and timeline. Two collectors with the same bankroll can get very different recommended splits."
            },
            {
                "example": "If the live signals show sealed wax momentum running hot with high volatility, the allocator nudges that bucket's expected return up but also nudges its risk score up — it doesn't just chase the hot hand blindly.",
                "meaning": "A bucket showing strong momentum won't automatically dominate your allocation — the model weighs that against the higher risk that usually comes with it."
            },
        ],
    )
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
    st.markdown("")
    explainer_expander(
        "this report",
        "The at-a-glance version of the whole market",
        "This is the summary page: <b>Composite YoY</b> and <b>Composite 3-Mo</b> are the average change across all ten categories over the last year and the last quarter. "
        "The <b>Momentum Map</b> plots every category by 3-Mo change (x-axis) against YoY change (y-axis) — top-right means both short- and long-term trends agree upward, bottom-left means both agree downward, and the other two corners mean the two timeframes disagree. "
        "<b>Breadth</b> is the % of categories with positive 3-month momentum — a high number means gains are broad-based, a low number means one or two categories are carrying the whole market.",
        [
            {
                "example": "A category sits in the top-right of the Momentum Map, and breadth is above 70%.",
                "meaning": "That's a broad, confirmed uptrend — not just one category spiking while everything else is flat or falling."
            },
            {
                "example": "A category sits in the bottom-right (positive 3-Mo, negative YoY) — it's had a good recent quarter, but is still down over the full year.",
                "meaning": "This is a recovery in progress, not a confirmed trend yet — worth watching whether the recent strength holds before treating it as a real turnaround."
            },
        ],
    )
    st.markdown("")
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
    fig_sc.add_hline(y=0, line_dash="dash", opacity=0.5, line_color=THEME["border"])
    fig_sc.add_vline(x=0, line_dash="dash", opacity=0.5, line_color=THEME["border"])
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
    st.markdown(f"<div class='pa-card fade-in'><h3>Category Analysis</h3><div class='muted'>Forecast + MACD (raw + deseasonalized) + seasonality</div></div>", unsafe_allow_html=True)
    st.markdown("")
    explainer_expander(
        "this page",
        "Three lenses on one category",
        "The <b>forecast chart</b> projects prices forward using historical trend + seasonal pattern, with a shaded confidence band — the wider the band, the less certain the model is. "
        "The two <b>MACD charts</b> show momentum: raw (which can include normal seasonal timing) and deseasonalized (which strips that out so you can see if the momentum is real). "
        "The bottom <b>seasonality bar chart</b> shows the average price by calendar month across all history — tall bars in a given month mean that category has historically run hot around that time of year, which matters when you're deciding whether a current spike is a new trend or just the usual seasonal pattern showing up again.",
        [
            {
                "example": "The forecast band is wide and the projected % change is small, while the MACD reads strongly positive.",
                "meaning": "Short-term momentum looks confident, but the long-range model isn't very sure where this settles — treat the near-term move as more reliable than the far-out forecast in that case."
            },
            {
                "example": "The seasonality chart shows this category always runs highest in October and November, and the current spike is happening in October.",
                "meaning": "Check the raw vs. deseasonalized MACD before assuming this is a new trend — if the deseasonalized version is flat, the spike is probably just this category's normal calendar pattern, not something new."
            },
        ],
    )
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
        fig.add_trace(go.Scatter(x=pd.concat([fc_df["Date"], fc_df["Date"][::-1]]), y=pd.concat([fc_df["Upper"], fc_df["Lower"][::-1]]), fill="toself", fillcolor="rgba(228, 0, 43, 0.12)", line=dict(color="rgba(228, 0, 43, 0)"), name=f"{int(ci*100)}% interval", hoverinfo="skip", showlegend=True))
        fig.update_layout(title=f"{cat} — {horizon}-Month Holt-Winters Forecast", xaxis_title="Month", yaxis_title="Market Value")
        apply_fig_theme(fig, height=420, slide_mode=slide_mode)
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")
        forecast_text = "—" if pd.isna(pct_change) else f"{pct_change:+.1f}%"
        section_card("Forecast Read", f"<div><b>What the Data Says:</b> Next {horizon} months project <b>{forecast_text}</b> vs last observed.</div><div style='margin-top:8px;'><b>What It Means:</b> Use this as directionally helpful — not an exact card price predictor.</div>")

        # --- Raw MACD ---
        m, sig, bucket = macd(d)
        macd_df = pd.DataFrame({"Date": d["Month_Year"], "MACD": m.values, "Signal": sig.values})
        fig_m = go.Figure()
        fig_m.add_trace(go.Scatter(x=macd_df["Date"], y=macd_df["MACD"], mode="lines", name="MACD"))
        fig_m.add_trace(go.Scatter(x=macd_df["Date"], y=macd_df["Signal"], mode="lines", name="Signal"))
        fig_m.add_hline(y=0, line_dash="dash", opacity=0.6, line_color=THEME["border"])
        fig_m.update_layout(title=f"{cat} — MACD Trend (raw, most recent: {bucket.iloc[-1]})", xaxis_title="Month", yaxis_title="MACD")
        apply_fig_theme(fig_m, height=340, slide_mode=slide_mode)
        st.plotly_chart(fig_m, use_container_width=True, theme="streamlit")

        # --- Deseasonalized MACD ---
        deseason_method = st.radio(
            "Deseasonalizing method",
            ["Ratio (multiplicative)", "Difference (additive)"],
            index=0,
            horizontal=True,
            key=f"deseason_method_{cat}",
        )
        method_key = "ratio" if deseason_method.startswith("Ratio") else "diff"
        d_adj = deseasonalize(d, method=method_key)
        m_ds, sig_ds, bucket_ds = macd(d_adj, value_col="deseasonalized")
        macd_ds_df = pd.DataFrame({"Date": d_adj["Month_Year"], "MACD": m_ds.values, "Signal": sig_ds.values})
        fig_ds = go.Figure()
        fig_ds.add_trace(go.Scatter(x=macd_ds_df["Date"], y=macd_ds_df["MACD"], mode="lines", name="MACD (deseasonalized)", line=dict(color=THEME["accent_red"])))
        fig_ds.add_trace(go.Scatter(x=macd_ds_df["Date"], y=macd_ds_df["Signal"], mode="lines", name="Signal (deseasonalized)", line=dict(color=THEME["secondary"])))
        fig_ds.add_hline(y=0, line_dash="dash", opacity=0.6, line_color=THEME["border"])
        fig_ds.update_layout(title=f"{cat} — MACD Trend (deseasonalized, most recent: {bucket_ds.iloc[-1]})", xaxis_title="Month", yaxis_title="MACD")
        apply_fig_theme(fig_ds, height=340, slide_mode=slide_mode)
        st.plotly_chart(fig_ds, use_container_width=True, theme="streamlit")

        raw_signal = str(bucket.iloc[-1])
        ds_signal = str(bucket_ds.iloc[-1])
        if raw_signal != ds_signal:
            divergence_note = (
                f"⚠️ Raw MACD reads <b>{raw_signal}</b> while deseasonalized MACD reads <b>{ds_signal}</b> — "
                f"current momentum may be partly (or mostly) seasonal timing rather than a genuine trend break."
            )
        else:
            divergence_note = (
                f"✅ Raw and deseasonalized MACD agree (<b>{raw_signal}</b>) — this signal looks like it holds up "
                f"beyond normal seasonal patterns."
            )
        section_card("Seasonality Check", divergence_note)

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
    st.markdown(f"<div class='pa-card fade-in'><h3>Market HeatMap</h3><div class='muted'>Buy-the-dip / sell-the-peak signals — a confirmed decline alone doesn't mean sell; it only means SELL THE PEAK if price is still near its own highs</div></div>", unsafe_allow_html=True)
    st.markdown("")

    with st.expander("📖 New here? How to read these signals (with examples)", expanded=False):
        st.markdown(f"""
<div class='pa-card fade-in' style='margin-bottom:14px;'>
<div style='font-weight:900; font-size:16px; margin-bottom:6px;'>This is built to buy low and sell high — not to follow the trend</div>
<div style='line-height:1.6;'>
Most momentum indicators say "it's declining, sell" and "it's rising, buy" — that's trend-following logic, and it's the opposite of what a collector usually wants, which is to sell <i>into</i> strength and buy <i>into</i> weakness. So this page checks four things and specifically uses the last one to keep a decline from being called SELL unless it's actually happening near a peak:<br><br>
<b>1. Momentum (raw MACD)</b> — is the category heating up or cooling off right now? Is the recent trend accelerating or decelerating?<br><br>
<b>2. Is that momentum real? (deseasonalized MACD)</b> — a lot of card categories spike every year around the same time (new set drops, holidays, restocks). We strip that out and check whether the momentum still holds up without it.<br><br>
<b>3. Where's it likely headed? (Holt-Winters forecast)</b> — projects prices forward based on historical trend + seasonal pattern. Used mainly for LONG-TERM HOLD — catching "short-term looks bad but this has legs" before the price action alone would show it.<br><br>
<b>4. Is it AT an extreme? (Overextension)</b> — price vs. its own trailing average. This is the key piece: it's what tells the model whether momentum is happening near a peak (sell zone), near a trough (buy zone), or somewhere in the boring middle (no clean call either way, so it says WATCH instead of guessing).
</div>
</div>
""", unsafe_allow_html=True)

        examples = [
            ("SELL THE PEAK", signal_badge("SELL THE PEAK"),
             "A card is 35% above its own trailing 6-month average and still climbing — momentum hasn't turned down yet, it's still reading High Up. Umbreon ex was in exactly this zone before it topped at $217 and slid to $182.",
             "This fires on the stretch itself, not on a confirmed reversal — waiting for momentum to visibly roll over before calling it a peak means you'd only ever sell after the drop has already started. Selling into the last leg of strength, while it's still going up, is the actual sell-at-the-peak move."),
            ("BUY THE DIP", signal_badge("BUY THE DIP"),
             "A card has fallen well below its own trailing average, but downside momentum has stopped accelerating — it's no longer making sharper lower lows, just drifting. That stabilization pattern is what separates a bottom from a falling knife.",
             "This is the actual buy-the-low signal. Note it does NOT require the long-term forecast to already agree — Holt-Winters extrapolates the recent trend, so it structurally lags a turn. Waiting for the forecast to confirm would mean missing the bottom entirely."),
            ("WATCH", signal_badge("WATCH"),
             "A category is down for a few months — real, confirmed weakness, not just seasonal softness — but it hasn't fallen far enough below its own average to call it cheap, and it isn't sitting near a recent high either. It's just... down, in the middle of its range.",
             "This is the important one: the model deliberately does NOT call this a SELL. A mid-range decline with no peak behind it and no clear bottom yet is exactly the kind of move a buy-low/sell-high collector shouldn't chase in either direction — there's no edge here yet."),
            ("CAUTION", signal_badge("CAUTION"),
             "A card has mild positive momentum and hasn't run far above its own trailing average — nowhere near overextended — but the long-term Holt-Winters forecast is already projecting a meaningful decline over the coming months.",
             "This is a different flag than SELL THE PEAK: the price hasn't gotten stretched, but the long-range model sees trouble ahead anyway. Not a sell signal on its own, but a reason not to add to a position here."),
            ("LONG-TERM HOLD", signal_badge("LONG-TERM HOLD"),
             "A card has cooled off short-term and hasn't technically reached \"oversold\" by the price-average check yet, but the long-term Holt-Winters forecast still projects meaningful upside over the next several months.",
             "This is the \"don't panic-sell the dip on a personal collection\" flag — a longer-view companion to BUY THE DIP for cases the price-position check alone hasn't caught yet."),
            ("BUY", signal_badge("BUY"),
             "A category is climbing, that climb holds up after removing seasonality, the forecast isn't fighting it, and — critically — price hasn't run far above its own trailing average yet.",
             "A genuine early-to-mid trend buy, not a chase. If this same momentum showed up after price had already run 25%+ above its trailing average, it would flip straight to SELL THE PEAK instead, not this."),
        ]
        for name, badge, example, meaning in examples:
            st.markdown(f"""
<div class='pa-card fade-in' style='margin-bottom:10px;'>
<div style='margin-bottom:8px;'>{badge}</div>
<div style='margin-bottom:6px;'><b>Illustrative example:</b> {example}</div>
<div class='muted'><b>What it means for a collector:</b> {meaning}</div>
</div>
""", unsafe_allow_html=True)

        st.markdown("<div class='muted'>These are illustrative walkthroughs of the logic, not live calls on specific cards — check the Signal Table below for what the model is actually flagging today.</div>", unsafe_allow_html=True)

    st.markdown("")
    c1, c2, c3 = st.columns(3)
    with c1:
        hw_horizon = st.slider("Long-term forecast horizon (months)", 6, 24, 12, step=1, key="heat_hw_horizon")
    with c2:
        hw_threshold = st.slider("Long-term bullish/bearish threshold (%)", 3.0, 25.0, 8.0, step=0.5, key="heat_hw_threshold")
    with c3:
        heat_deseason_method = st.radio("Deseasonalizing method", ["Ratio (multiplicative)", "Difference (additive)"], index=0, horizontal=True, key="heat_deseason_method")
    heat_method_key = "ratio" if heat_deseason_method.startswith("Ratio") else "diff"

    c4, c5 = st.columns(2)
    with c4:
        overextension_window = st.slider("Overextension lookback (months)", 3, 12, 6, step=1, key="heat_overext_window")
    with c5:
        overextension_threshold = st.slider("Overextension threshold (% above trailing avg)", 10.0, 60.0, 25.0, step=1.0, key="heat_overext_threshold")

    with st.spinner("Scoring momentum + long-term forecasts across categories..."):
        heat_rows = [
            compute_signal_snapshot(
                df_raw, c, hw_horizon=hw_horizon, hw_threshold=hw_threshold, deseason_method=heat_method_key,
                overextension_window=overextension_window, overextension_threshold=overextension_threshold,
            )
            for c in CATEGORIES
        ]
    heat_df = pd.DataFrame(heat_rows)
    hw_col = f"{hw_horizon}-Mo HW %"

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card("Buy signals", str(int(heat_df["Signal"].isin(["BUY", "BUY THE DIP"]).sum())), "BUY + BUY THE DIP")
    with k2:
        kpi_card("Sell the peak", str(int((heat_df["Signal"] == "SELL THE PEAK").sum())))
    with k3:
        kpi_card("Long-term hold candidates", str(int((heat_df["Signal"] == "LONG-TERM HOLD").sum())))
    with k4:
        kpi_card("Watch / Caution", str(int(heat_df["Signal"].isin(["WATCH", "CAUTION"]).sum())))

    st.markdown("")
    st.markdown("#### Price Position vs. Momentum")
    color_map = {
        "BUY": THEME["primary"], "BUY THE DIP": THEME["primary"],
        "SELL THE PEAK": "#4B5563", "WATCH": THEME["secondary"],
        "CAUTION": "#8E8E93", "LONG-TERM HOLD": THEME["primary"],
    }
    symbol_map = {
        "BUY": "circle", "BUY THE DIP": "diamond",
        "SELL THE PEAK": "circle", "WATCH": "circle",
        "CAUTION": "circle-open", "LONG-TERM HOLD": "diamond-open",
    }
    fig_q = go.Figure()
    for sig_name in ["BUY THE DIP", "LONG-TERM HOLD", "BUY", "WATCH", "CAUTION", "SELL THE PEAK"]:
        sub = heat_df[heat_df["Signal"] == sig_name]
        if sub.empty:
            continue
        fig_q.add_trace(go.Scatter(
            x=sub["Overextension %"], y=sub["Raw Score"], mode="markers+text", text=sub["Category"], textposition="top center",
            marker=dict(size=15, color=color_map.get(sig_name, THEME["muted"]), symbol=symbol_map.get(sig_name, "circle"), line=dict(width=1.5, color=THEME["text"])),
            name=sig_name,
        ))
    fig_q.add_vline(x=overextension_threshold, line_dash="dot", opacity=0.6, line_color=THEME["border"])
    fig_q.add_vline(x=-overextension_threshold, line_dash="dot", opacity=0.6, line_color=THEME["border"])
    fig_q.add_hline(y=0, line_dash="dash", opacity=0.6, line_color=THEME["border"])
    fig_q.update_layout(
        title="Price Position (vs. its own trailing average) vs. Momentum",
        xaxis_title=f"Overextension % (price vs. trailing {overextension_window}-mo average)",
        yaxis_title="Momentum score (raw MACD bucket)",
    )
    fig_q.update_yaxes(range=[-4, 4], tickvals=[-3, -2, -1, 1, 2, 3], ticktext=["High Down", "Med Down", "Low Down", "Low Up", "Med Up", "High Up"])
    apply_fig_theme(fig_q, height=460, slide_mode=slide_mode)
    st.plotly_chart(fig_q, use_container_width=True, theme="streamlit")
    section_card(
        "Reading this chart",
        "<div><b>Anywhere right of the dotted line:</b> overextended — SELL THE PEAK, whether momentum is still climbing or has already turned down. It doesn't wait for a confirmed reversal, since that confirmation only ever arrives after the price has started dropping.</div>"
        "<div style='margin-top:6px;'><b>Left of the dotted line + at Low Down or better:</b> oversold with downside momentum no longer accelerating — BUY THE DIP.</div>"
        "<div style='margin-top:6px;'><b>Left of the dotted line + still deep negative momentum:</b> oversold but still falling hard — a falling knife, not a confirmed bottom yet, so this stays WATCH.</div>"
        "<div style='margin-top:6px;'><b>Between the two dotted lines:</b> not stretched enough to call a peak, not cheap enough to call a low — WATCH, regardless of which way momentum currently points. LONG-TERM HOLD and CAUTION can still appear here — both depend on the long-range forecast, a third dimension this chart doesn't plot.</div>"
    )

    st.markdown("")
    st.markdown("#### Signal Changes Since Last Snapshot")
    snapshot_history = load_snapshot_history()
    if snapshot_history.empty:
        st.markdown("<div class='muted'>No saved snapshot yet — click \"Save today's snapshot\" below to start tracking week-over-week signal changes.</div>", unsafe_allow_html=True)
    else:
        last_date = snapshot_history["snapshot_date"].max()
        last_snap = snapshot_history[snapshot_history["snapshot_date"] == last_date].set_index("Category")["Signal"]
        changes = []
        for _, row in heat_df.iterrows():
            prev = last_snap.get(row["Category"])
            if prev is not None and prev != row["Signal"]:
                changes.append({"Category": row["Category"], "Previous": prev, "Current": row["Signal"]})
        if changes:
            chg_df = pd.DataFrame(changes)
            chg_df["Previous"] = chg_df["Previous"].apply(signal_badge)
            chg_df["Current"] = chg_df["Current"].apply(signal_badge)
            st.markdown(f"<div class='muted'>Since {last_date:%b %d, %Y}:</div>", unsafe_allow_html=True)
            st.markdown(chg_df.to_html(escape=False, index=False), unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='muted'>No signal changes since the last snapshot ({last_date:%b %d, %Y}).</div>", unsafe_allow_html=True)

    if st.button("📌 Save today's snapshot", key="heat_save_snapshot"):
        save_snapshot(heat_df)
        st.success("Snapshot saved — future visits will show changes against this.")

    st.markdown("")
    st.markdown("#### Signal Table")
    display_df = heat_df.copy()
    display_df["Raw MACD"] = display_df["Raw Bucket"].apply(bucket_badge)
    display_df["Deseasonalized MACD"] = display_df["Deseasonalized Bucket"].apply(bucket_badge)
    display_df["Signal"] = display_df["Signal"].apply(signal_badge)
    display_df[hw_col] = display_df[hw_col].apply(lambda v: "—" if pd.isna(v) else f"{v:+.1f}%")
    display_df["Overextension %"] = display_df["Overextension %"].apply(lambda v: "—" if pd.isna(v) else f"{v:+.0f}%")
    display_df = display_df[["Category", "Raw MACD", "Deseasonalized MACD", hw_col, "Overextension %", "Signal", "Why"]]
    st.markdown(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)

    st.markdown("")
    st.markdown("<div class='muted'>*Built to buy low and sell high, not to follow the trend — a confirmed decline only becomes SELL THE PEAK if price is still elevated near its own recent highs; otherwise it's WATCH. Blends raw MACD (momentum), a deseasonalized MACD (confirms it's not just calendar timing), an overextension check (price vs. its own trailing average — what actually identifies a peak or a trough), and a Holt-Winters forecast (long-term direction, used for LONG-TERM HOLD). Educational analytics, not financial advice.</div>", unsafe_allow_html=True)

    csv = heat_df.drop(columns=["Confirmed"]).to_csv(index=False).encode("utf-8")
    st.download_button("Download signal table CSV", data=csv, file_name="market_heatmap_signals.csv", mime="text/csv", key="heat_csv_download")
elif page == "State of Market":
    st.markdown(f"<div class='pa-card fade-in'><h3>State of Market</h3><div class='muted'>YoY vs 3-Mo momentum by category</div></div>", unsafe_allow_html=True)
    st.markdown("")
    explainer_expander(
        "this page",
        "Comparing the long view against the recent view, side by side",
        "Each category gets two bars: <b>YoY %</b> (change over the last 12 months) and <b>3-Mo %</b> (change over the last quarter). Reading them together tells you more than either alone — "
        "when both bars point the same direction, that's a confirmed trend. When they disagree, something's shifting.",
        [
            {
                "example": "A category shows a strongly positive YoY bar but a negative 3-Mo bar.",
                "meaning": "It had a great year overall, but has cooled off recently — worth checking if that's normal seasonal softness or an actual trend change before assuming the good year continues."
            },
            {
                "example": "A category shows a negative YoY bar but a strongly positive 3-Mo bar.",
                "meaning": "It's been a rough year, but something's turned recently — this is the kind of early-recovery pattern worth digging into further on the Category Analysis page before deciding it's real."
            },
        ],
    )
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
    fig.add_hline(y=0, line_dash="dash", opacity=0.6, line_color=THEME["border"])
    fig.update_layout(barmode="group", title=f"YoY vs 3-Mo Momentum (through {latest:%b %Y})", xaxis_title="Category", yaxis_title="% change")
    apply_fig_theme(fig, height=420, slide_mode=slide_mode)
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    st.table(mkt.sort_values("YoY %", ascending=False))
elif page == "Custom Index Builder":
    st.markdown(f"<div class='pa-card fade-in'><h3>Custom Index Builder</h3><div class='muted'>Blend categories into a custom market index</div></div>", unsafe_allow_html=True)
    st.markdown("")
    explainer_expander(
        "this page",
        "Build your own market line instead of tracking categories one at a time",
        "Pick the categories you actually care about — the ones that reflect your own collection or watchlist — assign each a weight, and this blends them into one custom-weighted index line over time. "
        "It's the same idea as a stock index fund: instead of watching ten separate category charts, you get one line that moves the way your specific mix moves.",
        [
            {
                "example": "Your collection is mostly Pokémon with a smaller Magic the Gathering position, so you weight it 70/30 instead of looking at the two category charts separately.",
                "meaning": "The resulting index line shows how your actual mix has performed — which can look quite different from either category alone if they move in different directions at different times."
            },
        ],
    )
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
    explainer_expander(
        "this heatmap",
        "When does each category historically run hot or cold?",
        "Every cell is the average month-over-month % change for that category in that calendar month, across all history. Red cells are months that have historically shown strong average gains; grey/black cells are months that have historically been flat or negative. "
        "This is the seasonal baseline the rest of the app checks momentum against — a category spiking in a historically-red month is less surprising than the same spike in a historically-grey one.",
        [
            {
                "example": "A TCG category shows consistent red cells every October, tied to annual new-set releases.",
                "meaning": "An October price jump in that category is expected, not necessarily a new trend — that's exactly why the Category Analysis and Market HeatMap pages check momentum against a deseasonalized version before calling it real."
            },
            {
                "example": "A category shows red in a month with no obvious release calendar tied to it.",
                "meaning": "That's worth a second look — an unexplained seasonal pattern might point to something structural (tax season liquidity, holiday gifting, etc.) worth knowing about rather than ignoring."
            },
        ],
    )
    st.markdown("")
    df_tmp = df_raw.copy()
    df_tmp["Month_Num"] = df_tmp["Month_Year"].dt.month
    wide = df_tmp.pivot_table(values="market_value", index="Category", columns="Month_Num", aggfunc="mean").reindex(index=CATEGORIES)
    pct = (wide.pct_change(axis=1) * 100).round(2)
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig = go.Figure(data=go.Heatmap(z=pct.values, x=month_labels, y=pct.index.tolist(), colorscale=HEATMAP_SCALE, zmin=-20, zmax=20))
    fig.update_layout(title="Seasonality — Avg MoM % Change", xaxis_title="Month", yaxis_title="Category")
    apply_fig_theme(fig, height=420, slide_mode=slide_mode)
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    st.table(pct.fillna("—"))
elif page == "Rolling Volatility":
    st.markdown(f"<div class='pa-card fade-in'><h3>Rolling Volatility</h3><div class='muted'>Coefficient of variation over time</div></div>", unsafe_allow_html=True)
    st.markdown("")
    explainer_expander(
        "this page",
        "How bumpy is the ride right now, compared to normal?",
        "This tracks the <b>coefficient of variation</b> — basically, how much prices are swinging relative to their average level — over a short window and a long window at the same time. "
        "When the short window is above the long window, volatility is currently rising above its normal level. When it's below, things have calmed down relative to the recent past.",
        [
            {
                "example": "The short-window line spikes well above the long-window line right after a big news event or set launch.",
                "meaning": "That's the market digesting new information — prices are less predictable right now, which matters if you're timing an entry or exit, not just picking direction."
            },
            {
                "example": "Both lines are low and close together for an extended stretch.",
                "meaning": "This category has been trading calmly — a more forgiving environment to buy or sell into without getting caught by a sudden swing."
            },
        ],
    )
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
    explainer_expander(
        "this heatmap",
        "Do these categories move together, or independently?",
        "Each cell shows how closely two categories' price movements track each other, from -1 (they move in exact opposite directions) to +1 (they move in perfect lockstep). Red means high positive correlation, grey/black means low or negative correlation. "
        "This matters for diversification: owning several highly-correlated categories isn't really spreading your risk, since they'll likely all dip or spike together.",
        [
            {
                "example": "Pokémon and Magic the Gathering show a high correlation reading.",
                "meaning": "Owning a lot of both doesn't diversify you much — a downturn hitting one is likely to hit the other around the same time, since whatever's driving the move (broad TCG demand, tariffs, etc.) probably affects both."
            },
            {
                "example": "Two categories show a low or negative correlation reading.",
                "meaning": "These tend to move somewhat independently — pairing them can smooth out your overall swings better than doubling down on categories that move in lockstep."
            },
        ],
    )
    st.markdown("")
    wide = df_raw.pivot_table(values="market_value", index="Month_Year", columns="Category", aggfunc="mean").sort_index()[CATEGORIES]
    basis = st.radio("Correlation basis", ["Monthly returns (pct_change)", "Levels (raw index)"], index=0, horizontal=True)
    mat = wide.pct_change().dropna() if basis.startswith("Monthly") else wide.dropna()
    corr = mat.corr()
    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(), zmin=-1, zmax=1, colorscale=HEATMAP_SCALE))
    fig.update_layout(title="Correlation Heatmap", xaxis_title="Category", yaxis_title="Category")
    apply_fig_theme(fig, height=520, slide_mode=slide_mode)
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    st.table(corr.round(2))
elif page == "Flip Forecast":
    st.markdown(f"<div class='pa-card fade-in'><h3>Flip Forecast</h3><div class='muted'>Monte Carlo projection based on category history</div></div>", unsafe_allow_html=True)
    st.markdown("")
    explainer_expander(
        "this page",
        "What are the realistic odds of hitting your asking price?",
        "This runs thousands of simulated price paths by resampling that category's own historical monthly moves — including its occasional big spikes, not just an average bell-curve move — then shows the median path plus a 10th-90th percentile band. "
        "The key number is the <b>probability of hitting your asking price</b> within your chosen time horizon, which is a much more honest answer than just eyeballing a trend line.",
        [
            {
                "example": "You bought a card at $75, you're asking $100, and the simulation shows only a 20% chance of hitting $100 within 12 months given this category's typical swings.",
                "meaning": "That's a signal to either lower your ask, extend your timeline, or accept you're holding for a lower-probability outcome — not a reason to panic, just useful information for setting expectations."
            },
            {
                "example": "The probability of hitting your target comes back above 70%.",
                "meaning": "Your asking price is well within this category's normal range of outcomes — a reasonable, achievable target rather than a stretch goal."
            },
        ],
    )
    st.markdown("")
    sim_category = st.selectbox("Choose Category for Forecast", CATEGORIES, index=CATEGORIES.index(cat1))
    d = preprocess(df_raw, sim_category).sort_values("Month_Year")
    d["pct_change"] = d["market_value"].pct_change()
    d = d.dropna()
    historical_returns = d["pct_change"].to_numpy()
    expected_return = float(np.mean(historical_returns)) if len(historical_returns) else 0.0
    monthly_volatility = min(max(float(np.std(historical_returns)) if len(historical_returns) else 0.05, 0.01), 0.30)
    asking_price = st.number_input("Your Asking Price ($)", min_value=0.0, value=100.0, step=1.0)
    purchase_price = st.number_input("Your Purchase Price ($)", min_value=0.0, value=75.0, step=1.0)
    num_months = st.slider("Horizon (months)", 6, 36, 12, step=1)
    num_simulations = st.slider("Number of Simulations", 200, 20000, 2000, step=200)
    rng = np.random.default_rng()

    # Bootstrap-resample actual historical monthly returns instead of assuming a Normal
    # distribution — card prices have fatter tails than a bell curve (an occasional
    # auction-headline-driven spike isn't well represented by mean/std alone). Falls
    # back to Normal only when there's too little history to resample meaningfully.
    use_bootstrap = len(historical_returns) >= 6
    if use_bootstrap:
        draws = rng.choice(historical_returns, size=(num_simulations, num_months), replace=True)
    else:
        draws = rng.normal(expected_return, monthly_volatility, size=(num_simulations, num_months))

    start_value = float(d["market_value"].tail(3).mean())
    growth = np.cumprod(1 + draws, axis=1)
    results = np.empty((num_simulations, num_months + 1), dtype=float)
    results[:, 0] = start_value
    results[:, 1:] = start_value * growth
    months_ahead = np.arange(num_months + 1)
    p10 = np.percentile(results, 10, axis=0)
    p50 = np.percentile(results, 50, axis=0)
    p90 = np.percentile(results, 90, axis=0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=months_ahead, y=p50, mode="lines", name="Median", line=dict(color=THEME["primary"], dash="dash", width=3)))
    fig.add_trace(go.Scatter(x=np.concatenate([months_ahead, months_ahead[::-1]]), y=np.concatenate([p90, p10[::-1]]), fill="toself", fillcolor="rgba(228, 0, 43, 0.10)", line=dict(color="rgba(0,0,0,0)"), name="10–90% band", hoverinfo="skip"))
    fig.add_hline(y=asking_price, line_dash="dot", opacity=0.8, line_color=THEME["accent_red"])
    fig.update_layout(title=f"Flip Forecast — {sim_category}", xaxis_title="Months Ahead", yaxis_title="Simulated Price ($)")
    apply_fig_theme(fig, height=420, slide_mode=slide_mode)
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    final_prices = results[:, -1]
    prob_hit = float(np.mean(final_prices >= asking_price) * 100)
    method_label = "Bootstrap (resampled from actual monthly moves)" if use_bootstrap else "Normal distribution (fallback — insufficient history to bootstrap)"
    st.table(pd.DataFrame({"Metric": ["Simulation method", "Historical avg monthly return", "Historical monthly volatility", "Probability Asking Price Hit"], "Value": [method_label, f"{expected_return:.2%}", f"{monthly_volatility:.2%}", f"{prob_hit:.1f}%"]}))
elif page == "Raw vs Grade Decision Engine":
    render_raw_vs_grade_engine()
elif page == "Liquidity + Exit Risk Monitor":
    render_liquidity_exit_monitor()
elif page == "Portfolio Allocator":
    render_portfolio_allocator()

st.markdown("""
---
**Cardboard Compass** — built by Pancake Analytics LLC 
*Analytics read, not financial advice.*
""")

if slide_mode:
    st.markdown("</div>", unsafe_allow_html=True)