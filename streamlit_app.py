import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────
st.image(
    "https://pancakebreakfaststats.com/wp-content/uploads/2025/04/pa001.png",
    use_container_width=True
)
st.title("Cardboard Bot")

st.markdown("""
Cardboard Bot turns eBay sales data into clear, visual insights powered by **Pancake Analytics**.  
Collecting doesn’t *need* analytics—but if you’re the kind of hobbyist who loves the numbers,
this app helps you spot trends, entry windows, and market shifts in seconds.
""")

# ─────────────────────────────────────────
#  CONSTANTS & COHORT RULES
# ─────────────────────────────────────────
CATEGORIES = [
    "Fortnite", "Marvel", "Pokemon", "Star Wars", "Magic the Gathering",
    "Baseball", "Basketball", "Football", "Hockey", "Soccer"
]

RULES = {
    "collector": {
        "High Up": ("No", "Yes"),  "Med Up": ("No", "Yes"),
        "Low Up" : ("Yes", "No"),  "Low Down": ("Yes", "No"),
        "Med Down":("No", "Yes"),  "High Down":("Yes", "No")
    },
    "flipper": {
        "High Up": ("Yes", "No"),  "Med Up": ("Yes", "No"),
        "Low Up" : ("Yes", "No"),  "Low Down":("No",  "Yes"),
        "Med Down":("No", "Yes"),  "High Down":("No",  "Yes")
    },
    "investor": {
        "High Up": ("No", "Yes"),  "Med Up": ("Yes", "No"),
        "Low Up" : ("Yes", "No"),  "Low Down":("No",  "Yes"),
        "Med Down":("No", "Yes"),  "High Down":("Yes","No")
    }
}

# ─────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────
def preprocess(df, category):
    d = df[df["Category"] == category].copy()
    d["Month_Year"] = pd.to_datetime(d["Month"] + " " + d["Year"].astype(str))
    d = (d.sort_values("Month_Year")
           .groupby("Month_Year")["market_value"]
           .mean()
           .reset_index())
    d["market_value"] = d["market_value"].astype(float)
    return d

def forecast(df):
    model = ExponentialSmoothing(
        df["market_value"], trend="add", seasonal="add", seasonal_periods=12
    ).fit()

    fcast = model.forecast(12)
    ci    = 1.96 * np.std(model.resid)
    dates = pd.date_range(
        df["Month_Year"].iloc[-1] + pd.DateOffset(months=1),
        periods=12, freq="MS"
    )
    return (
        pd.DataFrame({
            "Date":     dates,
            "Forecast": fcast,
            "Upper":    fcast + ci,
            "Lower":    fcast - ci
        }),
        model
    )

def macd_analysis(df):
    short  = df["market_value"].ewm(span=12, adjust=False).mean()
    long   = df["market_value"].ewm(span=26, adjust=False).mean()
    macd   = short - long
    signal = macd.ewm(span=9, adjust=False).mean()
    diff   = macd - signal

    bucket = pd.cut(
        diff,
        bins=[-np.inf, -1.5, -0.5, 0, 0.5, 1.5, np.inf],
        labels=["High Down", "Med Down", "Low Down",
                "Low Up",   "Med Up",   "High Up"]
    )

    return pd.DataFrame({
        "Month_Year": df["Month_Year"],
        "MACD":       macd,
        "Signal":     signal,
        "MACD_Bucket": bucket
    })

def best_month(df):
    df["Month"] = df["Month_Year"].dt.month_name()
    return df.groupby("Month")["market_value"].mean().sort_values()

def yoy_and_roll3(series, latest_month):
    today     = series.get(latest_month, np.nan)
    year_ago  = series.get(latest_month - pd.DateOffset(years=1), np.nan)
    three_ago = series.get(latest_month - pd.DateOffset(months=3), np.nan)

    yoy   = np.nan if np.isnan(today) or np.isnan(year_ago)  else (today - year_ago)  / year_ago  * 100
    roll3 = np.nan if np.isnan(today) or np.isnan(three_ago) else (today - three_ago) / three_ago * 100
    return yoy, roll3

# ─────────────────────────────────────────
#  UI CONTROLS
# ─────────────────────────────────────────
cat1 = st.selectbox("Select your primary category", CATEGORIES)
cat2 = st.selectbox(
    "Compare against another category (optional)",
    ["None"] + [c for c in CATEGORIES if c != cat1]
)
run_btn = st.button("Run Analysis")

# ─────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────
tab_analysis, tab_heatmap, tab_market = st.tabs(
    ["Category Analysis - What's the current and future outlook?", "Current Market HeatMap - Time to Buy?", "State of Market - Year over Year and Rolling 3 Months"]
)

df_raw = None
if run_btn:
    with st.spinner("Loading eBay data & crunching numbers…"):
        DATA_URL = "https://pancakebreakfaststats.com/wp-content/uploads/2025/06/data_file_006.xlsx"
        df_raw   = pd.read_excel(DATA_URL)

# ─────────────────────────────────────────
#  TAB 1 – CATEGORY ANALYSIS
# ─────────────────────────────────────────
with tab_analysis:
    if not run_btn:
        st.info("Click **Run Analysis** to generate charts and forecasts.")
    else:
        def display_analytics(cat):
            st.header(f"Analytics – {cat}")
            d = preprocess(df_raw, cat)

            # ── Forecast plot
            f_df, _ = forecast(d)
            pct = (f_df["Forecast"].iloc[-1] - d["market_value"].iloc[-1]) \
                  / d["market_value"].iloc[-1] * 100

            fig1, ax1 = plt.subplots()
            ax1.plot(d["Month_Year"], d["market_value"], label="Historical")
            ax1.plot(f_df["Date"],     f_df["Forecast"],  label="Forecast")
            ax1.fill_between(f_df["Date"], f_df["Lower"], f_df["Upper"], alpha=0.2)
            ax1.legend(); ax1.set_title("12-Month Forecast")
            st.pyplot(fig1)
            st.markdown(f"**Projected change by {f_df['Date'].iloc[-1]:%b %Y}: {pct:.2f}%**")

            # ── MACD trend plot
            macd_df = macd_analysis(d)
            bucket  = macd_df["MACD_Bucket"].iloc[-1]

            fig2, ax2 = plt.subplots()
            ax2.plot(macd_df["Month_Year"], macd_df["MACD"],   label="MACD")
            ax2.plot(macd_df["Month_Year"], macd_df["Signal"], label="Signal")
            ax2.set_title("MACD Trend"); ax2.legend()
            st.pyplot(fig2)
            st.markdown(f"**Current MACD Bucket:** {bucket}")

            # ── Seasonality
            bm = best_month(d)
            st.markdown(f"**Best buying month:** {bm.idxmin()} (avg value {bm.min():.2f})")

            fig3, ax3 = plt.subplots()
            bm.plot(kind="bar", ax=ax3)
            ax3.set_ylabel("Avg Market Value"); ax3.set_title("Average Value by Month")
            st.pyplot(fig3)

        if cat2 == "None":
            display_analytics(cat1)
        else:
            c1, c2 = st.columns(2)
            with c1: display_analytics(cat1)
            with c2: display_analytics(cat2)
            st.subheader("Side-by-side comparison complete.")

# ─────────────────────────────────────────
#  TAB 2 – MARKET HEATMAP
# ─────────────────────────────────────────
with tab_heatmap:
    if not run_btn:
        st.info("Run the analysis to view the MACD Market HeatMap.")
    else:
        rows = []
        for cat in CATEGORIES:
            d       = preprocess(df_raw, cat)
            bucket  = macd_analysis(d)["MACD_Bucket"].iloc[-1]
            rows.append({
                "Category"        : cat,
                "MACD Bucket"     : bucket,
                "Collector Buy?"  : RULES["collector"][bucket][0],
                "Collector Sell?" : RULES["collector"][bucket][1],
                "Flipper Buy?"    : RULES["flipper"][bucket][0],
                "Flipper Sell?"   : RULES["flipper"][bucket][1],
                "Investor Buy?"   : RULES["investor"][bucket][0],
                "Investor Sell?"  : RULES["investor"][bucket][1]
            })

        heat_df = pd.DataFrame(rows)
        st.dataframe(heat_df, use_container_width=True)

        st.markdown("""
        **How to read this table**

        * **Buy = Yes** → favorable entry conditions for that persona.  
        * **Sell = Yes** → consider trimming or flipping inventory.  

        **Cohort definitions**  
        • **Collector —** long-term enthusiast building a personal collection; focuses on rarity & cost basis.  
        • **Flipper —** short-term trader (weeks → months) riding hype & liquidity for quick profit.  
        • **Investor —** portfolio-minded holder seeking multi-month trend exposure with risk control.

        Mapping is based on Pancake Analytics’ MACD play-book (guidance — not financial advice).
        """)

# ─────────────────────────────────────────
#  TAB 3 – STATE OF MARKET
# ─────────────────────────────────────────
with tab_market:
    if not run_btn:
        st.info("Run the analysis to generate the State of Market report.")
    else:
        df_raw["Month_Year"] = pd.to_datetime(df_raw["Month"] + " " + df_raw["Year"].astype(str))
        latest = df_raw["Month_Year"].max()

        yoy_vals, r3_vals = [], []
        for cat in CATEGORIES:
            s = (df_raw[df_raw["Category"] == cat]
                 .groupby("Month_Year")["market_value"]
                 .mean())
            yoy, r3 = yoy_and_roll3(s, latest)
            yoy_vals.append(yoy); r3_vals.append(r3)

        mkt_df = pd.DataFrame({
            "Category": CATEGORIES,
            "YoY %"   : yoy_vals,
            "3-Mo %"  : r3_vals
        }).set_index("Category")

        # YoY bar chart
        st.subheader(f"Year-over-Year % Change (to {latest:%b %Y})")
        fig_y, ax_y = plt.subplots()
        mkt_df["YoY %"].plot(kind="bar", ax=ax_y)
        ax_y.axhline(0, color="gray", lw=0.8, ls="--")
        ax_y.set_ylabel("%"); st.pyplot(fig_y)

        # Rolling 3-month bar chart
        st.subheader("Rolling 3-Month % Change")
        fig_r, ax_r = plt.subplots()
        mkt_df["3-Mo %"].plot(kind="bar", color="orange", ax=ax_r)
        ax_r.axhline(0, color="gray", lw=0.8, ls="--")
        ax_r.set_ylabel("%"); st.pyplot(fig_r)

        # Data table
        st.dataframe(mkt_df.round(2), use_container_width=True, height=350)

        st.markdown(f"""
        **How to read these charts**

        • **Above 0 %** = prices higher than the comparison period  
          • *YoY %* compares to the same month last year.  
          • *3-Mo %* compares to three months ago.  

        • **Below 0 %** = prices lower than the comparison period.  

        • Categories showing **both positive YoY and positive 3-Mo gains** are heating up.  
        • Categories with **both negative values** may be cooling and could offer value opportunities.

        Use these signals alongside population reports, release calendars, and your own collecting goals.
        """)

# ─────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────
st.markdown("""
---
### Thank You for Using Cardboard Bot  
Built by **Pancake Analytics LLC** to help collectors dive deeper into their hobby.  
All data is sourced from eBay and analyzed for trends and opportunities.  
_This is an analytics read, not financial advice._
""")
