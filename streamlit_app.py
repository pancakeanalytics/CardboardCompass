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
Collecting doesn’t *need* analytics—but if you love the numbers, this app helps you spot trends,
entry windows, and market shifts in seconds.
            
*The market-index value you see here is **not** a product price; it’s the average selling
price of individual cards (“singles”) on eBay, **weighted for the number of sellers
and total items sold** to keep high-volume categories from skewing the picture. Cardboard Bot focuses on **macro** market trends—so while it can signal overall heat
or cool-offs, it won’t provide a precise valuation for the single card you’re holding
in your hand.*
""")

# ─────────────────────────────────────────
#  DATA LOADING (runs once per session)
# ─────────────────────────────────────────
@st.cache_data
def load_data(url: str) -> pd.DataFrame:
    return pd.read_excel(url)

DATA_URL = "https://pancakebreakfaststats.com/wp-content/uploads/2025/06/data_file_006.xlsx"

if "df_raw" not in st.session_state:
    if st.button("Run Analysis"):
        st.session_state["df_raw"] = load_data(DATA_URL)

df_raw: pd.DataFrame | None = st.session_state.get("df_raw")

# ─────────────────────────────────────────
#  CONSTANTS, RULES & HELPERS
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
                "Low Up", "Med Up", "High Up"]
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
#  CATEGORY & COMPARISON DROPDOWNS
# ─────────────────────────────────────────
cat1 = st.selectbox("Select your primary category", CATEGORIES)
cat2 = st.selectbox(
    "Compare against another category (optional)",
    ["None"] + [c for c in CATEGORIES if c != cat1]
)

# ─────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────
tab_analysis, tab_heatmap, tab_market, tab_index = st.tabs(
    ["Category Analysis", "Market HeatMap", "State of Market", "Custom Index Builder"]
)

# ============ TAB 1: CATEGORY ANALYSIS ============
with tab_analysis:
    if df_raw is None:
        st.info("Click **Run Analysis** to generate charts and forecasts.")
    else:
        def display_analytics(cat):
            st.header(f"Analytics – {cat}")
            d = preprocess(df_raw, cat)

            # Forecast
            f_df, _ = forecast(d)
            pct = (f_df["Forecast"].iloc[-1] - d["market_value"].iloc[-1]) / d["market_value"].iloc[-1] * 100

            fig1, ax1 = plt.subplots()
            ax1.plot(d["Month_Year"], d["market_value"], label="Historical")
            ax1.plot(f_df["Date"], f_df["Forecast"], label="Forecast")
            ax1.fill_between(f_df["Date"], f_df["Lower"], f_df["Upper"], alpha=0.2)
            ax1.legend(); ax1.set_title("12-Month Forecast")
            st.pyplot(fig1)
            st.markdown(f"**Projected change by {f_df['Date'].iloc[-1]:%b %Y}: {pct:.2f}%**")

            # MACD Trend
            macd_df = macd_analysis(d)
            bucket = macd_df["MACD_Bucket"].iloc[-1]

            fig2, ax2 = plt.subplots()
            ax2.plot(macd_df["Month_Year"], macd_df["MACD"],   label="MACD")
            ax2.plot(macd_df["Month_Year"], macd_df["Signal"], label="Signal")
            ax2.set_title("MACD Trend"); ax2.legend()
            st.pyplot(fig2)
            st.markdown(f"**Current MACD Bucket:** {bucket}")

            # Seasonality
            bm = best_month(d)
            st.markdown(f"**Best buying month:** {bm.idxmin()} (avg value {bm.min():.2f})")

            fig3, ax3 = plt.subplots()
            bm.plot(kind="bar", ax=ax3)
            ax3.set_ylabel("Avg Market Value"); ax3.set_title("Average Value by Month")
            st.pyplot(fig3)

        if cat2 == "None":
            display_analytics(cat1)
        else:
            col1, col2 = st.columns(2)
            with col1: display_analytics(cat1)
            with col2: display_analytics(cat2)
            st.subheader("Side-by-side comparison complete.")

# ============ TAB 2: MARKET HEATMAP ============
with tab_heatmap:
    if df_raw is None:
        st.info("Run the analysis to view the MACD Market HeatMap.")
    else:
        records = []
        for cat in CATEGORIES:
            d = preprocess(df_raw, cat)
            bucket = macd_analysis(d)["MACD_Bucket"].iloc[-1]
            records.append({
                "Category": cat,
                "MACD Bucket": bucket,
                "Collector Buy?": RULES["collector"][bucket][0],
                "Collector Sell?": RULES["collector"][bucket][1],
                "Flipper Buy?": RULES["flipper"][bucket][0],
                "Flipper Sell?": RULES["flipper"][bucket][1],
                "Investor Buy?": RULES["investor"][bucket][0],
                "Investor Sell?": RULES["investor"][bucket][1]
            })
        st.dataframe(pd.DataFrame(records), use_container_width=True)

        st.markdown("""
**How to read this table**

* **Buy = Yes** → favorable entry conditions for that persona.  
* **Sell = Yes** → consider trimming or flipping inventory.  

**Cohort definitions**  
• **Collector —** long-term enthusiast; focuses on rarity & cost basis.  
• **Flipper —** short-term trader riding hype & liquidity.  
• **Investor —** portfolio-minded holder targeting multi-month trends.

Mapping is based on Pancake Analytics’ MACD play-book (guidance — not financial advice).
""")

# ============ TAB 3: STATE OF MARKET ============
with tab_market:
    if df_raw is None:
        st.info("Run the analysis to generate the State of Market report.")
    else:
        df_raw["Month_Year"] = pd.to_datetime(df_raw["Month"] + " " + df_raw["Year"].astype(str))
        latest = df_raw["Month_Year"].max()

        yoy_vals, r3_vals = [], []
        for cat in CATEGORIES:
            s = df_raw[df_raw["Category"] == cat].groupby("Month_Year")["market_value"].mean()
            yoy, r3 = yoy_and_roll3(s, latest)
            yoy_vals.append(yoy); r3_vals.append(r3)

        mkt_df = pd.DataFrame({
            "Category": CATEGORIES,
            "YoY %": yoy_vals,
            "3-Mo %": r3_vals
        }).set_index("Category")

        st.subheader(f"Year-over-Year % Change (to {latest:%b %Y})")
        fig_y, ax_y = plt.subplots()
        mkt_df["YoY %"].plot(kind="bar", ax=ax_y)
        ax_y.axhline(0, color="gray", lw=0.8, ls="--"); ax_y.set_ylabel("%")
        st.pyplot(fig_y)

        st.subheader("Rolling 3-Month % Change")
        fig_r, ax_r = plt.subplots()
        mkt_df["3-Mo %"].plot(kind="bar", color="orange", ax=ax_r)
        ax_r.axhline(0, color="gray", lw=0.8, ls="--"); ax_r.set_ylabel("%")
        st.pyplot(fig_r)

        st.dataframe(mkt_df.round(2), use_container_width=True, height=340)

        st.markdown("""
**How to read these charts**

• **Above 0 %** = prices higher than the comparison period.  
• *YoY %* compares to the same month last year.  
• *3-Mo %* compares to three months ago.  

• **Below 0 %** = prices lower.  
• Dual positives ⇒ heating up.  Dual negatives ⇒ cooling → possible value buys.
""")

# ============ TAB 4: CUSTOM INDEX BUILDER ============
with tab_index:
    if df_raw is None:
        st.info("Run the analysis first, then design your own index here.")
    else:
        st.subheader("Build your Personal Portfolio Index")

        sel_categories = st.multiselect(
            "Select categories to include",
            CATEGORIES,
            default=["Pokemon", "Magic the Gathering"]
        )
        if not sel_categories:
            st.warning("Select at least one category to begin."); st.stop()

        st.markdown("**Assign weights (sliders auto-normalize):**")
        raw_w = {
            cat: st.slider(f"{cat} weight (%)", 0, 100, 20, 5)
            for cat in sel_categories
        }
        w_series = pd.Series(raw_w, dtype=float)
        if w_series.sum() == 0:
            st.warning("At least one weight must be greater than 0 %."); st.stop()
        weights = w_series / w_series.sum()

        pivot = (df_raw
                 .assign(Month_Year=lambda x: pd.to_datetime(x["Month"] + " " + x["Year"].astype(str)))
                 .pivot_table(values="market_value",
                              index="Month_Year",
                              columns="Category",
                              aggfunc="mean")
                 .sort_index())

        custom_idx   = (pivot[sel_categories] * weights).sum(axis=1)
        bench_poke   = pivot["Pokemon"]
        bench_tcg    = pivot[["Pokemon", "Magic the Gathering"]].mean(axis=1)
        sports_cols  = ["Baseball", "Basketball", "Football", "Soccer", "Hockey"]
        bench_sports = pivot[sports_cols].mean(axis=1)
        nons_cols    = ["Star Wars", "Marvel", "Fortnite"]
        bench_nons   = pivot[nons_cols].mean(axis=1)

        fig_idx, ax_idx = plt.subplots()
        custom_idx.plot(ax=ax_idx, linewidth=2, label="My Custom Index")
        bench_poke.plot(ax=ax_idx, alpha=0.7, label="Pokémon")
        bench_tcg.plot(ax=ax_idx,  alpha=0.7, label="TCGs")
        bench_sports.plot(ax=ax_idx, alpha=0.7, label="Sports")
        bench_nons.plot(ax=ax_idx,  alpha=0.7, label="Non-Sports")
        ax_idx.set_ylabel("Indexed Market Value"); ax_idx.legend(fontsize=8)
        ax_idx.set_title("Custom Index vs. Benchmarks")
        st.pyplot(fig_idx)

        latest = custom_idx.index.max()
        perf_df = pd.DataFrame({
            "Series": ["My Custom", "Pokémon", "TCGs", "Sports", "Non-Sports"],
            "YoY %":  [*map(lambda s: yoy_and_roll3(s, latest)[0],
                            [custom_idx, bench_poke, bench_tcg, bench_sports, bench_nons])],
            "3-Mo %": [*map(lambda s: yoy_and_roll3(s, latest)[1],
                            [custom_idx, bench_poke, bench_tcg, bench_sports, bench_nons])]
        }).set_index("Series").round(2)

        st.markdown(f"### Performance vs. {latest:%b %Y}")
        fig_y, ax_y = plt.subplots()
        perf_df["YoY %"].plot(kind="bar", ax=ax_y)
        ax_y.axhline(0, color="gray", lw=0.8, ls="--"); ax_y.set_ylabel("%")
        ax_y.set_title("Year-over-Year % Change"); st.pyplot(fig_y)

        fig_r, ax_r = plt.subplots()
        perf_df["3-Mo %"].plot(kind="bar", color="orange", ax=ax_r)
        ax_r.axhline(0, color="gray", lw=0.8, ls="--"); ax_r.set_ylabel("%")
        ax_r.set_title("Rolling 3-Month % Change"); st.pyplot(fig_r)

        st.markdown("#### Final Weights (%)")
        st.table(weights.mul(100).round(1).rename("Weight %"))

        st.markdown("#### Performance Table (%)")
        st.table(perf_df)

        st.markdown("""
**How to read the chart**

* **My Custom Index** – the blend you created.  
* Benchmarks: **Pokémon**, **TCGs**, **Sports**, **Non-Sports** (equal-weight blends).
""")

        st.markdown("""
**Collector tips**

• Re-run each month to track your personal market.  
• Rising 3-Mo before YoY often hints at a recovery.  
• If your index drops while a benchmark rises, bargains may be forming.  
• Save screenshots—the slope tells the story.
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
