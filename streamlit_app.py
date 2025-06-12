import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────
#  HEADER
# ──────────────────────────────────────────────────
st.image(
    "https://pancakebreakfaststats.com/wp-content/uploads/2025/04/pa001.png",
    use_container_width=True
)
st.title("Cardboard Bot")

st.markdown("""
Cardboard Bot helps you navigate the market with insights powered by **Pancake Analytics**.  
Collecting doesn’t need analytics – but for those who want a deeper hobby read, this app
turns eBay data into clear, visual insights.
""")

# ──────────────────────────────────────────────────
#  CONSTANTS & HELPERS
# ──────────────────────────────────────────────────
CATEGORIES = [
    'Fortnite', 'Marvel', 'Pokemon', 'Star Wars', 'Magic the Gathering',
    'Baseball', 'Basketball', 'Football', 'Hockey', 'Soccer'
]

def preprocess(df, category):
    d = df[df['Category'] == category].copy()
    d['Month_Year'] = pd.to_datetime(d['Month'] + ' ' + d['Year'].astype(str))
    d = d.sort_values('Month_Year')
    d = d.groupby('Month_Year').agg({'market_value': 'mean'}).reset_index()
    d['market_value'] = d['market_value'].astype(float)
    return d

def forecast(df):
    model = ExponentialSmoothing(
        df['market_value'], trend='add', seasonal='add', seasonal_periods=12
    ).fit()
    fcast = model.forecast(12)
    ci = 1.96 * np.std(model.resid)
    upper = fcast + ci
    lower = fcast - ci
    dates = pd.date_range(
        df['Month_Year'].iloc[-1] + pd.DateOffset(months=1),
        periods=12, freq='MS'
    )
    return pd.DataFrame({'Date': dates, 'Forecast': fcast,
                         'Upper': upper, 'Lower': lower}), model

def macd_analysis(df):
    short = df['market_value'].ewm(span=12, adjust=False).mean()
    long  = df['market_value'].ewm(span=26, adjust=False).mean()
    macd  = short - long
    signal = macd.ewm(span=9, adjust=False).mean()
    diff = macd - signal
    df['MACD_Trend'] = pd.cut(
        diff,
        bins=[-np.inf, -1.5, -0.5, 0, 0.5, 1.5, np.inf],
        labels=['High Down', 'Med Down', 'Low Down', 'Low Up', 'Med Up', 'High Up']
    )
    return df[['Month_Year', 'MACD_Trend']]

def best_month(df):
    df['Month'] = df['Month_Year'].dt.month_name()
    return df.groupby('Month')['market_value'].mean().sort_values()

# Buy/Sell rule-of-thumb mapping
RULES = {
    'collector': {
        'High Up': ('No', 'Yes'),  'Med Up': ('No', 'Yes'),
        'Low Up' : ('Yes','No'),   'Low Down': ('Yes','No'),
        'Med Down':('No','Yes'),   'High Down':('Yes','No')
    },
    'flipper': {
        'High Up': ('Yes','No'),   'Med Up': ('Yes','No'),
        'Low Up' : ('Yes','No'),   'Low Down':('No','Yes'),
        'Med Down':('No','Yes'),   'High Down':('No','Yes')
    },
    'investor': {
        'High Up': ('No','Yes'),   'Med Up': ('Yes','No'),
        'Low Up' : ('Yes','No'),   'Low Down':('No','Yes'),
        'Med Down':('No','Yes'),   'High Down':('Yes','No')
    }
}

# ──────────────────────────────────────────────────
#  UI CONTROLS
# ──────────────────────────────────────────────────
category_1 = st.selectbox(
    "Select your primary category",
    CATEGORIES
)

category_2 = st.selectbox(
    "Compare against another category (optional)",
    ["None"] + [c for c in CATEGORIES if c != category_1]
)

run_btn = st.button("Run Analysis")

# ──────────────────────────────────────────────────
#  TABS (always visible)
# ──────────────────────────────────────────────────
tab_analysis, tab_heatmap = st.tabs(["Category Analysis", "Current Market HeatMap"])

# Place-holder for data so both tabs can see it
df_raw = None

if run_btn:
    with st.spinner("Loading & crunching numbers…"):
        DATA_URL = "https://pancakebreakfaststats.com/wp-content/uploads/2025/06/data_file_006.xlsx"
        df_raw = pd.read_excel(DATA_URL)

# ──────────────────────────────────────────────────
#  TAB 1  –  CATEGORY ANALYSIS
# ──────────────────────────────────────────────────
with tab_analysis:
    if not run_btn:
        st.info("Click **Run Analysis** to generate forecasts and trend charts.")
    else:
        def display_analytics(cat):
            st.header(f"Analytics for {cat}")

            d = preprocess(df_raw, cat)

            # Forecast
            f_df, _ = forecast(d)
            pct_change = (f_df['Forecast'].iloc[-1] - d['market_value'].iloc[-1]) / d['market_value'].iloc[-1] * 100

            fig1, ax1 = plt.subplots()
            ax1.plot(d['Month_Year'], d['market_value'], label='Historical')
            ax1.plot(f_df['Date'], f_df['Forecast'], label='Forecast')
            ax1.fill_between(f_df['Date'], f_df['Lower'], f_df['Upper'], alpha=0.2)
            ax1.set_title("12-Month Forecast")
            ax1.legend()
            st.pyplot(fig1)

            st.markdown(f"**Projected change by {f_df['Date'].iloc[-1].strftime('%B %Y')}: {pct_change:.2f}%**")

            # MACD
            macd_df = macd_analysis(d)
            bucket = macd_df['MACD_Trend'].iloc[-1]
            st.markdown(f"**Current MACD Bucket:** {bucket}")

            fig2, ax2 = plt.subplots()
            ax2.plot(macd_df['Month_Year'], macd_df['MACD_Trend'].astype(str))
            ax2.set_title("MACD Bucket Over Time")
            ax2.tick_params(axis='x', rotation=45)
            st.pyplot(fig2)

            # Seasonality
            best = best_month(d)
            st.markdown(f"**Best buying month:** {best.idxmin()} (avg value {best.min():.2f})")

            fig3, ax3 = plt.subplots()
            best.plot(kind='bar', ax=ax3)
            ax3.set_ylabel("Avg Market Value")
            ax3.set_title("Average Market Value by Month")
            st.pyplot(fig3)

        if category_2 == "None":
            display_analytics(category_1)
        else:
            col1, col2 = st.columns(2)
            with col1:
                display_analytics(category_1)
            with col2:
                display_analytics(category_2)

            st.subheader("Side-by-side comparison complete.")

# ──────────────────────────────────────────────────
#  TAB 2  –  MARKET HEATMAP
# ──────────────────────────────────────────────────
with tab_heatmap:
    if not run_btn:
        st.info("Run the analysis to view the MACD market heat-map.")
    else:
        rows = []
        for cat in CATEGORIES:
            d = preprocess(df_raw, cat)
            bucket = macd_analysis(d)['MACD_Trend'].iloc[-1]

            rows.append({
                "Category"         : cat,
                "MACD Bucket"      : bucket,
                "Collector Buy?"   : RULES['collector'][bucket][0],
                "Collector Sell?"  : RULES['collector'][bucket][1],
                "Flipper Buy?"     : RULES['flipper'][bucket][0],
                "Flipper Sell?"    : RULES['flipper'][bucket][1],
                "Investor Buy?"    : RULES['investor'][bucket][0],
                "Investor Sell?"   : RULES['investor'][bucket][1],
            })

        heat_df = pd.DataFrame(rows)
        st.dataframe(heat_df, use_container_width=True)

        st.markdown("""
        **How to read this table**

        * **Buy = Yes** → favorable entry conditions for that persona.  
        * **Sell = Yes** → consider trimming or flipping inventory.  
        Mapping is based on Pancake Analytics’ MACD play-book.
        """)

# ──────────────────────────────────────────────────
#  FOOTER
# ──────────────────────────────────────────────────
st.markdown("""
---
### Thank You for Using Cardboard Bot
Built by **Pancake Analytics LLC** to help collectors dive deeper into their hobby.  
All data is sourced from eBay and analyzed for trends and opportunities.  
_This is an analytics read, not financial advice._
""")
