import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
from datetime import datetime

# ----------  STATIC SETUP  --------------------------------------------------
st.image(
    "https://pancakebreakfaststats.com/wp-content/uploads/2025/04/pa001.png",
    use_container_width=True
)
st.title("Cardboard Bot")

st.markdown("""
Cardboard Bot helps you navigate the market with insights powered by Pancake Analytics.
Collecting doesn’t need analytics – but for those looking to expand their hobby knowledge,
this app provides visual, predictive insights based on eBay data. Market value is weighted
by the number of sellers per month to give a collector-centric view.
""")

# Categories (Lorcana excluded per earlier spec)
CATEGORIES = [
    'Fortnite', 'Marvel', 'Pokemon', 'Star Wars', 'Magic the Gathering',
    'Baseball', 'Basketball', 'Football', 'Hockey', 'Soccer'
]

# ----------  INPUTS  ---------------------------------------------------------
category_1 = st.selectbox(
    "Select your primary category",
    [c for c in CATEGORIES if c != "Lorcana"]
)
category_2 = st.selectbox(
    "Compare against another category (optional)",
    ["None"] + [c for c in CATEGORIES if c not in ["Lorcana", category_1]]
)

run_analysis = st.button("Run Analysis")

# ----------  HELPER FUNCTIONS  ----------------------------------------------
def preprocess(df, category):
    d = df[df['Category'] == category].copy()
    d['Month_Year'] = pd.to_datetime(d['Month'] + ' ' + d['Year'].astype(str))
    d = d.sort_values('Month_Year')
    d = d.groupby('Month_Year').agg({'market_value': 'mean'}).reset_index()
    d['market_value'] = d['market_value'].astype(float)
    return d

def forecast(df):
    model = ExponentialSmoothing(
        df['market_value'],
        trend='add',
        seasonal='add',
        seasonal_periods=12
    ).fit()
    forecast_values = model.forecast(12)
    conf_int = 1.96 * np.std(model.resid)
    upper = forecast_values + conf_int
    lower = forecast_values - conf_int
    forecast_dates = pd.date_range(
        df['Month_Year'].iloc[-1] + pd.DateOffset(months=1),
        periods=12, freq='MS'
    )
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Forecast': forecast_values,
        'Upper': upper,
        'Lower': lower
    })
    return forecast_df, model

def macd_analysis(df):
    short_ema = df['market_value'].ewm(span=12, adjust=False).mean()
    long_ema  = df['market_value'].ewm(span=26, adjust=False).mean()
    macd  = short_ema - long_ema
    signal = macd.ewm(span=9, adjust=False).mean()
    diff = macd - signal
    df['MACD'] = macd
    df['Signal'] = signal
    df['MACD_Trend'] = pd.cut(
        diff,
        bins=[-np.inf, -1.5, -0.5, 0, 0.5, 1.5, np.inf],
        labels=['High Down', 'Med Down', 'Low Down', 'Low Up', 'Med Up', 'High Up']
    )
    return df[['Month_Year', 'MACD', 'Signal', 'MACD_Trend']]

def best_month(df):
    df['Month'] = df['Month_Year'].dt.month_name()
    return df.groupby('Month')['market_value'].mean().sort_values()

# ----------  BUY / SELL RULE MAP  -------------------------------------------
rule_map = {
    # Collector
    'collector': {
        'High Up': ('No', 'Yes'),   'Med Up': ('No', 'Yes'),
        'Low Up' : ('Yes','No'),    'Low Down': ('Yes','No'),
        'Med Down': ('No','Yes'),   'High Down': ('Yes','No')
    },
    # Flipper
    'flipper': {
        'High Up': ('Yes','No'),    'Med Up': ('Yes','No'),
        'Low Up' : ('Yes','No'),    'Low Down': ('No','Yes'),
        'Med Down': ('No','Yes'),   'High Down': ('No','Yes')
    },
    # Investor
    'investor': {
        'High Up': ('No','Yes'),    'Med Up': ('Yes','No'),
        'Low Up' : ('Yes','No'),    'Low Down': ('No','Yes'),
        'Med Down': ('No','Yes'),   'High Down': ('Yes','No')
    }
}

# ----------  MAIN  -----------------------------------------------------------
if run_analysis:
    with st.spinner("Running analysis…"):
        # Load data once
        DATA_URL = "https://pancakebreakfaststats.com/wp-content/uploads/2025/06/data_file_006.xlsx"
        df_raw = pd.read_excel(DATA_URL)

        # Create two tabs
        tab_analysis, tab_heatmap = st.tabs(["Category Analysis", "Market HeatMap"])

        # =====  TAB 1: PER-CATEGORY ANALYTICS (your existing logic)  ==========
        with tab_analysis:

            def display_analytics(category):
                st.header(f"Analytics for {category}")
                d = preprocess(df_raw, category)

                # ----- Forecast plot -----
                forecast_df, _ = forecast(d)
                fig1, ax1 = plt.subplots()
                ax1.plot(d['Month_Year'], d['market_value'], label='Historical')
                ax1.plot(forecast_df['Date'], forecast_df['Forecast'], label='Forecast')
                ax1.fill_between(forecast_df['Date'], forecast_df['Lower'], forecast_df['Upper'], alpha=0.2)
                ax1.set_title(f"12-Month Forecast: {category}")
                ax1.legend()
                st.pyplot(fig1)

                pct_change = ((forecast_df['Forecast'].iloc[-1] - d['market_value'].iloc[-1])
                              / d['market_value'].iloc[-1]) * 100
                st.markdown(f"**Forecasted % Change by {forecast_df['Date'].iloc[-1].strftime('%B %Y')}:** {pct_change:.2f}%")

                # ----- MACD plot -----
                macd_df = macd_analysis(d)
                fig2, ax2 = plt.subplots()
                ax2.plot(macd_df['Month_Year'], macd_df['MACD'], label='MACD')
                ax2.plot(macd_df['Month_Year'], macd_df['Signal'], label='Signal')
                ax2.set_title("MACD Trend")
                ax2.legend()
                st.pyplot(fig2)

                st.markdown(f"**Most Recent MACD Trend:** {macd_df['MACD_Trend'].iloc[-1]}")

                # ----- Buying-month plot -----
                best_buy = best_month(d)
                fig3, ax3 = plt.subplots()
                best_buy.plot(kind='bar', ax=ax3)
                ax3.set_title("Average Market Value by Month")
                st.pyplot(fig3)

                st.markdown(f"**Best Time to Buy {category} Cards:** {best_buy.idxmin()} "
                            f"(Lowest Avg Value: {best_buy.min():.2f})")

                # ----- Final read out -----
                st.subheader("Final Read Out")
                st.markdown(f"""
                Based on our analytics:
                - **Long-Term Forecast**: {pct_change:.2f}% change projected.
                - **Short-Term Trend**: {macd_df['MACD_Trend'].iloc[-1]}.
                - **Best Buying Month**: {best_buy.idxmin()}.

                _This is not financial advice. This is an analytics read. Collecting is part science, part art._
                """)

            # Single or dual display
            if category_2 == "None":
                display_analytics(category_1)
            else:
                col1, col2 = st.columns(2)
                with col1:
                    display_analytics(category_1)
                with col2:
                    display_analytics(category_2)
                st.subheader("Comparison Read Out")
                st.markdown("Comparison analytics based on forecast, MACD, and buying month shown above.")

        # =====  TAB 2: MARKET HEATMAP  =======================================
        with tab_heatmap:
            st.header("MACD Market HeatMap (All Categories)")

            heat_records = []
            for cat in CATEGORIES:
                d_cat = preprocess(df_raw, cat)
                macd_cat = macd_analysis(d_cat)
                bucket = macd_cat['MACD_Trend'].iloc[-1]

                buy_c, sell_c = rule_map['collector'][bucket]
                buy_f, sell_f = rule_map['flipper'][bucket]
                buy_i, sell_i = rule_map['investor'][bucket]

                heat_records.append({
                    'Category'        : cat,
                    'MACD Bucket'     : bucket,
                    'Collector Buy?'  : buy_c,
                    'Collector Sell?' : sell_c,
                    'Flipper Buy?'    : buy_f,
                    'Flipper Sell?'   : sell_f,
                    'Investor Buy?'   : buy_i,
                    'Investor Sell?'  : sell_i
                })

            heat_df = pd.DataFrame(heat_records)
            st.dataframe(heat_df, use_container_width=True)

            st.markdown("""
            **How to use this table**

            * **Buy = Yes** → favorable entry conditions for that persona.  
            * **Sell = Yes** → consider trimming / flipping inventory.  
            The mapping is derived from Pancake Analytics’ MACD play-book and is meant as a guide, not financial advice.
            """)

# ----------  FOOTER  ---------------------------------------------------------
st.markdown("""
---
### Thank You for Using Cardboard Bot
Cardboard Bot is built by **Pancake Analytics LLC** to help collectors dive deeper into their hobby.  
All data is sourced from eBay and analyzed for trends and opportunities.
""")
