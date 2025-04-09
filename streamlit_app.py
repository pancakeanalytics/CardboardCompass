import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
from datetime import datetime
import re
import openai
import json

st.set_page_config(page_title="Cardboard Bot", layout="wide")

# Title and Banner
st.image("https://pancakebreakfaststats.com/wp-content/uploads/2025/04/pa001.png")
st.title("Cardboard Bot 🤖📦")
st.markdown("""
Cardboard Bot helps you explore the secondary market value of trading cards.
Analytics aren’t necessary to collect — but they’re here if you want to grow your collecting knowledge.
All data is sourced from eBay, providing a broad look at card conditions and types of collectors. Market value is weighted by the number of sellers each month.
""")

# OpenAI Client
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Load data
@st.cache_data
def load_data():
    url = "https://pancakebreakfaststats.com/wp-content/uploads/2025/04/data_file_005.xlsx"
    df = pd.read_excel(url, engine="openpyxl")
    df['Month_Year'] = pd.to_datetime(df['Month'] + ' ' + df['Year'].astype(str))
    df = df.sort_values('Month_Year')
    return df

try:
    df = load_data()
    categories = ['Fortnite', 'Marvel', 'Pokemon', 'Star Wars', 'Magic the Gathering', 'Baseball', 'Basketball', 'Football', 'Hockey', 'Soccer']
    df = df[df['Category'] != 'Lorcana']
except:
    df = pd.DataFrame()
    categories = []

# Forecasting and MACD Analysis Functions
def run_holt_winters(data):
    ts = data.set_index('Month_Year')['market_value']
    model = ExponentialSmoothing(ts, trend='add', seasonal='add', seasonal_periods=12)
    fit = model.fit()
    forecast = fit.forecast(12)
    conf_int = 1.96 * np.std(forecast - fit.fittedvalues[-12:])
    return ts, forecast, conf_int

def calculate_macd(ts):
    ema12 = ts.ewm(span=12, adjust=False).mean()
    ema26 = ts.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    diff = macd - signal
    return macd, signal, diff

def get_gpt_response(messages):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return response.choices[0].message.content

# Session state
if "history" not in st.session_state:
    st.session_state.history = [
        {"role": "system", "content": "You are Cardboard Bot 🤖📦, a helpful and fun trading card advisor. Start by greeting the user, ask what card category they're interested in (like Pokémon, Marvel, Baseball, etc.), then wait for them to respond. Once they tell you a category, confirm it back and ask if they want to run a market analysis (forecast, MACD, best buy time). Be casual and friendly with emojis! Also, if the user asks a general question about trading cards not in the data, do your best to answer based on your broad knowledge."},
        {"role": "assistant", "content": "Hey there! 👋 I'm Cardboard Bot — ready to talk trading cards! What category are you curious about today? 🎴"}
    ]

# User input
user_input = st.chat_input("Talk to Cardboard Bot…")

if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    reply = get_gpt_response(st.session_state.history)
    st.session_state.history.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)

    # Try to detect a valid category
    category_match = next((cat for cat in categories if cat.lower() in user_input.lower()), None)
    if category_match:
        with st.chat_message("assistant"):
            st.markdown(f"Awesome choice! 🔍 Want me to run the full market analysis for **{category_match}** trading cards? Just say 'yes' or 'go for it' and I’m on it! 🧠📊")
        st.session_state.selected_category = category_match

# Run full analysis if user says yes
if "selected_category" in st.session_state:
    if user_input and re.search(r"\b(yes|yeah|go for it|sure|run it)\b", user_input.lower()):
        cat = st.session_state.selected_category
        df_cat = df[df['Category'] == cat]
        ts, forecast, conf = run_holt_winters(df_cat)
        future_index = pd.date_range(start=ts.index[-1] + pd.offsets.MonthBegin(), periods=12, freq='MS')

        fig1, ax1 = plt.subplots()
        ax1.plot(ts.index, ts, label='Actual')
        ax1.plot(future_index, forecast, label='Forecast')
        ax1.fill_between(future_index, forecast - conf, forecast + conf, alpha=0.3)
        ax1.set_title(f'{cat} 12-Month Forecast')
        ax1.legend()
        st.pyplot(fig1)

        pct_change = ((forecast[-1] - ts[-1]) / ts[-1]) * 100

        macd, signal, diff = calculate_macd(ts)
        recent_trend = diff.iloc[-1]

        if recent_trend > 1:
            trend_bucket = 'High Upward Trend 🚀'
        elif recent_trend > 0.25:
            trend_bucket = 'Medium Upward Trend 📈'
        elif recent_trend > 0:
            trend_bucket = 'Low Upward Trend ↗️'
        elif recent_trend > -0.25:
            trend_bucket = 'Low Downward Trend ↘️'
        elif recent_trend > -1:
            trend_bucket = 'Medium Downward Trend 📉'
        else:
            trend_bucket = 'High Downward Trend 🔻'

        fig2, ax2 = plt.subplots()
        ax2.plot(ts.index, macd, label='MACD')
        ax2.plot(ts.index, signal, label='Signal')
        ax2.fill_between(ts.index, diff, 0, where=(diff > 0), color='green', alpha=0.3)
        ax2.fill_between(ts.index, diff, 0, where=(diff < 0), color='red', alpha=0.3)
        ax2.set_title(f'{cat} MACD Trend')
        ax2.legend()
        st.pyplot(fig2)

        best_month_df = df_cat.groupby('Month')['market_value'].mean().sort_values()
        best_month = best_month_df.idxmin()
        st.bar_chart(best_month_df)

        # GPT-powered summary of all three
        summary_prompt = (
            f"Yo! Here's what I found for {cat} trading cards 📈:\n\n"
            f"- Forecast change over 12 months: {pct_change:.2f}%\n"
            f"- MACD trend: {trend_bucket}\n"
            f"- Best time to buy: {best_month}\n\n"
            "Give the collector some smart, casual advice about whether to buy, sell, or hold based on this data."
        )
        summary = get_gpt_response([
            {"role": "user", "content": summary_prompt}
        ])

        st.chat_message("assistant").markdown(summary)
        del st.session_state.selected_category

# Show history (excluding system messages)
for msg in st.session_state.history:
    if msg["role"] != "system":
        st.chat_message(msg["role"]).markdown(msg["content"])
