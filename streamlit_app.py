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
st.image("https://pancakebreakfaststats.com/wp-content/uploads/2024/04/Untitled-design-5.png")
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

df = load_data()
categories = ['Fortnite', 'Marvel', 'Pokemon', 'Star Wars', 'Magic the Gathering', 'Baseball', 'Basketball', 'Football', 'Hockey', 'Soccer']
df = df[df['Category'] != 'Lorcana']

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

def parse_input_with_gpt(user_input, valid_categories, memory):
    system_prompt = (
        "You're a casual, friendly trading card expert. Given a user's question, return a JSON object with the structure:\n"
        "{ 'categories': [valid_category_names], 'compare': true/false }\n"
        "Only use category names from this list:\n"
        f"{valid_categories}\n\n"
        "Note: 'Marvel' means Marvel trading cards.\n"
        f"Prior context (for follow-up questions): {memory}"
    )

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
    )

    try:
        parsed = json.loads(response.choices[0].message.content)
    except:
        parsed = {"categories": [], "compare": False}

    return parsed

def generate_summary_with_gpt(category, pct_change, trend_bucket, best_month):
    summary_prompt = (
        f"Yo! Here's what I've got for {category} trading cards 📈:\n\n"
        f"- Forecast change over the next 12 months: {pct_change:.2f}%\n"
        f"- MACD trend: {trend_bucket}\n"
        f"- Best time to snag deals: {best_month}\n\n"
        "Give the collector some chill, smart insights about whether to buy now, wait, or hold. Include emojis and keep it casual."
    )

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": summary_prompt}
        ]
    )

    return response.choices[0].message.content

# Chat memory setup
if "history" not in st.session_state:
    st.session_state.history = []
if "chat_context" not in st.session_state:
    st.session_state.chat_context = []

# Chat input
user_input = st.chat_input("Ask me about a trading card category or compare two (e.g., Compare Pokemon and Marvel)")

if user_input:
    st.session_state.history.append(("user", user_input))
    st.chat_message("user").markdown(user_input)

    memory = " ".join([msg for role, msg in st.session_state.history if role == "user"])
    parsed = parse_input_with_gpt(user_input, categories, memory)
    matched_categories = parsed["categories"]
    compare_mode = parsed["compare"]

    if not matched_categories:
        reply = "😅 Oops! I couldn't find a valid category in that. Try something like 'Show me Pokemon trends'!"
        st.session_state.history.append(("bot", reply))
        st.chat_message("assistant").markdown(reply)
    else:
        selected_categories = matched_categories[:2] if compare_mode else [matched_categories[0]]

        for cat in selected_categories:
            intro = f"🔍 Let’s dive into **{cat}** trading cards and see what the data’s sayin’..."
            st.session_state.history.append(("bot", intro))
            st.chat_message("assistant").markdown(intro)

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

            summary = generate_summary_with_gpt(cat, pct_change, trend_bucket, best_month)
            st.chat_message("assistant").markdown(summary)

        if compare_mode:
            final_text = f"📊 Between **{selected_categories[0]}** and **{selected_categories[1]}**, weigh those trends, MACD vibes, and best months — then pick your champion! 🏆"
        else:
            final_text = f"👍 That’s the scoop on **{selected_categories[0]}** trading cards! Wanna check out another set or compare a few? I got you."

        st.chat_message("assistant").markdown(final_text)

# Show chat history
for role, message in st.session_state.history:
    st.chat_message(role).markdown(message)