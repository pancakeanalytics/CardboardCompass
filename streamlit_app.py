import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt  # kept for a few legacy charts
import plotly.graph_objects as go

# ─────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────
st.image(
    "https://pancakebreakfaststats.com/wp-content/uploads/2025/06/pa.png",
    use_container_width=True
)
st.title("Cardboard Compass")

st.markdown("""
Cardboard Compass turns eBay sales data into clear, visual insights powered by **Pancake Analytics**.

*The “market-index” shown is the average eBay selling price of singles, **weighted by the
number of sellers and total items sold**. Cardboard Compass surfaces **macro** trends—it won’t
give you the exact price of the card in your hand.*
""")

# ─────────────────────────────────────────
#  LOAD DATA  (cached once per session)
# ─────────────────────────────────────────
@st.cache_data
def load_data(url: str) -> pd.DataFrame:
    return pd.read_excel(url)

DATA_URL = "https://pancakebreakfaststats.com/wp-content/uploads/2025/09/data_file_009.xlsx"

if "df_raw" not in st.session_state:
    if st.button("Run Analysis"):
        st.session_state["df_raw"] = load_data(DATA_URL)

df_raw: pd.DataFrame | None = st.session_state.get("df_raw")

# ─────────────────────────────────────────
#  CONSTANTS & RULES
# ─────────────────────────────────────────
CATEGORIES = [
    "Fortnite", "Marvel", "Pokemon", "Star Wars", "Magic the Gathering",
    "Baseball", "Basketball", "Football", "Hockey", "Soccer"
]

RULES = {
    "collector": {"High Up":("No","Yes"),"Med Up":("No","Yes"),
                  "Low Up":("Yes","No"),"Low Down":("Yes","No"),
                  "Med Down":("No","Yes"),"High Down":("Yes","No")},
    "flipper":   {"High Up":("Yes","No"),"Med Up":("Yes","No"),
                  "Low Up":("Yes","No"),"Low Down":("No","Yes"),
                  "Med Down":("No","Yes"),"High Down":("No","Yes")},
    "investor":  {"High Up":("No","Yes"),"Med Up":("Yes","No"),
                  "Low Up":("Yes","No"),"Low Down":("No","Yes"),
                  "Med Down":("No","Yes"),"High Down":("Yes","No")}
}

# ─────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────
def preprocess(df, cat):
    d = df[df["Category"] == cat].copy()
    d["Month_Year"] = pd.to_datetime(d["Month"] + " " + d["Year"].astype(str))
    return d.groupby("Month_Year")["market_value"].mean().reset_index()

def forecast(df, horizon=12, seasonal_periods=12, trend="add", seasonal="add", ci_level=0.95):
    """
    Holt-Winters forecast with simple residual-based CI.
    Returns (hist_df, fc_df, tidy_hist, tidy_fc).
    """
    y = df.market_value.astype(float)
    model = ExponentialSmoothing(y, trend=trend, seasonal=seasonal,
                                 seasonal_periods=seasonal_periods).fit()

    fc = model.forecast(horizon)

    # z multipliers for common CI levels
    z_table = {0.80:1.2816, 0.90:1.6449, 0.95:1.9600, 0.98:2.3263, 0.99:2.5758}
    z = z_table.get(round(ci_level, 2), 1.96)
    ci = z * np.std(model.resid)

    future = pd.date_range(df.Month_Year.iloc[-1] + pd.DateOffset(months=1),
                           periods=horizon, freq="MS")
    fc_df = pd.DataFrame({
        "Date": future,
        "Forecast": fc,
        "Upper": fc + ci,
        "Lower": fc - ci
    })

    hist_df = pd.DataFrame({
        "Date": df.Month_Year,
        "Historical": df.market_value.values
    })

    # Optional tidy frames if needed
    tidy_hist = hist_df.melt(id_vars="Date", var_name="Series", value_name="Value")
    tidy_fc   = fc_df.melt(id_vars="Date", var_name="Series", value_name="Value")

    return hist_df, fc_df, tidy_hist, tidy_fc

def macd(df):
    s = df.market_value.ewm(span=12,adjust=False).mean()
    l = df.market_value.ewm(span=26,adjust=False).mean()
    macd = s-l; signal = macd.ewm(span=9,adjust=False).mean()
    bucket = pd.cut(macd-signal,
        [-np.inf,-1.5,-0.5,0,0.5,1.5,np.inf],
        labels=["High Down","Med Down","Low Down","Low Up","Med Up","High Up"])
    return macd, signal, bucket

def yoy_3mo(series, latest):
    now  = series.get(latest,np.nan)
    yr   = series.get(latest-pd.DateOffset(years=1),np.nan)
    m3   = series.get(latest-pd.DateOffset(months=3),np.nan)
    yoy  = np.nan if np.isnan(now) or np.isnan(yr) else (now-yr)/yr*100
    r3   = np.nan if np.isnan(now) or np.isnan(m3) else (now-m3)/m3*100
    return yoy, r3

# ─────────────────────────────────────────
#  SIDEBAR NAV
# ─────────────────────────────────────────
PAGES = ["Category Analysis","Market HeatMap","State of Market",
         "Custom Index Builder","Seasonality HeatMap",
         "Rolling Volatility","Correlation Matrix","Flip Forecast"]

with st.sidebar:
    page = st.selectbox("Choose an analysis", PAGES)
    cat1 = st.selectbox("Primary category", CATEGORIES, index=CATEGORIES.index("Pokemon"))
    cat2 = st.selectbox("Compare against", ["None"]+[c for c in CATEGORIES if c!=cat1])

# ─────────────────────────────────────────
#  PAGE 1 ▸ CATEGORY ANALYSIS
# ─────────────────────────────────────────
if page == "Category Analysis":
    if df_raw is None:
        st.info("Click **Run Analysis** first.")
    else:
        def show_card(cat):
            d = preprocess(df_raw, cat)
            st.subheader(cat)

            # ── Forecast (interactive Plotly) ─────────────────────────────
            with st.expander("Forecast settings", expanded=False):
                horizon = st.slider("Horizon (months)", 6, 24, 12, step=1, key=f"h_{cat}")
                ci = st.select_slider("Confidence interval", options=[0.80,0.90,0.95,0.98,0.99],
                                      value=0.95, key=f"ci_{cat}")
                hw_trend = st.selectbox("Trend", ["add","mul"], index=0, key=f"t_{cat}")
                hw_seasonal = st.selectbox("Seasonal", ["add","mul"], index=0, key=f"s_{cat}")
                sp = st.number_input("Seasonal periods", min_value=4, max_value=24, value=12, step=1, key=f"sp_{cat}")

            hist_df, fc_df, _, _ = forecast(d, horizon=horizon,
                                             seasonal_periods=sp,
                                             trend=hw_trend, seasonal=hw_seasonal,
                                             ci_level=ci)

            pct = (fc_df.Forecast.iloc[-1]-d.market_value.iloc[-1])/d.market_value.iloc[-1]*100

            fig = go.Figure()

            # Historical line
            fig.add_trace(go.Scatter(
                x=hist_df["Date"], y=hist_df["Historical"],
                mode="lines", name="Historical",
                hovertemplate="Date=%{x|%b %Y}<br>Value=%{y:.2f}<extra></extra>"
            ))

            # Forecast line
            fig.add_trace(go.Scatter(
                x=fc_df["Date"], y=fc_df["Forecast"],
                mode="lines", name="Forecast",
                line=dict(dash="dash"),
                hovertemplate="Date=%{x|%b %Y}<br>Forecast=%{y:.2f}<extra></extra>"
            ))

            # Confidence band
            fig.add_trace(go.Scatter(
                x=pd.concat([fc_df["Date"], fc_df["Date"][::-1]]),
                y=pd.concat([fc_df["Upper"], fc_df["Lower"][::-1]]),
                fill="toself",
                fillcolor="rgba(66, 135, 245, 0.15)",
                line=dict(color="rgba(66,135,245,0)"),
                name=f"{int(ci*100)}% interval",
                hoverinfo="skip",
                showlegend=True
            ))

            fig.update_layout(
                title=f"{horizon}-Month Holt-Winters Forecast",
                xaxis_title="Month",
                yaxis_title="Market Value",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=10, r=10, t=50, b=10),
            )

            st.plotly_chart(fig, use_container_width=True, theme="streamlit")

            # Download forecast data
            csv = pd.concat([
                hist_df.rename(columns={"Historical":"Value"}).assign(Series="Historical"),
                fc_df.rename(columns={"Forecast":"Value"})[["Date","Value"]].assign(Series="Forecast")
            ]).sort_values("Date").to_csv(index=False)

            st.download_button(
                "⬇️ Download forecast data (CSV)",
                data=csv,
                file_name=f"{cat.replace(' ','_').lower()}_holtwinters_forecast.csv",
                mime="text/csv",
                key=f"dlfc_{cat}"
            )

            st.markdown(
                "**How to read** – Hover to see exact values; use the toolbar to zoom/pan; "
                "toggle series with the legend. Shaded area is your confidence band.\n\n"
                f"*Collector example:* Forecast shows **{pct:+.1f}%** change for {cat} over the next {horizon} months.*"
            )

            # ── MACD Trend (Plotly) ───────────────────────────────────────
            macd_line, signal_line, bucket = macd(d)
            macd_df = pd.DataFrame({
                "Date": d.Month_Year,
                "MACD": macd_line.values,
                "Signal": signal_line.values
            })

            fig_macd = go.Figure()

            fig_macd.add_trace(go.Scatter(
                x=macd_df["Date"], y=macd_df["MACD"],
                mode="lines", name="MACD",
                hovertemplate="Date=%{x|%b %Y}<br>MACD=%{y:.3f}<extra></extra>"
            ))
            fig_macd.add_trace(go.Scatter(
                x=macd_df["Date"], y=macd_df["Signal"],
                mode="lines", name="Signal",
                hovertemplate="Date=%{x|%b %Y}<br>Signal=%{y:.3f}<extra></extra>"
            ))

            # Zero line
            fig_macd.add_hline(y=0, line_width=1, line_dash="dash", opacity=0.6)

            fig_macd.update_layout(
                title=f"MACD Trend – bucket: {bucket.iloc[-1]}",
                xaxis_title="Month",
                yaxis_title="MACD value",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=10, r=10, t=50, b=10),
            )

            st.plotly_chart(fig_macd, use_container_width=True, theme="streamlit")
            st.markdown(
                "**How to read** – MACD above Signal & 0 ⇒ upbeat momentum; below 0 ⇒ downtrend.\n\n"
                "*Collector example:* MACD just crossed above zero on Marvel cards—grab key cards "
                "before the uptrend is obvious.*"
            )

            # ── Seasonality (Plotly) ──────────────────────────────────────
            d["Month"] = d.Month_Year.dt.month_name()
            month_order = ["January","February","March","April","May","June",
                           "July","August","September","October","November","December"]
            month_avg = (d.groupby("Month").market_value.mean()
                           .reindex(month_order))

            fig_seas = go.Figure()
            fig_seas.add_trace(go.Bar(
                x=month_avg.index, y=month_avg.values, name="Avg Value by Month",
                hovertemplate="Month=%{x}<br>Avg=%{y:.2f}<extra></extra>"
            ))

            fig_seas.update_layout(
                title="Average Value by Month (Seasonality)",
                xaxis_title="Month",
                yaxis_title="Average Value",
                margin=dict(l=10, r=10, t=50, b=10)
            )

            st.plotly_chart(fig_seas, use_container_width=True, theme="streamlit")
            st.markdown(
                "**How to read** – Short bars = historically cheaper months.\n\n"
                "*Collector example:* Star Wars shows its lowest average in **July**. "
                "Plan to splurge on lightsaber inserts mid-summer.*"
            )

        if cat2 == "None":
            show_card(cat1)
        else:
            c1,c2=st.columns(2)
            with c1: show_card(cat1)
            with c2: show_card(cat2)

# ─────────────────────────────────────────
#  PAGE 2 ▸ MARKET HEATMAP (YES-cell shading)
# ─────────────────────────────────────────
elif page == "Market HeatMap":
    if df_raw is None:
        st.info("Run the analysis to view the heat-map.")
    else:
        rows=[]
        for c in CATEGORIES:
            bucket=macd(preprocess(df_raw,c))[2].iloc[-1]
            rows.append({
                "Category":c,"MACD Bucket":bucket,
                "Collector Buy?":RULES["collector"][bucket][0],
                "Collector Sell?":RULES["collector"][bucket][1],
                "Flipper Buy?":RULES["flipper"][bucket][0],
                "Flipper Sell?":RULES["flipper"][bucket][1],
                "Investor Buy?":RULES["investor"][bucket][0],
                "Investor Sell?":RULES["investor"][bucket][1]
            })

        df_heat = pd.DataFrame(rows)
        yes_cols = [c for c in df_heat.columns if c.endswith("?")]

        def yes_green(val):
            return "background-color:#c6f6d5" if str(val).strip().lower()=="yes" else ""

        styled = df_heat.style.applymap(yes_green, subset=yes_cols)

        st.subheader("MACD Market HeatMap")
        st.dataframe(styled, use_container_width=True, height=350)

        st.markdown("""
**How to read**

* **MACD Bucket** shows momentum (High Up → High Down).  
* Green cells highlight suggested Buy/Sell actions.

| Persona | Hold horizon | Goal | Quick tip |
|---------|--------------|------|-----------|
| **Collector** | 1–5 yrs | Build PC cheaply | Look for green “Buy” cells. |
| **Flipper**   | Weeks–Months | Quick flips | Green “Sell” marks good exit points. |
| **Investor**  | 6–18 mths | Ride trends | Enter when green “Buy” first appears. |

*Collector example:* Marvel is “Med Down” with Collector Buy = **Yes** (highlighted green) — time to negotiate on Spidey slabs while prices soften.*
""")

# ─────────────────────────────────────────
#  PAGE 3 ▸ STATE OF MARKET
# ─────────────────────────────────────────
elif page == "State of Market":
    if df_raw is None:
        st.info("Run the analysis to generate the report.")
    else:
        df_raw["Month_Year"]=pd.to_datetime(df_raw.Month+" "+df_raw.Year.astype(str))
        latest=df_raw.Month_Year.max()
        yoy,mo3=[],[]
        for c in CATEGORIES:
            s=df_raw[df_raw.Category==c].groupby("Month_Year").market_value.mean()
            y,r=yoy_3mo(s,latest); yoy.append(y); mo3.append(r)
        mkt=pd.DataFrame({"YoY %":yoy,"3-Mo %":mo3},index=CATEGORIES)
        st.subheader("State of Market – Momentum Snapshot")

        # Keep Matplotlib here or convert to Plotly if you prefer
        fig,ax=plt.subplots(); mkt.plot(kind="bar",ax=ax); ax.axhline(0,color="gray",ls="--")
        ax.set_ylabel("%"); ax.set_title(f"Blue = YoY  |  Orange = 3-Mo  (to {latest:%b %Y})")
        st.pyplot(fig)

        st.markdown(
            "**How to read** – Two positives = heating; two negatives = cooling.\n\n"
            "*Collector example:* Pokémon is −8 % YoY but +9 % 3-Mo — buy before next bull run.*"
        )
        st.dataframe(mkt.round(2),use_container_width=True)

# ─────────────────────────────────────────
#  PAGE 4 ▸ CUSTOM INDEX BUILDER
# ─────────────────────────────────────────
elif page == "Custom Index Builder":
    if df_raw is None:
        st.info("Run the analysis first.")
    else:
        st.subheader("Custom Index Builder")
        sel = st.multiselect("Categories", CATEGORIES,
                             default=["Pokemon","Magic the Gathering"])
        if not sel: st.warning("Pick at least one."); st.stop()

        raw_w = {c: st.slider(f"{c} weight (%)",0,100,20,5) for c in sel}
        if sum(raw_w.values())==0:
            st.warning("Weights above 0 required."); st.stop()
        weights = pd.Series(raw_w,dtype=float)/sum(raw_w.values())

        pivot=(df_raw.assign(Month_Year=lambda x: pd.to_datetime(x.Month+" "+x.Year.astype(str)))
                      .pivot_table(values="market_value",index="Month_Year",
                                   columns="Category",aggfunc="mean")
                      .reindex(columns=CATEGORIES))
        custom=(pivot[sel]*weights).sum(axis=1)

        poke   = pivot["Pokemon"]
        tcg    = pivot[["Pokemon","Magic the Gathering"]].mean(axis=1)
        sports = pivot[["Baseball","Basketball","Football","Soccer","Hockey"]].mean(axis=1)
        nons   = pivot[["Star Wars","Marvel","Fortnite"]].mean(axis=1)

        fig_l,ax_l=plt.subplots()
        custom.plot(ax=ax_l,lw=2,label="My Index")
        for s,l in zip([poke,tcg,sports,nons],["Pokémon","TCGs","Sports","Non-Sports"]):
            s.plot(ax=ax_l,alpha=.7,label=l)
        ax_l.set_ylabel("Indexed Value"); ax_l.legend(fontsize=8)
        st.pyplot(fig_l)

        latest=custom.index.max()
        perf=pd.DataFrame({
            "Series":["My Index","Pokémon","TCGs","Sports","Non-Sports"],
            "YoY %":[yoy_3mo(s,latest)[0] for s in [custom,poke,tcg,sports,nons]],
            "3-Mo %":[yoy_3mo(s,latest)[1] for s in [custom,poke,tcg,sports,nons]]
        }).set_index("Series").round(2)

        col1,col2=st.columns(2)
        with col1:
            fig_y,ax_y=plt.subplots(); perf["YoY %"].plot(kind="bar",ax=ax_y)
            ax_y.axhline(0,color="gray",ls="--"); ax_y.set_ylabel("%")
            ax_y.set_title("YoY % change"); st.pyplot(fig_y)
        with col2:
            fig_r,ax_r=plt.subplots(); perf["3-Mo %"].plot(kind="bar",color="orange",ax=ax_r)
            ax_r.axhline(0,color="gray",ls="--"); ax_r.set_ylabel("%")
            ax_r.set_title("3-Month % change"); st.pyplot(fig_r)

        st.table(weights.mul(100).round(1).rename("Weight %"))
        st.table(perf)
        st.markdown(
            "**How to read** – Blue line = your blend. Bars show long- & short-term pace.\n\n"
            "*Collector example:* Your Marvel-heavy blend lags Sports by 10 % YoY — shift 15 % into Basketball to balance.*"
        )

# ─────────────────────────────────────────
#  PAGE 5 ▸ SEASONALITY HEATMAP (Plotly)
# ─────────────────────────────────────────
elif page == "Seasonality HeatMap":
    if df_raw is None:
        st.info("Run the analysis to view seasonality.")
    else:
        st.subheader("Seasonality – Avg Month-to-Month % Change")

        df_raw["Month_Year"]=pd.to_datetime(df_raw.Month+" "+df_raw.Year.astype(str))
        df_raw["Month_Num"]=df_raw.Month_Year.dt.month
        wide=(df_raw.pivot_table(values="market_value",index="Category",columns="Month_Num",aggfunc="mean")
              .reindex(index=CATEGORIES))
        pct=(wide.pct_change(axis=1)*100).round(2)

        # Build Plotly Heatmap
        month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        fig_hm = go.Figure(data=go.Heatmap(
            z=pct.values,
            x=month_labels,
            y=pct.index.tolist(),
            zmin=-20, zmax=20,
            colorscale="RdYlGn",
            hovertemplate="Category=%{y}<br>Month=%{x}<br>%Δ=%{z:.2f}%<extra></extra>"
        ))
        fig_hm.update_layout(
            title="Seasonality Heatmap – Avg MoM % Change",
            xaxis_title="Month",
            yaxis_title="Category",
            margin=dict(l=10, r=10, t=50, b=10)
        )
        st.plotly_chart(fig_hm, use_container_width=True, theme="streamlit")

        st.dataframe(pct.fillna("—"),use_container_width=True,height=300)
        st.markdown(
            "**How to read** – Red months = typical dips.\n\n"
            "*Collector example:* Marvel is red in November — hit Black-Friday deals for CGC slabs.*"
        )

# ─────────────────────────────────────────
#  PAGE 6 ▸ ROLLING VOLATILITY
# ─────────────────────────────────────────
elif page == "Rolling Volatility":
    if df_raw is None:
        st.info("Run the analysis to view volatility.")
    else:
        st.subheader("Rolling Volatility (CoV %)")
        pick=st.selectbox("Category",CATEGORIES,index=CATEGORIES.index(cat1))
        d=preprocess(df_raw,pick).set_index("Month_Year")
        cv6  = d.market_value.rolling(6 ).std()/d.market_value.rolling(6 ).mean()*100
        cv12 = d.market_value.rolling(12).std()/d.market_value.rolling(12).mean()*100
        fig_v,ax_v=plt.subplots(); cv6.plot(ax=ax_v,label="6-Mo"); cv12.plot(ax=ax_v,label="12-Mo")
        ax_v.set_ylabel("CoV %"); ax_v.legend(); ax_v.set_title(f"{pick} – Volatility")
        st.pyplot(fig_v)
        st.markdown(
            "**How to read** – Higher % = bigger swings.\n\n"
            "*Collector example:* Soccer volatility spiked — hold off on high-grade rookies until prices settle.*"
        )

# ─────────────────────────────────────────
#  PAGE 7 ▸ CORRELATION MATRIX
# ─────────────────────────────────────────
elif page == "Correlation Matrix":
    if df_raw is None:
        st.info("Run the analysis to view correlations.")
    else:
        st.subheader("Correlation Matrix – Monthly Returns")
        df_raw["Month_Year"]=pd.to_datetime(df_raw.Month+" "+df_raw.Year.astype(str))
        wide=(df_raw.pivot_table(values="market_value",index="Month_Year",
                                 columns="Category",aggfunc="mean").sort_index())
        corr=wide.pct_change().dropna().corr()

        # Keep matplotlib heatmap or convert similarly to Plotly if desired
        fig_c,ax_c=plt.subplots(figsize=(6,4.5))
        im=ax_c.imshow(corr,cmap="RdYlGn",vmin=-1,vmax=1)
        ax_c.set_xticks(range(len(CATEGORIES))); ax_c.set_xticklabels(CATEGORIES,rotation=45,ha="right")
        ax_c.set_yticks(range(len(CATEGORIES))); ax_c.set_yticklabels(CATEGORIES)
        fig_c.colorbar(im,ax=ax_c,label="ρ"); st.pyplot(fig_c)

        st.markdown(
            "**How to read** – Green ≈ +1 = move together; red ≈ −1 = opposite.\n\n"
            "*Collector example:* Marvel’s near-zero correlation with Baseball means a downturn in Topps Chrome won’t drag down your Spidey collection.*"
        )
        st.dataframe(corr.round(2),use_container_width=True,height=300)

# ─────────────────────────────────────────
#  FLIP FORECAST MODULE (Formerly Monte Carlo)
# ─────────────────────────────────────────
if page == "Flip Forecast":
    if df_raw is None:
        st.info("Run the analysis first to enable Flip Forecast.")
    else:
        st.subheader("🔄 Flip Forecast – Projecting Future Card Value")

        st.markdown("""
This tool simulates possible future prices based on historical trend, volatility, and seasonality.
""")

        # User selects category
        sim_category = st.selectbox("Choose Category for Forecast", CATEGORIES, index=CATEGORIES.index(cat1))

        # Preprocess selected category
        d = preprocess(df_raw, sim_category)
        d = d.sort_values("Month_Year")
        d["pct_change"] = d["market_value"].pct_change()
        d = d.dropna()

        # Calculate expected return and capped volatility
        expected_return = d["pct_change"].mean()
        monthly_volatility = min(max(d["pct_change"].std(), 0.01), 0.30)

        # Seasonality pattern
        d["Month"] = d["Month_Year"].dt.strftime("%B")
        month_avg = d.groupby("Month")["market_value"].mean()
        seasonality = month_avg / month_avg.mean()
        month_order = ["January","February","March","April","May","June",
                       "July","August","September","October","November","December"]
        seasonality = seasonality.reindex(month_order).fillna(1)

        # 3-month average & latest value
        real_3mo_avg= d["market_value"].tail(3).mean()
        latest_market_value = d["market_value"].iloc[-1]

        # Display current market value and 3mo avg for transparency
        st.write(f"Latest Market Value for {sim_category}: ${latest_market_value:.2f}")
        st.write(f"Computed 3-Month Average: ${real_3mo_avg:.2f}")

        # User inputs
        st.markdown("---")
        st.markdown("#### Your Card Details")
        asking_price = st.number_input("Your Asking Price ($)", min_value=0.0, value=100.0, step=1.0)
        purchase_price = st.number_input("Your Purchase Price ($)", min_value=0.0, value=75.0, step=1.0)
        avg_3mo_price = st.number_input("Average Market Price Over Last 3 Months ($)", min_value=0.0, value=float(round(real_3mo_avg, 2)), step=1.0)

        st.markdown("#### Simulation Settings")
        num_months = 12
        num_simulations = st.slider("Number of Simulations", 100, 100000, 500, step=100)

        initial_price = avg_3mo_price

        # Flip Forecast Simulation
        results = []
        for _ in range(num_simulations):
            prices = [initial_price]
            for m in range(num_months):
                seasonal_adj = seasonality.iloc[m % 12]
                rand_return = np.random.normal(expected_return, monthly_volatility)
                next_price = prices[-1] * (1 + rand_return * seasonal_adj)
                prices.append(next_price)
            results.append(prices)

        results = np.array(results)

        # Plot simulated paths (kept Matplotlib, can be changed to Plotly if you want)
        st.markdown("---")
        st.markdown("#### Simulated Price Paths")
        fig, ax = plt.subplots(figsize=(10,5))
        for r in results[:100]:
            ax.plot(range(num_months+1), r, alpha=0.2, color="blue")
        ax.set_title(f"Flip Forecast – {sim_category} ({num_simulations} runs)")
        ax.set_xlabel("Months Ahead")
        ax.set_ylabel("Simulated Price ($)")
        st.pyplot(fig)

        # Histogram of ending values
        st.markdown("---")
        st.markdown("#### Ending Price Distribution")
        final_prices = results[:, -1]
        fig2, ax2 = plt.subplots()
        ax2.hist(final_prices, bins=30, color="purple", alpha=0.7)
        ax2.axvline(np.median(final_prices), color="black", linestyle="--", label="Median")
        ax2.axvline(asking_price, color="red", linestyle=":", label="Your Asking Price")
        ax2.set_title("Distribution of Final Prices")
        ax2.set_xlabel("Price")
        ax2.legend()
        st.pyplot(fig2)

        # Probability of hitting asking price
        prob_hit_ask = np.mean(final_prices >= asking_price) * 100

        # ROI by percentile
        percentiles = [5, 25, 50, 75, 95]
        roi_dict = {
            f"{p}th Percentile Price": np.percentile(final_prices, p) for p in percentiles
        }
        roi_calc = {
            f"ROI at {p}th %": f"{(roi_dict[f'{p}th Percentile Price'] - purchase_price) / purchase_price:.2%}"
            for p in percentiles
        }

        # Stats summary
        st.markdown("---")
        st.markdown("#### Summary Statistics")
        st.write({
            "Selected Category": sim_category,
            "Starting Price": f"${initial_price:.2f}",
            "Expected Return": f"{expected_return:.2%}",
            "Monthly Volatility (Capped)": f"{monthly_volatility:.2%}",
            "Probability Your Asking Price is Hit": f"{prob_hit_ask:.1f}%",
            **roi_dict,
            **roi_calc
        })

#  FOOTER
# ─────────────────────────────────────────
st.markdown("""
---
### Thank You for Using Cardboard Compass  
Built by **Pancake Analytics LLC** – _analytics read, not financial advice._
""")
