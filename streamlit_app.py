import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt  # kept for nothing now, safe to remove if you want
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
    Holt-Winters forecast with residual-based CI.
    Returns (hist_df, fc_df, tidy_hist, tidy_fc).
    """
    y = df.market_value.astype(float)
    model = ExponentialSmoothing(y, trend=trend, seasonal=seasonal,
                                 seasonal_periods=seasonal_periods).fit()
    fc = model.forecast(horizon)

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

            # ── Forecast (interactive Plotly)
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
            fig.add_trace(go.Scatter(
                x=hist_df["Date"], y=hist_df["Historical"],
                mode="lines", name="Historical",
                hovertemplate="Date=%{x|%b %Y}<br>Value=%{y:.2f}<extra></extra>"
            ))
            fig.add_trace(go.Scatter(
                x=fc_df["Date"], y=fc_df["Forecast"],
                mode="lines", name="Forecast",
                line=dict(dash="dash"),
                hovertemplate="Date=%{x|%b %Y}<br>Forecast=%{y:.2f}<extra></extra>"
            ))
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

            # ── MACD Trend (interactive Plotly)
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
            fig_macd.add_hline(y=0, line_width=1, line_dash="dash", opacity=0.6)
            fig_macd.update_layout(
                title=f"MACD Trend – bucket: {bucket.iloc[-1]}",
                xaxis_title="Month",
                yaxis_title="MACD value",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(fig_macd, use_container_width=True, theme="streamlit")

            # ── Seasonality (interactive Plotly)
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

# ─────────────────────────────────────────
#  PAGE 3 ▸ STATE OF MARKET (Interactive)
# ─────────────────────────────────────────
elif page == "State of Market":
    if df_raw is None:
        st.info("Run the analysis to generate the report.")
    else:
        st.subheader("State of Market – Momentum Snapshot (Interactive)")

        df_raw["Month_Year"] = pd.to_datetime(df_raw.Month + " " + df_raw.Year.astype(str))
        latest = df_raw["Month_Year"].max()

        yoy_vals, mo3_vals = [], []
        for c in CATEGORIES:
            s = (df_raw[df_raw["Category"] == c]
                 .groupby("Month_Year")["market_value"].mean())
            y, r = yoy_3mo(s, latest)
            yoy_vals.append(y)
            mo3_vals.append(r)

        mkt = pd.DataFrame({
            "Category": CATEGORIES,
            "YoY %": yoy_vals,
            "3-Mo %": mo3_vals
        })

        col_a, col_b = st.columns([1,1])
        with col_a:
            sort_by = st.selectbox("Sort by", ["Category", "YoY %", "3-Mo %"], index=1)
        with col_b:
            ascending = st.toggle("Ascending", value=False)

        mkt_sorted = (mkt.sort_values(sort_by, ascending=ascending)
                          if sort_by != "Category" else mkt.sort_values("Category", ascending=True))

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=mkt_sorted["Category"], y=mkt_sorted["YoY %"],
            name="YoY %",
            hovertemplate="Category=%{x}<br>YoY=%{y:.2f}%<extra></extra>"
        ))
        fig.add_trace(go.Bar(
            x=mkt_sorted["Category"], y=mkt_sorted["3-Mo %"],
            name="3-Mo %",
            hovertemplate="Category=%{x}<br>3-Mo=%{y:.2f}%<extra></extra>"
        ))
        fig.add_hline(y=0, line_width=1, line_dash="dash", opacity=0.6)
        fig.update_layout(
            barmode="group",
            xaxis_title="Category",
            yaxis_title="Percent change",
            title=f"YoY vs 3-Month Momentum (through {latest:%b %Y})",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=60, b=10),
        )
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")

        st.dataframe(mkt_sorted.round(2), use_container_width=True)
        st.download_button(
            "⬇️ Download momentum table (CSV)",
            data=mkt_sorted.to_csv(index=False),
            file_name=f"momentum_snapshot_{latest:%Y_%m}.csv",
            mime="text/csv"
        )

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

        # Keep this one Matplotlib or convert if desired
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

        st.table(weights.mul(100).round(1).rename("Weight %"))
        st.table(perf)

# ─────────────────────────────────────────
#  PAGE 5 ▸ SEASONALITY HEATMAP (Interactive)
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

# ─────────────────────────────────────────
#  PAGE 6 ▸ ROLLING VOLATILITY (Interactive)
# ─────────────────────────────────────────
elif page == "Rolling Volatility":
    if df_raw is None:
        st.info("Run the analysis to view volatility.")
    else:
        st.subheader("Rolling Volatility (Interactive) – Coefficient of Variation %")

        pick = st.selectbox("Category", CATEGORIES, index=CATEGORIES.index(cat1))
        d = preprocess(df_raw, pick).set_index("Month_Year").sort_index()

        col1, col2 = st.columns(2)
        with col1:
            w1 = st.slider("Short window (months)", 3, 18, 6, step=1)
        with col2:
            w2 = st.slider("Long window (months)", 6, 36, 12, step=1)

        cv1 = (d.market_value.rolling(w1).std() / d.market_value.rolling(w1).mean() * 100).rename(f"{w1}-Mo")
        cv2 = (d.market_value.rolling(w2).std() / d.market_value.rolling(w2).mean() * 100).rename(f"{w2}-Mo")
        cv_df = pd.concat([cv1, cv2], axis=1)

        fig_v = go.Figure()
        fig_v.add_trace(go.Scatter(
            x=cv_df.index, y=cv_df.iloc[:,0], mode="lines", name=cv_df.columns[0],
            hovertemplate="Date=%{x|%b %Y}<br>CoV=%{y:.2f}%<extra></extra>"
        ))
        fig_v.add_trace(go.Scatter(
            x=cv_df.index, y=cv_df.iloc[:,1], mode="lines", name=cv_df.columns[1],
            hovertemplate="Date=%{x|%b %Y}<br>CoV=%{y:.2f}%<extra></extra>"
        ))
        fig_v.update_layout(
            title=f"{pick} – Rolling Volatility",
            xaxis_title="Month",
            yaxis_title="CoV (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=60, b=10),
        )
        st.plotly_chart(fig_v, use_container_width=True, theme="streamlit")

        st.dataframe(cv_df.round(2), use_container_width=True)
        st.download_button(
            "⬇️ Download volatility table (CSV)",
            data=cv_df.to_csv(index=True),
            file_name=f"volatility_{pick.replace(' ','_').lower()}.csv",
            mime="text/csv"
        )

# ─────────────────────────────────────────
#  PAGE 7 ▸ CORRELATION MATRIX (Interactive)
# ─────────────────────────────────────────
elif page == "Correlation Matrix":
    if df_raw is None:
        st.info("Run the analysis to view correlations.")
    else:
        st.subheader("Correlation Matrix – Monthly Returns (Interactive)")

        df_raw["Month_Year"]=pd.to_datetime(df_raw.Month+" "+df_raw.Year.astype(str))
        wide=(df_raw.pivot_table(values="market_value",index="Month_Year",
                                 columns="Category",aggfunc="mean")
                   .sort_index()[CATEGORIES])

        basis = st.radio("Correlation basis", ["Monthly returns (pct_change)", "Levels (raw index)"],
                         index=0, horizontal=True)
        mat = wide.pct_change().dropna() if basis.startswith("Monthly") else wide.dropna()
        corr = mat.corr()

        fig_c = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            zmin=-1, zmax=1,
            colorscale="RdYlGn",
            hovertemplate="X=%{x}<br>Y=%{y}<br>ρ=%{z:.2f}<extra></extra>"
        ))
        fig_c.update_layout(
            title="Correlation Heatmap",
            xaxis_title="Category",
            yaxis_title="Category",
            margin=dict(l=10, r=10, t=60, b=10)
        )
        st.plotly_chart(fig_c, use_container_width=True, theme="streamlit")

        st.dataframe(corr.round(2), use_container_width=True, height=300)
        st.download_button(
            "⬇️ Download correlation matrix (CSV)",
            data=corr.to_csv(index=True),
            file_name=f"correlations_{'returns' if basis.startswith('Monthly') else 'levels'}.csv",
            mime="text/csv"
        )

# ─────────────────────────────────────────
#  FLIP FORECAST (Interactive with Plotly)
# ─────────────────────────────────────────
if page == "Flip Forecast":
    if df_raw is None:
        st.info("Run the analysis first to enable Flip Forecast.")
    else:
        st.subheader("🔄 Flip Forecast – Projecting Future Card Value (Interactive)")

        sim_category = st.selectbox("Choose Category for Forecast", CATEGORIES, index=CATEGORIES.index(cat1))

        # Preprocess selected category
        d = preprocess(df_raw, sim_category).sort_values("Month_Year")
        d["pct_change"] = d["market_value"].pct_change()
        d = d.dropna()

        # Expected return & capped vol
        expected_return = d["pct_change"].mean()
        monthly_volatility = min(max(d["pct_change"].std(), 0.01), 0.30)

        # Seasonality pattern
        d["Month"] = d["Month_Year"].dt.strftime("%B")
        month_avg = d.groupby("Month")["market_value"].mean()
        seasonality = (month_avg / month_avg.mean()).reindex(
            ["January","February","March","April","May","June",
             "July","August","September","October","November","December"]
        ).fillna(1)

        # Transparency for inputs
        real_3mo_avg = d["market_value"].tail(3).mean()
        latest_market_value = d["market_value"].iloc[-1]
        st.write(f"Latest Market Value for {sim_category}: ${latest_market_value:.2f}")
        st.write(f"Computed 3-Month Average: ${real_3mo_avg:.2f}")

        # Inputs
        st.markdown("---")
        st.markdown("#### Your Card Details")
        asking_price   = st.number_input("Your Asking Price ($)",   min_value=0.0, value=100.0, step=1.0)
        purchase_price = st.number_input("Your Purchase Price ($)", min_value=0.0, value=75.0,  step=1.0)
        avg_3mo_price  = st.number_input("Average Market Price Over Last 3 Months ($)",
                                         min_value=0.0, value=float(round(real_3mo_avg, 2)), step=1.0)

        st.markdown("#### Simulation Settings")
        col_sim1, col_sim2 = st.columns(2)
        with col_sim1:
            num_months = st.slider("Horizon (months)", 6, 36, 12, step=1)
        with col_sim2:
            num_simulations = st.slider("Number of Simulations", 200, 50000, 2000, step=200)

        initial_price = avg_3mo_price

        # Monte Carlo simulation
        rng = np.random.default_rng()
        results = np.empty((num_simulations, num_months + 1), dtype=float)
        results[:, 0] = initial_price

        # Precompute seasonality multipliers for steps
        season_steps = np.array([seasonality.iloc[m % 12] for m in range(num_months)], dtype=float)

        for i in range(num_simulations):
            for m in range(num_months):
                rand_return = rng.normal(expected_return, monthly_volatility)
                results[i, m+1] = results[i, m] * (1 + rand_return * season_steps[m])

        months_ahead = np.arange(num_months + 1)

        # Percentile fan (10–90 and median)
        p10 = np.percentile(results, 10, axis=0)
        p50 = np.percentile(results, 50, axis=0)  # median
        p90 = np.percentile(results, 90, axis=0)

        # ── Interactive Plot: Simulated Paths + Percentile Band
        fig_paths = go.Figure()

        # Show a small sample of paths for visual texture
        sample = min(100, num_simulations)
        for idx, r in enumerate(results[:sample]):
            fig_paths.add_trace(go.Scattergl(
                x=months_ahead, y=r, mode="lines", line=dict(width=1),
                name="Sim path" if idx == 0 else None,  # single legend item
                hovertemplate="Month=%{x}<br>Price=$%{y:.2f}<extra></extra>",
                opacity=0.25,
                showlegend=(idx == 0)
            ))

        # Percentile band (10–90)
        fig_paths.add_trace(go.Scatter(
            x=np.concatenate([months_ahead, months_ahead[::-1]]),
            y=np.concatenate([p90, p10[::-1]]),
            fill="toself",
            fillcolor="rgba(66, 135, 245, 0.15)",
            line=dict(color="rgba(66,135,245,0)"),
            name="10–90% band",
            hoverinfo="skip"
        ))

        # Median line
        fig_paths.add_trace(go.Scatter(
            x=months_ahead, y=p50, mode="lines", name="Median",
            line=dict(width=2, dash="dash"),
            hovertemplate="Month=%{x}<br>Median=$%{y:.2f}<extra></extra>"
        ))

        # Asking price reference
        fig_paths.add_hline(y=asking_price, line_dash="dot", opacity=0.7,
                            annotation_text="Asking", annotation_position="top left")

        fig_paths.update_layout(
            title=f"Flip Forecast – {sim_category} ({num_simulations} runs)",
            xaxis_title="Months Ahead",
            yaxis_title="Simulated Price ($)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=60, b=10),
        )
        st.plotly_chart(fig_paths, use_container_width=True, theme="streamlit")

        # Downloads
        st.download_button(
            "⬇️ Download all simulated paths (CSV)",
            data=pd.DataFrame(results, columns=[f"M{m}" for m in months_ahead]).to_csv(index=False),
            file_name=f"flip_forecast_paths_{sim_category.replace(' ','_').lower()}.csv",
            mime="text/csv"
        )

        # ── Interactive Histogram of Ending Prices
        final_prices = results[:, -1]
        median_final = float(np.median(final_prices))

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=final_prices, nbinsx=40, name="Final prices",
            hovertemplate="Count=%{y}<br>Price=$%{x:.2f}<extra></extra>"
        ))
        # Vertical markers (median & asking)
        fig_hist.add_vline(x=median_final, line_dash="dash",
                           annotation_text=f"Median ${median_final:,.2f}")
        fig_hist.add_vline(x=asking_price, line_dash="dot",
                           annotation_text=f"Asking ${asking_price:,.2f}")

        fig_hist.update_layout(
            title="Distribution of Final Prices",
            xaxis_title="Price ($)",
            yaxis_title="Count",
            bargap=0.05,
            margin=dict(l=10, r=10, t=60, b=10)
        )
        st.plotly_chart(fig_hist, use_container_width=True, theme="streamlit")

        st.download_button(
            "⬇️ Download final price distribution (CSV)",
            data=pd.DataFrame({"final_price": final_prices}).to_csv(index=False),
            file_name=f"flip_forecast_final_prices_{sim_category.replace(' ','_').lower()}.csv",
            mime="text/csv"
        )

        # Stats
        prob_hit_ask = np.mean(final_prices >= asking_price) * 100
        percentiles = [5, 25, 50, 75, 95]
        roi_prices = {p: np.percentile(final_prices, p) for p in percentiles}
        roi_calc = {f"ROI at {p}th %": f"{(roi_prices[p] - purchase_price) / purchase_price:.2%}" for p in percentiles}

        st.markdown("---")
        st.markdown("#### Summary Statistics")
        st.write({
            "Selected Category": sim_category,
            "Starting Price": f"${initial_price:.2f}",
            "Expected Return": f"{expected_return:.2%}",
            "Monthly Volatility (Capped)": f"{monthly_volatility:.2%}",
            "Probability Your Asking Price is Hit": f"{prob_hit_ask:.1f}%",
            **{f"{p}th Percentile Price": f"${roi_prices[p]:.2f}" for p in percentiles},
            **roi_calc
        })

#  FOOTER
# ─────────────────────────────────────────
st.markdown("""
---
### Thank You for Using Cardboard Compass  
Built by **Pancake Analytics LLC** – _analytics read, not financial advice._
""")
