import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

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

DATA_URL = "https://pancakebreakfaststats.com/wp-content/uploads/2025/06/data_file_006.xlsx"

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

def forecast(df):
    model = ExponentialSmoothing(df.market_value, trend="add",
                                 seasonal="add", seasonal_periods=12).fit()
    fc = model.forecast(12)
    ci = 1.96 * np.std(model.resid)
    future = pd.date_range(df.Month_Year.iloc[-1] + pd.DateOffset(months=1),
                           periods=12, freq="MS")
    return pd.DataFrame({"Date":future,"Forecast":fc,
                         "Upper":fc+ci,"Lower":fc-ci})

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
         "Rolling Volatility","Correlation Matrix"]

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

            # Forecast
            fc_df = forecast(d)
            pct = (fc_df.Forecast.iloc[-1]-d.market_value.iloc[-1])/d.market_value.iloc[-1]*100
            fig,ax=plt.subplots(); ax.plot(d.Month_Year,d.market_value,label="Historical")
            ax.plot(fc_df.Date,fc_df.Forecast,label="Forecast")
            ax.fill_between(fc_df.Date,fc_df.Lower,fc_df.Upper,alpha=.2)
            ax.set_title("12-Month Forecast"); ax.legend()
            st.pyplot(fig)
            st.markdown(
                "**How to read** – Blue = history, orange = forecast, shaded band = ±1.96σ.\n\n"
                f"*Collector example:* Forecast shows **{pct:+.1f}%** upside for {cat}. "
                "If you’re missing a flagship rookie, locking it in now beats buying next year.*"
            )

            # MACD Trend
            macd_line, signal_line, bucket = macd(d)
            fig2,ax2=plt.subplots(); ax2.plot(d.Month_Year,macd_line,label="MACD")
            ax2.plot(d.Month_Year,signal_line,label="Signal")
            ax2.axhline(0,color="gray",ls="--",lw=.7)
            ax2.set_title(f"MACD Trend – bucket: {bucket.iloc[-1]} (Y-axis = MACD value)")
            ax2.legend(); st.pyplot(fig2)
            st.markdown(
                "**How to read** – MACD above Signal & 0 ⇒ upbeat momentum; below 0 ⇒ downtrend.\n\n"
                "*Collector example:* MACD just crossed above zero on Marvel cards—grab key cards "
                "before the uptrend is obvious.*"
            )

            # Seasonality
            d["Month"]=d.Month_Year.dt.month_name()
            month_avg=d.groupby("Month").market_value.mean().reindex(
                ["January","February","March","April","May","June",
                 "July","August","September","October","November","December"])
            fig3,ax3=plt.subplots(); month_avg.plot(kind="bar",ax=ax3)
            ax3.set_title("Average Value by Month"); st.pyplot(fig3)
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
#  PAGE 5 ▸ SEASONALITY HEATMAP
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
        pct=wide.pct_change(axis=1)*100
        fig_h,ax_h=plt.subplots(figsize=(10,4))
        im=ax_h.imshow(pct,cmap="RdYlGn",vmin=-20,vmax=20,aspect="auto")
        ax_h.set_xticks(range(12)); ax_h.set_xticklabels(
            ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
            rotation=45,ha="right")
        ax_h.set_yticks(range(len(CATEGORIES))); ax_h.set_yticklabels(CATEGORIES)
        fig_h.colorbar(im,ax=ax_h,label="% change")
        st.pyplot(fig_h)
        st.markdown(
            "**How to read** – Red months = typical dips.\n\n"
            "*Collector example:* Marvel is red in November — hit Black-Friday deals for CGC slabs.*"
        )
        st.dataframe(pct.round(2).fillna("—"),use_container_width=True,height=300)

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
#  FOOTER
# ─────────────────────────────────────────
st.markdown("""
---
### Thank You for Using Cardboard Compass  
Built by **Pancake Analytics LLC** – _analytics read, not financial advice._
""")
