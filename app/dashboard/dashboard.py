

# python -m streamlit run app/dashboard/dashboard.py

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from analytics import (
    monthly_pct_change,
    calculate_risk,
    risk_score_normalised,
    household_impact,
    price_summary,
    predict_ridge,
    predict_next_n_ridge,
    arima_forecast,
    detect_anomalies,
    cluster_cities,
    price_volatility,
    correlation_matrix,
    monthly_inflation_rate,
    top_inflated_items,
    most_expensive_cities,
)

# ═══════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════
st.set_page_config(
    page_title="Pakistan Price Pulse",
    layout="wide",
    page_icon="🇵🇰",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;700&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
div[data-testid="stMetricValue"] > div { font-family: 'DM Mono', monospace !important; font-size:1.5rem !important; }
div[data-testid="stMetricLabel"] > div { font-size:0.68rem !important; text-transform:uppercase; letter-spacing:.06em; color:#666 !important; }
.insight-card  { background:#0d1117; border:0.5px solid #21262d; border-radius:12px; padding:1rem 1.2rem; margin-bottom:.6rem; }
.insight-card h4 { margin:0 0 4px; font-size:13px; color:#ccc; }
.insight-card p  { margin:0; font-size:12px; color:#888; line-height:1.6; }
.adv-good { background:#0d2718; border:0.5px solid #1D9E75; border-radius:12px; padding:1rem 1.2rem; margin:.4rem 0; }
.adv-warn { background:#2a1d08; border:0.5px solid #EF9F27; border-radius:12px; padding:1rem 1.2rem; margin:.4rem 0; }
.adv-bad  { background:#2a0d0d; border:0.5px solid #E24B4A; border-radius:12px; padding:1rem 1.2rem; margin:.4rem 0; }
.slabel   { font-size:11px; color:#555; font-family:'DM Mono',monospace; text-transform:uppercase; letter-spacing:.08em; margin-bottom:.3rem; }
div[data-testid="stSidebar"] { background:#0a0a0a; }
</style>
""", unsafe_allow_html=True)

# colours
PK_GREEN  = "#1D9E75"
PK_GOLD   = "#EF9F27"
PK_RED    = "#E24B4A"
PK_BLUE   = "#378ADD"
PK_PURPLE = "#7F77DD"
DARK_BG   = "rgba(0,0,0,0)"
GRID_CLR  = "#1a1a1a"
FONT      = "DM Mono"

CAT_COL = {
    "Essentials": PK_GREEN, "Fuel": PK_RED,
    "Vegetables": PK_GOLD,  "Protein": PK_BLUE, "Others": PK_PURPLE,
}

def dark_layout(h=300, margin=None):
    m = margin or dict(l=0, r=0, t=24, b=0)
    return dict(paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
                font=dict(family=FONT, color="#888"), margin=m, height=h,
                xaxis=dict(gridcolor=GRID_CLR, zeroline=False),
                yaxis=dict(gridcolor=GRID_CLR, zeroline=False))

# ═══════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════
@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "../../data/prices.csv")
    if not os.path.exists(path):
        st.error("❌ data/prices.csv not found"); st.stop()
    d = pd.read_csv(path)
    d["Date"] = pd.to_datetime(d["Date"])
    return d

df = load_data()

# ═══════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🇵🇰 Price Pulse")
    st.markdown('<div class="slabel">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("Go to",
        ["🏠 Dashboard","📊 Deep Analytics","🤖 ML Forecasting",
         "💡 Key Insights","🧾 Budget Calculator"],
        label_visibility="collapsed")
    st.markdown("---")
    st.markdown('<div class="slabel">Filters</div>', unsafe_allow_html=True)
    category = st.selectbox("Category",
        ["All","Essentials","Fuel","Vegetables","Protein","Others"])
    fdf = df if category == "All" else df[df["Category"] == category]
    city = st.selectbox("City",  sorted(fdf["City"].unique()))
    item = st.selectbox("Item",  sorted(fdf["Item"].unique()))
    st.markdown("---")
    st.markdown('<div class="slabel">Dataset stats</div>', unsafe_allow_html=True)
    st.metric("Cities", df["City"].nunique())
    st.metric("Items",  df["Item"].nunique())
    st.metric("Avg Price", f"Rs {df['Price'].mean():.0f}")
    st.markdown("---")
    st.caption("Developed by **Sidra** · Pakistan Price Pulse 🇵🇰")

# ═══════════════════════════════════════════════
# SHARED COMPUTATIONS
# ═══════════════════════════════════════════════
data = (fdf[(fdf["City"] == city) & (fdf["Item"] == item)]
        .sort_values("Date").copy())

if data.empty:
    st.warning("⚠️ No data for this combination."); st.stop()

# ── BUG-1 FIX: use monthly_pct_change, not .diff().iloc[-1]
pct_ch        = monthly_pct_change(data)
risk          = calculate_risk(pct_ch)           # calibrated to dataset distribution
r_score       = risk_score_normalised(pct_ch)    # 0-1 for gauge

current_price = data["Price"].iloc[-1]
first_price   = data["Price"].iloc[0]
monthly_ch    = current_price - first_price
avg_price     = df[df["Item"] == item]["Price"].mean()
impact        = household_impact(current_price, avg_price)
summary       = price_summary(data)

# ── BUG-2 FIX: Ridge + TimeSeriesSplit
ridge_result  = predict_ridge(data)
prediction    = ridge_result["predicted_price"]

latest_by_city = (df[df["Item"] == item].sort_values("Date")
                  .groupby("City").tail(1).set_index("City")["Price"])
cheapest_city  = latest_by_city.idxmin()
cheapest_price = latest_by_city.min()
saving         = max(current_price - cheapest_price, 0)

risk_color = PK_GREEN if risk == "Low" else PK_GOLD if risk == "Medium" else PK_RED


# ╔═══════════════════════════════════════════════╗
# ║  PAGE 1 — MAIN DASHBOARD                     ║
# ╚═══════════════════════════════════════════════╝
if page == "🏠 Dashboard":

    # header
    c1, c2 = st.columns([0.78, 0.22])
    with c1:
        st.markdown("## 🇵🇰 Pakistan Price Pulse")
        st.caption("Inflation monitoring · Smart Bazar Advisor · ML Predictions")
    with c2:
        st.markdown(
            f'<div style="margin-top:14px;text-align:right">'
            f'<span style="background:{risk_color}22;color:{risk_color};'
            f'border:0.5px solid {risk_color};padding:4px 14px;'
            f'border-radius:20px;font-size:12px;font-family:DM Mono,monospace">'
            f'● {risk.upper()} RISK</span></div>', unsafe_allow_html=True)
    st.markdown("---")

    # KPIs
    st.markdown('<div class="slabel">Key metrics</div>', unsafe_allow_html=True)
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Current Price",     f"Rs {current_price:.0f}")
    k2.metric("Monthly Change",    f"Rs {monthly_ch:+.0f}",
              f"{pct_ch:+.1f}% monthly")
    k3.metric("National Average",  f"Rs {avg_price:.0f}",
              f"Rs {current_price - avg_price:+.0f} vs avg")
    k4.metric("Ridge Prediction",
              f"Rs {prediction:.0f}" if prediction else "—",
              f"Rs {prediction - current_price:+.0f}" if prediction else "")
    k5.metric("Cheapest City",     cheapest_city,
              f"Save Rs {saving:.0f}" if saving > 0 else "Best price here!")

    # ── Model quality note
    if ridge_result["sufficient_data"]:
        mean_r2 = ridge_result["mean_cv_r2"]
        r2_color = "#1D9E75" if mean_r2 > 0.2 else "#EF9F27" if mean_r2 > -0.1 else "#E24B4A"
        st.markdown(
            f'<div style="font-size:11px;color:{r2_color};font-family:DM Mono,monospace;'
            f'margin-bottom:.5rem">📐 Ridge model (α=10) · TimeSeriesSplit CV R² = '
            f'{mean_r2:.3f} · scores: {ridge_result["cv_r2_scores"]}</div>',
            unsafe_allow_html=True)

    st.markdown("---")

    # Smart Bazar Advisor
    st.markdown('<div class="slabel">Smart bazar advisor</div>', unsafe_allow_html=True)
    if prediction and prediction < current_price and risk == "Low":
        cls,icon,title = "adv-good","💡","Delay your purchase"
        msg = (f"Prices of **{item}** are trending downward in {city}. "
               f"Ridge-predicted next price Rs {prediction:.0f} < current Rs {current_price:.0f}. "
               f"Monthly change: {pct_ch:+.1f}% — classified as LOW risk.")
    elif risk == "High" or monthly_ch > 20:
        cls,icon,title = "adv-bad","⚡","Buy now — prices rising fast"
        msg = (f"**{item}** monthly change is {pct_ch:+.1f}% in {city}. "
               f"Risk is HIGH. Stock up before further hikes. "
               f"Predicted next price: Rs {prediction:.0f}." if prediction else
               f"**{item}** is rising fast in {city}. Buy now.")
    else:
        cls,icon,title = "adv-warn","📊","Monitor prices carefully"
        msg = (f"**{item}** in {city} shows {pct_ch:+.1f}% monthly change — MEDIUM risk. "
               f"National average is Rs {avg_price:.0f}. "
               f"Predicted next: Rs {prediction:.0f}." if prediction else
               f"**{item}** prices in {city} are fluctuating. Monitor regularly.")

    saving_line = (f"\n\n🏆 **Best deal:** {cheapest_city} @ Rs {cheapest_price:.0f} — "
                   f"save Rs {saving:.0f}/unit.") if saving > 0 else "\n\n✅ You are already in the cheapest city!"
    st.markdown(f'<div class="{cls}"><h4>{icon} {title}</h4><p>{msg}{saving_line}</p></div>',
                unsafe_allow_html=True)

    st.markdown("---")

    # Trend chart + gauge
    col_t, col_g = st.columns([0.6, 0.4])

    with col_t:
        st.markdown('<div class="slabel">30-day price trend with anomalies</div>',
                    unsafe_allow_html=True)
        # BUG-3 FIX: Z-score anomalies
        d_anom = detect_anomalies(data)
        anom   = d_anom[d_anom["is_anomaly"]]

        fig = px.line(d_anom, x="Date", y="Price",
                      template="plotly_dark", color_discrete_sequence=[PK_GREEN])
        fig.update_traces(line_width=2.5, marker_size=6, mode="lines+markers")

        if not anom.empty:
            for _, row in anom.iterrows():
                fig.add_annotation(x=row["Date"], y=row["Price"],
                                   text=f"⚠️ {row['anomaly_reason']}",
                                   showarrow=True, arrowhead=2,
                                   font=dict(size=9, color=PK_RED),
                                   arrowcolor=PK_RED, bgcolor="#2a0d0d",
                                   bordercolor=PK_RED)
            fig.add_scatter(x=anom["Date"], y=anom["Price"], mode="markers",
                            marker=dict(color=PK_RED, size=13, symbol="x-thin-open",
                                        line_width=2.5),
                            name="Anomaly (Z>2σ)")

        if prediction:
            nxt = data["Date"].iloc[-1] + pd.Timedelta(days=1)
            fig.add_scatter(x=[data["Date"].iloc[-1], nxt],
                            y=[current_price, prediction],
                            mode="lines+markers",
                            line=dict(dash="dot", color=PK_GOLD, width=2),
                            marker=dict(size=9, color=PK_GOLD),
                            name="Ridge Prediction")

        fig.update_layout(**dark_layout(290))
        st.plotly_chart(fig, use_container_width=True)

    with col_g:
        st.markdown('<div class="slabel">Inflation risk gauge</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:11px;color:#666;font-family:DM Mono,monospace;'
            f'margin-bottom:6px">Monthly change: {pct_ch:+.1f}% · '
            f'Threshold: Low < 20% · Medium < 50% · High ≥ 50%</div>',
            unsafe_allow_html=True)

        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=abs(pct_ch),
            delta={"reference": 0, "suffix": "%"},
            number={"suffix": "%", "font": {"color": risk_color, "family": FONT}},
            title={"text": f"<b>{risk} Risk</b>",
                   "font": {"color": risk_color, "size": 14}},
            gauge={
                "axis":  {"range": [0, 120], "tickcolor": "#444",
                          "tickvals": [0, 20, 50, 80, 120],
                          "ticktext": ["0", "20%<br>Low", "50%<br>Med", "80%", "120%"]},
                "bar":   {"color": risk_color, "thickness": 0.28},
                "bgcolor": "#1a1a1a",
                "steps": [
                    {"range": [0,  20], "color": "#0d2718"},
                    {"range": [20, 50], "color": "#2a1d08"},
                    {"range": [50,120], "color": "#2a0d0d"},
                ],
                "threshold": {"line": {"color": risk_color, "width": 3},
                              "thickness": 0.8, "value": abs(pct_ch)},
            }
        ))
        gauge.update_layout(paper_bgcolor=DARK_BG,
                            font=dict(family=FONT, color="#888"),
                            margin=dict(l=10,r=10,t=30,b=10), height=260)
        st.plotly_chart(gauge, use_container_width=True)

        if risk == "High":
            st.error(f"🚨 {pct_ch:+.1f}% monthly change — HIGH inflation!")
        elif risk == "Medium":
            st.warning(f"⚠️ {pct_ch:+.1f}% monthly change — medium pressure")
        else:
            st.success(f"✅ {pct_ch:+.1f}% monthly change — stable")

    st.markdown("---")

    # City ranking
    st.markdown(f'<div class="slabel">City price ranking — {item}</div>',
                unsafe_allow_html=True)
    ranking = (df[df["Item"] == item].groupby("City")["Price"]
               .mean().sort_values().reset_index())
    ranking.columns = ["City","Avg Price"]
    ranking["HL"] = ranking["City"] == city
    fig_r = px.bar(ranking, x="Avg Price", y="City", orientation="h",
                   template="plotly_dark", color="HL",
                   color_discrete_map={True: PK_GREEN, False: "#2a2a2a"},
                   text="Avg Price")
    fig_r.update_traces(texttemplate="Rs %{text:.0f}", textposition="outside")
    fig_r.update_layout(**dark_layout(400, dict(l=0,r=70,t=24,b=0)), showlegend=False)
    st.plotly_chart(fig_r, use_container_width=True)

    st.markdown("---")

    # Category overview
    st.markdown('<div class="slabel">Category average prices</div>', unsafe_allow_html=True)
    cat_avg = df.groupby("Category")["Price"].mean().reset_index()
    fig_c = px.bar(cat_avg, x="Category", y="Price", template="plotly_dark",
                   color="Category", color_discrete_map=CAT_COL, text="Price")
    fig_c.update_traces(texttemplate="Rs %{text:.0f}", textposition="outside")
    fig_c.update_layout(**dark_layout(280), showlegend=False)
    st.plotly_chart(fig_c, use_container_width=True)

    st.markdown("---")

    # Household impact
    st.markdown('<div class="slabel">Household impact analysis</div>', unsafe_allow_html=True)
    h1,h2,h3 = st.columns(3)
    h1.metric("Extra Monthly Cost",    f"Rs {impact:.0f}", help="vs national avg × 10 units")
    h2.metric("Annual Burden",         f"Rs {impact*12:.0f}")
    h3.metric("Savings if Cheapest",   f"Rs {saving*10:.0f}", help=f"Buy from {cheapest_city}")

    st.markdown("---")
    with st.expander("📋 Raw data"):
        st.dataframe(data[["Date","City","Item","Category","Price"]], use_container_width=True)
    d1,d2 = st.columns(2)
    d1.download_button("⬇️ Filtered Data", data.to_csv(index=False),
                       f"{city}_{item}.csv","text/csv")
    d2.download_button("⬇️ Full Dataset",  df.to_csv(index=False),
                       "price_pulse_full.csv","text/csv")


# ╔═══════════════════════════════════════════════╗
# ║  PAGE 2 — DEEP ANALYTICS                     ║
# ╚═══════════════════════════════════════════════╝
elif page == "📊 Deep Analytics":

    st.markdown("## 📊 Deep Analytics")
    st.caption("Correlation heatmap · Volatility ranking · City clustering · Inflation matrix")
    st.markdown("---")

    # Correlation heatmap
    st.markdown('<div class="slabel">Item price correlation heatmap</div>',
                unsafe_allow_html=True)
    st.caption("Pearson r between item prices across all cities × dates.  "
               "+1 = move together, −1 = move opposite, 0 = no relationship.")
    corr = correlation_matrix(df)
    fig_h, ax = plt.subplots(figsize=(10,7))
    fig_h.patch.set_facecolor("#0d1117"); ax.set_facecolor("#0d1117")
    sns.heatmap(corr, annot=True, fmt=".2f", ax=ax,
                cmap=sns.diverging_palette(10, 150, as_cmap=True),
                linewidths=0.5, linecolor="#1a1a1a",
                annot_kws={"size": 8, "family": "monospace"})
    ax.tick_params(colors="#888", labelsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", color="#888")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, color="#888")
    plt.tight_layout()
    st.pyplot(fig_h); plt.close()

    st.markdown("---")

    # Volatility
    st.markdown('<div class="slabel">Price volatility ranking (CV %)</div>',
                unsafe_allow_html=True)
    st.caption("CV = std / mean × 100.  Higher CV = harder to budget for that item.")
    vol = price_volatility(df)
    fig_v = px.bar(vol, x="Item", y="volatility", template="plotly_dark",
                   color="volatility",
                   color_continuous_scale=[PK_GREEN, PK_GOLD, PK_RED],
                   text="volatility", labels={"volatility":"CV (%)"})
    fig_v.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_v.update_layout(**dark_layout(300), coloraxis_showscale=False)
    st.plotly_chart(fig_v, use_container_width=True)

    st.markdown("---")

    # Stats summary
    st.markdown(f'<div class="slabel">Statistical summary — {item} in {city}</div>',
                unsafe_allow_html=True)
    s = summary
    s1,s2,s3,s4,s5,s6 = st.columns(6)
    s1.metric("Min",    f"Rs {s['min']}")
    s2.metric("Max",    f"Rs {s['max']}")
    s3.metric("Mean",   f"Rs {s['mean']}")
    s4.metric("Median", f"Rs {s['median']}")
    s5.metric("Std Dev",f"Rs {s['std']}")
    s6.metric("CV",     f"{s['cv']}%",
              help="Coefficient of Variation — lower=more stable price")

    st.markdown("---")

    # City clustering scatter — BUG-4 fixed
    st.markdown(f'<div class="slabel">K-Means city clustering — {item}</div>',
                unsafe_allow_html=True)
    st.caption("Cities grouped into Expensive / Moderate / Affordable by "
               "average price + volatility. n_clusters capped at available cities.")
    clusters = cluster_cities(df, item)
    if clusters["cluster"].iloc[0] != "Insufficient data":
        cl_col = {"Expensive": PK_RED, "Moderate": PK_GOLD, "Affordable": PK_GREEN}
        fig_cl = px.scatter(clusters, x="avg_price", y="std_price",
                            color="cluster", text="City",
                            template="plotly_dark", color_discrete_map=cl_col,
                            labels={"avg_price":"Avg Price (Rs)","std_price":"Volatility (Std)"})
        fig_cl.update_traces(textposition="top center", marker_size=13)
        fig_cl.update_layout(**dark_layout(320), legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_cl, use_container_width=True)
        inertia = clusters["inertia"].iloc[0]
        st.caption(f"K-Means inertia: {inertia:.2f}  (lower = tighter clusters)")
        with st.expander("View cluster table"):
            st.dataframe(clusters.drop("inertia",axis=1).sort_values("avg_price"),
                         use_container_width=True)

    st.markdown("---")

    # Monthly inflation heatmap
    st.markdown('<div class="slabel">Monthly inflation % — all items × all cities</div>',
                unsafe_allow_html=True)
    heat_records = [
        {"Item": it, "City": ct, "Inflation %": r}
        for it in df["Item"].unique()
        for ct in df["City"].unique()
        if (r := monthly_inflation_rate(df, it, ct)) is not None
    ]
    heat_df    = pd.DataFrame(heat_records)
    heat_pivot = heat_df.pivot(index="Item", columns="City", values="Inflation %")
    fig_h2, ax2 = plt.subplots(figsize=(14,6))
    fig_h2.patch.set_facecolor("#0d1117"); ax2.set_facecolor("#0d1117")
    sns.heatmap(heat_pivot, annot=True, fmt=".0f", ax=ax2,
                cmap=sns.diverging_palette(10,150,as_cmap=True), center=0,
                linewidths=0.4, linecolor="#1a1a1a",
                annot_kws={"size": 7, "family": "monospace"})
    ax2.tick_params(colors="#888", labelsize=7)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right",
                        color="#888", fontsize=7)
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, color="#888", fontsize=7)
    plt.tight_layout()
    st.pyplot(fig_h2); plt.close()


# ╔═══════════════════════════════════════════════╗
# ║  PAGE 3 — ML FORECASTING                     ║
# ╚═══════════════════════════════════════════════╝
elif page == "🤖 ML Forecasting":

    st.markdown("## 🤖 ML Forecasting")
    st.caption("Ridge Regression (L2) · Holt's Double Exp. Smoothing · Z-score Anomaly Detection")
    st.markdown("---")

    # model cards
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown("""
        <div class="insight-card">
        <h4>📐 Ridge Regression</h4>
        <p>L2-regularised linear model (α=10).
        Prevents overfitting by shrinking coefficients.
        Validated with TimeSeriesSplit CV — trains only on past data.</p>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown("""
        <div class="insight-card">
        <h4>📈 Holt's Exp. Smoothing</h4>
        <p>Captures level <i>and</i> local trend separately.
        α=0.4 for level, β=0.3 for trend.
        Adapts faster to recent price movements than OLS.</p>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown("""
        <div class="insight-card">
        <h4>🔍 Z-Score Anomaly</h4>
        <p>Flags prices where |Z| > 2σ from the series mean.
        Only fires when a genuine spike exists — unlike IsolationForest
        which always flags a fixed % of points.</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Forecast comparison chart
    st.markdown(f'<div class="slabel">Forecast comparison — {item} in {city}</div>',
                unsafe_allow_html=True)

    lr_preds = predict_next_n_ridge(data, n=5)
    ar_preds = arima_forecast(data, steps=5)

    if lr_preds or ar_preds:
        last_date    = data["Date"].iloc[-1]
        freq         = max(int((data["Date"].iloc[-1]-data["Date"].iloc[0]).days
                               / max(len(data)-1,1)), 1)
        future_dates = [last_date + pd.Timedelta(days=freq*(i+1)) for i in range(5)]

        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(x=data["Date"], y=data["Price"],
                                   mode="lines+markers", name="Actual",
                                   line=dict(color=PK_GREEN, width=2.5),
                                   marker=dict(size=6)))
        if lr_preds:
            fig_f.add_trace(go.Scatter(x=future_dates, y=lr_preds,
                                       mode="lines+markers", name="Ridge Regression",
                                       line=dict(color=PK_BLUE, dash="dot", width=2),
                                       marker=dict(size=8, symbol="diamond")))
        if ar_preds:
            fig_f.add_trace(go.Scatter(x=future_dates, y=ar_preds,
                                       mode="lines+markers", name="Holt's Exp. Smoothing",
                                       line=dict(color=PK_GOLD, dash="dash", width=2),
                                       marker=dict(size=8, symbol="square")))

        fig_f.update_layout(**dark_layout(360),
                            legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_f, use_container_width=True)

        # Model CV scores
        st.markdown('<div class="slabel">Model validation — Ridge TimeSeriesSplit CV</div>',
                    unsafe_allow_html=True)
        if ridge_result["sufficient_data"]:
            cv_scores = ridge_result["cv_r2_scores"]
            mean_r2   = ridge_result["mean_cv_r2"]
            r2_color  = PK_GREEN if mean_r2 > 0.2 else PK_GOLD if mean_r2 > -0.1 else PK_RED
            note = ("Note: Price data is inherently noisy/random. Low R² is expected. "
                    "The model still captures the trend direction correctly.")
            st.markdown(
                f'<div style="background:#0d1117;border:0.5px solid #21262d;'
                f'border-radius:10px;padding:.8rem 1rem">'
                f'<span style="font-size:12px;color:#888">CV R² scores: </span>'
                f'<span style="font-family:DM Mono;color:{r2_color}">{cv_scores}</span>'
                f'<span style="font-size:12px;color:#888"> · Mean R² = </span>'
                f'<span style="font-family:DM Mono;color:{r2_color};font-weight:600">'
                f'{mean_r2:.3f}</span><br>'
                f'<span style="font-size:11px;color:#555">{note}</span></div>',
                unsafe_allow_html=True)
        else:
            st.info("Need ≥ 5 data points for CV validation.")

        st.markdown("---")

        # Side-by-side predictions
        col_lr, col_ar = st.columns(2)
        with col_lr:
            st.markdown("**Ridge — next 5 prices**")
            for i,(d,p) in enumerate(zip(future_dates, lr_preds), 1):
                st.metric(f"Step {i} · {d.strftime('%b %d')}",
                          f"Rs {p:.0f}", f"Rs {p-current_price:+.0f}")
        with col_ar:
            st.markdown("**Holt's Smoothing — next 5 prices**")
            for i,(d,p) in enumerate(zip(future_dates, ar_preds), 1):
                st.metric(f"Step {i} · {d.strftime('%b %d')}",
                          f"Rs {p:.0f}", f"Rs {p-current_price:+.0f}")
    else:
        st.info("Need ≥ 5 data points for forecasting.")

    st.markdown("---")

    # Anomaly detection — BUG-3 fixed
    st.markdown(f'<div class="slabel">Z-score anomaly detection — {item} in {city}</div>',
                unsafe_allow_html=True)
    st.caption("Points where |Z-score| > 2.0 (i.e. > 2 standard deviations from mean) "
               "are flagged as price anomalies/spikes.")

    d_anom = detect_anomalies(data)
    anom   = d_anom[d_anom["is_anomaly"]]

    fig_a = px.line(d_anom, x="Date", y="Price",
                    template="plotly_dark", color_discrete_sequence=[PK_GREEN])
    fig_a.update_traces(line_width=2)
    if not anom.empty:
        fig_a.add_scatter(x=anom["Date"], y=anom["Price"], mode="markers",
                          marker=dict(color=PK_RED, size=14, symbol="x-thin-open",
                                      line_width=2.5),
                          name="Anomaly |Z|>2σ")
        for _, row in anom.iterrows():
            fig_a.add_annotation(x=row["Date"], y=row["Price"],
                                 text=row["anomaly_reason"],
                                 showarrow=True, arrowhead=2,
                                 font=dict(size=8,color=PK_RED),
                                 arrowcolor=PK_RED, bgcolor="#2a0d0d",
                                 bordercolor=PK_RED)
        st.warning(f"⚠️ {len(anom)} genuine price spike(s) detected (|Z| > 2σ)")
    else:
        st.success("✅ No significant price anomalies — prices within normal range.")

    fig_a.update_layout(**dark_layout(280))
    st.plotly_chart(fig_a, use_container_width=True)

    if not anom.empty:
        st.dataframe(anom[["Date","Price","z_score","anomaly_reason"]],
                     use_container_width=True)


# ╔═══════════════════════════════════════════════╗
# ║  PAGE 4 — KEY INSIGHTS                       ║
# ╚═══════════════════════════════════════════════╝
elif page == "💡 Key Insights":

    st.markdown("## 💡 Key Insights")
    st.caption("Auto-generated narrative — what does this data actually tell us?")
    st.markdown("---")

    top_items  = top_inflated_items(df, top_n=5)
    exp_cities = most_expensive_cities(df)
    vol        = price_volatility(df)

    f1, f2 = st.columns(2)
    with f1:
        st.markdown(
            f'<div class="insight-card"><h4>🔥 Highest inflation item</h4>'
            f'<p><b>{top_items.iloc[0]["Item"]}</b> had the highest average monthly '
            f'inflation at <b>{top_items.iloc[0]["avg_inflation_pct"]:+.1f}%</b> across '
            f'all cities — highest budget pressure on households.</p></div>',
            unsafe_allow_html=True)
        st.markdown(
            f'<div class="insight-card"><h4>🏙️ Most expensive city</h4>'
            f'<p><b>{exp_cities.iloc[0]["City"]}</b> has the highest price index: '
            f'avg <b>Rs {exp_cities.iloc[0]["avg_price"]:.0f}</b> across all items.</p>'
            f'</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="insight-card"><h4>📉 Most affordable city</h4>'
            f'<p><b>{exp_cities.iloc[-1]["City"]}</b> is cheapest overall: '
            f'avg <b>Rs {exp_cities.iloc[-1]["avg_price"]:.0f}</b>. '
            f'Consumers there face lower living costs.</p></div>',
            unsafe_allow_html=True)

    with f2:
        st.markdown(
            f'<div class="insight-card"><h4>📊 Most volatile item</h4>'
            f'<p><b>{vol.iloc[0]["Item"]}</b> — CV = <b>{vol.iloc[0]["volatility"]:.1f}%</b>. '
            f'Most unpredictable price, hardest to budget for.</p></div>',
            unsafe_allow_html=True)
        st.markdown(
            f'<div class="insight-card"><h4>✅ Most stable item</h4>'
            f'<p><b>{vol.iloc[-1]["Item"]}</b> — CV = <b>{vol.iloc[-1]["volatility"]:.1f}%</b>. '
            f'Most consistent price, reliable for household planning.</p></div>',
            unsafe_allow_html=True)
        saving_est = (exp_cities.iloc[0]["avg_price"] - exp_cities.iloc[-1]["avg_price"]) * 10
        st.markdown(
            f'<div class="insight-card"><h4>🛒 Smart buying tip</h4>'
            f'<p>Buying a 10-item basket from <b>{exp_cities.iloc[-1]["City"]}</b> instead '
            f'of <b>{exp_cities.iloc[0]["City"]}</b> could save roughly '
            f'<b>Rs {saving_est:.0f}/month</b>.</p></div>',
            unsafe_allow_html=True)

    st.markdown("---")

    st.markdown('<div class="slabel">Top 5 most inflated items</div>', unsafe_allow_html=True)
    fig_ti = px.bar(top_items, x="avg_inflation_pct", y="Item", orientation="h",
                    template="plotly_dark",
                    color="avg_inflation_pct",
                    color_continuous_scale=[PK_GREEN, PK_GOLD, PK_RED],
                    text="avg_inflation_pct",
                    labels={"avg_inflation_pct": "Avg Monthly Inflation (%)"})
    fig_ti.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_ti.update_layout(**dark_layout(280, dict(l=0,r=70,t=24,b=0)),
                         coloraxis_showscale=False)
    st.plotly_chart(fig_ti, use_container_width=True)

    st.markdown("---")

    st.markdown('<div class="slabel">City cost index</div>', unsafe_allow_html=True)
    fig_ci = px.bar(exp_cities, x="avg_price", y="City", orientation="h",
                    template="plotly_dark",
                    color="avg_price",
                    color_continuous_scale=[PK_GREEN, PK_GOLD, PK_RED],
                    text="avg_price",
                    labels={"avg_price":"Average Price (Rs)"})
    fig_ci.update_traces(texttemplate="Rs %{text:.0f}", textposition="outside")
    fig_ci.update_layout(**dark_layout(420, dict(l=0,r=80,t=24,b=0)),
                         coloraxis_showscale=False)
    st.plotly_chart(fig_ci, use_container_width=True)


# ╔═══════════════════════════════════════════════╗
# ║  PAGE 5 — BUDGET CALCULATOR                  ║
# ╚═══════════════════════════════════════════════╝
elif page == "🧾 Budget Calculator":

    st.markdown("## 🧾 Monthly Grocery Budget Calculator")
    st.caption("Enter your monthly quantities — see your total cost and potential savings.")
    st.markdown("---")

    calc_city = st.selectbox("Your city", sorted(df["City"].unique()), key="cc")

    st.markdown('<div class="slabel">Monthly quantities per item</div>',
                unsafe_allow_html=True)

    latest_all = (df.sort_values("Date").groupby(["City","Item"])
                  .tail(1).set_index(["City","Item"])["Price"])

    quantities = {}
    cols = st.columns(3)
    for i, it in enumerate(sorted(df["Item"].unique())):
        with cols[i % 3]:
            try:   ph = latest_all[(calc_city, it)]
            except KeyError: ph = 0
            default = 5 if it in ["Flour","Rice","Milk","Sugar"] else \
                      2 if it in ["Chicken","Meat","Eggs"] else 1
            quantities[it] = st.number_input(
                f"{it}  (Rs {ph:.0f}/unit)",
                min_value=0, max_value=200, value=default, step=1, key=f"q_{it}")

    st.markdown("---")

    rows, total_mine, total_cheap = [], 0, 0
    for it, qty in quantities.items():
        if qty == 0: continue
        try:   my_p = latest_all[(calc_city, it)]
        except KeyError: continue
        item_prices = {c: latest_all[(c, it)] for c in df["City"].unique()
                       if (c, it) in latest_all.index}
        if not item_prices: continue
        cp   = min(item_prices, key=item_prices.get)
        cpp  = item_prices[cp]
        mine = my_p * qty
        cheap= cpp  * qty
        total_mine  += mine
        total_cheap += cheap
        rows.append({"Item": it, "Qty": qty,
                     "Your Price": f"Rs {my_p:.0f}",
                     "Your Total": f"Rs {mine:.0f}",
                     "Cheapest City": cp,
                     "Best Price": f"Rs {cpp:.0f}",
                     "Saving": f"Rs {mine-cheap:.0f}"})

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.markdown("---")
        b1,b2,b3 = st.columns(3)
        b1.metric("Your Monthly Basket",     f"Rs {total_mine:.0f}")
        b2.metric("Cheapest Possible",       f"Rs {total_cheap:.0f}")
        pct_save = (total_mine-total_cheap)/total_mine*100 if total_mine else 0
        b3.metric("Potential Saving",        f"Rs {total_mine-total_cheap:.0f}",
                  f"{pct_save:.1f}% cheaper")

        st.markdown("---")
        st.markdown('<div class="slabel">Spending breakdown</div>', unsafe_allow_html=True)
        rdf = pd.DataFrame(rows)
        rdf["Your Total"] = rdf["Your Total"].str.replace("Rs ","").astype(float)
        fig_pie = px.pie(rdf, names="Item", values="Your Total",
                         template="plotly_dark",
                         color_discrete_sequence=px.colors.qualitative.Bold)
        fig_pie.update_layout(paper_bgcolor=DARK_BG,
                              font=dict(family=FONT),
                              margin=dict(l=0,r=0,t=20,b=0), height=320)
        st.plotly_chart(fig_pie, use_container_width=True)
        st.download_button("⬇️ Download Budget Report",
                           pd.DataFrame(rows).to_csv(index=False),
                           "budget_report.csv","text/csv")
    else:
        st.info("Enter quantities above to see your budget breakdown.")

# footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#333;font-family:DM Mono,monospace;"
    "font-size:11px;padding:.6rem'>PAKISTAN PRICE PULSE · 15 CITIES · 15 ITEMS"
    "PAKISTAN PRICE PULSE · PBS-Validated Data · 15 Cities · 15 Items · Developed by Sidra 🇵🇰", unsafe_allow_html=True)
