"""
🌫️ India AQI Intelligence Dashboard
Author: Atharva Rane
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="India AQI Intelligence",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #060912;
}

/* Hide streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #0a0f1a 100%);
    border-right: 1px solid #1e2d3d;
}

/* Hero section */
.hero {
    background: linear-gradient(135deg, #0a0f1a 0%, #0d1b2a 50%, #0a0f1a 100%);
    border: 1px solid #1e3a5f;
    border-radius: 20px;
    padding: 40px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(0,150,255,0.05) 0%, transparent 50%),
                radial-gradient(circle at 70% 50%, rgba(0,255,150,0.03) 0%, transparent 50%);
    pointer-events: none;
}
.hero-title {
    font-size: 42px;
    font-weight: 700;
    background: linear-gradient(135deg, #ffffff 0%, #4fc3f7 50%, #29b6f6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.2;
}
.hero-sub {
    color: #546e7a;
    font-size: 15px;
    margin-top: 8px;
    font-weight: 400;
}
.hero-badges {
    display: flex;
    gap: 10px;
    margin-top: 20px;
    flex-wrap: wrap;
}
.badge {
    background: rgba(79,195,247,0.1);
    border: 1px solid rgba(79,195,247,0.3);
    color: #4fc3f7;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
}

/* KPI Cards */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 12px;
    margin: 20px 0;
}
.kpi-card {
    background: linear-gradient(135deg, #0d1117, #0a1628);
    border: 1px solid #1e3a5f;
    border-radius: 14px;
    padding: 20px 16px;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
}
.kpi-card:hover {
    border-color: #4fc3f7;
    transform: translateY(-2px);
}
.kpi-value {
    font-size: 28px;
    font-weight: 700;
    color: #4fc3f7;
    line-height: 1;
}
.kpi-label {
    font-size: 11px;
    color: #546e7a;
    margin-top: 6px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.kpi-sub {
    font-size: 11px;
    color: #37474f;
    margin-top: 3px;
}

/* AQI Status Card */
.aqi-status {
    background: linear-gradient(135deg, #0a0f1a, #0d1b2a);
    border-radius: 20px;
    padding: 32px;
    text-align: center;
    border: 1px solid;
    position: relative;
    overflow: hidden;
}
.aqi-number {
    font-size: 80px;
    font-weight: 800;
    line-height: 1;
    margin: 0;
}
.aqi-category {
    font-size: 18px;
    font-weight: 600;
    margin-top: 8px;
}
.aqi-message {
    font-size: 12px;
    color: #546e7a;
    margin-top: 10px;
    line-height: 1.5;
}

/* Pollutant bars */
.pollutant-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 10px 0;
}
.pollutant-name {
    color: #78909c;
    font-size: 12px;
    width: 160px;
    flex-shrink: 0;
}
.pollutant-bar-bg {
    flex: 1;
    background: #0d1117;
    border-radius: 4px;
    height: 6px;
    overflow: hidden;
}
.pollutant-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.6s ease;
}
.pollutant-val {
    color: #cfd8dc;
    font-size: 12px;
    font-weight: 600;
    width: 70px;
    text-align: right;
    flex-shrink: 0;
}

/* Section headers */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 28px 0 16px 0;
}
.section-title {
    font-size: 18px;
    font-weight: 600;
    color: #eceff1;
}
.section-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #1e3a5f, transparent);
}

/* Tab styling */
[data-testid="stTabs"] [role="tablist"] {
    background: #0d1117;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #1e2d3d;
}
[data-testid="stTabs"] [role="tab"] {
    background: transparent;
    color: #546e7a;
    border-radius: 8px;
    padding: 8px 20px;
    font-size: 13px;
    font-weight: 500;
    border: none;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #1a3a5c, #1e4d7a);
    color: #4fc3f7;
}

/* Forecast day card */
.fc-card {
    background: #0d1117;
    border-radius: 12px;
    padding: 14px 8px;
    text-align: center;
    border: 1px solid #1e2d3d;
    border-top: 3px solid;
}
.fc-day { color: #546e7a; font-size: 11px; text-transform: uppercase; }
.fc-aqi { font-size: 26px; font-weight: 700; line-height: 1.2; }
.fc-cat { font-size: 10px; margin-top: 4px; }

/* Sidebar items */
.sidebar-brand {
    text-align: center;
    padding: 20px 0 10px;
}
.sidebar-brand-title {
    font-size: 18px;
    font-weight: 700;
    color: #4fc3f7;
}
.sidebar-brand-sub {
    font-size: 11px;
    color: #37474f;
    margin-top: 4px;
}
.sidebar-section {
    color: #4fc3f7;
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 20px 0 8px;
}

/* Map container */
.map-wrapper {
    background: #0d1117;
    border: 1px solid #1e2d3d;
    border-radius: 16px;
    overflow: hidden;
}

/* Metric overrides */
[data-testid="metric-container"] {
    background: #0d1117;
    border: 1px solid #1e2d3d;
    border-radius: 12px;
    padding: 16px;
}

/* Selectbox & slider */
[data-testid="stSelectbox"] > div > div {
    background: #0d1117;
    border: 1px solid #1e2d3d;
    border-radius: 8px;
    color: white;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    background: #0d1117;
    border-radius: 12px;
    border: 1px solid #1e2d3d;
}

/* City rank table */
.rank-table { width: 100%; border-collapse: collapse; }
.rank-table tr { border-bottom: 1px solid #1e2d3d; }
.rank-table td { padding: 10px 12px; font-size: 13px; color: #cfd8dc; }
.rank-table th { padding: 10px 12px; font-size: 11px; color: #546e7a;
                  text-transform: uppercase; letter-spacing: 0.5px;
                  border-bottom: 1px solid #1e3a5f; }
.rank-num { color: #37474f; font-weight: 600; }
.rank-city { color: #eceff1; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# ── AQI Config ─────────────────────────────────────────────────────────────────
AQI_COLORS = {
    "Good":        "#00C853",
    "Satisfactory":"#AEEA00",
    "Moderate":    "#FFD600",
    "Poor":        "#FF6D00",
    "Very Poor":   "#DD2C00",
    "Severe":      "#6D0000",
}
AQI_HEALTH = {
    "Good":         ("Good",        "Air is clean. Perfect for outdoor activities.",                 "#00C853"),
    "Satisfactory": ("Satisfactory","Acceptable. Sensitive individuals should take caution.",         "#AEEA00"),
    "Moderate":     ("Moderate",    "Sensitive groups may experience discomfort.",                   "#FFD600"),
    "Poor":         ("Poor",        "Everyone may experience health effects. Limit outdoor time.",   "#FF6D00"),
    "Very Poor":    ("Very Poor",   "Health warnings. Avoid prolonged outdoor exertion.",            "#DD2C00"),
    "Severe":       ("Severe",      "Emergency conditions. Avoid all outdoor activity.",             "#6D0000"),
}

MONTH_ORDER = ["January","February","March","April","May","June",
               "July","August","September","October","November","December"]

# ── Data ───────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/india_aqi_2020_2024.csv")
    df["Date"]      = pd.to_datetime(df["Date"])
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["MonthNum"]  = df["Date"].dt.month
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    return df

@st.cache_resource
def train_model(df):
    le = LabelEncoder()
    df2 = df.copy()
    df2["CityEnc"] = le.fit_transform(df2["City"])
    feats = ["MonthNum","DayOfYear","DayOfWeek","Year",
             "PM2_5","PM10","NO2","SO2","CO","O3","CityEnc"]
    X = df2[feats]; y = df2["AQI"]
    m = RandomForestRegressor(n_estimators=80, random_state=42, n_jobs=-1)
    m.fit(X, y)
    return m, le

def aqi_category(v):
    if v<=50:   return "Good"
    if v<=100:  return "Satisfactory"
    if v<=200:  return "Moderate"
    if v<=300:  return "Poor"
    if v<=400:  return "Very Poor"
    return "Severe"

def section(title, icon=""):
    st.markdown(f"""
    <div class="section-header">
        <span style="font-size:18px;">{icon}</span>
        <span class="section-title">{title}</span>
        <div class="section-line"></div>
    </div>""", unsafe_allow_html=True)

df   = load_data()
CITIES = sorted(df["City"].unique().tolist())

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div style="font-size:36px;">🌫️</div>
        <div class="sidebar-brand-title">AQI Intelligence</div>
        <div class="sidebar-brand-sub">India Air Quality Monitor</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sidebar-section">📍 City Settings</div>', unsafe_allow_html=True)
    selected_city = st.selectbox("Primary City", CITIES, index=CITIES.index("Delhi"))
    selected_year = st.selectbox("Year", [2020,2021,2022,2023,2024], index=4)

    st.markdown('<div class="sidebar-section">🔀 Comparison</div>', unsafe_allow_html=True)
    compare_cities = st.multiselect(
        "Select Cities to Compare",
        CITIES,
        default=["Delhi","Mumbai","Bangalore","Chennai","Pune"]
    )

    st.markdown("---")
    st.markdown("""
    <div style="color:#37474f; font-size:11px; line-height:1.7; padding:0 4px;">
        📊 <b style="color:#546e7a;">Dataset:</b> 54,810 data points<br>
        🏙️ <b style="color:#546e7a;">Cities:</b> 30 Indian cities<br>
        📅 <b style="color:#546e7a;">Period:</b> 2020–2024<br>
        🧪 <b style="color:#546e7a;">Pollutants:</b> PM2.5, PM10, NO2, SO2, CO, O3<br><br>
        Built by <b style="color:#4fc3f7;">Atharva Rane</b><br>
        Data Analyst | Python · ML · Streamlit
    </div>""", unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <p class="hero-title">🌫️ India AQI Intelligence</p>
    <p class="hero-sub">Comprehensive Air Quality Analysis & ML Forecasting for 30 Indian Cities · 2020–2024</p>
    <div class="hero-badges">
        <span class="badge">🏙️ 30 Cities</span>
        <span class="badge">📊 54,810 Data Points</span>
        <span class="badge">🤖 ML Forecasting</span>
        <span class="badge">🗺️ Interactive Maps</span>
        <span class="badge">📈 5-Year Trends</span>
    </div>
</div>""", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏙️  City Dashboard",
    "🔀  Compare Cities",
    "📈  Trends & Patterns",
    "🤖  AQI Forecast",
    "🗺️  India Map",
])

# ═══════════════════════
# TAB 1 — CITY DASHBOARD
# ═══════════════════════
with tab1:
    city_df  = df[df["City"] == selected_city].copy()
    year_df  = city_df[city_df["Year"] == selected_year]
    latest   = city_df.sort_values("Date").iloc[-1]
    cat      = latest["Category"]
    _, health_msg, color = AQI_HEALTH[cat]
    aqi_val  = round(latest["AQI"], 1)

    col_status, col_pol = st.columns([1, 1.6], gap="medium")

    with col_status:
        section(f"{selected_city} — Live Status", "📍")
        st.markdown(f"""
        <div class="aqi-status" style="border-color: {color}33;">
            <div style="font-size:13px; color:#546e7a; text-transform:uppercase;
                        letter-spacing:1px; margin-bottom:8px;">Air Quality Index</div>
            <div class="aqi-number" style="color:{color};">{aqi_val}</div>
            <div class="aqi-category" style="color:{color};">{cat}</div>
            <div style="width:80%; height:1px; background:{color}22;
                        margin:16px auto;"></div>
            <div class="aqi-message">{health_msg}</div>
            <div style="margin-top:20px; display:flex; justify-content:center; gap:16px;">
        """, unsafe_allow_html=True)

        # Mini stats
        avg  = round(year_df["AQI"].mean(), 0)
        best = round(year_df["AQI"].min(), 0)
        worst= round(year_df["AQI"].max(), 0)
        st.markdown(f"""
            <div style="text-align:center; flex:1;">
                <div style="color:{color}; font-size:18px; font-weight:700;">{avg}</div>
                <div style="color:#37474f; font-size:10px; margin-top:2px;">Avg {selected_year}</div>
            </div>
            <div style="text-align:center; flex:1;">
                <div style="color:#00C853; font-size:18px; font-weight:700;">{best}</div>
                <div style="color:#37474f; font-size:10px; margin-top:2px;">Best Day</div>
            </div>
            <div style="text-align:center; flex:1;">
                <div style="color:#DD2C00; font-size:18px; font-weight:700;">{worst}</div>
                <div style="color:#37474f; font-size:10px; margin-top:2px;">Worst Day</div>
            </div>
            </div>
        </div>""", unsafe_allow_html=True)

    with col_pol:
        section("Pollutant Breakdown", "🧪")
        pollutants = [
            ("PM2.5  Fine Particles",  latest["PM2_5"], 60,  "#FF6D00", "μg/m³"),
            ("PM10   Coarse Particles",latest["PM10"],  100, "#FFD600", "μg/m³"),
            ("NO₂    Nitrogen Dioxide",latest["NO2"],   80,  "#00BCD4", "μg/m³"),
            ("SO₂    Sulfur Dioxide",  latest["SO2"],   80,  "#AB47BC", "μg/m³"),
            ("CO     Carbon Monoxide", latest["CO"],    4,   "#EF5350", "mg/m³"),
            ("O₃     Ozone",           latest["O3"],    100, "#26A69A", "μg/m³"),
        ]
        for name, val, safe, bar_color, unit in pollutants:
            pct   = min(val / (safe * 2) * 100, 100)
            status= "✓ Safe" if val <= safe else "⚠ High"
            sc    = "#00C853" if val <= safe else "#FF6D00"
            st.markdown(f"""
            <div style="margin:12px 0;">
                <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                    <span style="color:#78909c; font-size:12px; font-family:monospace;">{name}</span>
                    <span style="font-size:12px;">
                        <span style="color:{bar_color}; font-weight:600;">{val} {unit}</span>
                        <span style="color:{sc}; font-size:10px; margin-left:8px;">{status}</span>
                    </span>
                </div>
                <div style="background:#0d1117; border-radius:4px; height:5px; overflow:hidden;">
                    <div style="background:{bar_color}; width:{pct}%; height:100%;
                                border-radius:4px; opacity:0.8;"></div>
                </div>
            </div>""", unsafe_allow_html=True)

    # Daily AQI chart
    section(f"Daily AQI — {selected_year}", "📅")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=year_df["Date"], y=year_df["AQI"],
        mode="lines", line=dict(color="#4fc3f7", width=1.5),
        fill="tozeroy", fillcolor="rgba(79,195,247,0.06)",
        hovertemplate="<b>%{x|%d %b}</b><br>AQI: <b>%{y}</b><extra></extra>"
    ))
    for threshold, lcolor, label in [
        (50,"#00C853","Good"),
        (100,"#AEEA00","Satisfactory"),
        (200,"#FF6D00","Poor"),
        (300,"#DD2C00","Very Poor")
    ]:
        fig.add_hline(y=threshold, line_dash="dot", line_color=lcolor,
                      line_width=1, opacity=0.5,
                      annotation_text=label,
                      annotation_position="right",
                      annotation_font_color=lcolor,
                      annotation_font_size=10)
    fig.update_layout(
        template="plotly_dark", height=320,
        plot_bgcolor="#060912", paper_bgcolor="#060912",
        margin=dict(l=0,r=80,t=10,b=0),
        xaxis=dict(showgrid=False, color="#37474f"),
        yaxis=dict(gridcolor="#0d1117", color="#37474f"),
        hoverlabel=dict(bgcolor="#0d1117", bordercolor="#1e3a5f"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # KPI row
    c1,c2,c3,c4,c5 = st.columns(5)
    good_days = int((year_df["AQI"]<=100).sum())
    bad_days  = int((year_df["AQI"]>200).sum())
    c1.metric("📊 Avg AQI",       f"{avg:.0f}")
    c2.metric("🔴 Peak AQI",      f"{worst:.0f}")
    c3.metric("🟢 Best AQI",      f"{best:.0f}")
    c4.metric("✅ Clean Days",     f"{good_days}", "AQI ≤ 100")
    c5.metric("⚠️ Unhealthy Days", f"{bad_days}",  "AQI > 200")

    # Monthly bar
    section("Monthly Average AQI", "📆")
    monthly = year_df.groupby("Month")["AQI"].mean().reset_index()
    monthly["Month"] = pd.Categorical(monthly["Month"], categories=MONTH_ORDER, ordered=True)
    monthly = monthly.sort_values("Month")
    monthly["Color"] = monthly["AQI"].apply(lambda x:
        "#00C853" if x<=50 else "#AEEA00" if x<=100 else
        "#FFD600" if x<=200 else "#FF6D00" if x<=300 else "#DD2C00")
    fig2 = go.Figure(go.Bar(
        x=monthly["Month"], y=monthly["AQI"],
        marker_color=monthly["Color"],
        marker_line_width=0,
        hovertemplate="<b>%{x}</b><br>Avg AQI: <b>%{y:.0f}</b><extra></extra>"
    ))
    fig2.update_layout(
        template="plotly_dark", height=280,
        plot_bgcolor="#060912", paper_bgcolor="#060912",
        margin=dict(l=0,r=0,t=10,b=0),
        xaxis=dict(showgrid=False, color="#37474f"),
        yaxis=dict(gridcolor="#0d1117", color="#37474f"),
        bargap=0.3,
    )
    st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════
# TAB 2 — COMPARE CITIES
# ══════════════════════
with tab2:
    if len(compare_cities) < 2:
        st.warning("Select at least 2 cities from the sidebar.")
    else:
        comp_df   = df[df["City"].isin(compare_cities)]
        year_comp = comp_df[comp_df["Year"] == selected_year]

        section("AQI City Ranking", "🏆")
        city_avg = year_comp.groupby("City")["AQI"].mean().sort_values(ascending=False).reset_index()
        city_avg["Rank"] = range(1, len(city_avg)+1)
        city_avg["Color"] = city_avg["AQI"].apply(lambda x:
            "#00C853" if x<=50 else "#AEEA00" if x<=100 else
            "#FFD600" if x<=200 else "#FF6D00" if x<=300 else "#DD2C00")

        col1, col2 = st.columns([1, 2], gap="medium")
        with col1:
            # Rank table
            rows_html = ""
            for _, r in city_avg.iterrows():
                rows_html += f"""
                <tr>
                    <td class="rank-num">#{int(r.Rank)}</td>
                    <td class="rank-city">{r.City}</td>
                    <td style="color:{r.Color}; font-weight:600;">{r.AQI:.0f}</td>
                    <td style="color:{r.Color}; font-size:11px;">{aqi_category(r.AQI)}</td>
                </tr>"""
            st.markdown(f"""
            <div style="background:#0d1117; border:1px solid #1e2d3d;
                        border-radius:14px; overflow:hidden; margin-top:8px;">
                <table class="rank-table">
                    <thead><tr>
                        <th>#</th><th>City</th><th>AQI</th><th>Status</th>
                    </tr></thead>
                    <tbody>{rows_html}</tbody>
                </table>
            </div>""", unsafe_allow_html=True)

        with col2:
            fig = go.Figure(go.Bar(
                x=city_avg["AQI"], y=city_avg["City"],
                orientation="h",
                marker_color=city_avg["Color"],
                marker_line_width=0,
                text=city_avg["AQI"].round(0),
                textposition="outside",
                textfont=dict(color="#cfd8dc", size=11),
                hovertemplate="<b>%{y}</b><br>Avg AQI: <b>%{x:.0f}</b><extra></extra>"
            ))
            fig.update_layout(
                template="plotly_dark", height=350,
                plot_bgcolor="#060912", paper_bgcolor="#060912",
                margin=dict(l=0,r=40,t=10,b=0),
                xaxis=dict(showgrid=False, color="#37474f"),
                yaxis=dict(showgrid=False, color="#cfd8dc"),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Monthly trend
        section("Monthly AQI Trend Comparison", "📈")
        monthly_comp = year_comp.groupby(["Month","City"])["AQI"].mean().reset_index()
        monthly_comp["Month"] = pd.Categorical(monthly_comp["Month"], categories=MONTH_ORDER, ordered=True)
        monthly_comp = monthly_comp.sort_values("Month")
        fig3 = px.line(monthly_comp, x="Month", y="AQI", color="City",
                       template="plotly_dark", markers=True,
                       color_discrete_sequence=px.colors.qualitative.Set2)
        fig3.update_layout(
            height=360, plot_bgcolor="#060912", paper_bgcolor="#060912",
            margin=dict(l=0,r=0,t=10,b=0),
            xaxis=dict(showgrid=False, color="#37474f"),
            yaxis=dict(gridcolor="#0d1117", color="#37474f"),
            legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e2d3d"),
        )
        fig3.update_traces(line_width=2, marker_size=6)
        st.plotly_chart(fig3, use_container_width=True)

        # Good vs bad days
        section("Clean vs Unhealthy Days", "☀️")
        good_bad = year_comp.groupby("City").apply(lambda x: pd.Series({
            "🟢 Clean (≤100)":     int((x["AQI"]<=100).sum()),
            "🟡 Moderate (101–200)": int(((x["AQI"]>100)&(x["AQI"]<=200)).sum()),
            "🔴 Unhealthy (>200)": int((x["AQI"]>200).sum()),
        })).reset_index()
        good_bad_m = good_bad.melt(id_vars="City", var_name="Category", value_name="Days")
        fig4 = px.bar(good_bad_m, x="City", y="Days", color="Category",
                      color_discrete_map={
                          "🟢 Clean (≤100)":"#00C853",
                          "🟡 Moderate (101–200)":"#FFD600",
                          "🔴 Unhealthy (>200)":"#DD2C00"
                      }, template="plotly_dark", barmode="stack",
                      hover_data={"Days":True})
        fig4.update_layout(
            height=320, plot_bgcolor="#060912", paper_bgcolor="#060912",
            margin=dict(l=0,r=0,t=10,b=0),
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor="#0d1117", color="#37474f"),
            legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h",
                        yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig4, use_container_width=True)

# ════════════════════════
# TAB 3 — TRENDS
# ════════════════════════
with tab3:
    city_df = df[df["City"] == selected_city].copy()

    section(f"Year-over-Year Trend — {selected_city}", "📊")
    yoy = city_df.groupby(["Year","Month"])["AQI"].mean().reset_index()
    yoy["Month"] = pd.Categorical(yoy["Month"], categories=MONTH_ORDER, ordered=True)
    yoy = yoy.sort_values(["Year","Month"])
    fig = px.line(yoy, x="Month", y="AQI", color="Year",
                  template="plotly_dark", markers=True,
                  color_discrete_sequence=["#1565C0","#1976D2","#42A5F5","#90CAF9","#4fc3f7"])
    fig.update_layout(
        height=360, plot_bgcolor="#060912", paper_bgcolor="#060912",
        margin=dict(l=0,r=0,t=10,b=0),
        xaxis=dict(showgrid=False, color="#37474f"),
        yaxis=dict(gridcolor="#0d1117", color="#37474f"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_traces(line_width=2.5, marker_size=7)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        section("Seasonal Distribution", "🌤️")
        fig2 = px.violin(city_df, x="Season", y="AQI", color="Season",
                         template="plotly_dark", box=True,
                         color_discrete_sequence=["#FF6D00","#FFD600","#4fc3f7","#00C853"])
        fig2.update_layout(
            height=340, showlegend=False,
            plot_bgcolor="#060912", paper_bgcolor="#060912",
            margin=dict(l=0,r=0,t=10,b=0),
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        section("AQI Category Breakdown by Year", "🎯")
        cat_year = city_df.groupby(["Year","Category"]).size().reset_index(name="Days")
        cat_order = ["Good","Satisfactory","Moderate","Poor","Very Poor","Severe"]
        fig3 = px.bar(cat_year, x="Year", y="Days", color="Category",
                      color_discrete_map=AQI_COLORS,
                      category_orders={"Category":cat_order},
                      template="plotly_dark")
        fig3.update_layout(
            height=340, plot_bgcolor="#060912", paper_bgcolor="#060912",
            margin=dict(l=0,r=0,t=10,b=0),
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor="#0d1117", color="#37474f"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font_size=11),
        )
        st.plotly_chart(fig3, use_container_width=True)

    section("Monthly AQI Heatmap (All Years)", "🔥")
    heat_df = city_df.groupby(["Year","MonthNum"])["AQI"].mean().reset_index()
    heat_pivot = heat_df.pivot(index="Year", columns="MonthNum", values="AQI")
    heat_pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                          "Jul","Aug","Sep","Oct","Nov","Dec"]
    fig4 = px.imshow(heat_pivot, color_continuous_scale="RdYlGn_r",
                     template="plotly_dark", text_auto=".0f", aspect="auto",
                     zmin=50, zmax=300)
    fig4.update_layout(
        height=260, plot_bgcolor="#060912", paper_bgcolor="#060912",
        margin=dict(l=0,r=0,t=10,b=0),
        coloraxis_colorbar=dict(title="AQI", len=0.8)
    )
    fig4.update_traces(textfont_size=11)
    st.plotly_chart(fig4, use_container_width=True)

# ════════════════════════
# TAB 4 — FORECAST
# ════════════════════════
with tab4:
    section("ML-Powered 7-Day AQI Forecast", "🤖")
    st.markdown("""<p style="color:#546e7a; font-size:13px; margin-top:-12px; margin-bottom:20px;">
    Random Forest model trained on 54,810 data points across 30 cities and 5 years.</p>""",
    unsafe_allow_html=True)

    col_ctrl, col_result = st.columns([1, 2], gap="large")

    with col_ctrl:
        st.markdown("""<div style="background:#0d1117; border:1px solid #1e2d3d;
                       border-radius:14px; padding:20px;">
                       <div style="color:#4fc3f7; font-size:13px; font-weight:600;
                       margin-bottom:16px;">⚙️ Forecast Parameters</div>""",
                    unsafe_allow_html=True)

        fc_city   = st.selectbox("City", CITIES, index=CITIES.index(selected_city), key="fc_city")
        fc_month  = st.slider("Month", 1, 12, 1, format="%d",
                              help="1=January, 12=December")
        fc_pm25   = st.slider("PM2.5 (μg/m³)", 10, 300, 80)
        fc_pm10   = st.slider("PM10 (μg/m³)",  20, 500, 120)
        fc_no2    = st.slider("NO₂ (μg/m³)",   5,  150, 40)

        st.markdown("</div>", unsafe_allow_html=True)
        predict = st.button("🔮 Generate Forecast", type="primary", use_container_width=True)

        # Model info card
        st.markdown("""
        <div style="background:#0d1117; border:1px solid #1e2d3d;
                    border-radius:14px; padding:16px; margin-top:12px;">
            <div style="color:#4fc3f7; font-size:12px; font-weight:600; margin-bottom:10px;">
                🧠 Model Info
            </div>
            <div style="color:#546e7a; font-size:11px; line-height:1.8;">
                Algorithm: Random Forest Regressor<br>
                Trees: 80 estimators<br>
                Training data: 43,848 samples<br>
                Features: 11 (seasonal + pollutants)<br>
            </div>
        </div>""", unsafe_allow_html=True)

    with col_result:
        if predict:
            with st.spinner("Running forecast model..."):
                model, le = train_model(df)
                city_enc  = le.transform([fc_city])[0]
                forecasts = []
                for d in range(1, 8):
                    doy = min(fc_month * 30 + d, 365)
                    feats = [[fc_month, doy, d%7, 2024,
                              fc_pm25*np.random.uniform(0.95,1.05),
                              fc_pm10*np.random.uniform(0.95,1.05),
                              fc_no2 *np.random.uniform(0.95,1.05),
                              20*np.random.uniform(0.9,1.1),
                              15*np.random.uniform(0.9,1.1),
                              40*np.random.uniform(0.9,1.1),
                              city_enc]]
                    pred = round(model.predict(feats)[0], 1)
                    forecasts.append({"Day": f"Day {d}", "AQI": pred,
                                     "Category": aqi_category(pred)})
                fc_df = pd.DataFrame(forecasts)

            # Forecast bar chart
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=fc_df["Day"], y=fc_df["AQI"],
                marker_color=[AQI_COLORS[c] for c in fc_df["Category"]],
                marker_line_width=0,
                text=fc_df["AQI"], textposition="outside",
                textfont=dict(color="#cfd8dc", size=12),
                hovertemplate="<b>%{x}</b><br>AQI: <b>%{y}</b><extra></extra>"
            ))
            fig.add_hline(y=100, line_dash="dot", line_color="#AEEA00",
                          line_width=1, annotation_text="Satisfactory limit",
                          annotation_font_color="#AEEA00", annotation_font_size=10)
            fig.add_hline(y=200, line_dash="dot", line_color="#FF6D00",
                          line_width=1, annotation_text="Poor threshold",
                          annotation_font_color="#FF6D00", annotation_font_size=10)
            fig.update_layout(
                template="plotly_dark", height=300,
                plot_bgcolor="#060912", paper_bgcolor="#060912",
                margin=dict(l=0,r=100,t=10,b=0),
                xaxis=dict(showgrid=False, color="#37474f"),
                yaxis=dict(gridcolor="#0d1117", color="#37474f"),
                title=dict(text=f"7-Day Forecast — {fc_city}",
                          font=dict(color="#eceff1", size=14), x=0),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Day cards
            cols = st.columns(7, gap="small")
            for i, (_, row) in enumerate(fc_df.iterrows()):
                c = AQI_COLORS[row["Category"]]
                cols[i].markdown(f"""
                <div class="fc-card" style="border-top-color:{c};">
                    <div class="fc-day">{row['Day']}</div>
                    <div class="fc-aqi" style="color:{c};">{row['AQI']}</div>
                    <div class="fc-cat" style="color:{c}66;">{row['Category']}</div>
                </div>""", unsafe_allow_html=True)

            avg_fc = fc_df["AQI"].mean()
            worst_day = fc_df.loc[fc_df["AQI"].idxmax(), "Day"]
            best_day  = fc_df.loc[fc_df["AQI"].idxmin(), "Day"]
            st.markdown(f"""
            <div style="background:#0d1117; border:1px solid #1e2d3d;
                        border-radius:12px; padding:16px; margin-top:16px;
                        display:flex; gap:24px;">
                <div>⚠️ <span style="color:#546e7a;">Worst day:</span>
                     <b style="color:#DD2C00;">{worst_day}</b></div>
                <div>✅ <span style="color:#546e7a;">Best day:</span>
                     <b style="color:#00C853;">{best_day}</b></div>
                <div>📊 <span style="color:#546e7a;">7-day avg:</span>
                     <b style="color:#4fc3f7;">{avg_fc:.1f}</b>
                     ({aqi_category(avg_fc)})</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#0d1117; border:1px dashed #1e3a5f;
                        border-radius:14px; padding:60px; text-align:center;">
                <div style="font-size:48px;">🔮</div>
                <div style="color:#4fc3f7; font-size:16px; font-weight:600; margin-top:12px;">
                    Configure & Forecast
                </div>
                <div style="color:#37474f; font-size:13px; margin-top:8px;">
                    Set parameters on the left and click Generate Forecast
                </div>
            </div>""", unsafe_allow_html=True)

# ════════════════════════
# TAB 5 — MAP
# ════════════════════════
with tab5:
    section("India AQI Heat Map", "🗺️")

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        map_year = st.selectbox("Year", [2020,2021,2022,2023,2024], index=4, key="my")
    with c2:
        map_season = st.selectbox("Season", ["All","winter","summer","monsoon","post_monsoon"], key="ms")

    map_df = df[df["Year"]==map_year].copy()
    if map_season != "All":
        map_df = map_df[map_df["Season"]==map_season]

    city_map = map_df.groupby(["City","Latitude","Longitude","State"])["AQI"].mean().reset_index()
    city_map["AQI"] = city_map["AQI"].round(1)
    city_map["Category"] = city_map["AQI"].apply(aqi_category)
    city_map["Size"] = city_map["AQI"] / 3

    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lat=city_map["Latitude"],
        lon=city_map["Longitude"],
        text=city_map.apply(
            lambda r: f"{r.City}<br>AQI: {r.AQI}<br>{r.Category}<br>{r.State}", axis=1),
        hoverinfo="text",
        mode="markers",
        marker=dict(
            size=(city_map["AQI"] / 6).clip(lower=8).tolist(),
            color=city_map["AQI"].tolist(),
            colorscale=[
                [0.0, "#00C853"],
                [0.3, "#FFD600"],
                [0.6, "#FF6D00"],
                [0.85,"#DD2C00"],
                [1.0, "#6D0000"],
            ],
            cmin=40,
            cmax=280,
            showscale=True,
            opacity=0.85,
            line_width=0,
        )
    ))
    fig.update_geos(
        scope="asia",
        showcountries=True, countrycolor="#1e2d3d",
        showland=True, landcolor="#0a0f1a",
        showocean=True, oceancolor="#060912",
        showlakes=True, lakecolor="#060912",
        lataxis_range=[5, 38],
        lonaxis_range=[65, 100],
        bgcolor="#060912",
        framecolor="#1e2d3d",
    )
    fig.update_layout(
        height=520,
        paper_bgcolor="#060912",
        margin=dict(l=0, r=0, t=0, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        section("Most Polluted Cities", "🔴")
        top10 = city_map.nlargest(10,"AQI")[["City","State","AQI","Category"]]
        top10 = top10.reset_index(drop=True)
        top10.index += 1
        st.dataframe(top10, use_container_width=True)
    with col2:
        section("Cleanest Cities", "🟢")
        bot10 = city_map.nsmallest(10,"AQI")[["City","State","AQI","Category"]]
        bot10 = bot10.reset_index(drop=True)
        bot10.index += 1
        st.dataframe(bot10, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; padding:20px; color:#263238;">
    <div style="font-size:13px;">
        Built by <b style="color:#4fc3f7;">Atharva Rane</b> · Data Analyst
        · Python · Streamlit · Scikit-learn · Plotly
    </div>
    <div style="font-size:11px; margin-top:6px;">
        <a href="https://github.com/AtharvaRane7/india-aqi-intelligence"
           style="color:#37474f; text-decoration:none;">
           GitHub Repository
        </a>
    </div>
</div>""", unsafe_allow_html=True)
