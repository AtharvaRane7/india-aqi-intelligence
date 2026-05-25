"""
🌫️ India AQI Intelligence Dashboard
Author: Atharva Rane
A comprehensive air quality analysis and forecasting app for 30 Indian cities
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="India AQI Intelligence",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252a3d);
        border-radius: 12px;
        padding: 20px;
        border-left: 4px solid #4CAF50;
        margin: 8px 0;
    }
    .aqi-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 14px;
        margin: 4px;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e2130;
        border-radius: 8px;
        padding: 8px 20px;
        color: white;
    }
    h1 { color: #ffffff; }
    .sidebar-text { color: #aaaaaa; font-size: 12px; }
</style>
""", unsafe_allow_html=True)

# ── AQI Color & Health mapping ────────────────────────────────────────────────
AQI_COLORS = {
    "Good":        "#00E400",
    "Satisfactory":"#92D050",
    "Moderate":    "#FFFF00",
    "Poor":        "#FF7E00",
    "Very Poor":   "#FF0000",
    "Severe":      "#8B0000",
}

AQI_HEALTH = {
    "Good":        ("😊 Good", "Air quality is satisfactory. Enjoy outdoor activities.", "#00E400"),
    "Satisfactory":("🙂 Satisfactory", "Acceptable air quality. Unusually sensitive people should consider limiting prolonged outdoor exertion.", "#92D050"),
    "Moderate":    ("😐 Moderate", "Members of sensitive groups may experience health effects. General public less likely to be affected.", "#FFFF00"),
    "Poor":        ("😷 Poor", "Everyone may begin to experience health effects. Sensitive groups should avoid prolonged outdoor exertion.", "#FF7E00"),
    "Very Poor":   ("🤢 Very Poor", "Health warnings of emergency conditions. The entire population is more likely to be affected.", "#FF0000"),
    "Severe":      ("☠️ Severe", "Health alert: everyone may experience serious health effects. Avoid all outdoor activity.", "#8B0000"),
}

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/india_aqi_2020_2024.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["MonthNum"]  = df["Date"].dt.month
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    return df

@st.cache_resource
def train_forecast_model(df):
    """Train RF model to forecast AQI"""
    features = ["MonthNum", "DayOfYear", "DayOfWeek", "Year",
                "PM2_5", "PM10", "NO2", "SO2", "CO", "O3"]
    le = LabelEncoder()
    df2 = df.copy()
    df2["CityEnc"] = le.fit_transform(df2["City"])
    features = features + ["CityEnc"]
    X = df2[features].dropna()
    y = df2.loc[X.index, "AQI"]
    model = RandomForestRegressor(n_estimators=80, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model, le

df = load_data()
CITIES = sorted(df["City"].unique().tolist())

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://raw.githubusercontent.com/AtharvaRane7/india-aqi-intelligence/main/assets/logo.png",
             use_column_width=True) if False else None
    st.markdown("## 🌫️ India AQI Intelligence")
    st.markdown("---")

    selected_city = st.selectbox("🏙️ Select City", CITIES, index=CITIES.index("Delhi"))
    selected_year = st.selectbox("📅 Year", [2020, 2021, 2022, 2023, 2024], index=4)
    compare_cities = st.multiselect(
        "🔀 Compare Cities",
        CITIES,
        default=["Delhi", "Mumbai", "Bangalore", "Chennai"]
    )
    st.markdown("---")
    st.markdown("**About**")
    st.markdown("""
    <p class='sidebar-text'>
    This dashboard analyses air quality data for 30 Indian cities from 2020–2024.
    Built with Python, Streamlit, and Scikit-learn.<br><br>
    <b>Data Source:</b> CPCB-style synthetic dataset based on real pollution patterns.<br><br>
    Built by <b>Atharva Rane</b> | Data Analyst
    </p>
    """, unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🌫️ India AQI Intelligence Dashboard")
st.markdown("*Comprehensive Air Quality Analysis & Forecasting for 30 Indian Cities (2020–2024)*")
st.markdown("---")

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏙️ City Overview",
    "🔀 City Comparison",
    "📈 Trends & Patterns",
    "🤖 AQI Forecast",
    "🗺️ India Map"
])

# ═══════════════════════════════════════════════════════════
# TAB 1: CITY OVERVIEW
# ═══════════════════════════════════════════════════════════
with tab1:
    city_df = df[df["City"] == selected_city].copy()
    year_df = city_df[city_df["Year"] == selected_year]
    latest  = city_df.sort_values("Date").iloc[-1]

    # Current AQI status
    cat    = latest["Category"]
    emoji, health_msg, color = AQI_HEALTH[cat]
    aqi_val = round(latest["AQI"], 1)

    st.markdown(f"## {selected_city} — Air Quality Report {selected_year}")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #1a1a2e, #16213e);
                    border-radius: 16px; padding: 24px; text-align: center;
                    border: 2px solid {color};'>
            <div style='font-size: 64px; font-weight: bold; color: {color};'>{aqi_val}</div>
            <div style='font-size: 20px; color: {color}; margin-top: 8px;'>{emoji}</div>
            <div style='color: #aaa; margin-top: 8px; font-size: 13px;'>{health_msg}</div>
            <div style='margin-top: 16px; color: #888; font-size: 11px;'>Latest Reading</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Pollutant breakdown
        pollutants = {
            "PM2.5 (Fine Particles)": (latest["PM2_5"], 60,  "μg/m³"),
            "PM10 (Coarse Particles)": (latest["PM10"], 100, "μg/m³"),
            "NO₂ (Nitrogen Dioxide)": (latest["NO2"], 80,   "μg/m³"),
            "SO₂ (Sulfur Dioxide)":   (latest["SO2"], 80,   "μg/m³"),
            "CO (Carbon Monoxide)":   (latest["CO"],  4,    "mg/m³"),
            "O₃ (Ozone)":             (latest["O3"],  100,  "μg/m³"),
        }
        st.markdown("#### 🧪 Pollutant Breakdown")
        for pol, (val, safe_limit, unit) in pollutants.items():
            pct = min(val / (safe_limit * 2) * 100, 100)
            bar_color = "#00E400" if val <= safe_limit else "#FF7E00" if val <= safe_limit*1.5 else "#FF0000"
            st.markdown(f"""
            <div style='margin: 6px 0;'>
                <div style='display: flex; justify-content: space-between; color: #ccc; font-size: 13px;'>
                    <span>{pol}</span>
                    <span style='color: {bar_color};'><b>{val} {unit}</b></span>
                </div>
                <div style='background: #2a2a3e; border-radius: 4px; height: 8px; margin-top: 4px;'>
                    <div style='background: {bar_color}; width: {pct}%; height: 8px; border-radius: 4px;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # KPI Row
    k1, k2, k3, k4, k5 = st.columns(5)
    avg_aqi  = year_df["AQI"].mean()
    max_aqi  = year_df["AQI"].max()
    min_aqi  = year_df["AQI"].min()
    good_days= (year_df["AQI"] <= 100).sum()
    bad_days = (year_df["AQI"] > 200).sum()

    k1.metric("📊 Avg AQI", f"{avg_aqi:.0f}", f"{selected_year}")
    k2.metric("🔴 Peak AQI", f"{max_aqi:.0f}", "Worst day")
    k3.metric("🟢 Best AQI", f"{min_aqi:.0f}", "Best day")
    k4.metric("✅ Good Days", f"{good_days}", "AQI ≤ 100")
    k5.metric("⚠️ Unhealthy Days", f"{bad_days}", "AQI > 200")

    # Daily AQI line chart
    st.markdown("#### 📅 Daily AQI — " + str(selected_year))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=year_df["Date"], y=year_df["AQI"],
        mode="lines", name="AQI",
        line=dict(color="#4FC3F7", width=1.5),
        fill="tozeroy", fillcolor="rgba(79,195,247,0.1)"
    ))
    # Add category threshold lines
    for level, color in [("Good", "#00E400"), ("Satisfactory", "#92D050"),
                          ("Moderate", "#FFFF00"), ("Poor", "#FF7E00"),
                          ("Very Poor", "#FF0000")]:
        thresholds = {"Good": 50, "Satisfactory": 100, "Moderate": 200,
                      "Poor": 300, "Very Poor": 400}
        fig.add_hline(y=thresholds[level], line_dash="dot",
                      line_color=color, opacity=0.4,
                      annotation_text=level, annotation_position="right")
    fig.update_layout(
        template="plotly_dark", height=350,
        xaxis_title="Date", yaxis_title="AQI",
        margin=dict(l=0, r=60, t=20, b=0),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    # Monthly avg bar
    st.markdown("#### 📆 Monthly Average AQI")
    monthly = year_df.groupby("Month")["AQI"].mean().reset_index()
    month_order = ["January","February","March","April","May","June",
                   "July","August","September","October","November","December"]
    monthly["Month"] = pd.Categorical(monthly["Month"], categories=month_order, ordered=True)
    monthly = monthly.sort_values("Month")
    monthly["Color"] = monthly["AQI"].apply(
        lambda x: "#00E400" if x<=50 else "#92D050" if x<=100 else
                  "#FFFF00" if x<=200 else "#FF7E00" if x<=300 else "#FF0000")
    fig2 = px.bar(monthly, x="Month", y="AQI", color="Color",
                  color_discrete_map="identity", template="plotly_dark")
    fig2.update_layout(height=300, showlegend=False,
                       margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# TAB 2: CITY COMPARISON
# ═══════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 🔀 Multi-City Comparison")
    if len(compare_cities) < 2:
        st.warning("Please select at least 2 cities from the sidebar to compare.")
    else:
        comp_df = df[df["City"].isin(compare_cities)]
        year_comp = comp_df[comp_df["Year"] == selected_year]

        # Average AQI ranking
        city_avg = year_comp.groupby("City")["AQI"].mean().sort_values(ascending=False).reset_index()
        city_avg["Color"] = city_avg["AQI"].apply(
            lambda x: "#00E400" if x<=50 else "#92D050" if x<=100 else
                      "#FFFF00" if x<=200 else "#FF7E00" if x<=300 else "#FF0000")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"#### 🏆 AQI Ranking — {selected_year}")
            fig = px.bar(city_avg, x="AQI", y="City", orientation="h",
                        color="Color", color_discrete_map="identity",
                        template="plotly_dark")
            fig.update_layout(height=350, showlegend=False,
                              margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### 🍃 Good Air Days vs Bad Air Days")
            good_bad = year_comp.groupby("City").apply(
                lambda x: pd.Series({
                    "Good (≤100)": (x["AQI"]<=100).sum(),
                    "Moderate (101-200)": ((x["AQI"]>100)&(x["AQI"]<=200)).sum(),
                    "Unhealthy (>200)": (x["AQI"]>200).sum()
                })
            ).reset_index()
            good_bad_melt = good_bad.melt(id_vars="City", var_name="Category", value_name="Days")
            fig2 = px.bar(good_bad_melt, x="City", y="Days", color="Category",
                         color_discrete_map={
                             "Good (≤100)": "#00E400",
                             "Moderate (101-200)": "#FFFF00",
                             "Unhealthy (>200)": "#FF0000"
                         }, template="plotly_dark", barmode="stack")
            fig2.update_layout(height=350, margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig2, use_container_width=True)

        # Time series comparison
        st.markdown(f"#### 📈 Monthly AQI Trend Comparison — {selected_year}")
        monthly_comp = year_comp.groupby(["Month","City"])["AQI"].mean().reset_index()
        month_order = ["January","February","March","April","May","June",
                       "July","August","September","October","November","December"]
        monthly_comp["Month"] = pd.Categorical(monthly_comp["Month"], categories=month_order, ordered=True)
        monthly_comp = monthly_comp.sort_values("Month")
        fig3 = px.line(monthly_comp, x="Month", y="AQI", color="City",
                       template="plotly_dark", markers=True)
        fig3.update_layout(height=380, margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig3, use_container_width=True)

        # Pollutant radar
        st.markdown("#### 🕸️ Pollutant Profile Comparison")
        radar_df = year_comp.groupby("City")[["PM2_5","PM10","NO2","SO2","CO","O3"]].mean()
        fig4 = go.Figure()
        for city in compare_cities:
            if city in radar_df.index:
                vals = radar_df.loc[city].tolist()
                vals_norm = [v/radar_df[col].max() for v, col in zip(vals, radar_df.columns)]
                fig4.add_trace(go.Scatterpolar(
                    r=vals_norm + [vals_norm[0]],
                    theta=["PM2.5","PM10","NO2","SO2","CO","O3","PM2.5"],
                    fill="toself", name=city, opacity=0.6
                ))
        fig4.update_layout(template="plotly_dark", height=400,
                          polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                          margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig4, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# TAB 3: TRENDS & PATTERNS
# ═══════════════════════════════════════════════════════════
with tab3:
    st.markdown("## 📈 Trends & Seasonal Patterns")
    city_df = df[df["City"] == selected_city].copy()

    col1, col2 = st.columns(2)
    with col1:
        # Year over year
        st.markdown(f"#### 📊 Year-over-Year AQI Trend — {selected_city}")
        yoy = city_df.groupby(["Year","Month"])["AQI"].mean().reset_index()
        month_order = ["January","February","March","April","May","June",
                       "July","August","September","October","November","December"]
        yoy["Month"] = pd.Categorical(yoy["Month"], categories=month_order, ordered=True)
        yoy = yoy.sort_values(["Year","Month"])
        fig = px.line(yoy, x="Month", y="AQI", color="Year",
                      template="plotly_dark", markers=True,
                      color_discrete_sequence=px.colors.sequential.Blues_r)
        fig.update_layout(height=380, margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Seasonal box
        st.markdown(f"#### 🌤️ AQI by Season — {selected_city}")
        fig2 = px.box(city_df, x="Season", y="AQI",
                      color="Season", template="plotly_dark",
                      color_discrete_sequence=["#FF7E00","#FFFF00","#4FC3F7","#92D050"])
        fig2.update_layout(height=380, showlegend=False,
                           margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig2, use_container_width=True)

    # Category distribution over years
    st.markdown(f"#### 🎯 AQI Category Distribution Over Years — {selected_city}")
    cat_year = city_df.groupby(["Year","Category"]).size().reset_index(name="Days")
    cat_order = ["Good","Satisfactory","Moderate","Poor","Very Poor","Severe"]
    fig3 = px.bar(cat_year, x="Year", y="Days", color="Category",
                  color_discrete_map=AQI_COLORS,
                  category_orders={"Category": cat_order},
                  template="plotly_dark")
    fig3.update_layout(height=350, margin=dict(l=0,r=0,t=20,b=0))
    st.plotly_chart(fig3, use_container_width=True)

    # Heatmap
    st.markdown(f"#### 🔥 Monthly AQI Heatmap — {selected_city}")
    heat_df = city_df.groupby(["Year","MonthNum"])["AQI"].mean().reset_index()
    heat_pivot = heat_df.pivot(index="Year", columns="MonthNum", values="AQI")
    heat_pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                          "Jul","Aug","Sep","Oct","Nov","Dec"]
    fig4 = px.imshow(heat_pivot, color_continuous_scale="RdYlGn_r",
                     template="plotly_dark", text_auto=".0f",
                     aspect="auto")
    fig4.update_layout(height=280, margin=dict(l=0,r=0,t=20,b=0))
    st.plotly_chart(fig4, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# TAB 4: AQI FORECAST
# ═══════════════════════════════════════════════════════════
with tab4:
    st.markdown("## 🤖 AQI Forecasting — Machine Learning Model")
    st.markdown("*Predicts AQI for the next 7 days based on historical patterns and seasonal trends*")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("#### ⚙️ Forecast Settings")
        forecast_city   = st.selectbox("City", CITIES, index=CITIES.index(selected_city), key="fc")
        forecast_month  = st.slider("Month", 1, 12, 1)
        pm25_input  = st.slider("PM2.5 (μg/m³)", 10, 300, 80)
        pm10_input  = st.slider("PM10 (μg/m³)",  20, 500, 120)
        no2_input   = st.slider("NO2 (μg/m³)",   5,  150, 40)
        predict_btn = st.button("🔮 Predict AQI", type="primary", use_container_width=True)

    with col2:
        if predict_btn:
            with st.spinner("Running forecast model..."):
                model, le = train_forecast_model(df)

                city_enc = le.transform([forecast_city])[0]
                # Generate 7-day forecast
                forecasts = []
                base_so2 = 20; base_co = 15; base_o3 = 40
                for day_offset in range(1, 8):
                    day_of_year = min(forecast_month * 30 + day_offset, 365)
                    features = [[
                        forecast_month,
                        day_of_year,
                        day_offset % 7,
                        2024,
                        pm25_input * np.random.uniform(0.95, 1.05),
                        pm10_input * np.random.uniform(0.95, 1.05),
                        no2_input  * np.random.uniform(0.95, 1.05),
                        base_so2   * np.random.uniform(0.90, 1.10),
                        base_co    * np.random.uniform(0.90, 1.10),
                        base_o3    * np.random.uniform(0.90, 1.10),
                        city_enc
                    ]]
                    pred = model.predict(features)[0]
                    forecasts.append({"Day": f"Day {day_offset}", "AQI": round(pred, 1)})

                fc_df = pd.DataFrame(forecasts)
                fc_df["Category"] = fc_df["AQI"].apply(
                    lambda x: "Good" if x<=50 else "Satisfactory" if x<=100 else
                              "Moderate" if x<=200 else "Poor" if x<=300 else
                              "Very Poor" if x<=400 else "Severe")
                fc_df["Color"] = fc_df["Category"].map(AQI_COLORS)

                st.markdown(f"#### 📅 7-Day AQI Forecast — {forecast_city}")
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=fc_df["Day"], y=fc_df["AQI"],
                    marker_color=fc_df["Color"],
                    text=fc_df["AQI"], textposition="outside",
                    name="Forecasted AQI"
                ))
                fig.add_hline(y=100, line_dash="dot", line_color="#92D050",
                              annotation_text="Satisfactory threshold")
                fig.add_hline(y=200, line_dash="dot", line_color="#FF7E00",
                              annotation_text="Poor threshold")
                fig.update_layout(template="plotly_dark", height=350,
                                  margin=dict(l=0,r=60,t=20,b=0))
                st.plotly_chart(fig, use_container_width=True)

                # Category summary
                st.markdown("#### 📋 Forecast Summary")
                cols = st.columns(7)
                for i, (_, row) in enumerate(fc_df.iterrows()):
                    emoji, _, color = AQI_HEALTH[row["Category"]]
                    cols[i].markdown(f"""
                    <div style='text-align:center; background:#1e2130;
                                border-radius:10px; padding:10px;
                                border-top: 3px solid {color};'>
                        <div style='color:#aaa; font-size:11px;'>{row['Day']}</div>
                        <div style='color:{color}; font-size:20px; font-weight:bold;'>{row['AQI']}</div>
                        <div style='font-size:10px; color:#888;'>{row['Category']}</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("👈 Configure settings and click **Predict AQI** to run the forecast model.")

            # Model info
            st.markdown("#### 🧠 Model Details")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("""
                **Algorithm:** Random Forest Regressor
                - 80 decision trees
                - Trained on 54,810 data points
                - 30 Indian cities, 5 years
                """)
            with col_b:
                st.markdown("""
                **Features Used:**
                - Month, Day of Year, Day of Week
                - PM2.5, PM10, NO2, SO2, CO, O3
                - City encoding
                - Year (trend capture)
                """)

# ═══════════════════════════════════════════════════════════
# TAB 5: INDIA MAP
# ═══════════════════════════════════════════════════════════
with tab5:
    st.markdown("## 🗺️ India AQI Map")

    map_year = st.selectbox("Select Year for Map", [2020,2021,2022,2023,2024],
                             index=4, key="map_year")
    map_season = st.selectbox("Season", ["All","winter","summer","monsoon","post_monsoon"], key="map_season")

    map_df = df[df["Year"] == map_year].copy()
    if map_season != "All":
        map_df = map_df[map_df["Season"] == map_season]

    city_map = map_df.groupby(["City","Latitude","Longitude","State"])["AQI"].mean().reset_index()
    city_map["AQI"] = city_map["AQI"].round(1)
    city_map["Category"] = city_map["AQI"].apply(
        lambda x: "Good" if x<=50 else "Satisfactory" if x<=100 else
                  "Moderate" if x<=200 else "Poor" if x<=300 else
                  "Very Poor" if x<=400 else "Severe")

    fig = px.scatter_mapbox(
        city_map,
        lat="Latitude", lon="Longitude",
        color="AQI",
        size="AQI",
        size_max=35,
        hover_name="City",
        hover_data={"State": True, "AQI": True, "Category": True,
                    "Latitude": False, "Longitude": False},
        color_continuous_scale=["#00E400","#FFFF00","#FF7E00","#FF0000","#8B0000"],
        range_color=[50, 300],
        mapbox_style="carto-darkmatter",
        zoom=4.2,
        center={"lat": 22.5, "lon": 80.0},
        title=f"Average AQI — {map_year} ({map_season.title()})"
    )
    fig.update_layout(
        height=600,
        template="plotly_dark",
        margin=dict(l=0, r=0, t=40, b=0),
        coloraxis_colorbar=dict(title="AQI")
    )
    st.plotly_chart(fig, use_container_width=True)

    # Top 10 most polluted
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🔴 Top 10 Most Polluted Cities")
        top10 = city_map.nlargest(10, "AQI")[["City","State","AQI","Category"]]
        top10["AQI"] = top10["AQI"].round(1)
        st.dataframe(top10.reset_index(drop=True), use_container_width=True,
                     hide_index=True)
    with col2:
        st.markdown("#### 🟢 Top 10 Cleanest Cities")
        bot10 = city_map.nsmallest(10, "AQI")[["City","State","AQI","Category"]]
        bot10["AQI"] = bot10["AQI"].round(1)
        st.dataframe(bot10.reset_index(drop=True), use_container_width=True,
                     hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #555; font-size: 12px; padding: 20px;'>
    Built by <b>Atharva Rane</b> | Data Analyst | Python • Streamlit • Scikit-learn • Plotly<br>
    <a href='https://github.com/AtharvaRane7/india-aqi-intelligence' style='color: #4FC3F7;'>
    GitHub Repository</a>
</div>
""", unsafe_allow_html=True)
