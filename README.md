# 🌫️ India AQI Intelligence Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://india-aqi-intelligence.streamlit.app)

A comprehensive **Air Quality Analysis & Forecasting** web application for 30 Indian cities (2020–2024). Built with Python, Streamlit, Plotly, and Scikit-learn.

---

## 🔴 Live App
👉 **[india-aqi-intelligence.streamlit.app](https://india-aqi-intelligence.streamlit.app)**

---

## 🎯 What This App Does

| Feature | Description |
|---|---|
| 🏙️ City Overview | Real-time AQI dashboard with pollutant breakdown for any city |
| 🔀 City Comparison | Compare AQI, pollutants, and seasonal patterns across multiple cities |
| 📈 Trends | Year-over-year analysis, seasonal heatmaps, category distributions |
| 🤖 AQI Forecast | ML-powered 7-day AQI prediction using Random Forest |
| 🗺️ India Map | Interactive choropleth map showing AQI across all cities |

---

## 📊 Dataset

- **30 Indian cities** — Delhi, Mumbai, Bangalore, Chennai, Kolkata, Pune and more
- **5 years** of daily data (2020–2024) — 54,810 data points
- **Pollutants tracked** — PM2.5, PM10, NO2, SO2, CO, O3
- Based on real seasonal and regional pollution patterns from CPCB data

---

## 🤖 ML Model

- **Algorithm:** Random Forest Regressor
- **Features:** Month, Day, Season, Pollutant concentrations, City
- **Training data:** 43,848 samples | **Test data:** 10,962 samples
- **Output:** 7-day AQI forecast with category and health risk

---

## 🛠️ Tech Stack

```
Python 3.11     → Core language
Streamlit       → Web app framework
Plotly          → Interactive visualizations
Scikit-learn    → Machine learning (Random Forest)
Pandas / NumPy  → Data processing
```

---

## ▶️ Run Locally

```bash
git clone https://github.com/AtharvaRane7/india-aqi-intelligence
cd india-aqi-intelligence
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Project Structure

```
india-aqi-intelligence/
├── app.py                    ← Main Streamlit application
├── generate_data.py          ← Dataset generation script
├── data/
│   └── india_aqi_2020_2024.csv  ← 54,810 rows of AQI data
├── .streamlit/
│   └── config.toml           ← Dark theme configuration
├── requirements.txt
└── README.md
```

---

## 🏥 AQI Health Categories

| AQI Range | Category | Health Impact |
|---|---|---|
| 0–50 | 🟢 Good | Minimal impact |
| 51–100 | 🟡 Satisfactory | Minor breathing discomfort |
| 101–200 | 🟠 Moderate | Breathing discomfort for sensitive people |
| 201–300 | 🔴 Poor | Breathing discomfort for most |
| 301–400 | 🔴 Very Poor | Respiratory illness risk |
| 400+ | ⚫ Severe | Serious health effects for all |

---

Built by **Atharva Rane** | Data Analyst | [GitHub](https://github.com/AtharvaRane7) | [LinkedIn](https://linkedin.com/in/atharvarane)
