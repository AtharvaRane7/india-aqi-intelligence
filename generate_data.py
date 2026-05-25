"""
Generate realistic India AQI dataset based on actual pollution patterns
"""
import pandas as pd
import numpy as np

np.random.seed(42)

CITIES = {
    "Delhi":       {"base": 180, "winter": 80,  "summer": -20, "monsoon": -60, "lat": 28.66, "lon": 77.21, "state": "Delhi"},
    "Mumbai":      {"base": 95,  "winter": 20,  "summer": 10,  "monsoon": -40, "lat": 19.08, "lon": 72.88, "state": "Maharashtra"},
    "Pune":        {"base": 85,  "winter": 15,  "summer": 10,  "monsoon": -35, "lat": 18.52, "lon": 73.86, "state": "Maharashtra"},
    "Kolkata":     {"base": 140, "winter": 60,  "summer": -10, "monsoon": -50, "lat": 22.57, "lon": 88.36, "state": "West Bengal"},
    "Chennai":     {"base": 75,  "winter": 10,  "summer": 15,  "monsoon": -20, "lat": 13.08, "lon": 80.27, "state": "Tamil Nadu"},
    "Bangalore":   {"base": 70,  "winter": 10,  "summer": 5,   "monsoon": -25, "lat": 12.97, "lon": 77.59, "state": "Karnataka"},
    "Hyderabad":   {"base": 90,  "winter": 20,  "summer": 15,  "monsoon": -30, "lat": 17.38, "lon": 78.49, "state": "Telangana"},
    "Ahmedabad":   {"base": 130, "winter": 50,  "summer": 20,  "monsoon": -40, "lat": 23.03, "lon": 72.58, "state": "Gujarat"},
    "Lucknow":     {"base": 160, "winter": 70,  "summer": 10,  "monsoon": -55, "lat": 26.85, "lon": 80.95, "state": "Uttar Pradesh"},
    "Kanpur":      {"base": 175, "winter": 75,  "summer": 15,  "monsoon": -60, "lat": 26.46, "lon": 80.33, "state": "Uttar Pradesh"},
    "Jaipur":      {"base": 145, "winter": 55,  "summer": 25,  "monsoon": -45, "lat": 26.91, "lon": 75.79, "state": "Rajasthan"},
    "Bhopal":      {"base": 110, "winter": 35,  "summer": 10,  "monsoon": -40, "lat": 23.26, "lon": 77.41, "state": "Madhya Pradesh"},
    "Nagpur":      {"base": 100, "winter": 25,  "summer": 15,  "monsoon": -35, "lat": 21.15, "lon": 79.09, "state": "Maharashtra"},
    "Patna":       {"base": 165, "winter": 75,  "summer": 10,  "monsoon": -55, "lat": 25.60, "lon": 85.12, "state": "Bihar"},
    "Surat":       {"base": 115, "winter": 40,  "summer": 15,  "monsoon": -40, "lat": 21.17, "lon": 72.83, "state": "Gujarat"},
    "Vadodara":    {"base": 105, "winter": 30,  "summer": 15,  "monsoon": -35, "lat": 22.31, "lon": 73.19, "state": "Gujarat"},
    "Indore":      {"base": 105, "winter": 30,  "summer": 10,  "monsoon": -35, "lat": 22.72, "lon": 75.86, "state": "Madhya Pradesh"},
    "Coimbatore":  {"base": 65,  "winter": 5,   "summer": 10,  "monsoon": -20, "lat": 11.01, "lon": 76.96, "state": "Tamil Nadu"},
    "Kochi":       {"base": 55,  "winter": 5,   "summer": 5,   "monsoon": -15, "lat": 9.93,  "lon": 76.27, "state": "Kerala"},
    "Guwahati":    {"base": 95,  "winter": 25,  "summer": 5,   "monsoon": -30, "lat": 26.14, "lon": 91.74, "state": "Assam"},
    "Chandigarh":  {"base": 130, "winter": 50,  "summer": 10,  "monsoon": -45, "lat": 30.73, "lon": 76.78, "state": "Punjab"},
    "Amritsar":    {"base": 150, "winter": 65,  "summer": 15,  "monsoon": -50, "lat": 31.63, "lon": 74.87, "state": "Punjab"},
    "Varanasi":    {"base": 170, "winter": 80,  "summer": 10,  "monsoon": -55, "lat": 25.32, "lon": 83.01, "state": "Uttar Pradesh"},
    "Agra":        {"base": 160, "winter": 75,  "summer": 15,  "monsoon": -55, "lat": 27.18, "lon": 78.01, "state": "Uttar Pradesh"},
    "Meerut":      {"base": 165, "winter": 75,  "summer": 10,  "monsoon": -55, "lat": 28.98, "lon": 77.71, "state": "Uttar Pradesh"},
    "Visakhapatnam":{"base": 80, "winter": 15,  "summer": 10,  "monsoon": -25, "lat": 17.69, "lon": 83.22, "state": "Andhra Pradesh"},
    "Thiruvananthapuram":{"base":55,"winter": 5, "summer": 5,  "monsoon": -15, "lat": 8.52,  "lon": 76.94, "state": "Kerala"},
    "Jodhpur":     {"base": 140, "winter": 50,  "summer": 30,  "monsoon": -35, "lat": 26.29, "lon": 73.02, "state": "Rajasthan"},
    "Ranchi":      {"base": 100, "winter": 30,  "summer": 5,   "monsoon": -35, "lat": 23.34, "lon": 85.31, "state": "Jharkhand"},
    "Raipur":      {"base": 110, "winter": 35,  "summer": 10,  "monsoon": -35, "lat": 21.25, "lon": 81.63, "state": "Chhattisgarh"},
}

def get_season(month):
    if month in [12, 1, 2]:   return "winter"
    elif month in [3, 4, 5]:  return "summer"
    elif month in [6, 7, 8, 9]: return "monsoon"
    else:                       return "post_monsoon"

def generate_pollutants(aqi):
    pm25  = aqi * np.random.uniform(0.4, 0.6)
    pm10  = aqi * np.random.uniform(0.6, 0.9)
    no2   = aqi * np.random.uniform(0.15, 0.25)
    so2   = aqi * np.random.uniform(0.05, 0.12)
    co    = aqi * np.random.uniform(0.08, 0.15)
    o3    = max(20, aqi * np.random.uniform(0.10, 0.20))
    return round(pm25,1), round(pm10,1), round(no2,1), round(so2,1), round(co,1), round(o3,1)

rows = []
dates = pd.date_range("2020-01-01", "2024-12-31", freq="D")

for city, params in CITIES.items():
    for date in dates:
        month = date.month
        season = get_season(month)
        seasonal_adj = params.get(season, params.get("post_monsoon", 0) if season == "post_monsoon" else 0)
        if season == "post_monsoon":
            seasonal_adj = params["winter"] * 0.5

        day_noise = np.random.normal(0, 20)
        weekday_effect = -5 if date.weekday() >= 5 else 0
        trend = (date.year - 2020) * -3

        aqi = max(10, params["base"] + seasonal_adj + day_noise + weekday_effect + trend)
        aqi = round(aqi, 1)

        pm25, pm10, no2, so2, co, o3 = generate_pollutants(aqi)

        if aqi <= 50:       category = "Good"
        elif aqi <= 100:    category = "Satisfactory"
        elif aqi <= 200:    category = "Moderate"
        elif aqi <= 300:    category = "Poor"
        elif aqi <= 400:    category = "Very Poor"
        else:               category = "Severe"

        rows.append({
            "Date": date.strftime("%Y-%m-%d"),
            "City": city,
            "State": params["state"],
            "Latitude": params["lat"],
            "Longitude": params["lon"],
            "AQI": aqi,
            "Category": category,
            "Season": season,
            "Month": date.strftime("%B"),
            "Year": date.year,
            "PM2_5": pm25,
            "PM10": pm10,
            "NO2": no2,
            "SO2": so2,
            "CO": co,
            "O3": o3,
        })

df = pd.DataFrame(rows)
df.to_csv("/home/claude/projects/india-aqi-intelligence/data/india_aqi_2020_2024.csv", index=False)
print(f"✓ Dataset: {len(df):,} rows across {df.City.nunique()} cities")
print(df.groupby("City")["AQI"].mean().sort_values(ascending=False).head(10))
