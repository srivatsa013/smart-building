"""
fetch_weather_profiles.py
Fetches ~1 year of real hourly outdoor temperatures for Vellore, India
from the Open-Meteo archive API (free, no API key required).

Saves: weather_profiles.npy  — shape (N_days, 24)  float32 array
       Each row is one day's 24-hour outdoor temperature profile (°C).

Run once before training:
    python fetch_weather_profiles.py

train.py will automatically use weather_profiles.npy if it exists.
"""

import numpy as np
import datetime
import sys

try:
    import requests
except ImportError:
    print("ERROR: requests is not installed. Run: pip install requests")
    sys.exit(1)

# ── Location: Vellore, Tamil Nadu, India ─────────────────────────────────────
LATITUDE  = 12.9165
LONGITUDE = 79.1325
CITY_NAME = "Vellore, India"

# ── Fetch last 365 days of hourly temperature ─────────────────────────────────
end_date   = datetime.date.today() - datetime.timedelta(days=1)   # yesterday
start_date = end_date - datetime.timedelta(days=364)

print(f"Fetching hourly temperatures for {CITY_NAME}")
print(f"  Period : {start_date}  →  {end_date}")
print(f"  Source : Open-Meteo Archive API (free, no key needed)\n")

resp = requests.get(
    "https://archive-api.open-meteo.com/v1/archive",
    params={
        "latitude":        LATITUDE,
        "longitude":       LONGITUDE,
        "start_date":      str(start_date),
        "end_date":        str(end_date),
        "hourly":          "temperature_2m",
        "timezone":        "Asia/Kolkata",
    },
    timeout=30,
)
resp.raise_for_status()
data = resp.json()

hourly_temps = data["hourly"]["temperature_2m"]   # list of 8760 floats
print(f"  Received {len(hourly_temps)} hourly readings")

# ── Reshape into daily 24-hour profiles ──────────────────────────────────────
n_complete_days = len(hourly_temps) // 24
profiles = []

for day in range(n_complete_days):
    day_temps = hourly_temps[day * 24 : (day + 1) * 24]
    # Skip days with missing data (Open-Meteo returns None for gaps)
    if any(t is None for t in day_temps):
        continue
    profiles.append(day_temps)

profiles = np.array(profiles, dtype=np.float32)   # shape (N_days, 24)

print(f"  Valid days  : {len(profiles)}")
print(f"  Temp range  : {profiles.min():.1f}°C  –  {profiles.max():.1f}°C")
print(f"  Daily mean  : {profiles.mean(axis=1).mean():.1f}°C")

np.save("weather_profiles.npy", profiles)
print(f"\nSaved: weather_profiles.npy  ({profiles.shape[0]} days × 24 hours)")
print("Run python train.py to retrain the model with real weather data.")
