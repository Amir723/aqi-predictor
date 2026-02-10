"""
Fetch Historical AQI Data from OpenMeteo API
Fetches 3 months of historical air quality data for Karachi
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def fetch_historical_data(days=90):
    """
    Fetch historical air quality AND weather data from OpenMeteo
    
    Args:
        days: Number of days to fetch (default 90 = 3 months)
    
    Returns:
        DataFrame with hourly air quality and weather data
    """
    print(f"🌍 Fetching {days} days of data for {config.CITY}...")
    
    # Use yesterday as end date (API doesn't support future dates)
    end_date = (datetime.now() - timedelta(days=1)).date()
    start_date = end_date - timedelta(days=days)
    
    # STEP 1: Fetch AIR QUALITY data
    print("   Fetching air quality data...")
    params_air = {
        "latitude": config.LATITUDE,
        "longitude": config.LONGITUDE,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": ",".join(config.AIR_QUALITY_PARAMS),
        "timezone": "auto"
    }
    
    try:
        response = requests.get(config.OPENMETEO_BASE_URL, params=params_air, timeout=30)
        response.raise_for_status()
        data_air = response.json()
        hourly_air = data_air.get("hourly", {})
    except Exception as e:
        print(f"❌ Air quality fetch failed: {e}")
        raise
    
    # STEP 2: Fetch WEATHER data from Archive API
    print("   Fetching weather data...")
    ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
    
    params_weather = {
        "latitude": config.LATITUDE,
        "longitude": config.LONGITUDE,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": ",".join(config.WEATHER_PARAMS),
        "timezone": "auto"
    }
    
    try:
        response = requests.get(ARCHIVE_URL, params=params_weather, timeout=30)
        response.raise_for_status()
        data_weather = response.json()
        hourly_weather = data_weather.get("hourly", {})
    except Exception as e:
        print(f"❌ Weather fetch failed: {e}")
        raise
    
    # STEP 3: Combine both datasets
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly_air["time"]),
        "city": config.CITY,
        
        # Air Quality
        "pm25": hourly_air.get("pm2_5", [None] * len(hourly_air["time"])),
        "pm10": hourly_air.get("pm10", [None] * len(hourly_air["time"])),
        "co": hourly_air.get("carbon_monoxide", [None] * len(hourly_air["time"])),
        "no2": hourly_air.get("nitrogen_dioxide", [None] * len(hourly_air["time"])),
        "so2": hourly_air.get("sulphur_dioxide", [None] * len(hourly_air["time"])),
        "o3": hourly_air.get("ozone", [None] * len(hourly_air["time"])),
        "dust": hourly_air.get("dust", [None] * len(hourly_air["time"])),
        "uv_index": hourly_air.get("uv_index", [None] * len(hourly_air["time"])),
        
        # Weather
        "temp": hourly_weather.get("temperature_2m", [None] * len(hourly_air["time"])),
        "humidity": hourly_weather.get("relative_humidity_2m", [None] * len(hourly_air["time"])),
        "wind_speed": hourly_weather.get("wind_speed_10m", [None] * len(hourly_air["time"])),
        "pressure": hourly_weather.get("surface_pressure", [None] * len(hourly_air["time"])),
    })
    
    # Calculate AQI from PM2.5 if not present
    if "pm25" in df.columns and not df["pm25"].isna().all():
        print("   Calculating AQI from PM2.5...")
        df["aqi"] = calculate_aqi_from_pm25(df["pm25"])
    
    # Remove rows with all nulls
    df = df.dropna(subset=["pm25", "pm10"], how="all")
    
    print(f"✅ Fetched {len(df)} hourly records with air quality + weather")
    
    return df


def calculate_aqi_from_pm25(pm25_series):
    """
    Calculate AQI from PM2.5 using EPA standard
    """
    def pm25_to_aqi(pm25):
        if pd.isna(pm25):
            return None
        
        # EPA AQI breakpoints for PM2.5
        if pm25 <= 12.0:
            return linear_scale(pm25, 0, 12.0, 0, 50)
        elif pm25 <= 35.4:
            return linear_scale(pm25, 12.1, 35.4, 51, 100)
        elif pm25 <= 55.4:
            return linear_scale(pm25, 35.5, 55.4, 101, 150)
        elif pm25 <= 150.4:
            return linear_scale(pm25, 55.5, 150.4, 151, 200)
        elif pm25 <= 250.4:
            return linear_scale(pm25, 150.5, 250.4, 201, 300)
        elif pm25 <= 350.4:
            return linear_scale(pm25, 250.5, 350.4, 301, 400)
        else:
            return linear_scale(pm25, 350.5, 500.4, 401, 500)
    
    return pm25_series.apply(pm25_to_aqi)


def linear_scale(value, in_min, in_max, out_min, out_max):
    """Linear interpolation"""
    return ((value - in_min) / (in_max - in_min)) * (out_max - out_min) + out_min


def fetch_current_data():
    """
    Fetch current hour's data (for real-time updates)
    """
    print(f"📡 Fetching current data for {config.CITY}...")
    
    # Get just today's data
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    
    # Air quality
    params_air = {
        "latitude": config.LATITUDE,
        "longitude": config.LONGITUDE,
        "start_date": yesterday.strftime("%Y-%m-%d"),
        "end_date": today.strftime("%Y-%m-%d"),
        "hourly": ",".join(config.AIR_QUALITY_PARAMS),
        "timezone": "auto"
    }
    
    # Weather
    ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
    params_weather = {
        "latitude": config.LATITUDE,
        "longitude": config.LONGITUDE,
        "start_date": yesterday.strftime("%Y-%m-%d"),
        "end_date": today.strftime("%Y-%m-%d"),
        "hourly": ",".join(config.WEATHER_PARAMS),
        "timezone": "auto"
    }
    
    try:
        # Fetch air quality
        response_air = requests.get(config.OPENMETEO_BASE_URL, params=params_air, timeout=10)
        response_air.raise_for_status()
        data_air = response_air.json()
        hourly_air = data_air.get("hourly", {})
        
        # Fetch weather
        response_weather = requests.get(ARCHIVE_URL, params=params_weather, timeout=10)
        response_weather.raise_for_status()
        data_weather = response_weather.json()
        hourly_weather = data_weather.get("hourly", {})
        
        # Combine
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(hourly_air["time"]),
            "city": config.CITY,
            "pm25": hourly_air.get("pm2_5", [None] * len(hourly_air["time"])),
            "pm10": hourly_air.get("pm10", [None] * len(hourly_air["time"])),
            "co": hourly_air.get("carbon_monoxide", [None] * len(hourly_air["time"])),
            "no2": hourly_air.get("nitrogen_dioxide", [None] * len(hourly_air["time"])),
            "so2": hourly_air.get("sulphur_dioxide", [None] * len(hourly_air["time"])),
            "o3": hourly_air.get("ozone", [None] * len(hourly_air["time"])),
            "dust": hourly_air.get("dust", [None] * len(hourly_air["time"])),
            "uv_index": hourly_air.get("uv_index", [None] * len(hourly_air["time"])),
            "temp": hourly_weather.get("temperature_2m", [None] * len(hourly_air["time"])),
            "humidity": hourly_weather.get("relative_humidity_2m", [None] * len(hourly_air["time"])),
            "wind_speed": hourly_weather.get("wind_speed_10m", [None] * len(hourly_air["time"])),
            "pressure": hourly_weather.get("surface_pressure", [None] * len(hourly_air["time"])),
        })
        
        # Calculate AQI
        if not df["pm25"].isna().all():
            df["aqi"] = calculate_aqi_from_pm25(df["pm25"])
        
        # Get latest hour
        latest = df.iloc[-1].to_dict()
        
        print(f"✅ Current AQI: {latest.get('aqi', 'N/A')}")
        
        return latest
        
    except Exception as e:
        print(f"❌ Failed to fetch current data: {e}")
        raise


if __name__ == "__main__":
    # Test fetching
    print(f"\n{'='*60}")
    print("🧪 TESTING OPENMETEO API")
    print(f"{'='*60}\n")
    
    # Test historical data
    try:
        df = fetch_historical_data(days=7)
        print(f"\n📊 Data Summary:")
        print(df[["timestamp", "aqi", "pm25", "pm10", "temp", "humidity"]].head(10))
        print(f"\n✅ Columns: {list(df.columns)}")
        
    except Exception as e:
        print(f"\n❌ Historical fetch failed: {e}")
    
    # Test current data
    try:
        current = fetch_current_data()
        print(f"\n📍 Current Data:")
        print(f"   Timestamp: {current['timestamp']}")
        print(f"   AQI: {current['aqi']:.1f}" if current.get('aqi') else "   AQI: N/A")
        print(f"   PM2.5: {current['pm25']:.1f}" if current.get('pm25') else "   PM2.5: N/A")
        print(f"   Temperature: {current['temp']:.1f}°C" if current.get('temp') else "   Temperature: N/A")
        
    except Exception as e:
        print(f"\n❌ Current fetch failed: {e}")