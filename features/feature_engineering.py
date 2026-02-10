"""
Feature Engineering for AQI Prediction
Creates time-based features, rolling aggregations, and derived features
"""
import pandas as pd
import numpy as np
import config

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create comprehensive features for AQI prediction
    
    Features include:
    - Time-based: hour, day, weekday, month, season
    - Rolling aggregations: 24h, 7-day windows
    - Derived features: change rates, ratios, interactions
    """
    df = df.copy()
    
    # ==================== TIMESTAMP HANDLING ====================
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # ==================== CREATE AQI IF MISSING ====================
    # If AQI not present, calculate from PM2.5
    if "aqi" not in df.columns and "pm25" in df.columns:
        print("   Calculating AQI from PM2.5...")
        df["aqi"] = calculate_aqi_from_pm25(df["pm25"])
    
    # ==================== TIME-BASED FEATURES ====================
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["weekday"] = df["timestamp"].dt.weekday
    df["month"] = df["timestamp"].dt.month
    df["year"] = df["timestamp"].dt.year
    
    # Cyclical encoding for hour (sine/cosine transformation)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    
    # Cyclical encoding for month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    # Season (1=Winter, 2=Spring, 3=Summer, 4=Fall)
    df["season"] = df["month"].apply(lambda x: (x % 12 + 3) // 3)
    
    # Weekend flag
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    
    # Rush hour flag (7-9 AM, 5-7 PM)
    df["is_rush_hour"] = df["hour"].apply(
        lambda x: 1 if (7 <= x <= 9) or (17 <= x <= 19) else 0
    )
    
    # ==================== ROLLING FEATURES (24H) ====================
    rolling_cols = ["pm25", "pm10", "o3", "no2", "so2", "co", "temp", "humidity"]
    
    for col in rolling_cols:
        if col in df.columns:
            # 24-hour rolling mean
            df[f"{col}_rolling_24h_mean"] = df[col].rolling(
                window=24, min_periods=1
            ).mean()
            
            # 24-hour rolling std (volatility)
            df[f"{col}_rolling_24h_std"] = df[col].rolling(
                window=24, min_periods=1
            ).std()
            
            # 24-hour rolling max
            df[f"{col}_rolling_24h_max"] = df[col].rolling(
                window=24, min_periods=1
            ).max()
    
    # ==================== ROLLING FEATURES (7 DAYS) ====================
    for col in ["pm25", "pm10", "aqi"]:
        if col in df.columns:
            df[f"{col}_rolling_7d_mean"] = df[col].rolling(
                window=168, min_periods=1
            ).mean()
    
    # ==================== CHANGE RATES ====================
    for col in ["pm25", "pm10", "o3", "no2", "aqi"]:
        if col in df.columns:
            # 1-hour change
            df[f"{col}_change_1h"] = df[col].diff(1)
            
            # 3-hour change
            df[f"{col}_change_3h"] = df[col].diff(3)
            
            # 24-hour change
            df[f"{col}_change_24h"] = df[col].diff(24)
            
            # Percentage change
            df[f"{col}_pct_change"] = df[col].pct_change(1) * 100
    
    # ==================== DERIVED FEATURES ====================
    # PM ratio (fine vs coarse particles)
    if "pm25" in df.columns and "pm10" in df.columns:
        df["pm_ratio"] = df["pm25"] / (df["pm10"] + 1)  # +1 to avoid division by zero
    
    # Combined pollutant index
    pollutant_cols = ["pm25", "pm10", "o3", "no2", "so2"]
    available_pollutants = [col for col in pollutant_cols if col in df.columns]
    if available_pollutants:
        df["combined_pollutant_index"] = df[available_pollutants].mean(axis=1)
    
    # Temperature-humidity interaction (heat index proxy)
    if "temp" in df.columns and "humidity" in df.columns:
        df["temp_humidity_interaction"] = df["temp"] * df["humidity"] / 100
    
    # AQI change rate (critical feature)
    if "aqi" in df.columns:
        df["aqi_change_rate"] = df["aqi"].diff() / df["aqi"].shift(1)
        df["aqi_volatility"] = df["aqi"].rolling(window=24, min_periods=1).std()
    
    # ==================== LAG FEATURES ====================
    # Previous hour values (important for time series)
    for col in ["pm25", "pm10", "aqi"]:
        if col in df.columns:
            df[f"{col}_lag_1h"] = df[col].shift(1)
            df[f"{col}_lag_3h"] = df[col].shift(3)
            df[f"{col}_lag_24h"] = df[col].shift(24)
    
    # ==================== TARGET VARIABLE ====================
    # Predict AQI for next hour
    if "aqi" in df.columns:
        df["target_aqi"] = df["aqi"].shift(-1)
    
    # ==================== CLEANUP ====================
    # Drop rows with NaN in target
    if "target_aqi" in df.columns:
        df = df[df["target_aqi"].notna()]
    
    # Fill remaining NaNs with forward fill, then backward fill
    df = df.ffill().bfill()
    
    # Drop any remaining NaNs
    df = df.dropna()
    
    # Reset index
    df = df.reset_index(drop=True)
    
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Get list of feature columns (excluding metadata and target)
    """
    exclude_cols = config.DROP_FEATURES + [config.TARGET_COLUMN]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols


def calculate_aqi_from_pm25(pm25_series):
    """Calculate AQI from PM2.5 using EPA standard"""
    def pm25_to_aqi(pm25):
        if pd.isna(pm25):
            return None
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
        else:
            return linear_scale(pm25, 250.5, 500.4, 301, 500)
    
    return pm25_series.apply(pm25_to_aqi)


def linear_scale(value, in_min, in_max, out_min, out_max):
    """Linear interpolation"""
    return ((value - in_min) / (in_max - in_min)) * (out_max - out_min) + out_min


if __name__ == "__main__":
    # Test feature engineering
    print("🧪 Testing feature engineering...")
    
    # Create sample data
    dates = pd.date_range(start="2024-01-01", periods=200, freq="H")
    sample_data = pd.DataFrame({
        "timestamp": dates,
        "city": "karachi",
        "pm25": np.random.randint(20, 100, 200),
        "pm10": np.random.randint(30, 150, 200),
        "o3": np.random.randint(10, 80, 200),
        "no2": np.random.randint(5, 50, 200),
        "so2": np.random.randint(2, 30, 200),
        "co": np.random.uniform(0.5, 3.0, 200),
        "temp": np.random.randint(15, 35, 200),
        "humidity": np.random.randint(30, 90, 200),
    })
    
    # Apply feature engineering
    features_df = create_features(sample_data)
    
    print(f"\n✅ Feature engineering complete!")
    print(f"   Original columns: {len(sample_data.columns)}")
    print(f"   Feature columns: {len(features_df.columns)}")
    print(f"   Rows: {len(features_df)}")
    print(f"\n📊 Feature columns created:")
    feature_cols = get_feature_columns(features_df)
    for i, col in enumerate(feature_cols[:10], 1):
        print(f"   {i}. {col}")
    if len(feature_cols) > 10:
        print(f"   ... and {len(feature_cols) - 10} more")