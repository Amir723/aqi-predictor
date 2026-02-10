"""
Configuration for AQI Prediction System
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ==================== DATABASE ====================
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "aqi_db"
FEATURES_COLLECTION = "features"
PREDICTIONS_COLLECTION = "predictions"

# ==================== LOCATION (KARACHI) ====================
CITY = "Karachi"
LATITUDE = 24.8607
LONGITUDE = 67.0011

# ==================== OPENMETEO API ====================
OPENMETEO_BASE_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

AIR_QUALITY_PARAMS = [
    "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
    "sulphur_dioxide", "ozone", "dust", "uv_index"  # ✅ aqi removed
]

WEATHER_PARAMS = [
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
    "surface_pressure"
]

# ==================== MLFLOW ====================
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
MLFLOW_EXPERIMENT_NAME = "Karachi_AQI_Prediction"
MODEL_REGISTRY_NAME = "Best_AQI_Model"

# ==================== MODELS TO TRAIN ====================
MODELS_CONFIG = {
    "RandomForest": {
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_split": 5,
        "random_state": 42
    },
    "GradientBoosting": {
        "n_estimators": 200,
        "learning_rate": 0.1,
        "max_depth": 5,
        "random_state": 42
    },
    "Ridge": {
        "alpha": 1.0,
        "random_state": 42
    }
}

# ==================== FEATURE ENGINEERING ====================
ROLLING_WINDOW_24H = 24
ROLLING_WINDOW_7D = 168  # 7 days * 24 hours

# ==================== TRAINING ====================
TEST_SIZE = 0.2
RANDOM_STATE = 42
TARGET_COLUMN = "target_aqi"
DROP_FEATURES = ["_id", "timestamp", "aqi", "city"]

# ==================== AQI THRESHOLDS ====================
AQI_CATEGORIES = {
    "Good": (0, 50),
    "Moderate": (51, 100),
    "Unhealthy for Sensitive Groups": (101, 150),
    "Unhealthy": (151, 200),
    "Very Unhealthy": (201, 300),
    "Hazardous": (301, 500)
}

HAZARDOUS_THRESHOLD = 150

# ==================== PREDICTION ====================
PREDICTION_DAYS = 3
TOTAL_PREDICTION_HOURS = 72  # 3 days * 24 hours
