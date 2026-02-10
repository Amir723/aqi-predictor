"""
Generate 3-Day Forecast for Karachi
"""
import pandas as pd
import numpy as np
from pymongo import MongoClient
import joblib
import sys
import os
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

def load_best_model():
    """Load best model"""
    print("🔍 Loading model...")
    
    if not os.path.exists("models/best_model.pkl"):
        raise Exception("❌ Model not found! Run: python training/train_model.py")
    
    model = joblib.load("models/best_model.pkl")
    metadata = joblib.load("models/model_metadata.pkl")
    
    print(f"✅ Model: {metadata['model_name']}")
    print(f"   RMSE: {metadata['metrics']['rmse']:.3f}")
    
    return model, metadata


def load_latest_features():
    """Load latest features from MongoDB"""
    print(f"\n📚 Loading latest data...")
    
    client = MongoClient(config.MONGO_URI)
    db = client[config.DB_NAME]
    collection = db[config.FEATURES_COLLECTION]
    
    latest = collection.find_one(sort=[("timestamp", -1)])
    
    if not latest:
        raise Exception("❌ No data! Run: python data/backfill.py --days 90")
    
    df = pd.DataFrame([latest])
    df = df.drop(columns=config.DROP_FEATURES + [config.TARGET_COLUMN], errors="ignore")
    df = df.select_dtypes(include=[np.number])
    
    timestamp = latest.get("timestamp")
    print(f"✅ Loaded from: {timestamp}")
    
    return df, timestamp


def predict_next_3_days(model, features):
    """Generate 72-hour forecast"""
    print(f"\n🔮 Generating {config.PREDICTION_DAYS}-day forecast...")
    
    predictions = []
    
    for hour in range(config.TOTAL_PREDICTION_HOURS):
        pred = model.predict(features)[0]
        predictions.append(pred)
        
        if (hour + 1) % 24 == 0:
            day = (hour + 1) // 24
            print(f"   Day {day}: Avg AQI = {np.mean(predictions[-24:]):.1f}")
    
    return predictions


def save_predictions(predictions, start_timestamp):
    """Save to MongoDB"""
    print(f"\n💾 Saving predictions...")
    
    client = MongoClient(config.MONGO_URI)
    db = client[config.DB_NAME]
    collection = db[config.PREDICTIONS_COLLECTION]
    
    # Clear old
    collection.delete_many({})
    
    # Create records
    records = []
    for hour, pred_aqi in enumerate(predictions):
        forecast_time = start_timestamp + timedelta(hours=hour + 1)
        
        category = get_aqi_category(pred_aqi)
        is_hazardous = bool(pred_aqi >= config.HAZARDOUS_THRESHOLD)
        
        records.append({
            "prediction_timestamp": datetime.now(),
            "forecast_timestamp": forecast_time,
            "forecast_hour": hour + 1,
            "forecast_day": (hour // 24) + 1,
            "predicted_aqi": float(pred_aqi),
            "category": category,
            "is_hazardous": is_hazardous,
            "city": config.CITY
        })
    
    collection.insert_many(records)
    print(f"✅ Saved {len(records)} predictions")
    
    return records


def get_aqi_category(aqi):
    """Get category"""
    for category, (low, high) in config.AQI_CATEGORIES.items():
        if low <= aqi <= high:
            return category
    return "Unknown"


def generate_summary(predictions):
    """Show summary"""
    print(f"\n{'='*60}")
    print(f"📊 3-DAY FORECAST - {config.CITY}")
    print(f"{'='*60}\n")
    
    print(f"Overall:")
    print(f"   Avg AQI: {np.mean(predictions):.1f}")
    print(f"   Min: {min(predictions):.1f}")
    print(f"   Max: {max(predictions):.1f}")
    
    for day in range(1, 4):
        start = (day - 1) * 24
        end = day * 24
        day_preds = predictions[start:end]
        avg = np.mean(day_preds)
        
        print(f"\nDay {day}:")
        print(f"   Avg: {avg:.1f} ({get_aqi_category(avg)})")
        print(f"   Range: {min(day_preds):.1f} - {max(day_preds):.1f}")
    
    hazardous = sum(1 for p in predictions if p >= config.HAZARDOUS_THRESHOLD)
    if hazardous > 0:
        print(f"\n⚠️ WARNING: {hazardous} hours unhealthy!")
    else:
        print(f"\n✅ No hazardous levels")


def run_prediction_pipeline():
    """Main prediction"""
    print(f"\n{'='*60}")
    print(f"🔮 PREDICTION - {datetime.now()}")
    print(f"{'='*60}\n")
    
    try:
        model, metadata = load_best_model()
        features, last_timestamp = load_latest_features()
        predictions = predict_next_3_days(model, features)
        save_predictions(predictions, last_timestamp)
        generate_summary(predictions)
        
        print(f"\n{'='*60}")
        print("✅ PREDICTION COMPLETED")
        print(f"{'='*60}\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_prediction_pipeline()
    sys.exit(0 if success else 1)