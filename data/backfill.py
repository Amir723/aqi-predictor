"""
Backfill 3 Months Historical Data from OpenMeteo
"""
import pandas as pd
from pymongo import MongoClient
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from features.fetch_data import fetch_historical_data
from features.feature_engineering import create_features

def backfill_openmeteo(days=90):
    """Fetch 3 months of data from OpenMeteo and store in MongoDB"""
    print(f"\n{'='*60}")
    print(f"🔄 BACKFILL STARTED - {datetime.now()}")
    print(f"   Fetching {days} days ({days//30} months) of data")
    print(f"   City: {config.CITY}")
    print(f"{'='*60}\n")
    
    # STEP 1: FETCH DATA
    print("📡 Step 1: Fetching data from OpenMeteo API...")
    try:
        df = fetch_historical_data(days=days)
        df = fetch_historical_data(days=days)
        print(f"✅ Fetched {len(df)} hourly records")
        
        # DEBUG: Check columns
        print(f"\n🔍 DEBUG - Columns in data:")
        print(f"   {list(df.columns)}")
        print(f"\n🔍 DEBUG - First row:")
        print(df.iloc[0])
        print(f"\n🔍 DEBUG - AQI column present? {'aqi' in df.columns}")
        print(f"🔍 DEBUG - Sample AQI values: {df['aqi'].head() if 'aqi' in df.columns else 'NO AQI COLUMN'}")
        print(f"✅ Fetched {len(df)} hourly records")
    except Exception as e:
        print(f"❌ Failed to fetch data: {e}")
        return False
    
    # STEP 2: FEATURE ENGINEERING
    print("\n⚙️ Step 2: Applying feature engineering...")
    try:
        features_df = create_features(df)
        print(f"✅ Features created:")
        print(f"   Rows: {len(features_df)}")
        print(f"   Columns: {len(features_df.columns)}")
    except Exception as e:
        print(f"❌ Feature engineering failed: {e}")
        return False
    
    # STEP 3: STORE IN MONGODB
    print("\n💾 Step 3: Storing in MongoDB Feature Store...")
    try:
        client = MongoClient(config.MONGO_URI)
        db = client[config.DB_NAME]
        collection = db[config.FEATURES_COLLECTION]
        
        # Clear old data
        print("   Clearing old data...")
        collection.delete_many({})
        
        # Insert in batches
        batch_size = 1000
        records = features_df.to_dict("records")
        
        total_inserted = 0
        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]
            collection.insert_many(batch)
            total_inserted += len(batch)
            print(f"   Inserted: {total_inserted}/{len(records)} records")
        
        print(f"\n✅ Backfill completed!")
        print(f"   Total records: {collection.count_documents({})}")
        
        # Save backup
        os.makedirs("data", exist_ok=True)
        features_df.to_csv(f"data/karachi_features_{days}days.csv", index=False)
        print(f"   Backup saved: data/karachi_features_{days}days.csv")
        
    except Exception as e:
        print(f"❌ MongoDB error: {e}")
        return False
    
    print(f"\n{'='*60}")
    print("✅ BACKFILL COMPLETED SUCCESSFULLY")
    print(f"{'='*60}\n")
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Backfill AQI data")
    parser.add_argument("--days", type=int, default=90, help="Days to fetch")
    
    args = parser.parse_args()
    
    success = backfill_openmeteo(days=args.days)
    sys.exit(0 if success else 1)