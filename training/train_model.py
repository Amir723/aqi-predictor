"""
Training Pipeline - 3 Models
RandomForest, GradientBoosting, Ridge
"""
import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import mlflow
import mlflow.sklearn
import joblib
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

def load_data_from_mongodb():
    """Load training data from MongoDB"""
    print("📚 Loading data from MongoDB...")
    
    client = MongoClient(config.MONGO_URI)
    db = client[config.DB_NAME]
    collection = db[config.FEATURES_COLLECTION]
    
    data = list(collection.find())
    
    if not data:
        raise Exception("❌ No data! Run: python data/backfill.py --days 90")
    
    df = pd.DataFrame(data)
    print(f"✅ Loaded {len(df)} records")
    
    return df


def prepare_data(df):
    """Prepare features and target"""
    print("\n🔧 Preparing data...")
    
    # Drop metadata
    df = df.drop(columns=config.DROP_FEATURES, errors="ignore")
    
    # Check target
    if config.TARGET_COLUMN not in df.columns:
        raise Exception(f"❌ Target '{config.TARGET_COLUMN}' not found!")
    
    X = df.drop(columns=[config.TARGET_COLUMN])
    y = df[config.TARGET_COLUMN]
    
    # Numeric only
    X = X.select_dtypes(include=[np.number])
    
    # ========== FIX: HANDLE INFINITY & NaN ==========
    print("   Cleaning infinity and NaN values...")
    
    # Replace infinity with NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN with column median
    X = X.fillna(X.median())
    
    # If still any NaN (empty columns), fill with 0
    X = X.fillna(0)
    
    # Verify no infinity/NaN left
    assert not X.isin([np.inf, -np.inf]).any().any(), "Still has infinity!"
    assert not X.isna().any().any(), "Still has NaN!"
    print("   ✅ Data cleaned")
    # ================================================
    
    print(f"✅ Features: {X.shape[1]} columns, {len(X)} samples")
    print(f"✅ Target range: [{y.min():.1f}, {y.max():.1f}]")
    
    return X, y


def evaluate_model(model, X_test, y_test, model_name):
    """Calculate metrics"""
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n📊 {model_name}:")
    print(f"   RMSE: {rmse:.3f}")
    print(f"   MAE:  {mae:.3f}")
    print(f"   R²:   {r2:.3f}")
    
    return {"rmse": rmse, "mae": mae, "r2": r2}


def train_three_models(X_train, X_test, y_train, y_test):
    """Train 3 models"""
    print(f"\n{'='*60}")
    print("🚀 TRAINING 3 MODELS")
    print(f"{'='*60}\n")
    
    # MLflow setup
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    
    models = {}
    results = {}
    
    # MODEL 1: Random Forest
    print("🌳 Model 1: Random Forest...")
    with mlflow.start_run(run_name="RandomForest"):
        model = RandomForestRegressor(**config.MODELS_CONFIG["RandomForest"], n_jobs=-1)
        model.fit(X_train, y_train)
        
        metrics = evaluate_model(model, X_test, y_test, "Random Forest")
        
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        mlflow.log_params(config.MODELS_CONFIG["RandomForest"])
        mlflow.sklearn.log_model(model, "model")
        
        models["RandomForest"] = model
        results["RandomForest"] = metrics
    
    # MODEL 2: Gradient Boosting
    print("\n⚡ Model 2: Gradient Boosting...")
    with mlflow.start_run(run_name="GradientBoosting"):
        model = GradientBoostingRegressor(**config.MODELS_CONFIG["GradientBoosting"])
        model.fit(X_train, y_train)
        
        metrics = evaluate_model(model, X_test, y_test, "Gradient Boosting")
        
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        mlflow.log_params(config.MODELS_CONFIG["GradientBoosting"])
        mlflow.sklearn.log_model(model, "model")
        
        models["GradientBoosting"] = model
        results["GradientBoosting"] = metrics
    
    # MODEL 3: Ridge
    print("\n📈 Model 3: Ridge...")
    with mlflow.start_run(run_name="Ridge"):
        model = Ridge(**config.MODELS_CONFIG["Ridge"])
        model.fit(X_train, y_train)
        
        metrics = evaluate_model(model, X_test, y_test, "Ridge")
        
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        mlflow.log_params(config.MODELS_CONFIG["Ridge"])
        mlflow.sklearn.log_model(model, "model")
        
        models["Ridge"] = model
        results["Ridge"] = metrics
    
    return models, results


def select_best_model(models, results):
    """Select best by RMSE and register in MLflow"""
    print(f"\n{'='*60}")
    print("🏆 SELECTING BEST MODEL")
    print(f"{'='*60}\n")
    
    # Find best
    best_name = min(results, key=lambda x: results[x]["rmse"])
    best_model = models[best_name]
    best_metrics = results[best_name]
    
    print(f"🥇 Best Model: {best_name}")
    print(f"   RMSE: {best_metrics['rmse']:.3f}")
    print(f"   MAE:  {best_metrics['mae']:.3f}")
    print(f"   R²:   {best_metrics['r2']:.3f}")
    
    # Register in MLflow
    print(f"\n📦 Registering in MLflow Model Registry...")
    with mlflow.start_run(run_name=f"BEST_{best_name}"):
        for key, value in best_metrics.items():
            mlflow.log_metric(key, value)
        
        mlflow.sklearn.log_model(
            best_model,
            "model",
            registered_model_name=config.MODEL_REGISTRY_NAME
        )
    
    # Save locally
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_model.pkl")
    
    metadata = {
        "model_name": best_name,
        "metrics": best_metrics,
        "trained_at": datetime.now().isoformat()
    }
    joblib.dump(metadata, "models/model_metadata.pkl")
    
    print(f"✅ Saved: models/best_model.pkl")
    print(f"✅ Registered: '{config.MODEL_REGISTRY_NAME}'")
    
    return best_name, best_model


def run_training_pipeline():
    """Main training pipeline"""
    print(f"\n{'='*60}")
    print(f"🚀 TRAINING - {datetime.now()}")
    print(f"{'='*60}\n")
    
    try:
        # Load
        df = load_data_from_mongodb()
        
        # Prepare
        X, y = prepare_data(df)
        
        # Split
        print(f"\n✂️ Splitting data ({config.TEST_SIZE*100:.0f}% test)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
            shuffle=False  # Time series!
        )
        print(f"✅ Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Train 3 models
        models, results = train_three_models(X_train, X_test, y_train, y_test)
        
        # Select best
        best_name, best_model = select_best_model(models, results)
        
        # Summary
        print(f"\n{'='*60}")
        print("✅ TRAINING COMPLETED")
        print(f"{'='*60}\n")
        
        for name, metrics in results.items():
            symbol = "🥇" if name == best_name else "  "
            print(f"{symbol} {name:20s} - RMSE: {metrics['rmse']:.3f}")
        print()
        
        return True
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_training_pipeline()
    sys.exit(0 if success else 1)