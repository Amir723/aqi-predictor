# 🌤️ Karachi AQI Prediction System

**3-Day Air Quality Forecast for Karachi using Machine Learning**

Predicts Air Quality Index (AQI) for Karachi for the next 3 days using historical data from OpenMeteo API, MongoDB feature store, and MLflow model registry.

---

## 🎯 Features

- ✅ **3 Months Historical Data** - Fetches 90 days of hourly AQI data from OpenMeteo API
- ✅ **85+ Engineered Features** - Time-based, rolling averages, change rates, lag features
- ✅ **MongoDB Feature Store** - Scalable data storage
- ✅ **3 ML Models** - Random Forest, Gradient Boosting, Ridge Regression
- ✅ **MLflow Model Registry** - Experiment tracking and model versioning
- ✅ **3-Day Forecast** - 72-hour hourly predictions
- ✅ **Interactive Dashboard** - Streamlit web app with real-time AQI
- ✅ **GitHub Actions Automation** - Hourly data updates, daily model retraining

---

## 📊 Model Performance

| Model | RMSE | MAE | R² | Status |
|-------|------|-----|-----|---------|
| **Gradient Boosting** | **7.490** | 5.176 | **0.922** | ✅ **Best** |
| Random Forest | 8.026 | 5.059 | 0.910 | Good |
| Ridge Regression | 10.291 | 5.988 | 0.852 | Baseline |

**R² = 0.922** means the model is **92.2% accurate**!

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- MongoDB Atlas account (free tier works)
- Git

### 1. Clone Repository
```bash
git clone https://github.com/Amir723/aqi-predictor.git
cd aqi-predictor
```

### 2. Install Dependencies
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### 3. Configure Environment

Create `.env` file:
```bash
MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/
MLFLOW_TRACKING_URI=mlruns
```

### 4. Fetch Historical Data (3 months)
```bash
python data/backfill.py --days 90
```

**Output:** Fetches 2000+ hourly records from OpenMeteo

### 5. Train Models
```bash
python training/train_model.py
```

**Output:** Trains 3 models, selects best, saves to MLflow

### 6. Generate 3-Day Forecast
```bash
python prediction/predict_next_3_days.py
```

**Output:** Creates 72-hour predictions, stores in MongoDB

### 7. Launch Dashboard
```bash
streamlit run dashboard/app.py
```

**Dashboard:** Opens at http://localhost:8501

---

## 📁 Project Structure
```
aqi-predictor/
├── .github/workflows/      # GitHub Actions automation
│   ├── feature_hourly.yml  # Hourly data fetching
│   └── train_daily.yml     # Daily model retraining
├── config.py               # Configuration (location, models)
├── requirements.txt        # Dependencies
├── features/
│   ├── fetch_data.py       # OpenMeteo API integration
│   └── feature_engineering.py  # Feature creation
├── data/
│   └── backfill.py         # Historical data fetching
├── training/
│   └── train_model.py      # 3 models training pipeline
├── prediction/
│   └── predict_next_3_days.py  # 72-hour forecast
└── dashboard/
    └── app.py              # Streamlit web interface
```

---

## 🌍 Data Source

**OpenMeteo Air Quality API** - Completely free, no API key required!

- **Historical Data:** 90 days of hourly records
- **Parameters:** PM2.5, PM10, CO, NO₂, SO₂, O₃, dust, UV index
- **Weather:** Temperature, humidity, wind speed, pressure
- **Location:** Karachi (24.8607°N, 67.0011°E)

API Documentation: https://open-meteo.com/en/docs/air-quality-api

---

## 🤖 Automation (GitHub Actions)

### Hourly Pipeline
- **Schedule:** Every hour (`0 * * * *`)
- **Action:** Fetches current AQI data from OpenMeteo
- **Updates:** MongoDB feature store

### Daily Pipeline
- **Schedule:** Daily at 2 AM UTC (`0 2 * * *`)
- **Actions:**
  1. Retrains all 3 models
  2. Selects best model (by RMSE)
  3. Registers in MLflow
  4. Generates fresh 3-day forecast

### Setup GitHub Actions

1. Add secrets to GitHub repo:
   - Settings → Secrets and variables → Actions
   - Add: `MONGO_URI` (MongoDB connection string)
   - Add: `MLFLOW_TRACKING_URI` (set to `mlruns`)

2. Workflows run automatically on schedule or manually via Actions tab

---

## 📊 Dashboard Features

- **Current AQI** - Color-coded health category (Good/Moderate/Unhealthy)
- **3-Day Forecast** - Hourly predictions chart
- **Daily Averages** - Summary for each day
- **Pollutant Levels** - PM2.5, PM10 concentrations
- **Weather Data** - Temperature, humidity

---

## 🧪 Feature Engineering

**100+ features including:**

- **Time-based:** hour, day, weekday, month, season, cyclical encodings
- **Rolling stats:** 24-hour & 7-day means, std, max
- **Change rates:** 1h, 3h, 24h changes and percentages
- **Derived:** PM ratio, pollutant index, temp-humidity interaction
- **Lag features:** Previous 1h, 3h, 24h values
- **Volatility:** Rolling standard deviations

---

## 🛠️ Technology Stack

- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Model Registry:** MLflow
- **Database:** MongoDB Atlas
- **Dashboard:** Streamlit, Plotly
- **API:** OpenMeteo (free)
- **Automation:** GitHub Actions

---

## 📈 AQI Categories

| AQI Range | Category | Color | Health Impact |
|-----------|----------|-------|---------------|
| 0-50 | Good | 🟢 Green | No health risk |
| 51-100 | Moderate | 🟡 Yellow | Acceptable |
| 101-150 | Unhealthy for Sensitive | 🟠 Orange | Sensitive groups affected |
| 151-200 | Unhealthy | 🔴 Red | Everyone affected |
| 201-300 | Very Unhealthy | 🟣 Purple | Health warnings |
| 301+ | Hazardous | 🟤 Maroon | Emergency conditions |

---

## 🔧 Configuration

Edit `config.py` to customize:

- **Location:** Change `LATITUDE` and `LONGITUDE` for different cities
- **Models:** Adjust hyperparameters in `MODELS_CONFIG`
- **Prediction Horizon:** Change `PREDICTION_DAYS` (default: 3)
- **Thresholds:** Modify `HAZARDOUS_THRESHOLD` for alerts

---

## 📝 Requirements

- Python 3.10+
- MongoDB Atlas account (free tier)
- 2GB RAM minimum
- Internet connection for API calls

---

## 🐛 Troubleshooting

### No data in Feature Store
```bash
python data/backfill.py --days 90
```

### Model not found
```bash
python training/train_model.py
```

### MongoDB connection failed
- Check `MONGO_URI` in `.env` file
- Verify IP whitelist (set to `0.0.0.0/0` for testing)
- Ensure database user has read/write permissions

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 📄 License

This project is open source and available under the MIT License.

---

## 🙏 Acknowledgments

- **OpenMeteo** - Free air quality API
- **MongoDB Atlas** - Cloud database
- **MLflow** - Model registry
- **Streamlit** - Dashboard framework

---

## 📧 Contact

**Amir** - [@Amir723](https://github.com/Amir723)

Project Link: [https://github.com/Amir723/aqi-predictor](https://github.com/Amir723/aqi-predictor)

---

**Made with ❤️ for cleaner air in Karachi 🌤️**
