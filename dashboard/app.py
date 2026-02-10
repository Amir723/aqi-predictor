"""
Streamlit Dashboard - Karachi AQI 3-Day Forecast
Shows current AQI, historical trends, and next 3 days predictions
"""
import streamlit as st
import pandas as pd
import numpy as np
from pymongo import MongoClient
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

# Page config
st.set_page_config(
    page_title=f"{config.CITY} AQI Forecast",
    page_icon="🌤️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .big-font {
        font-size: 50px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_db():
    """Get MongoDB connection"""
    client = MongoClient(config.MONGO_URI)
    return client[config.DB_NAME]

def get_aqi_color(aqi):
    """Get color based on AQI value"""
    if aqi <= 50:
        return "#00E400"  # Good
    elif aqi <= 100:
        return "#FFFF00"  # Moderate
    elif aqi <= 150:
        return "#FF7E00"  # Unhealthy for Sensitive
    elif aqi <= 200:
        return "#FF0000"  # Unhealthy
    elif aqi <= 300:
        return "#8F3F97"  # Very Unhealthy
    else:
        return "#7E0023"  # Hazardous

def get_aqi_category(aqi):
    """Get AQI category"""
    for category, (low, high) in config.AQI_CATEGORIES.items():
        if low <= aqi <= high:
            return category
    return "Unknown"

def load_current_aqi():
    """Load current AQI"""
    db = get_db()
    collection = db[config.FEATURES_COLLECTION]
    
    latest = collection.find_one(sort=[("timestamp", -1)])
    
    if latest:
        return {
            "aqi": latest.get("aqi", 0),
            "pm25": latest.get("pm25", 0),
            "pm10": latest.get("pm10", 0),
            "temp": latest.get("temp", 0),
            "humidity": latest.get("humidity", 0),
            "timestamp": latest.get("timestamp")
        }
    return None

def load_historical_data(days=7):
    """Load historical data"""
    db = get_db()
    collection = db[config.FEATURES_COLLECTION]
    
    cutoff = datetime.now() - timedelta(days=days)
    data = list(collection.find(
        {"timestamp": {"$gte": cutoff}},
        {"timestamp": 1, "aqi": 1, "pm25": 1, "pm10": 1, "temp": 1}
    ).sort("timestamp", 1))
    
    if data:
        return pd.DataFrame(data)
    return pd.DataFrame()

def load_predictions():
    """Load 3-day predictions"""
    db = get_db()
    collection = db[config.PREDICTIONS_COLLECTION]
    
    # Get latest predictions
    data = list(collection.find().sort("forecast_hour", 1))
    
    if data:
        return pd.DataFrame(data)
    return pd.DataFrame()

def plot_historical_trend(df):
    """Plot historical AQI"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["aqi"],
        mode='lines',
        name='AQI',
        line=dict(color='#3366cc', width=2),
        fill='tozeroy',
        fillcolor='rgba(51, 102, 204, 0.1)'
    ))
    
    # Thresholds
    fig.add_hline(y=50, line_dash="dash", line_color="green", annotation_text="Good", annotation_position="right")
    fig.add_hline(y=100, line_dash="dash", line_color="yellow", annotation_text="Moderate", annotation_position="right")
    fig.add_hline(y=150, line_dash="dash", line_color="orange", annotation_text="Unhealthy", annotation_position="right")
    
    fig.update_layout(
        title=f"AQI Trend - Last 7 Days ({config.CITY})",
        xaxis_title="Date",
        yaxis_title="AQI",
        hovermode='x unified',
        height=400,
        showlegend=False
    )
    
    return fig

def plot_3day_forecast(df_pred):
    """Plot 3-day forecast"""
    fig = go.Figure()
    
    # Add predictions
    fig.add_trace(go.Scatter(
        x=df_pred["forecast_timestamp"],
        y=df_pred["predicted_aqi"],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#dc3912', width=3, dash='dot'),
        marker=dict(size=4),
        fill='tozeroy',
        fillcolor='rgba(220, 57, 18, 0.1)'
    ))
    
    # Add hazardous threshold
    fig.add_hline(y=config.HAZARDOUS_THRESHOLD, line_dash="dash", line_color="red", annotation_text="Hazardous Threshold")
    
    fig.update_layout(
        title=f"3-Day AQI Forecast ({config.CITY})",
        xaxis_title="Date & Time",
        yaxis_title="Predicted AQI",
        hovermode='x unified',
        height=400,
        showlegend=False
    )
    
    return fig

def plot_daily_averages(df_pred):
    """Plot daily average predictions"""
    df_pred['day'] = df_pred['forecast_day']
    daily_avg = df_pred.groupby('day')['predicted_aqi'].mean().reset_index()
    daily_avg['day_label'] = ['Day ' + str(int(d)) for d in daily_avg['day']]
    
    fig = go.Figure()
    
    colors = [get_aqi_color(aqi) for aqi in daily_avg['predicted_aqi']]
    
    fig.add_trace(go.Bar(
        x=daily_avg['day_label'],
        y=daily_avg['predicted_aqi'],
        marker_color=colors,
        text=daily_avg['predicted_aqi'].round(1),
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Daily Average AQI Forecast",
        xaxis_title="",
        yaxis_title="Average AQI",
        height=350,
        showlegend=False
    )
    
    return fig

# ==================== MAIN DASHBOARD ====================

def main():
    # Header
    st.title(f"🌤️ {config.CITY} Air Quality Forecast")
    st.markdown("### Real-time AQI & 3-Day Predictions")
    st.markdown("---")
    
    # Load data
    current = load_current_aqi()
    df_historical = load_historical_data(days=7)
    df_predictions = load_predictions()
    
    if current is None:
        st.error("❌ No data available! Run backfill first:\n```python data/backfill.py --days 90```")
        return
    
    # ========== CURRENT AQI ==========
    st.header("📍 Current Air Quality")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        aqi = current["aqi"]
        category = get_aqi_category(aqi)
        color = get_aqi_color(aqi)
        
        st.markdown(f"""
        <div style="background-color: {color}; padding: 30px; border-radius: 15px; text-align: center;">
            <h1 style="color: white; margin: 0; font-size: 60px;">{aqi:.0f}</h1>
            <h3 style="color: white; margin: 0;">{category}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric("PM2.5", f"{current['pm25']:.1f} µg/m³", help="Fine particulate matter")
    
    with col3:
        st.metric("PM10", f"{current['pm10']:.1f} µg/m³", help="Coarse particulate matter")
    
    with col4:
        st.metric("Temperature", f"{current['temp']:.1f}°C")
        st.metric("Humidity", f"{current['humidity']:.0f}%")
    
    # Hazard alert
    if aqi >= config.HAZARDOUS_THRESHOLD:
        st.error(f"⚠️ **UNHEALTHY AIR QUALITY!** Current AQI is {aqi:.0f} ({category}). Limit outdoor activities.")
    
    st.markdown("---")
    
    # ========== FORECASTS ==========
    st.header("🔮 3-Day Forecast")
    
    if df_predictions.empty:
        st.warning("⚠️ No predictions available. Run prediction pipeline:\n```python prediction/predict_next_3_days.py```")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Hourly forecast chart
            fig_forecast = plot_3day_forecast(df_predictions)
            st.plotly_chart(fig_forecast, use_container_width=True)
        
        with col2:
            # Daily averages
            fig_daily = plot_daily_averages(df_predictions)
            st.plotly_chart(fig_daily, use_container_width=True)
        
        # Summary metrics
        st.subheader("📊 Forecast Summary")
        col1, col2, col3 = st.columns(3)
        
        for day in range(1, 4):
            day_data = df_predictions[df_predictions['forecast_day'] == day]
            avg_aqi = day_data['predicted_aqi'].mean()
            max_aqi = day_data['predicted_aqi'].max()
            hazard_hours = (day_data['predicted_aqi'] >= config.HAZARDOUS_THRESHOLD).sum()
            
            with [col1, col2, col3][day-1]:
                st.markdown(f"**Day {day}**")
                st.metric("Avg AQI", f"{avg_aqi:.1f}", help=get_aqi_category(avg_aqi))
                st.metric("Max AQI", f"{max_aqi:.1f}")
                if hazard_hours > 0:
                    st.warning(f"⚠️ {hazard_hours}h unhealthy")
    
    st.markdown("---")
    
    # ========== HISTORICAL TREND ==========
    st.header("📈 Historical Trend")
    
    if not df_historical.empty:
        fig_hist = plot_historical_trend(df_historical)
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("No historical data available")
    
    # ========== FOOTER ==========
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: gray;">
        <p>Last updated: {current['timestamp'].strftime('%Y-%m-%d %H:%M') if isinstance(current['timestamp'], datetime) else 'Unknown'}</p>
        <p>Data source: OpenMeteo API | Model: MLflow Registry</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
