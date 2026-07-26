import streamlit as str
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def load_data_and_train_model():
    df = pd.read_csv("clean_aqi.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    
    X = df[["hour", "day_of_week"]]
    y = df["value"]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return df, model

try:
    df, model = load_data_and_train_model()
    
    # Page Header Banner
    str.markdown("<h1 style='color: #003366; font-size: 32px; font-weight: 700; margin-bottom: 0px;'>Global Air Quality Insights Platform</h1>", unsafe_allow_html=True)
    str.markdown("<p style='color: #475569; font-size: 16px; margin-top: 4px;'>Interactive exploration and machine learning forecasting dashboard.</p>", unsafe_allow_html=True)
    str.markdown("<hr style='border-top: 2px solid #e2e8f0; margin-bottom: 30px;'>", unsafe_allow_html=True)
    
    # Section 1: Data View
    str.markdown("<h2 style='color: #0056b3; font-size: 22px; font-weight: 600;'>1. Cleaned Air Quality Dataset</h2>", unsafe_allow_html=True)
    str.dataframe(df.head(10), use_container_width=True)
    
    csv_data = df.to_csv(index=False).encode('utf-8')
    str.download_button(
        label="Download Prepared Dataset as CSV",
        data=csv_data,
        file_name="crea_processed_aqi.csv",
        mime="text/csv"
    )
    
    str.markdown("<br><br>", unsafe_allow_html=True)
    
    # Section 2: Visual Charts
    str.markdown("<h2 style='color: #0056b3; font-size: 22px; font-weight: 600;'>2. Exploratory Trends</h2>", unsafe_allow_html=True)
    chart_choice = str.selectbox("Select Visual Data Chart View:", ["Hourly Fluctuations", "Concentration Timeline"])
    
    if chart_choice == "Hourly Fluctuations":
        str.image("hourly_trends.png", caption="Analysis of hourly rush-hour pollution spikes")
    else:
        str.image("pollution_timeline.png", caption="Complete historical time-series observation")
        
    str.markdown("<br><br>", unsafe_allow_html=True)
    
    # Section 3: Forecasting UI Split Layout
    str.markdown("<h2 style='color: #0056b3; font-size: 22px; font-weight: 600;'>3. Predictive Forecaster & Policy Simulator</h2>", unsafe_allow_html=True)
    
    col1, col2 = str.columns([1, 1], gap="large")
    
    with col1:
        str.markdown("<p style='font-weight: 600; color: #1e293b;'>Scenario Parameters</p>", unsafe_allow_html=True)
        input_hour = str.slider("Select Hour of Day (24h clock)", 0, 23, 12)
        input_day = str.selectbox("Select Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        policy_reduction = str.slider("Simulate Emission Reduction Policy (%)", 0, 100, 0)
        
    with col2:
        day_mapping = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
        numerical_day = day_mapping[input_day]
        
        query_features = pd.DataFrame([[input_hour, numerical_day]], columns=["hour", "day_of_week"])
        predicted_metric = model.predict(query_features)
        final_prediction = float(predicted_metric[0] * (1 - (policy_reduction / 100.0)))
        
        # Display targeted metrics card block
        str.markdown("<p style='font-weight: 600; color: #1e293b; margin-bottom: 2px;'>Model Output</p>", unsafe_allow_html=True)
        str.metric(label="Predicted PM2.5 Concentration", value=f"{final_prediction:.2f} µg/m³")
        
        str.markdown("<p style='font-weight: 600; color: #1e293b; margin-top: 15px; margin-bottom: 5px;'>WHO Safety Assessment</p>", unsafe_allow_html=True)
        WHO_24H_LIMIT = 15.0
        
        if final_prediction <= WHO_24H_LIMIT:
            str.markdown(f"<div style='background-color: #ecfdf5; border-left: 4px solid #10b981; padding: 12px; border-radius: 4px; color: #065f46; font-size: 14px;'><strong>SAFE:</strong> Target prediction ({final_prediction:.2f} µg/m³) is within WHO limits of {WHO_24H_LIMIT} µg/m³.</div>", unsafe_allow_html=True)
        else:
            excess = final_prediction - WHO_24H_LIMIT
            str.markdown(f"<div style='background-color: #fef2f2; border-left: 4px solid #ef4444; padding: 12px; border-radius: 4px; color: #991b1b; font-size: 14px;'><strong>WARNING:</strong> Prediction exceeds WHO guidelines by {excess:.2f} µg/m³.</div>", unsafe_allow_html=True)

except FileNotFoundError:
    str.error("Missing baseline dataset components. Please execute your preprocessing script pipeline first.")
