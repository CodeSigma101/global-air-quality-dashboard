import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_predictive_model(csv_path):
    print("Loading dataset for machine learning training...")
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Feature Engineering: Extract numerical time features for the model
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    
    # Define our inputs (X) and the target variable we want to predict (y)
    X = df[["hour", "day_of_week"]]
    y = df["value"]
    
    # Split data: 80% for training the model, 20% for testing its accuracy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest Regressor model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    print("\nModel Evaluation Results:")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} µg/m³")
    print(f"R² Score (Accuracy Metric): {r2:.2f}")
    
    # Proactively test a future scenario: Predict pollution level at 8:00 AM on a Monday (Day 0)
    sample_input = pd.DataFrame([[8, 0]], columns=["hour", "day_of_week"])
    predicted_val = model.predict(sample_input)[0]
    print(f"\nProactive Forecast: Predicted PM2.5 at 8:00 AM on Monday: {predicted_val:.2f} µg/m³")

if __name__ == "__main__":
    try:
        train_predictive_model("clean_aqi.csv")
        print("\nPredictive modeling phase completed successfully.")
    except FileNotFoundError:
        print("Error: 'clean_aqi.csv' missing. Please run 'make_data.py' first.")
