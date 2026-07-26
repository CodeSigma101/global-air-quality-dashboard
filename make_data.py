import pandas as pd
import numpy as np

def generate_local_dataset():
    print("Generating synthetic air quality time-series data...")
    
    # Create a range of timestamps for the last 2 weeks (hourly records)
    date_range = pd.date_range(start="2026-07-12", end="2026-07-26", freq="h")
    
    extracted_data = []
    for dt in date_range:
        # Generate baseline values with a simulated rush-hour peak cycle
        hour = dt.hour
        base_value = 12.5
        
        # Simulate traffic spikes at 8 AM and 6 PM
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            base_value += np.random.uniform(15.0, 30.0)
        else:
            base_value += np.random.uniform(2.0, 10.0)
            
        extracted_data.append({
            "timestamp": dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "parameter": "pm25",
            "value": round(base_value, 2),
            "unit": "µg/m³"
        })
        
    df = pd.DataFrame(extracted_data)
    df.to_csv("clean_aqi.csv", index=False)
    print("Successfully generated and saved local dataset to 'clean_aqi.csv'!")

if __name__ == "__main__":
    generate_local_dataset()
