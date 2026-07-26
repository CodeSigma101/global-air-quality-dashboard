import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_exploratory_analysis(csv_path):
    print("Loading prepared dataset for visual exploration...")
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Extract structural time components for metric charting
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.day_name()
    df["month"] = df["timestamp"].dt.month
    
    # Chart 1: The Pollution Concentration Timeline
    plt.figure(figsize=(12, 5))
    plt.plot(df["timestamp"], df["value"], color="teal", linewidth=1.5)
    plt.title("Air Quality Concentration Timeline (PM2.5)")
    plt.xlabel("Date Timeline")
    plt.ylabel("Concentration Value")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig("pollution_timeline.png", bbox_inches="tight")
    plt.close()
    print("Saved timeline chart to: pollution_timeline.png")
    
    # Chart 2: Hourly Breakdown (Analyzing Rush Hour Triggers)
    plt.figure(figsize=(10, 5))
    sns.lineplot(x="hour", y="value", data=df, color="crimson", marker="o")
    plt.title("Hourly Air Pollution Fluctuations")
    plt.xlabel("Hour of Day (24h Clock)")
    plt.ylabel("Average PM2.5 Concentration")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig("hourly_trends.png", bbox_inches="tight")
    plt.close()
    print("Saved hourly variance graph to: hourly_trends.png")

if __name__ == "__main__":
    try:
        perform_exploratory_analysis("clean_aqi.csv")
        print("\nExploratory data analysis phase completed successfully.")
        print("Check your folder to view 'pollution_timeline.png' and 'hourly_trends.png'!")
    except FileNotFoundError:
        print("Error: 'clean_aqi.csv' not found. Run your data fetch file first.")
