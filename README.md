# Global Air Quality Insights & Forecasting Platform
####App Link:https://global-air-quality-dashboard.streamlit.app/

An end-to-end data science pipeline and predictive modeling platform built to ingest, analyze, and forecast global ambient PM2.5 air pollution concentration trends. This project simulates the complete lifecycle of environmental data engineering and policy-driven data science.

## 🚀 Key Features

* **Data Engineering Pipeline**: Automated preprocessing module that structures irregular time-series measurements, handles data validation, and enforces data integrity rules.
* **Exploratory Data Analysis Suite**: Analytics engine that isolates and models systemic seasonal variances and hourly pollution spikes during high-density urban traffic frames.
* **Machine Learning Engine**: Trained Random Forest Regressor optimized using Scikit-Learn to map temporal variations, achieving a performance metric of R² = 0.80.
* **Interactive Policy Dashboard**: Responsive Streamlit application featuring live scenario parameter slide controls, a simulated emission-reduction policy injector, and real-time risk evaluations aligned with World Health Organization (WHO) safety guidelines.

## 📁 Repository Architecture

* `data_fetch.py`: Production API ingestion module using native HTTP protocols.
* `make_data.py`: High-fidelity synthetic time-series generation engine.
* `eda.py`: Matplotlib and Seaborn analytics generator for media-ready data visuals.
* `model.py`: Random Forest configuration, dataset split vectors, and execution logs.
* `app.py`: Interface code supporting adaptive dark/light system profiles.

## 🛠️ Local Setup & Deployment

1. **Clone the Workspace**
   ```bash
   git clone https://github.com
   cd global-air-quality-dashboard
   ```

2. **Install Core System Dependencies**
   ```bash
   pip install pandas requests matplotlib seaborn scikit-learn streamlit
   ```

3. **Initialize the Pipeline Layers**
   ```bash
   python make_data.py
   python eda.py
   python model.py
   ```

4. **Launch the User Interface**
   ```bash
   streamlit run app.py
   ```
