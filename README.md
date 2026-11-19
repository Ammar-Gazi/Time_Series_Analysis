# 🎬 Theatrical Box Office Demand Forecasting & Capacity Planning

An end-to-end time series forecasting and studio capacity planning pipeline built with **Python, SQL (SQLite), Statsmodels, Scikit-Learn, and Streamlit**. 

The system ingests real-world theatrical release records from the RapidAPI IMDb API, resolves upstream schema anomalies, aggregates monthly worldwide gross revenue (2010–2025), benchmarks statistical vs. machine learning forecasters via **5-fold walk-forward validation**, and translates forecast accuracy into **asymmetric theatrical operational cost savings**.

---

## 🏗️ End-to-End Architecture

```text
[RapidAPI IMDb (/api/imdb/search)]
               │
               ▼
[Data Engineering & Anomaly Patch] ──► Fixes 32-bit int overflow (> $2.147B)
               │
               ▼
[SQLite Warehouse (box_office.db)] ──► 902 clean theatrical releases (2010–2025)
               │
               ▼
[SQL Monthly Aggregation] ───────────► 192 continuous monthly observations
               │
               ▼
[Statistical Diagnostics] ───────────► STL Decomposition + ADF Test + ACF/PACF
               │
               ▼
[Leak-Free Feature Engineering] ─────► Cyclical sin/cos + ACF lags + shifted rolling stats
               │
               ▼
[5-Fold Walk-Forward Validation]
   ├──► Tier 1: Seasonal Naive Baseline (t - 12)
   ├──► Tier 2: SARIMAX (1,1,1)x(1,1,0,12) + COVID-19 Structural Break Exog
   └──► Tier 3: Quantile Gradient Boosting (Point + 90% Prediction Intervals)
               │
               ▼
[Asymmetric Business Cost Engine] ───► $0.35M Under-Forecast vs. $0.10M Over-Forecast
               │
               ▼
[Streamlit Interactive Dashboard] ───► 2026 H1 Recursive Forecast & Cost Simulator
