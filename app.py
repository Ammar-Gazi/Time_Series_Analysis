import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Box Office Demand Forecasting", layout="wide")
st.title("🎬 Theatrical Box Office Forecasting & Capacity Planning")
st.caption("End-to-End Time Series Pipeline: RapidAPI IMDb Ingestion → SQLite → Walk-Forward Validation → Quantile Gradient Boosting")

# Load exported pipeline artifacts
df_ts = pd.read_csv("monthly_box_office_ts.csv", parse_dates=["month_date"])
df_cv = pd.read_csv("walk_forward_cv_metrics.csv")
df_oof = pd.read_csv("oof_predictions.csv", parse_dates=["month_date"])
df_future = pd.read_csv("future_forecast_6m.csv", parse_dates=["month_date"])

# Sidebar Interactive Cost Simulator
st.sidebar.header("⚙️ Studio Cost Parameters")
under_rate = st.sidebar.slider("Under-Forecast Loss ($M per $1M missed)", 0.10, 0.80, 0.35, 0.05)
over_rate = st.sidebar.slider("Over-Forecast Overhead ($M per $1M excess)", 0.05, 0.50, 0.10, 0.05)

def calc_cost(y_true, y_pred):
    diff = y_true - y_pred
    return float(np.sum(np.maximum(0, diff) * under_rate + np.maximum(0, -diff) * over_rate))

# Dynamic Cost Calculation on Out-of-Fold Validation Set
naive_total_cost = calc_cost(df_oof["total_gross_musd"], df_oof["Naive_Pred"])
gb_total_cost = calc_cost(df_oof["total_gross_musd"], df_oof["XGB_Pred"])
cost_savings_pct = ((naive_total_cost - gb_total_cost) / naive_total_cost) * 100

mae_naive = df_cv["Naive_MAE ($M)"].mean()
mae_gb = df_cv["XGB_MAE ($M)"].mean()
mae_imp = ((mae_naive - mae_gb) / mae_naive) * 100

# Top KPI Row
c1, c2, c3 = st.columns(3)
c1.metric("6-Month Forecasted Gross (2026 H1)", f"${df_future['Forecast_Gross ($M)'].sum():,.1f}M")
c2.metric("Walk-Forward MAE", f"${mae_gb:.1f}M", f"-{mae_imp:.1f}% vs. Seasonal Naive")
c3.metric("Simulated Operational Cost Savings", f"${(naive_total_cost - gb_total_cost):,.1f}M", f"-{cost_savings_pct:.1f}% vs. Baseline")

tab1, tab2 = st.tabs(["📈 2026 Forecast & 90% Confidence Band", "📊 Walk-Forward Validation Benchmark"])

with tab1:
    recent_hist = df_ts[df_ts["month_date"] >= "2022-01-01"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent_hist["month_date"], y=recent_hist["total_gross_musd"],
                             mode="lines+markers", name="Actual Monthly Gross ($M)", line=dict(color="#1f77b4", width=2.5)))
    fig.add_trace(go.Scatter(x=df_future["month_date"], y=df_future["Forecast_Gross ($M)"],
                             mode="lines+markers", name="2026 Forecast ($M)", line=dict(color="#ff7f0e", width=3, dash="dash")))
    fig.add_trace(go.Scatter(
        x=pd.concat([df_future["month_date"], df_future["month_date"][::-1]]),
        y=pd.concat([df_future["Upper_90_CI ($M)"], df_future["Lower_90_CI ($M)"][::-1]]),
        fill="toself", fillcolor="rgba(255, 127, 14, 0.2)", line=dict(color="rgba(255,255,255,0)"),
        name="90% Prediction Interval"
    ))
    fig.update_layout(height=430, yaxis_title="Worldwide Gross ($M)", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df_future, use_container_width=True)

with tab2:
    st.subheader("5-Fold Expanding Window Walk-Forward Validation")
    st.dataframe(df_cv, use_container_width=True)