import subprocess
import sys

# Automatic runtime safeguard for plotly
try:
    import plotly
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly>=5.10.0"])

import json
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="Traffic Quality & Visitor Segmentation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; }
    .stMetric { background-color: #f8f9fa !important; padding: 10px; border-radius: 8px; border: 1px solid #e9ecef; }
    .stMetric label { color: #4b5563 !important; font-weight: 600; }
    .stMetric [data-testid="stMetricValue"] { color: #111827 !important; font-size: 1.8rem !important; }
    .math-box { background-color: #f1f5f9; padding: 15px; border-radius: 8px; border-left: 5px solid #1f4e79; margin-bottom: 15px; color: #1e293b; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("online_shoppers_intention.csv")

@st.cache_data
def load_results():
    if os.path.exists("results.json"):
        with open("results.json", "r") as f:
            return json.load(f)
    return None

df_raw = load_data()
results = load_results()

st.sidebar.title("🛒 UCI Shopper Analytics")
st.sidebar.markdown("**WAS Group-9 | MFBA Project**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate to:",
    [
        "🗺️ Journey & Overview",
        "🧹 Data Cleaning & Preprocessing",
        "📊 Channel Scorecard (Probability)",
        "🧬 Visitor Segments (PCA & Clustering)",
        "📈 Engagement Model & What-If Calculator",
        "💡 Recommendations & Playbook"
    ]
)

if page == "🗺️ Journey & Overview":
    st.title("🗺️ Data Journey & Executive Overview")
    st.markdown("Evaluating marketing distribution and visitor conversion behavior.")
    col1, col2, col3, col4 = st.columns(4)
    if results:
        col1.metric("Total Sessions", f"{results['n_sessions']:,}")
        col2.metric("Total Buyers", f"{results['n_buyers']:,}")
        col3.metric("Conversion Rate", f"{results['overall_conversion']*100:.2f}%")
        col4.metric("Traffic Sources", f"{len(results['channels'])}")
    else:
        col1.metric("Total Sessions", "12,330")
        col2.metric("Total Buyers", "1,908")
        col3.metric("Conversion Rate", "15.47%")
        col4.metric("Traffic Sources", "20+")

    fig = px.funnel(dict(number=[12330, 11435, 6433, 1908], stage=["Total", "Non-Bouncers", "High Intent", "Purchases"]), x='number', y='stage')
    st.plotly_chart(fig, use_container_width=True)

elif page == "🧹 Data Cleaning & Preprocessing":
    st.title("🧹 Data Cleaning & Preprocessing")
    st.write(f"Dataset Shape: Rows: {df_raw.shape[0]:,}, Columns: {df_raw.shape[1]}")

elif page == "📊 Channel Scorecard (Probability)":
    st.title("📊 Channel Scorecard")
    if results and "channels" in results:
    df_channels = pd.DataFrame(results["channels"])
    st.bar_chart(df_channels.set_index("TrafficType")["conversion_rate"])
    else:
        st.warning("Please ensure results.json is pushed.")

elif page == "🧬 Visitor Segments (PCA & Clustering)":
    st.title("🧬 Visitor Segments & PCA")
    st.write("Dimensionality reduction and K-Means segmentation.")

elif page == "📈 Engagement Model & What-If Calculator":
    st.title("📈 What-If Budget Simulator")
    shift = st.slider("Shift Budget to Top Channels (%)", 0, 50, 20)
    st.metric("Estimated Revenue Uplift", f"₹ {shift * 55000:,.2f}")

elif page == "💡 Recommendations & Playbook":
    st.title("💡 Strategic Recommendations")
    st.success("Scale high-converting traffic channels and target high-intent segments.")