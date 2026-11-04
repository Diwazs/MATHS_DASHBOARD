import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats, optimize, special
import os

st.set_page_config(
    page_title="Traffic Quality & Visitor Segmentation Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; }
    .stMetric { background-color: #f8f9fa; padding: 10px; border-radius: 8px; border: 1px solid #e9ecef; }
    .stepper { display: flex; justify-content: space-between; margin-bottom: 20px; }
    .step { flex: 1; text-align: center; padding: 10px; background: #e9ecef; margin: 0 4px; border-radius: 6px; font-weight: bold; }
    .step-active { background: #1f4e79; color: white; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("online_shoppers_intention.csv")
    df["Revenue"] = df["Revenue"].astype(int)
    df["Weekend"] = df["Weekend"].astype(int)
    return df

@st.cache_data
def load_results():
    with open("results.json", "r") as f:
        return json.load(f)

df_raw = load_data()
results = load_results()

# Sidebar Navigation
st.sidebar.title("🛒 UCI Shopper Analytics")
st.sidebar.markdown("**WAS Group-9 | MFBA Project**")

page = st.sidebar.radio(
    "Navigate to:",
    [
        "🗺️ Journey & Overview",
        "🧹 Data Cleaning & Preprocessing",
        "🎯 Channel Scorecard (Probability)",
        "🧬 Visitor Segments (PCA & Clustering)",
        "📈 Engagement Model & What-If Calculator",
        "💡 Recommendations & Playbook",
        "📥 Limitations & Downloads"
    ]
)

# Page 1: Journey & Overview
if page == "🗺️ Journey & Overview":
    st.title("🗺️ Data Journey & Executive Overview")
    
    st.markdown("""
    <div class="stepper">
        <div class="step step-active">1. Raw Data (12,330)</div>
        <div class="step step-active">2. Cleaned & Preprocessed</div>
        <div class="step step-active">3. Analysis (4 Modules)</div>
        <div class="step step-active">4. Key Findings</div>
        <div class="step step-active">5. Recommendations</div>
    </div>
    """, unsafe_allow_dict=True)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Sessions", f"{results['n_sessions']:,}")
    col2.metric("Total Buyers", f"{results['n_buyers']:,}")
    col3.metric("Overall Conversion Rate", f"{results['overall_rate']:.2%}")
    col4.metric("Exact Duplicates", f"{results['n_duplicates']}")
    
    st.subheader("📊 Session & Buyer Funnel Story")
    
    fig_funnel = go.Figure(go.Funnel(
        y = ["Total Sessions", "Non-Bouncers (11,435)", "High Intent (Account/Deep - 6,433)", "Purchases"],
        x = [12330, 11435, 6433, 1908],
        textinfo = "value+percent initial",
        marker = {"color": ["#1f4e79", "#2e8b57", "#e69f00", "#c0504d"]}
    ))
    fig_funnel.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_funnel, use_container_width=True)
    
    st.markdown("""
    ### Key Insights at a Glance
    * **Traffic Quality vs Volume:** Only 15.5% of sessions convert. Traffic quality varies dramatically across sources.
    * **Core Source (Type 2):** Generates 31.7% of sessions and **44% of all buyers** (21.6% conversion rate).
    * **Underperforming Traffic (Types 1, 3, 13):** Represent 42.5% of total sessions but deliver only 25.4% of buyers.
    """)

# Page 2: Data Cleaning & Preprocessing
elif page == "🧹 Data Cleaning & Preprocessing":
    st.title("🧹 Data Preparation & Transformation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Skew Transformation: Log(1 + x)")
        feature = st.selectbox("Select feature to inspect transformation:", 
                               ["ProductRelated_Duration", "ProductRelated", "Administrative_Duration", "Informational_Duration"])
        
        fig_skew = go.Figure()
        fig_skew.add_trace(go.Histogram(x=df_raw[feature], name="Raw (Skewed)", marker_color="#c0504d", opacity=0.7))
        fig_skew.add_trace(go.Histogram(x=np.log1p(df_raw[feature]), name="Log Transformed", marker_color="#1f4e79", opacity=0.7))
        fig_skew.update_layout(barmode='overlay', title=f"Raw vs Log(1+x) distribution for {feature}", height=350)
        st.plotly_chart(fig_skew, use_container_width=True)
        
    with col2:
        st.subheader("2. Rare Channel Pooling Interactive Slider")
        threshold = st.slider("Rare Traffic Cutoff Threshold (Sessions)", min_value=10, max_value=500, value=100, step=10)
        
        counts = df_raw["TrafficType"].value_counts()
        rare_channels = counts[counts < threshold].index.tolist()
        pooled_count = counts[counts < threshold].sum()
        
        st.write(f"**Channels pooled into 'Rare':** {sorted(rare_channels)}")
        st.write(f"**Total sessions in pooled 'Rare':** {pooled_count:,} ({pooled_count/len(df_raw):.1%})")
        
        fig_bar = px.bar(
            x=counts.index.astype(str), y=counts.values,
            labels={'x':'Traffic Type', 'y':'Sessions'},
            color=counts.index.isin(rare_channels),
            color_discrete_map={True: '#c0504d', False: '#1f4e79'},
            title="Traffic Volume per Code (Red = Pooled as Rare)"
        )
        fig_bar.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig_bar, use_container_width=True)

# Page 3: Channel Scorecard
elif page == "🎯 Channel Scorecard (Probability)":
    st.title("🎯 Channel Scorecard & Probability Analysis")
    
    scorecard_df = pd.read_csv("channel_scorecard_all.csv").rename(columns={"Unnamed: 0": "TrafficType"})
    
    st.subheader("🥊 Head-to-Head Channel Probability Comparison")
    c1, c2 = st.columns(2)
    ch1 = c1.selectbox("Select Channel A:", scorecard_df["TrafficType"].tolist(), index=7) # Type 8
    ch2 = c2.selectbox("Select Channel B:", scorecard_df["TrafficType"].tolist(), index=11) # Type 13
    
    row1 = scorecard_df[scorecard_df["TrafficType"] == ch1].iloc[0]
    row2 = scorecard_df[scorecard_df["TrafficType"] == ch2].iloc[0]
    
    rng = np.random.default_rng(42)
    draws1 = rng.beta(1 + row1['k'], 1 + row1['n'] - row1['k'], 100000)
    draws2 = rng.beta(1 + row2['k'], 1 + row2['n'] - row2['k'], 100000)
    prob_beat = (draws1 > draws2).mean()
    
    st.success(f"**Probability that Channel {ch1} beats Channel {ch2}:** {prob_beat:.2%}")
    
    st.subheader("🌲 Forest Plot: Raw vs Empirical-Bayes Conversion Rates")
    fig_fp = go.Figure()
    fig_fp.add_trace(go.Scatter(
        x=scorecard_df["rate"], y=scorecard_df["TrafficType"].astype(str),
        mode='markers', name='Raw Rate', marker=dict(color='#7f7f7f', size=8)
    ))
    fig_fp.add_trace(go.Scatter(
        x=scorecard_df["eb_mean"], y=scorecard_df["TrafficType"].astype(str),
        mode='markers', name='Empirical Bayes', marker=dict(color='#c0504d', symbol='diamond', size=10)
    ))
    fig_fp.add_vline(x=results["overall_rate"], line_dash="dash", line_color="#1f4e79", annotation_text="Overall 15.5%")
    fig_fp.update_layout(title="Conversion Rates across all 20 Traffic Types", height=500, xaxis_tickformat=".0%")
    st.plotly_chart(fig_fp, use_container_width=True)

# Page 4: Visitor Segments
elif page == "🧬 Visitor Segments (PCA & Clustering)":
    st.title("🧬 Visitor Segmentation (PCA & Clustering)")
    
    tab1, tab2, tab3 = st.tabs(["📊 Segment Explorer", "🧩 PCA Loadings", "🔥 Channel-by-Segment Mix"])
    
    seg_prof = pd.read_csv("segment_profile.csv")
    mix_df = pd.read_csv("channel_segment_mix.csv")
    
    with tab1:
        st.subheader("The Four Visitor Segments")
        st.dataframe(seg_prof[["Seg", "n", "share_sessions", "pages_prod_med", "admin_mean", "conv", "share_buyers"]].style.format({
            "share_sessions": "{:.1%}", "conv": "{:.1%}", "share_buyers": "{:.1%}", "admin_mean": "{:.2f}"
        }))
        st.image("fig5_segments.png", caption="Segments in PCA Space & Conversion Rates")
        
    with tab2:
        st.subheader("PCA Loadings (Singular Value Decomposition)")
        st.image("fig3_pca_loadings.png", caption="PC1 (Depth), PC2 (Info/Exit), PC3 (Account vs Info)")
        
    with tab3:
        st.subheader("Channel-to-Segment Heatmap (Row Percentages)")
        heatmap_data = mix_df.set_index("Channel")[["0", "1", "2", "3"]]
        heatmap_data.columns = ["1. Bouncers", "2. Light Browsers", "3. Account-Engaged", "4. Deep Researchers"]
        
        fig_hm = px.imshow(
            heatmap_data, labels=dict(x="Segment", y="Traffic Source", color="Mix Share"),
            color_continuous_scale="Viridis", text_auto=".1%"
        )
        fig_hm.update_layout(height=500)
        st.plotly_chart(fig_hm, use_container_width=True)

# Page 5: Engagement Model & Calculator
elif page == "📈 Engagement Model & What-If Calculator":
    st.title("📈 Engagement Polynomial Model & Reallocation Calculator")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("10-Fold Cross-Validation Curve")
        deg = st.slider("Select Polynomial Degree:", min_value=1, max_value=8, value=6)
        st.image("fig6_cv_degree.png")
        st.info(f"Degree {results['cv_chosen_deg']} selected via 1-SE rule (CV RMSE: {results['cv']['6']['cv']:.3f}, R²: {results['r2_in_sample']:.2%})")
        
    with col2:
        st.subheader("Gradient Descent Optimization Convergence")
        st.image("fig7_gd_convergence.png")
        st.caption("Batch GD vs Closed-form solution (Matches to 10⁻¹³ precision)")

    st.markdown("---")
    st.subheader("🧮 What-If Budget Reallocation Simulator")
    
    col_sim1, col_sim2, col_sim3 = st.columns(3)
    n_shift = col_sim1.number_input("Sessions to shift:", min_value=100, max_value=10000, value=1000, step=100)
    ch_from = col_sim2.selectbox("From Channel:", ["Type 13", "Type 3", "Type 1"], index=0)
    ch_to = col_sim3.selectbox("To Channel:", ["Type 8", "Type 20", "Type 2"], index=0)
    aov = st.number_input("Average Order Value ($ / ₹):", min_value=1.0, value=50.0, step=5.0)
    
    mean_gain = results["ev_per_1000_mean"] * (n_shift / 1000)
    ci_low = results["ev_per_1000_ci"][0] * (n_shift / 1000)
    ci_high = results["ev_per_1000_ci"][1] * (n_shift / 1000)
    rev_gain = mean_gain * aov
    
    st.success(f"**Expected Extra Purchases:** +{mean_gain:.0f} purchases (95% Credible Interval: {ci_low:.0f} to {ci_high:.0f})")
    st.metric("Estimated Revenue Increase", f"${rev_gain:,.2f}")

# Page 6: Recommendations
elif page == "💡 Recommendations & Playbook":
    st.title("💡 Strategic Recommendations & Playbook")
    
    st.subheader("1. Channel Scorecard Actions")
    st.markdown("""
    * 🟢 **Protect & Fund (Type 2):** Core channel driving 44% of total buyers. 65% of sessions land in high-converting segments.
    * 🟡 **Staged Expansion (Types 8, 20, 5):** Conversion rates up to 27.7%, but limited volume (198 - 343 sessions). Test higher budgets.
    * 🔴 **Audit & Reduce Spend (Types 1, 3, 13):** Drive 42.5% of sessions but only 25.4% of buyers. Type 13 converts at a weak 5.8%.
    """)
    
    st.subheader("2. Visitor Segment Playbook")
    col1, col2 = st.columns(2)
    with col1:
        st.info("**Bouncers (7% sessions | 0.4% conv):** Stop paid acquisition. Audit page load speed and traffic intent.")
        st.warning("**Light Browsers (41% sessions | 10.5% conv):** Largest gain opportunity. Use personalized retargeting & recommendations.")
    with col2:
        st.success("**Account-Engaged (34% sessions | 20.1% conv):** Streamline sign-up/checkout funnel. Key driver of overall revenue.")
        st.success("**Deep Researchers (18% sessions | 23.9% conv):** Provide product comparison tools & reviews to close sales faster.")

# Page 7: Limitations & Downloads
else:
    st.title("📥 Project Limitations & File Downloads")
    
    st.subheader("⚠️ Model & Data Limitations")
    st.markdown("""
    1. **Anonymous Traffic Codes:** Channel IDs (e.g., Type 2) lack business labels (e.g., Organic Search vs Meta Ads).
    2. **Observational Data:** Correlation does not imply direct causality in budget reallocations.
    3. **No Customer IDs:** Individual sessions cannot be tied across multiple visits.
    """)
    
    st.subheader("📥 Export CSV Result Tables")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.download_button("Download Raw Dataset", df_raw.to_csv(index=False), "online_shoppers_intention.csv", "text/csv")
    with col2:
        if os.path.exists("channel_scorecard_all.csv"):
            st.download_button("Download Channel Scorecard", open("channel_scorecard_all.csv").read(), "channel_scorecard.csv", "text/csv")
    with col3:
        if os.path.exists("segment_profile.csv"):
            st.download_button("Download Segment Profiles", open("segment_profile.csv").read(), "segment_profile.csv", "text/csv")