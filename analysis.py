"""
Mathematical Foundation of Business Analytics (MFBA) - Group 9 Analysis Pipeline
Dataset: online_shoppers_intention.csv (12,330 sessions)
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')
np.random.seed(42)

def run_analysis(csv_path="online_shoppers_intention.csv", k_clusters=4):
    print("Loading and cleaning raw data...")
    df = pd.read_csv(csv_path)
    n_sessions = len(df)
    n_buyers = int(df['Revenue'].sum())
    overall_conv = n_buyers / n_sessions
    
    df = df.dropna()
    
    # Log transform right-skewed duration variables
    for col in ['Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration']:
        if col in df.columns:
            df[col + '_log'] = np.log1p(df[col])
        
    print("Computing channel scorecard and Bayesian shrinkage...")
    channel_summary = df.groupby('TrafficType').agg(
        sessions=('Revenue', 'count'),
        purchases=('Revenue', 'sum')
    ).reset_index()
    
    channel_summary['conversion_rate'] = channel_summary['purchases'] / channel_summary['sessions']
    
    # Bayesian Beta-Binomial shrinkage (Prior: a=2, b=13)
    a_prior, b_prior = 2.0, 13.0
    channel_summary['bayes_posterior'] = (a_prior + channel_summary['purchases']) / (a_prior + b_prior + channel_summary['sessions'])
    channel_summary['ci_lower'] = channel_summary['conversion_rate'] - 1.96 * np.sqrt(channel_summary['conversion_rate']*(1-channel_summary['conversion_rate'])/channel_summary['sessions'].replace(0,1))
    channel_summary['ci_upper'] = channel_summary['conversion_rate'] + 1.96 * np.sqrt(channel_summary['conversion_rate']*(1-channel_summary['conversion_rate'])/channel_summary['sessions'].replace(0,1))
    
    print("Running PCA via SVD...")
    numeric_cols = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration', 
                    'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues']
    X_num = df[numeric_cols].select_dtypes(include=[np.number])
    scaler = StandardScaler()
    Z = scaler.fit_transform(X_num)
    
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    eigenvalues = (s ** 2) / (len(Z) - 1)
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
    
    pca = PCA(n_components=3)
    pca_scores = pca.fit_transform(Z)
    
    print("Running K-Means Clustering...")
    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(pca_scores)
    df['Segment'] = clusters
    
    # Save results to JSON for app.py
    results = {
        "n_sessions": n_sessions,
        "n_buyers": n_buyers,
        "overall_conversion": float(overall_conv),
        "channels": channel_summary.to_dict(orient="records"),
        "pca_explained_variance": explained_variance_ratio.tolist()
    }
    
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    channel_summary.to_csv("channel_scorecard_summary.csv", index=False)
    print("Analysis complete successfully!")

if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "online_shoppers_intention.csv"
    k_val = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    run_analysis(csv_file, k_val)