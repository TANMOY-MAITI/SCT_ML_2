import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

st.set_page_config(page_title="Customer Segmentation", layout="centered")

st.title("🛍️ Retail Customer Segmentation using K-Means")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview", df.head())

    # Data pre-processing
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

    features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    X = df[features]

    # Cluster Selection
    k = st.slider("Select number of clusters (K)", 2, 10, 5)

    model = KMeans(n_clusters=k, random_state=42)
    df['Cluster'] = model.fit_predict(X)

    # Visualize
    st.subheader("📊 Cluster Visualization")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='viridis')
    st.pyplot(fig)

    # Cluster Summary
    st.subheader("🧮 Cluster Summary")
    st.write(df.groupby('Cluster')[features].mean())

