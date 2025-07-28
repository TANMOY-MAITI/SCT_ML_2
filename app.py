import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

st.set_page_config(page_title="Mall Customer Clustering", layout="wide")
st.title("🛍️ Mall Customer Segmentation")

# Column layout: 1/3 (left) and 2/3 (right)
left_col, right_col = st.columns([1, 2])

with left_col:
    st.header("📁 Upload & Cluster Settings")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        k = left_col.slider("Select number of clusters (K)", 2, 10, 5)
        start = left_col.button("🚀 Start Clustering")
        

with right_col:
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### 🔍 Data Preview")
        st.dataframe(df.head())

        if 'Gender' in df.columns:
            df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

        features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
        X = df[features]

        if start:
            model = KMeans(n_clusters=k, random_state=42)
            df['Cluster'] = model.fit_predict(X)

            # Visualize
            st.subheader("📊 Cluster Visualization")
            fig, ax = plt.subplots(figsize=(3, 2.5), dpi=100) # smaller figure
            sns.scatterplot(
                      data=df,
                      x='Annual Income (k$)',
                      y='Spending Score (1-100)',
                      hue='Cluster',
                      palette='viridis',
                      s=20,  # smaller points
                      ax=ax)
            st.pyplot(fig)

            st.subheader("📋 Cluster Summary")
            st.write(df.groupby('Cluster')[features].mean())

            # Optional: Download CSV
            st.download_button("📥 Download Result", df.to_csv(index=False), file_name="clustered_data.csv", mime="text/csv")
