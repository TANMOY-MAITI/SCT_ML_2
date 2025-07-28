import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Mall Customer Clustering", layout="wide")
st.title("🛍️ Mall Customer Segmentation")

# 1/3 and 2/3 column layout
left_col, right_col = st.columns([1, 2])

with left_col:
    st.header("📁 Upload & Controls")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    st.image("https://cdn-icons-png.flaticon.com/512/3208/3208753.png", width=180)
    start = False
    if uploaded_file:
        start = st.button("🚀 Start Clustering")

with right_col:
    st.header("📊 Output & Visualization")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("🔍 Data Preview")
        st.dataframe(df.head())

        if start:
            st.subheader("🌀 Scatter Plot")
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x='Age', y='Annual Income (k$)', hue='Gender', s=100, alpha=0.7)
            plt.title("Age vs Income by Gender")
            st.pyplot(fig)
