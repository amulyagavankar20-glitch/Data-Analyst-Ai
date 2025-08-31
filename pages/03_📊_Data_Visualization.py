import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 👉 Chatbot import
from chatbot import chatbot_sidebar

st.session_state["page_name"] = "Data Visualization"

st.set_option("client.showErrorDetails", True)

st.title("📊 Data Visualization")

st.markdown(
    """
    <style>
    .scrollable-table { overflow-x: auto; }
    </style>
    """,
    unsafe_allow_html=True
)

if "dataset" in st.session_state:
    df = st.session_state["dataset"]

    # ---- Quick stats
    st.caption(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=["int64", "float64"]).columns.tolist()
    st.caption(f"Numeric: {len(num_cols)} | Categorical: {len(cat_cols)}")

    # ---- Dataset preview
    st.subheader("🔍 Dataset Preview")
    with st.container():
        st.markdown('<div class="scrollable-table">', unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True, height=420)
        st.markdown('</div>', unsafe_allow_html=True)

    if len(num_cols) > 1:
        st.subheader("📌 Correlation Heatmap")
        corr = df[num_cols].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    if num_cols:
        st.subheader("📦 Box Plot")
        col = st.selectbox("Select numeric column:", num_cols, key="boxplot_col")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x=df[col], ax=ax, color="skyblue")
        ax.set_title(f"Box Plot of {col}")
        st.pyplot(fig)

    if num_cols:
        st.subheader("📊 Histogram")
        col = st.selectbox("Select column for histogram:", num_cols, key="hist_col")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df[col], bins=30, kde=True, ax=ax, color="orange")
        ax.set_title(f"Histogram of {col}")
        st.pyplot(fig)

    if cat_cols:
        st.subheader("📋 Bar Graph")
        col = st.selectbox("Select categorical column:", cat_cols, key="bar_col")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(x=df[col], ax=ax, palette="Set2")
        ax.set_title(f"Bar Graph of {col}")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        st.pyplot(fig)

else:
    st.warning("⚠️ Please upload a dataset first.")


chatbot_sidebar()
