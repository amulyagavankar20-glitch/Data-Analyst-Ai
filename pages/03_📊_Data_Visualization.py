import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

    # ---- Dataset preview (scrollable)
    st.subheader("🔍 Dataset Preview")
    with st.container():
        st.markdown('<div class="scrollable-table">', unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True, height=420)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---- Correlation Heatmap (if numeric)
    if len(num_cols) > 1:
        st.subheader("📌 Correlation Heatmap")
        corr = df[num_cols].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.info("Not enough numeric columns for a correlation heatmap.")

else:
    st.warning("⚠️ Please upload a dataset first.")

chatbot_sidebar()
