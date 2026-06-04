import streamlit as st
import pandas as pd
import plotly.express as px

from analysis import (
    load_data,
    get_dataset_info,
    get_summary,
    get_missing_values
)

from model import (
    train_classification_model
)

st.set_page_config(
    page_title="Classification Dashboard",
    layout="wide"
)

st.title("Machine Learning Classification Dashboard")

uploaded_file = st.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

if uploaded_file:

    df = load_data(uploaded_file)

    st.success("Dataset Uploaded Successfully")

    st.header("Dataset Preview")

    st.dataframe(df.head())

    info = get_dataset_info(df)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Rows", info["Rows"])
    c2.metric("Columns", info["Columns"])
    c3.metric("Missing", info["Missing Values"])
    c4.metric("Duplicates", info["Duplicates"])

    st.header("Summary Statistics")

    st.dataframe(get_summary(df))

    st.header("Missing Values")

    st.dataframe(
        get_missing_values(df)
    )

    st.header("Target Column")

    target_column = st.selectbox(
        "Select Target",
        df.columns
    )

    if st.button("Train Model"):

        (
            accuracy,
            report,
            matrix,
            importance
        ) = train_classification_model(
            df,
            target_column
        )

        st.success("Training Complete")

        st.header("Model Accuracy")

        st.metric(
            "Accuracy",
            f"{accuracy:.4f}"
        )

        st.header("Classification Report")

        report_df = pd.DataFrame(
            report
        ).transpose()

        st.dataframe(report_df)

        st.header("Confusion Matrix")

        st.dataframe(
            pd.DataFrame(matrix)
        )

        st.header("Feature Importance")

        st.dataframe(
            importance
        )

        fig = px.bar(
            importance,
            x="Feature",
            y="Importance",
            title="Feature Importance"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )
