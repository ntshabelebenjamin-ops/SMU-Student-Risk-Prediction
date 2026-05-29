import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="SMU Student Profile Dashboard",
    layout="wide"
)

st.title("SMU First-Time Entering Students Dashboard")

uploaded_file = st.file_uploader(
    "Upload Excel File",
    type=["xlsx"]
)

if uploaded_file is not None:

    df = pd.read_excel(uploaded_file)

    st.success("Data loaded successfully")

    # Dataset Overview
    st.header("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Students", len(df))
    col2.metric("Variables", len(df.columns))
    col3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.dataframe(df.head())

    # Show Column Names
    st.header("Column Names")

    st.write(df.columns.tolist())

    # Frequency Analysis
    st.header("Frequency Analysis")

    selected_col = st.selectbox(
        "Select Variable",
        df.columns
    )

    freq = (
        df[selected_col]
        .fillna("Missing")
        .value_counts()
        .reset_index()
    )

    freq.columns = ["Category", "Frequency"]

    freq["Percentage"] = round(
        freq["Frequency"] / len(df) * 100,
        2
    )

    st.dataframe(freq)

    fig = px.bar(
        freq,
        x="Category",
        y="Frequency",
        title=f"Distribution of {selected_col}"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ==========================================
    # CHANGE THIS TO YOUR EXACT COLUMN NAME
    # ==========================================

    TARGET = "22. Academic Achievement"

    if TARGET in df.columns:

        st.header("Academic Achievement Classification")

        model_df = df.copy()

        model_df = model_df.fillna("Missing")

        for col in model_df.columns:

            encoder = LabelEncoder()

            model_df[col] = encoder.fit_transform(
                model_df[col].astype(str)
            )

        X = model_df.drop(columns=[TARGET])

        y = model_df[TARGET]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.30,
            random_state=42
        )

        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        st.metric(
            "Classification Accuracy",
            f"{accuracy:.2%}"
        )

        # Variable Importance

        st.subheader("Top Predictors")

        importance = pd.DataFrame({
            "Variable": X.columns,
            "Importance": model.feature_importances_
        })

        importance = importance.sort_values(
            "Importance",
            ascending=False
        ).head(15)

        fig2 = px.bar(
            importance,
            x="Importance",
            y="Variable",
            orientation="h"
        )

        st.plotly_chart(fig2, use_container_width=True)

        # PCA Visualisation

        st.subheader("Student Groups")

        pca = PCA(n_components=2)

        pca_results = pca.fit_transform(X)

        pca_df = pd.DataFrame({
            "PCA1": pca_results[:, 0],
            "PCA2": pca_results[:, 1],
            "Academic Achievement": y
        })

        fig3 = px.scatter(
            pca_df,
            x="PCA1",
            y="PCA2",
            color="Academic Achievement",
            title="Student Clusters"
        )

        st.plotly_chart(fig3, use_container_width=True)

    else:

        st.warning(
            f"Column '{TARGET}' not found."
        )

else:

    st.info(
        "Please upload your SMU Excel dataset."
    )
