```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA

# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="SMU FTEN Student Profile Dashboard",
    layout="wide"
)

st.title("SMU First-Time Entering Students Profile Dashboard")
st.markdown("""
### Student Biographical Profile, Academic Pressures,
### Social and Emotional Wellbeing Analysis
""")

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload SMU Biographical Questionnaire Dataset",
    type=["xlsx"]
)

if uploaded_file:

    df = pd.read_excel(uploaded_file)

    st.success("Dataset uploaded successfully")

    # --------------------------------------------------
    # OVERVIEW
    # --------------------------------------------------

    st.header("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Students", len(df))
    col2.metric("Variables", len(df.columns))
    col3.metric("Missing Values", df.isnull().sum().sum())

    st.dataframe(df.head())

    # --------------------------------------------------
    # STUDENT PROFILE
    # --------------------------------------------------

    st.header("1. Biographical Data and Demographics")

    categorical_columns = df.select_dtypes(include='object').columns

    selected_variable = st.selectbox(
        "Select Variable",
        categorical_columns
    )

    profile_table = (
        df[selected_variable]
        .value_counts(dropna=False)
        .reset_index()
    )

    profile_table.columns = [
        selected_variable,
        "Frequency"
    ]

    profile_table["Percentage"] = round(
        profile_table["Frequency"]
        / len(df) * 100,
        2
    )

    st.dataframe(profile_table)

    fig = px.bar(
        profile_table,
        x=selected_variable,
        y="Frequency",
        title=f"{selected_variable} Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------
    # ACADEMIC SOCIAL EMOTIONAL
    # --------------------------------------------------

    st.header(
        "2. Academic, Social and Emotional Pressures"
    )

    pressure_variable = st.selectbox(
        "Select Pressure Variable",
        df.columns
    )

    pressure_table = (
        df[pressure_variable]
        .value_counts(dropna=False)
        .reset_index()
    )

    pressure_table.columns = [
        "Response",
        "Frequency"
    ]

    pressure_table["Percentage"] = round(
        pressure_table["Frequency"]
        / len(df) * 100,
        2
    )

    st.dataframe(pressure_table)

    fig2 = px.pie(
        pressure_table,
        names="Response",
        values="Frequency",
        title=pressure_variable
    )

    st.plotly_chart(fig2)

    # --------------------------------------------------
    # MACHINE LEARNING
    # --------------------------------------------------

    st.header(
        "3. Academic Achievement Classification"
    )

    target_variable = st.selectbox(
        "Select Academic Achievement Variable",
        df.columns
    )

    model_df = df.copy()

    model_df = model_df.fillna("Missing")

    encoders = {}

    for column in model_df.columns:

        le = LabelEncoder()

        model_df[column] = (
            le.fit_transform(
                model_df[column].astype(str)
            )
        )

        encoders[column] = le

    X = model_df.drop(
        columns=[target_variable]
    )

    y = model_df[target_variable]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.30,
            random_state=42
        )
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

    # --------------------------------------------------
    # VARIABLE IMPORTANCE
    # --------------------------------------------------

    st.header(
        "4. Factors Influencing Academic Achievement"
    )

    importance_df = pd.DataFrame({

        "Variable": X.columns,

        "Importance":
        model.feature_importances_

    })

    importance_df = (
        importance_df
        .sort_values(
            by="Importance",
            ascending=False
        )
        .head(20)
    )

    fig3 = px.bar(
        importance_df,
        x="Importance",
        y="Variable",
        orientation="h",
        title="Top Predictors"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

    # --------------------------------------------------
    # STUDENT SEGMENTATION
    # --------------------------------------------------

    st.header(
        "5. Student Groups Based on Academic Achievement"
    )

    pca = PCA(
        n_components=2
    )

    pca_results = pca.fit_transform(X)

    pca_df = pd.DataFrame({

        "Dimension 1":
        pca_results[:, 0],

        "Dimension 2":
        pca_results[:, 1],

        "Achievement Group":
        y

    })

    fig4 = px.scatter(
        pca_df,
        x="Dimension 1",
        y="Dimension 2",
        color="Achievement Group",
        title="Student Segmentation"
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )

    # --------------------------------------------------
    # AT RISK STUDENTS
    # --------------------------------------------------

    st.header(
        "6. Student Risk Categories"
    )

    median_score = y.median()

    pca_df["Risk Category"] = np.where(

        y <= median_score,

        "At Risk",

        "Not At Risk"

    )

    fig5 = px.scatter(
        pca_df,
        x="Dimension 1",
        y="Dimension 2",
        color="Risk Category",
        title="At-Risk Students"
    )

    st.plotly_chart(
        fig5,
        use_container_width=True
    )

    # --------------------------------------------------
    # EXECUTIVE SUMMARY
    # --------------------------------------------------

    st.header(
        "7. Executive Summary"
    )

    st.write(
        f"Total students analysed: {len(df)}"
    )

    st.write(
        f"Model accuracy: {accuracy:.2%}"
    )

    st.write(
        "The chart above identifies student groups "
        "with similar biographical, academic, social "
        "and emotional characteristics."
    )

else:

    st.info(
        "Please upload the SMU Biographical Questionnaire Excel file."
    )
```

