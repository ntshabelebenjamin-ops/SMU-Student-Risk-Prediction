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

# PAGE CONFIGURATION

# --------------------------------------------------

st.set_page_config(
page_title="SMU FTEN Student Profile Dashboard",
layout="wide"
)

st.title("SMU First-Time Entering Students Profile Dashboard")
st.markdown(
"Biographical Profile, Academic, Social and Emotional Analysis"
)

# --------------------------------------------------

# UPLOAD FILE

# --------------------------------------------------

uploaded_file = st.file_uploader(
"Upload SMU Biographical Questionnaire Dataset",
type=["xlsx"]
)

if uploaded_file is not None:

```
df = pd.read_excel(uploaded_file)

st.success("Dataset uploaded successfully")

# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------

st.header("Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Students", len(df))
col2.metric("Variables", len(df.columns))
col3.metric("Missing Values", int(df.isnull().sum().sum()))

st.subheader("Dataset Preview")
st.dataframe(df.head())

# --------------------------------------------------
# STUDENT PROFILE
# --------------------------------------------------

st.header("1. Biographical Data and Demographics")

categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

if len(categorical_cols) > 0:

    selected_variable = st.selectbox(
        "Select Demographic Variable",
        categorical_cols
    )

    freq_table = (
        df[selected_variable]
        .fillna("Missing")
        .value_counts()
        .reset_index()
    )

    freq_table.columns = ["Category", "Frequency"]

    freq_table["Percentage"] = round(
        (freq_table["Frequency"] / len(df)) * 100,
        2
    )

    st.dataframe(freq_table)

    fig = px.bar(
        freq_table,
        x="Category",
        y="Frequency",
        title=f"Distribution of {selected_variable}"
    )

    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# ALL FREQUENCY TABLES
# --------------------------------------------------

st.header("2. Frequency and Percentage Analysis")

selected_col = st.selectbox(
    "Select Any Variable",
    df.columns
)

profile = (
    df[selected_col]
    .fillna("Missing")
    .value_counts()
    .reset_index()
)

profile.columns = ["Response", "Frequency"]

profile["Percentage"] = round(
    profile["Frequency"] / len(df) * 100,
    2
)

st.dataframe(profile)

# --------------------------------------------------
# MACHINE LEARNING
# --------------------------------------------------

st.header("3. Academic Achievement Classification")

target_variable = "22. Academic Achievement"

if target_variable in df.columns:

    model_df = df.copy()

    model_df = model_df.fillna("Missing")

    for col in model_df.columns:

        le = LabelEncoder()

        model_df[col] = le.fit_transform(
            model_df[col].astype(str)
        )

    X = model_df.drop(columns=[target_variable])

    y = model_df[target_variable]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    st.metric(
        "Model Accuracy",
        f"{accuracy:.2%}"
    )

    # --------------------------------------------------
    # VARIABLE IMPORTANCE
    # --------------------------------------------------

    st.header("4. Top Predictors of Academic Achievement")

    importance_df = pd.DataFrame({
        "Variable": X.columns,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    top_20 = importance_df.head(20)

    fig2 = px.bar(
        top_20,
        x="Importance",
        y="Variable",
        orientation="h",
        title="Top 20 Predictors"
    )

    st.plotly_chart(fig2, use_container_width=True)

    # --------------------------------------------------
    # PCA STUDENT GROUPS
    # --------------------------------------------------

    st.header("5. Student Groups")

    pca = PCA(n_components=2)

    pca_results = pca.fit_transform(X)

    pca_df = pd.DataFrame({
        "Dimension 1": pca_results[:, 0],
        "Dimension 2": pca_results[:, 1],
        "Achievement Group": y
    })

    fig3 = px.scatter(
        pca_df,
        x="Dimension 1",
        y="Dimension 2",
        color="Achievement Group",
        title="Student Segmentation Based on Academic Achievement"
    )

    st.plotly_chart(fig3, use_container_width=True)

    # --------------------------------------------------
    # RISK GROUPS
    # --------------------------------------------------

    st.header("6. At-Risk Student Identification")

    median_score = y.median()

    pca_df["Risk Category"] = np.where(
        y <= median_score,
        "At Risk",
        "Not At Risk"
    )

    fig4 = px.scatter(
        pca_df,
        x="Dimension 1",
        y="Dimension 2",
        color="Risk Category",
        title="At-Risk Student Groups"
    )

    st.plotly_chart(fig4, use_container_width=True)

    # --------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------

    st.header("7. Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        predictions
    )

    cm_df = pd.DataFrame(cm)

    st.dataframe(cm_df)

else:

    st.error(
        f"Column '{target_variable}' not found in dataset."
    )

    st.write("Available columns:")

    st.write(df.columns.tolist())
```

else:

```
st.info(
    "Upload the SMU Biographical Questionnaire Excel file to begin."
)
```
