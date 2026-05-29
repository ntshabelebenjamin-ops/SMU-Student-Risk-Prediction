import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="SMU FTEN Profile Dashboard",
    layout="wide"
)

st.title("SMU First-Time Entering Students Profile Dashboard")
st.markdown(
    "Biographical Profile, Academic, Social and Emotional Analysis"
)

# =====================================================
# FILE UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    "Upload SMU Biographical Questionnaire Dataset",
    type=["xlsx"]
)

if uploaded_file is not None:

    df = pd.read_excel(uploaded_file)

    # =====================================================
    # EXECUTIVE SUMMARY
    # =====================================================

    st.header("Executive Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "First-Time Entering Students",
        len(df)
    )

    col2.metric(
        "Variables",
        len(df.columns)
    )

    col3.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )

    st.divider()

    # =====================================================
    # HELPER FUNCTION
    # =====================================================

    def frequency_table(dataframe, column_name):

        freq = (
            dataframe[column_name]
            .fillna("Missing")
            .value_counts()
            .reset_index()
        )

        freq.columns = ["Response", "Frequency"]

        freq["Percentage"] = round(
            freq["Frequency"] / len(dataframe) * 100,
            2
        )

        return freq

    # =====================================================
    # DEMOGRAPHIC PROFILE
    # =====================================================

    st.header("Biographical and Demographic Profile")

    demographic_vars = [
        "3. What was the main language used by teachers in class?",
        "5. Was your language learning at school the same as your home language?",
        "6. Is English your first, second or third language?",
        "7. How did you pay for your school fees?",
        "8. What was the average number of learners in your matric classroom?",
        "9. How would you describe the place in which your school is situated?"
    ]

    for col in demographic_vars:

        if col in df.columns:

            st.subheader(col)

            freq = frequency_table(
                df,
                col
            )

            st.dataframe(
                freq,
                use_container_width=True
            )

            fig = px.bar(
                freq,
                x="Response",
                y="Frequency",
                title=col
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    st.divider()
```

