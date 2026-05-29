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
    page_title="SMU FTEN Dashboard 2025",
    layout="wide"
)

# =====================================================
# DASHBOARD TITLE
# =====================================================

st.title(
    "SMU First-Time Entering Students Profile and Success Prediction Dashboard: 2025 Cohort"
)

st.caption(
    "Prepared by Benjamin Ntshabele | Institutional Researcher | Institutional Planning and Quality Assurance | SMU"
)

st.caption(
    "Institutional Research and Student Success Analytics Dashboard"
)

st.markdown("""
This executive dashboard provides a comprehensive profile of the 2025 First-Time Entering Students (FTEN) cohort at Sefako Makgatho Health Sciences University (SMU).
""")

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

    st.header("2025 FTEN Cohort Overview")

    total_cells = df.shape[0] * df.shape[1]

    completeness = round(
        (1 - (df.isnull().sum().sum() / total_cells)) * 100,
        1
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "FTEN Students",
        len(df)
    )

    col2.metric(
        "Questionnaire Variables",
        len(df.columns)
    )

    col3.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )

    col4.metric(
        "Data Completeness",
        f"{completeness}%"
    )

    st.divider()

    # =====================================================
    # HELPER FUNCTIONS
    # =====================================================

    def create_profile_table(dataframe, column_name):

        freq = (
            dataframe[column_name]
            .fillna("Missing")
            .value_counts()
            .reset_index()
        )

        freq.columns = [
            "Response",
            "Frequency"
        ]

        freq["Percentage"] = round(
            freq["Frequency"] /
            freq["Frequency"].sum() * 100,
            1
        )

        total = pd.DataFrame({
            "Response": ["TOTAL"],
            "Frequency": [freq["Frequency"].sum()],
            "Percentage": [100.0]
        })

        freq = pd.concat(
            [freq, total],
            ignore_index=True
        )

        return freq

    def download_table(
        dataframe,
        filename,
        button_text
    ):

        csv = dataframe.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label=button_text,
            data=csv,
            file_name=filename,
            mime="text/csv"
        )
            # =====================================================
    # STUDENT DEMOGRAPHIC PROFILE
    # =====================================================

    st.header("2025 Student Demographic Profile")

    demographic_labels = {

        "3. What was the main language used by teachers in class?":
        "Language of Learning and Teaching",

        "5. Was your language learning at school the same as your home language?":
        "Alignment Between Home and School Language",

        "6. Is English your first, second or third language?":
        "English Language Exposure",

        "7. How did you pay for your school fees?":
        "School Funding Background",

        "8. What was the average number of learners in your matric classroom?":
        "Classroom Size",

        "9. How would you describe the place in which your school is situated?":
        "School Geographic Context"
    }

    for original_var, executive_label in demographic_labels.items():

        if original_var in df.columns:

            st.subheader(executive_label)

            freq = create_profile_table(
                df,
                original_var
            )

            chart_data = freq[
                freq["Response"] != "TOTAL"
            ]

            left, right = st.columns([1, 2])

            with left:

                st.dataframe(
                    freq,
                    use_container_width=True
                )

                download_table(
                    freq,
                    f"{executive_label}.csv",
                    f"📥 Download {executive_label}"
                )

            with right:

                if "Language" in executive_label:

                    fig = px.pie(
                        chart_data,
                        names="Response",
                        values="Percentage",
                        hole=0.45,
                        title=executive_label
                    )

                elif "Funding" in executive_label:

                    fig = px.treemap(
                        chart_data,
                        path=["Response"],
                        values="Frequency",
                        title=executive_label
                    )

                elif "Classroom" in executive_label:

                    fig = px.bar(
                        chart_data,
                        x="Response",
                        y="Percentage",
                        text="Percentage",
                        title=executive_label
                    )

                else:

                    fig = px.bar(
                        chart_data,
                        y="Response",
                        x="Percentage",
                        orientation="h",
                        text="Percentage",
                        title=executive_label
                    )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            largest_group = chart_data.iloc[0]["Response"]

            largest_pct = chart_data.iloc[0]["Percentage"]

            st.info(
                f"Key Finding: The largest student group falls under '{largest_group}', representing {largest_pct}% of the 2025 FTEN cohort."
            )

    st.divider()
