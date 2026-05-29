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
    "Institutional Research and Student Success Analytics Dashboard"
)

st.markdown("""
This executive dashboard provides a comprehensive profile of the 2025 First-Time Entering Students (FTEN) cohort at Sefako Makgatho Health Sciences University (SMU). The analysis examines student demographics, academic preparedness, home and school learning environments, social and emotional wellbeing, and factors associated with academic success.
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
    # HELPER FUNCTION
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

            left, right = st.columns([1, 2])

            with left:
                st.dataframe(
                    freq,
                    use_container_width=True
                )

            chart_data = freq[
                freq["Response"] != "TOTAL"
            ]

            with right:

                if "Language" in executive_label:

                    fig = px.pie(
                        chart_data,
                        names="Response",
                        values="Percentage",
                        hole=0.4,
                        title=executive_label
                    )

                elif "Funding" in executive_label:

                    fig = px.treemap(
                        chart_data,
                        path=["Response"],
                        values="Frequency",
                        title=executive_label
                    )

                else:

                    fig = px.bar(
                        chart_data,
                        x="Response",
                        y="Percentage",
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
                # =====================================================
    # 2025 ACADEMIC READINESS PROFILE
    # =====================================================

    st.header("2025 Academic Readiness Profile")

    academic_labels = {

        "13. Did your high school make use of online remote learning services?":
        "Exposure to Online Learning",

        "16. How often did you make use of online material, online learning sites or ‘apps’?":
        "Use of Digital Learning Resources",

        "20. Did you participate in any leadership (extracurricular) activities during high school?\n(e.g., School Representative Council, Junior City Council etc.)":
        "Leadership Participation",

        "22. Academic Achievement":
        "Academic Achievement Profile"
    }

    for original_var, executive_label in academic_labels.items():

        if original_var in df.columns:

            st.subheader(executive_label)

            freq = create_profile_table(
                df,
                original_var
            )

            left, right = st.columns([1, 2])

            with left:
                st.dataframe(
                    freq,
                    use_container_width=True
                )

            chart_data = freq[
                freq["Response"] != "TOTAL"
            ]

            with right:

                fig = px.bar(
                    chart_data,
                    x="Response",
                    y="Percentage",
                    text="Percentage",
                    title=executive_label
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            largest_group = chart_data.iloc[0]["Response"]
            largest_pct = chart_data.iloc[0]["Percentage"]

            st.success(
                f"Insight: {largest_pct}% of students fall within '{largest_group}' for {executive_label.lower()}."
            )

    st.divider()

    # =====================================================
    # 2025 SCHOOL AND HOME LEARNING ENVIRONMENT
    # =====================================================

    st.header("2025 School and Home Learning Environment")

    home_labels = {

        "24. Please answer the following questions about your home [Does your home have access to running water ?]":
        "Access to Running Water",

        "24. Please answer the following questions about your home [Does your home have access to electricity?]":
        "Access to Electricity",

        "24. Please answer the following questions about your home [Does your home have a computer/laptop?]":
        "Computer or Laptop Ownership",

        "24. Please answer the following questions about your home [Does your home have access to internet connection ( eg. Wifi...)?]":
        "Internet Access at Home",

        "24. Please answer the following questions about your home [Do you have a private area to study in your home?]":
        "Private Study Space"
    }

    for original_var, executive_label in home_labels.items():

        if original_var in df.columns:

            st.subheader(executive_label)

            freq = create_profile_table(
                df,
                original_var
            )

            left, right = st.columns([1, 2])

            with left:
                st.dataframe(
                    freq,
                    use_container_width=True
                )

            chart_data = freq[
                freq["Response"] != "TOTAL"
            ]

            with right:

                fig = px.pie(
                    chart_data,
                    names="Response",
                    values="Percentage",
                    hole=0.5,
                    title=executive_label
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

    st.divider()

    # =====================================================
    # 2025 STUDENT SUPPORT AND WELLBEING PROFILE
    # =====================================================

    st.header("2025 Student Support and Wellbeing Profile")

    social_labels = {

        "29. Do you have any dependents (e.g., Children, cousins, siblings, or nephews that you financially provide for)?":
        "Dependents",

        "33. Do you plan to work part-time during your studies?":
        "Part-Time Employment Intentions",

        "39. Where will you be staying during your studies?":
        "Accommodation Arrangements",

        "43. How far away from campus do you live?":
        "Distance from Campus",

        "45. How long will it take you to travel to campus to arrive at 8am lecture on time?":
        "Travel Time to Campus"
    }

    for original_var, executive_label in social_labels.items():

        if original_var in df.columns:

            st.subheader(executive_label)

            freq = create_profile_table(
                df,
                original_var
            )

            left, right = st.columns([1, 2])

            with left:
                st.dataframe(
                    freq,
                    use_container_width=True
                )

            chart_data = freq[
                freq["Response"] != "TOTAL"
            ]

            with right:

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

    emotional_labels = {

        "35. What fears or concerns do you have regarding your first year at university?":
        "First-Year Concerns",

        "36. What are you looking forward to the most in your first year at university?":
        "First-Year Expectations"
    }

    for original_var, executive_label in emotional_labels.items():

        if original_var in df.columns:

            st.subheader(executive_label)

            freq = create_profile_table(
                df,
                original_var
            )

            st.dataframe(
                freq,
                use_container_width=True
            )

    st.divider()

    # =====================================================
    # STRATEGIC INSIGHTS
    # =====================================================

    st.header("Strategic Insights for Student Success: 2025 Cohort")

    st.info("""
    Key strategic themes emerging from the 2025 FTEN cohort include:

    • Digital readiness and access to technology
    • Academic preparedness for university studies
    • Availability of private study environments
    • Student accommodation and commuting challenges
    • Financial vulnerability and part-time employment intentions
    • Emotional support structures and first-year concerns
    • Factors associated with academic success and student retention
    """)
        # =====================================================
    # 2025 STUDENT SUCCESS PREDICTION MODEL
    # =====================================================

    st.header("2025 Student Success Prediction Model")

    TARGET = "22. Academic Achievement"

    if TARGET in df.columns:

        ml_df = df.copy()

        ml_df = ml_df.fillna("Missing")

        for col in ml_df.columns:

            encoder = LabelEncoder()

            ml_df[col] = encoder.fit_transform(
                ml_df[col].astype(str)
            )

        X = ml_df.drop(columns=[TARGET])

        y = ml_df[TARGET]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.30,
            random_state=42
        )

        model = RandomForestClassifier(
            n_estimators=500,
            random_state=42
        )

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        st.subheader("Prediction Accuracy")

        col1, col2 = st.columns(2)

        col1.metric(
            "Model Accuracy",
            f"{accuracy:.1%}"
        )

        col2.metric(
            "Predictor Variables",
            len(X.columns)
        )

        st.divider()

        # =====================================================
        # TOP PREDICTORS
        # =====================================================

        st.header("Key Predictors of Academic Success")

        importance = pd.DataFrame({
            "Variable": X.columns,
            "Importance": model.feature_importances_
        })

        importance = importance.sort_values(
            by="Importance",
            ascending=False
        )

        top_predictors = importance.head(20)

        fig = px.bar(
            top_predictors,
            x="Importance",
            y="Variable",
            orientation="h",
            text="Importance",
            title="Top 20 Predictors of Academic Achievement"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.info(
            "The chart above identifies the strongest factors associated with academic achievement within the 2025 FTEN cohort."
        )

        st.divider()

        # =====================================================
        # STUDENT SUCCESS SEGMENTS
        # =====================================================

        st.header("Student Success Segments")

        pca = PCA(
            n_components=2
        )

        pca_results = pca.fit_transform(X)

        segments = pd.DataFrame({

            "Dimension 1":
            pca_results[:, 0],

            "Dimension 2":
            pca_results[:, 1],

            "Academic Achievement":
            y.astype(str)

        })

        fig = px.scatter(
            segments,
            x="Dimension 1",
            y="Dimension 2",
            color="Academic Achievement",
            title="Student Success Segments"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.info(
            "Students positioned close together share similar characteristics and achievement profiles."
        )

        st.divider()

        # =====================================================
        # STUDENTS REQUIRING ADDITIONAL SUPPORT
        # =====================================================

        st.header("Students Requiring Additional Support")

        threshold = y.median()

        segments["Support Category"] = np.where(
            y <= threshold,
            "Additional Support Required",
            "Progressing Well"
        )

        fig = px.scatter(
            segments,
            x="Dimension 1",
            y="Dimension 2",
            color="Support Category",
            title="Student Support Segmentation"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        support_summary = (
            segments["Support Category"]
            .value_counts()
            .reset_index()
        )

        support_summary.columns = [
            "Category",
            "Students"
        ]

        support_summary["Percentage"] = round(
            support_summary["Students"] /
            support_summary["Students"].sum() * 100,
            1
        )

        st.dataframe(
            support_summary,
            use_container_width=True
        )

        st.warning(
            "Students classified as requiring additional support may benefit from targeted academic, financial, psychosocial and mentoring interventions."
        )

    else:

        st.error(
            "Academic Achievement variable not found in the dataset."
        )

else:

    st.info(
        "Please upload the SMU Biographical Questionnaire dataset to begin."
    )

    st.divider()
