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

            st.success(
                f"Strategic Insight: {chart_data.iloc[0]['Percentage']}% of students fall within the dominant category for {executive_label.lower()}."
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
    # STRATEGIC STUDENT SUCCESS INDICATORS
    # =====================================================

    st.header("Strategic Student Success Indicators")

    st.markdown("""
    The following indicators have been identified as critical student success and retention factors:

    - First-generation university students
    - Rural versus urban background
    - Financial vulnerability and NSFAS dependency
    - Language diversity and schooling background
    - Family responsibilities
    - Commuting and travel challenges
    """)

    st.divider()
        # =====================================================
    # FIRST-GENERATION STUDENTS
    # =====================================================

    st.header("First-Generation University Students")

    st.info(
        "Update the variable below with the actual questionnaire field that identifies first-generation university students."
    )

    # Example column name - replace with actual variable
    first_gen_col = "First Generation Student"

    if first_gen_col in df.columns:

        first_gen = create_profile_table(
            df,
            first_gen_col
        )

        st.dataframe(
            first_gen,
            use_container_width=True
        )

        download_table(
            first_gen,
            "First_Generation_Students.csv",
            "📥 Download First-Generation Student Data"
        )

        fig = px.pie(
            first_gen[first_gen["Response"] != "TOTAL"],
            names="Response",
            values="Frequency",
            title="First-Generation University Students"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    # =====================================================
    # RURAL VS URBAN BACKGROUND
    # =====================================================

    st.header("Rural vs Urban Background")

    location_col = (
        "9. How would you describe the place in which your school is situated?"
    )

    if location_col in df.columns:

        rural_urban = create_profile_table(
            df,
            location_col
        )

        st.dataframe(
            rural_urban,
            use_container_width=True
        )

        download_table(
            rural_urban,
            "Rural_Urban_Background.csv",
            "📥 Download Rural vs Urban Data"
        )

        fig = px.bar(
            rural_urban[
                rural_urban["Response"] != "TOTAL"
            ],
            x="Response",
            y="Percentage",
            text="Percentage",
            title="Rural vs Urban Background"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    # =====================================================
    # LANGUAGE DIVERSITY
    # =====================================================

    st.header("Language Diversity and Schooling Background")

    language_col = (
        "3. What was the main language used by teachers in class?"
    )

    if language_col in df.columns:

        language = create_profile_table(
            df,
            language_col
        )

        st.dataframe(
            language,
            use_container_width=True
        )

        download_table(
            language,
            "Language_Diversity.csv",
            "📥 Download Language Diversity Data"
        )

        fig = px.pie(
            language[
                language["Response"] != "TOTAL"
            ],
            names="Response",
            values="Frequency",
            hole=0.4,
            title="Language Diversity"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    # =====================================================
    # FAMILY RESPONSIBILITIES
    # =====================================================

    st.header("Family Responsibilities")

    dependents_col = (
        "29. Do you have any dependents (e.g., Children, cousins, siblings, or nephews that you financially provide for)?"
    )

    if dependents_col in df.columns:

        dependents = create_profile_table(
            df,
            dependents_col
        )

        st.dataframe(
            dependents,
            use_container_width=True
        )

        download_table(
            dependents,
            "Family_Responsibilities.csv",
            "📥 Download Family Responsibility Data"
        )

        fig = px.bar(
            dependents[
                dependents["Response"] != "TOTAL"
            ],
            x="Response",
            y="Percentage",
            text="Percentage",
            title="Family Responsibilities"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    # =====================================================
    # COMMUTING CHALLENGES
    # =====================================================

    st.header("Commuting Challenges")

    travel_col = (
        "45. How long will it take you to travel to campus to arrive at 8am lecture on time?"
    )

    if travel_col in df.columns:

        travel = create_profile_table(
            df,
            travel_col
        )

        st.dataframe(
            travel,
            use_container_width=True
        )

        download_table(
            travel,
            "Commuting_Challenges.csv",
            "📥 Download Commuting Data"
        )

        fig = px.bar(
            travel[
                travel["Response"] != "TOTAL"
            ],
            y="Response",
            x="Percentage",
            orientation="h",
            text="Percentage",
            title="Commuting Challenges"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    # =====================================================
    # STUDENT SUCCESS PREDICTION MODEL
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

        predictions = model.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        st.metric(
            "Prediction Accuracy",
            f"{accuracy:.1%}"
        )

        st.divider()

        st.header(
            "Key Predictors of Academic Success"
        )

        importance = pd.DataFrame({

            "Variable":
            X.columns,

            "Importance":
            model.feature_importances_

        })

        importance = importance.sort_values(
            by="Importance",
            ascending=False
        )

        top20 = importance.head(20)

        st.dataframe(
            top20,
            use_container_width=True
        )

        fig = px.bar(
            top20,
            x="Importance",
            y="Variable",
            orientation="h",
            title="Top Predictors of Academic Achievement"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        download_table(
            top20,
            "Top_Predictors.csv",
            "📥 Download Predictor Analysis"
        )

        st.divider()

        st.header(
            "Student Success Segments"
        )

        pca = PCA(
            n_components=2
        )

        pca_results = pca.fit_transform(
            X
        )

        segments = pd.DataFrame({

            "Dimension 1":
            pca_results[:, 0],

            "Dimension 2":
            pca_results[:, 1],

            "Achievement":
            y

        })

        fig = px.scatter(
            segments,
            x="Dimension 1",
            y="Dimension 2",
            color="Achievement",
            title="Student Success Segments"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.divider()

        st.header(
            "Students Requiring Additional Support"
        )

        segments["Support Category"] = np.where(
            y <= y.median(),
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

    st.divider()

    # =====================================================
    # DOWNLOAD COMPLETE DATASET
    # =====================================================

    st.header("Download Results")

    csv = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "📥 Download Complete Dataset",
        csv,
        "SMU_FTEN_2025.csv",
        "text/csv"
    )

else:

    st.info(
        "Please upload the SMU Biographical Questionnaire dataset to begin."
    )
    def executive_chart(freq, title):

    chart_data = freq[
        freq["Response"] != "TOTAL"
    ]

    fig = px.bar(
        chart_data,
        y="Response",
        x="Percentage",
        orientation="h",
        text="Percentage"
    )

    fig.update_layout(
        title={
            "text": title,
            "font": {"size": 16}
        },
        font={
            "size": 14
        },
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    fig.update_traces(
        textposition="outside"
    )

    return fig
