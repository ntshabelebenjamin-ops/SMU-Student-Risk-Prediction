st.header("Executive Summary")

col1, col2, col3 = st.columns(3)

col1.metric("First-Time Entering Students", len(df))
col2.metric("Variables", len(df.columns))
col3.metric("Missing Values", int(df.isnull().sum().sum()))
demographic_vars = [
    "3. What was the main language used by teachers in class?",
    "5. Was your language learning at school the same as your home language?",
    "6. Is English your first, second or third language?",
    "7. How did you pay for your school fees?",
    "8. What was the average number of learners in your matric classroom?",
    "9. How would you describe the place in which your school is situated?"
]
for col in demographic_vars:

    st.subheader(col)

    freq = (
        df[col]
        .fillna("Missing")
        .value_counts()
        .reset_index()
    )

    freq.columns = ["Response", "Frequency"]

    freq["Percentage"] = round(
        freq["Frequency"] / len(df) * 100,
        2

    academic_vars = [
    "13. Did your high school make use of online remote learning services?",
    "16. How often did you make use of online material, online learning sites or ‘apps’?",
    "20. Did you participate in any leadership (extracurricular) activities during high school?\n(e.g., School Representative Council, Junior City Council etc.)",
    "22. Academic Achievement"
]
    )
    home_vars = [
    "24. Please answer the following questions about your home [Does your home have access to running water ?]",
    "24. Please answer the following questions about your home [Does your home have access to electricity?]",
    "24. Please answer the following questions about your home [Does your home have a computer/laptop?]",
    "24. Please answer the following questions about your home [Does your home have access to internet connection ( eg. Wifi...)?]",
    "24. Please answer the following questions about your home [Do you have a private area to study in your home?]"
]
    social_vars = [
    "29. Do you have any dependents (e.g., Children, cousins, siblings, or nephews that you\nfinancially provide for)?",
    "33. Do you plan to work part-time during your studies?",
    "39. Where will you be staying during your studies?",
    "43. How far away from campus do you live?",
    "45. How long will it take you to travel to campus to arrive at 8am lecture on time?"
]
    emotional_vars = [
    "28. While you are at university, who will be providing you with general support e.g.,\nemotional, guidance, parental etc.",
    "35. What fears or concerns do you have regarding your first year at university?",
    "36. What are you looking forward to the most in your first year at university?"
]
    st.header("Key Insights")

st.write("• Academic achievement distribution")
st.write("• Access to internet and technology")
st.write("• Home study environment")
st.write("• Distance from campus")
st.write("• Funding sources")
st.write("• Support structures")
TARGET = "22. Academic Achievement"
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

ml_df = df.copy()

ml_df = ml_df.fillna("Missing")

for col in ml_df.columns:
    ml_df[col] = LabelEncoder().fit_transform(
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
    n_estimators=300,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

st.metric(
    "Classification Accuracy",
    f"{accuracy:.2%}"
)
importance = pd.DataFrame({
    "Variable": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
).head(20)

fig = px.bar(
    importance,
    x="Importance",
    y="Variable",
    orientation="h",
    title="Top Predictors of Academic Achievement"
)

st.plotly_chart(fig)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

results = pca.fit_transform(X)

groups = pd.DataFrame({
    "PCA1": results[:,0],
    "PCA2": results[:,1],
    "Achievement": y
})

fig = px.scatter(
    groups,
    x="PCA1",
    y="PCA2",
    color="Achievement",
    title="Student Groups"
)

st.plotly_chart(fig)
groups["Risk"] = np.where(
    y <= y.median(),
    "At Risk",
    "Not At Risk"
)

fig = px.scatter(
    groups,
    x="PCA1",
    y="PCA2",
    color="Risk",
    title="At-Risk Student Groups"
)

st.plotly_chart(fig)

    st.dataframe(freq)
