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
