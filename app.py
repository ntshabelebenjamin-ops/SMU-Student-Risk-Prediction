st.header("Executive Summary")

col1, col2, col3 = st.columns(3)

col1.metric("First-Time Entering Students", len(df))
col2.metric("Variables", len(df.columns))
col3.metric("Missing Values", int(df.isnull().sum().sum()))
