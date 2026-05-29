# SMU First-Time Entering Students Profile Dashboard

## Overview

This Streamlit dashboard provides an Institutional Research analysis of the SMU First-Time Entering Student (FTEN) cohort using data collected through the Biographical Questionnaire.

The dashboard profiles students according to their demographic characteristics, academic preparedness, social circumstances, emotional support structures, and home environment. In addition, the dashboard applies machine learning techniques to identify factors associated with Academic Achievement and classify students into achievement and risk groups.

---

## Dashboard Components

### 1. Executive Summary

Provides an overview of:

- Total number of students
- Total number of questionnaire variables
- Missing values in the dataset

---

### 2. Biographical and Demographic Profile

Analyses:

- Language background
- School location
- School fee payment
- Class size
- Language of learning and teaching
- English language exposure

Outputs:

- Frequency tables
- Percentages

---

### 3. Academic Pressures

Analyses:

- Online learning exposure
- Use of learning technologies
- Leadership participation
- Academic Achievement

Outputs:

- Frequency tables
- Percentages

---

### 4. Home Environment Profile

Analyses:

- Electricity access
- Running water access
- Internet access
- Computer/laptop ownership
- Private study space

Outputs:

- Frequency tables
- Percentages

---

### 5. Social Pressures

Analyses:

- Dependents
- Accommodation
- Travel distance
- Travel time
- Employment intentions

Outputs:

- Frequency tables
- Percentages

---

### 6. Emotional Pressures

Analyses:

- Emotional support systems
- First-year concerns
- First-year expectations

Outputs:

- Frequency tables
- Percentages

---

### 7. Academic Achievement Classification

Machine Learning Model:

- Random Forest Classifier

Response Variable:

- 22. Academic Achievement

Predictor Variables:

- All remaining questionnaire variables

Outputs:

- Classification accuracy
- Variable importance rankings
- Student groups visualisation
- At-risk student visualisation

---

### 8. Student Groups

Principal Component Analysis (PCA) is used to visualise student groups according to their characteristics and academic achievement.

Outputs:

- Student clusters
- Achievement groups

---

### 9. At-Risk Students

Students are grouped into:

- At Risk
- Not At Risk

based on Academic Achievement outcomes.

Outputs:

- At-risk student visualisation

---

## Installation

Install all required packages:

```bash
pip install -r requirements.txt
```

---

## Running the Dashboard

Run the application:

```bash
streamlit run app.py
```

---

## Data Requirements

Upload the SMU Biographical Questionnaire dataset in Microsoft Excel format (.xlsx).

---

## Author

Benjamin Tlhale Ntshabele

Institutional Research Unit

Sefako Makgatho Health Sciences University
