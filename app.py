import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="SMU FTEN Profile Dashboard",
    layout="wide"
)

st.title("SMU First-Time Entering Students Profile Dashboard")

uploaded_file = st.file_uploader(
    "Upload SMU Biographical Questionnaire Dataset",
    type=["xlsx"]
)

if uploaded_file is not None:

    df = pd.read_excel(uploaded_file)
