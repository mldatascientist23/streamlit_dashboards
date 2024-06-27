import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# Web App title
st.markdown('''
# **Exploratory Data Analysis Web App**
This app is developed by DigitalGuruTech Youtube channel called **EDA App**
''')

# How to upload a file from PC
with st.sidebar.header("Upload your dataset (.csv)"):
    upload_file = st.sidebar.file_uploader("Upload your file", type=['csv'])
    st.sidebar.markdown("[Example CSV file](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)")

# Profiling report for pandas
if upload_file is not None:
    @st.cache_data
    def load_csv():
        csv = pd.read_csv(upload_file)
        return csv
    df = load_csv()
    pr = ProfileReport(df, explorative=True)
    st.header('**Input DataFrame**')
    st.write(df)
    st.markdown('---')
    st.header('**Profiling Report with Pandas Profiling**')
    st_profile_report(pr)
else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to use example data'):
        # Example dataset
        def load_data():
            a = pd.DataFrame(np.random.rand(100, 5),
                             columns=['age', 'banana', 'codanics', 'Deutchland', 'Ear'])
            return a

        df = load_data()
        pr = ProfileReport(df, explorative=True)
        st.header('**Input DataFrame**')
        st.write(df)
        st.markdown('---')
        st.header('**Profiling Report with Pandas Profiling**')
        st_profile_report(pr)