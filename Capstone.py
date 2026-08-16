#!/usr/bin/env python
# coding: utf-8

# In[4]:

import streamlit as st  # for interactive web apps from data scripts
import plotly.express as px     # graphing library makes interactive, publication-quality graphs.
import pandas as pd     # statistical calculations
import os               # to download and upload files
import warnings 
warnings.filterwarnings('ignore')

st.set_page_config(page_title = "Tokenizer Analysis Dashboard", page_icon = ":pushpin:",layout = "wide")

st.title(":pushpin: Tokenizer Analysis Dashboard")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "Sample - Superstore.xls")

st.markdown('<style>div.block-container{padding-top:2rem;}</style>',unsafe_allow_html=True)  # padding for the web page

fl = st.file_uploader("Upload your dataset file", type=["csv","txt","xlsx","xls"])  # upload csv file
if fl is not None:   # if the file loads correctly
    filename = fl.name
    st.write(f"Uploaded file: {filename}")

    if filename.endswith('.csv'):
        df = pd.read_csv(fl, encoding = "ISO-8859-1")  # read the uploaded file into a pandas dataframe

    elif filename.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(fl)  # read the uploaded file into a pandas dataframe

    elif filename.endswith('.txt'):
        df = pd.read_csv(fl, sep="\t", encoding = "ISO-8859-1")  # read the uploaded file into a pandas dataframe
else:
    os.chdir(r"C:\Users\27783\Downloads") 

st.sidebar.header("Choose your language for analysis:")  # header for the language sele
region = st.sidebar.selectbox("Select your language", ["English", "isiXhosa", "isiZulu", "Sesotho", "Sepedi", "Setswana", "siSwati", "Xitsonga"])  # side bar showing the language options for analysis

st.sidebar.header("Choose your tokenizer for analysis:")  # header for the tokenizer selection
tokenizer = st.sidebar.selectbox("Select your tokenizer", ["Tokenizer 1", "Tokenizer 2", "Tokenizer 3"])  # side bar showing the tokenizer options for analysis


