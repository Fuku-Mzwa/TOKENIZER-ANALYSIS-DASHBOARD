#!/usr/bin/env python

# coding: utf-8


import streamlit as st  # for interactive web apps from data scripts
import plotly.express as px     # graphing library makes interactive, publication-quality graphs.
import pandas as pd     # statistical calculations
import os               # to download and upload files
import warnings 
warnings.filterwarnings('ignore')
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline  # for tokenization and named entity recognition
from streamlit_option_menu import option_menu
from huggingface_hub import HfApi

st.set_page_config(page_title = "Tokenizer Analysis Dashboard", page_icon = ":pushpin:",layout = "wide")  # layout of the web page is set to wide to use the entire screen space

st.title(":pushpin: Tokenizer Analysis Dashboard") #  code for title of the home page

selected = option_menu(   # navigation bar for the web page
        menu_title = None,
        options = ["Home", "Inspect", "Compare", "Rank"],
        icons = ["house", "eye", "arrow-left-right", "trophy"],
        menu_icon = "cast",
        default_index = 0,
        orientation = "horizontal",
)

if selected == "Home":  # home page of the web app
    st.title(f"You are on page {selected}")
    
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

    st.sidebar.header("Choose your language for analysis:")  # header for the language selection
    language = st.sidebar.selectbox("Select your language", ["English", "isiXhosa", "isiZulu", "Sesotho", "Sepedi", "Setswana", "siSwati", "Xitsonga"])  # side bar showing the language options for analysis

    st.sidebar.header("Choose your tokenizer for analysis:")  # header for the tokenizer selection
    tokenizer = st.sidebar.selectbox("Select your tokenizer", ["Tokenizer 1", "Tokenizer 2", "Tokenizer 3"])  # side bar showing the tokenizer options for analysis

    api = HfApi()  # Hugging Face API to access the models and tokenizers

    search_input = st.text_input("Search for a tokenizer on Hugging Face Hub")  # search bar for the user to search for a tokenizer or model

     
elif selected == "Inspect":
    st.title(f"You are on page {selected}")
    st.markdown('<style>div.block-container{padding-top:2rem;}</style>',unsafe_allow_html=True)  # padding for the web page

elif selected ==  "Compare":
    st.title(f"You are on page {selected}")
    st.markdown('<style>div.block-container{padding-top:2rem;}</style>',unsafe_allow_html=True)  # padding for the web page

    
elif selected == "Rank":
    st.title(f"You are on page {selected}")
    st.markdown('<style>div.block-container{padding-top:2rem;}</style>',unsafe_allow_html=True)  # padding for the web page


