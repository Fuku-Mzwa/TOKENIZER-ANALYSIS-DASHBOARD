#!/usr/bin/env python

# coding: utf-8


import streamlit as st  # for interactive web apps from data scripts
import plotly.express as px     # graphing library makes interactive, publication-quality graphs.
import pandas as pd     # statistical calculations
import os               # to download and upload files
import warnings 
warnings.filterwarnings('ignore')
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline  # for tokenization and named entity recognition
from tokenizers import Tokenizer
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
    st.markdown('<style>div.block-container{padding-top:2rem;}</style>',unsafe_allow_html=True)  # padding for the web page

elif selected == "Inspect":  # home page of the web app
    st.title(f"You are on page {selected}")
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, "Sample - Superstore.xls")

    st.markdown('<style>div.block-container{padding-top:2rem;}</style>',unsafe_allow_html=True)  # padding for the web page

    st.subheader("Dataset upload")  # subheader for the file upload section
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

    
    POPULAR_TOKENIZERS = {  # popular/most common tokenizers as a dictionary with the name of the tokenizer as the key and the model ID as the value
        "GPT-2": "gpt2",
        "BERT": "bert-base-uncased",
        "RoBERTa": "roberta-base",
        "CodeBERT": "microsoft/codebert-base",
        "Llama 3.2": "meta-llama/Llama-3.2-1B",
        "T5": "google-t5/t5-base"
    }

    
    if "tokenizer" not in st.session_state:  # session state saves the variable values across reruns of the app, so that the user does not have to reselect the tokenizer every time they run the app
        st.session_state.tokenizer = None

    if "tokenizer_name" not in st.session_state:
        st.session_state.tokenizer_name = None

    st.subheader("Choose a tokenizer") # subheader

    popular_names = list(POPULAR_TOKENIZERS.keys())  # saves the names of the popular tokenizers in a list

    selected_popular = st.selectbox(  # displays the names of the popular tokenizers in a dropdown menu, with "Other" as the last option
        "Popular tokenizers",
        popular_names + ["Other"],
    )

    if selected_popular != "Other":   # chosen popular tokenizer is saved in the variable selected_tokenizer from the dictionary POPULAR_TOKENIZERS
        selected_tokenizer = POPULAR_TOKENIZERS[selected_popular]

    elif selected_popular == "Other":  # typed model/tokenizer is checked to be nonempty with space stripped, and saved in the variable selected_tokenizer
        search_query = st.text_input( # model/tokenizer search
            "Search for an AI model / tokenizer",
            help="e.g. google/gemma-2b, xlm-roberta-base, meta-llama/Llama-3.2-1B",
        )
        selected_tokenizer = search_query.strip()

    else:   # if no popular tokenizer is selected and no model/tokenizer is typed, the variable selected_tokenizer is set to None
        selected_tokenizer = None



    # code to load the selected tokenizer using the AutoTokenizer class from the transformers library, and save it in the session state
    if selected_tokenizer:

        st.info(f"Selected tokenizer: {selected_tokenizer}")  # shows the selected tokenizer in an info box

        if st.button("Load tokenizer", type="primary"): # if button is clicked, the tokenizer is loaded and saved in the session state

            try:  # code to load the selected tokenizer using the AutoTokenizer class from the transformers library, and save it in the session state
                with st.spinner(  # shows a spinner while the tokenizer is loading (the tokenizer code runs while the spinner displays the loading indicator)
                    f"Loading {selected_tokenizer}..."
                ):
                    tokenizer = AutoTokenizer.from_pretrained(  # loads a tokenizer from the Hugging Face model hub using the model ID of the selected tokenizer
                        selected_tokenizer
                    )

                st.session_state.tokenizer = tokenizer
                st.session_state.tokenizer_name = selected_tokenizer

                st.success(   # shows a success message if the tokenizer is loaded successfully
                    f"Successfully loaded {selected_tokenizer}"
                )

            except OSError:  # shows an error message if the tokenizer could not be loaded, and resets the session state variables to None
                st.session_state.tokenizer = None
                st.session_state.tokenizer_name = None

                st.error(
                    f"{selected_tokenizer} is not a valid Hugging Face model/tokenizer ID. Please check the name and try again."
                )
    
            except Exception as e:  # shows an error message if the tokenizer could not be loaded, and resets the session state variables to None
                st.session_state.tokenizer = None
                st.session_state.tokenizer_name = None

                st.error(
                    f"{selected_tokenizer} is not a valid Hugging Face model/tokenizer ID. Please check the name and try again."
                )

                st.exception(e)
    else:
        st.warning("Please select a tokenizer or enter a model/tokenizer name to load.")
        

elif selected ==  "Compare":
    st.title(f"You are on page {selected}")
    st.markdown('<style>div.block-container{padding-top:2rem;}</style>',unsafe_allow_html=True)  # padding for the web page

    
elif selected == "Rank":
    st.title(f"You are on page {selected}")
    st.markdown('<style>div.block-container{padding-top:2rem;}</style>',unsafe_allow_html=True)  # padding for the web page


