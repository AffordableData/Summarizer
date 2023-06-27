# To run: > streamlit run app.py
# Name: Affordable Data Summarizer
# Description: Summarizes text files for comparison between OpenAI and Azure OpenAI
# Author: Chris Harris
# GitHub: https://github.com/AffordableData/Summarizer

# region: Imports
import streamlit as st

from langchain.llms import OpenAI, AzureOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain import PromptTemplate

from rich import print
from datetime import datetime
import time

import pyperclip

from os import getenv
from dotenv import load_dotenv, find_dotenv

# Load the .env file into the environment
load_dotenv(find_dotenv())

# endregion

# Raise an error if not running inside Streamlit
if st.runtime.exists() == False:
    raise RuntimeError("This must be run from Streamlit. Use > streamlit run app.py")


# region: Functions

def copy_to_clipboard(text):
    pyperclip.copy(text)
    st.info("Text copied to clipboard!")

# endregion

# region: Initialize variables

# Define the prompt template
prompt_template = """Write a concise summary of the following {doc_type}:


{text}


CONCISE SUMMARY:"""

uploaded_file = None
# endregion

# region: Initialize Streamlit
st.set_page_config(
    page_title="AD Summarizer",
    page_icon="🗜️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/AffordableData/Summarizer',
        'Report a bug': "https://github.com/AffordableData/Summarizer/issues",
        'About': "Summarizer is a proof of concept project for [Affordable Data](https://affordabledata.substack.com/)."
    }
)

st.title("Affordable Data Summarizer")
st.sidebar.title("Settings")

with st.sidebar:
    TEMPERATURE = st.slider(
        "Temperature (higher values = more creativity)",
        min_value=0.0,
        max_value=1.0,
        value=.5,
        step=0.01,
        format="%f",
    )
    doc_type = st.text_input("Document type (used in prompt)", value="article")
    openai_enabled = st.checkbox("Use OpenAI", value=True)
    azure_enabled = st.checkbox("Use Azure OpenAI", value=True)
    
# endregion

# region: Set up AI providers
prompt_template = prompt_template.replace("{doc_type}", doc_type)
PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

# Connection for OpenAI
llm_openai = OpenAI(
    openai_api_key=getenv("OpenAI_API_KEY"),
    temperature=TEMPERATURE,
)

# Connection for  AzureOpenAI
llm_azure = AzureOpenAI(
    openai_api_type="azure",
    openai_api_key=getenv("AZURE_OPENAI_KEY"),
    openai_api_base=getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_version=getenv("AZURE_OPENAI_API_VERSION"),
    deployment_name=getenv("DEPLOYMENT_NAME"),
    model_name=getenv("MODEL_NAME"),
    temperature=TEMPERATURE,
)

# endregion

# region: Set up session state
try:
    summary_openai = st.session_state['summary_openai']
    summary_openai_time = st.session_state['summary_openai_time']
except KeyError:
    summary_openai = None
    st.session_state['summary_openai'] = summary_openai
    summary_openai_time = None

try:
    summary_azure = st.session_state['summary_azure']
    summary_azure_time = st.session_state['summary_azure_time']
except KeyError:
    summary_azure = None
    st.session_state['summary_azure'] = summary_azure
    summary_azure_time = None

# endregion

# Get the file
uploaded_file = st.file_uploader("Choose a text file, then click Summarize", type=["txt"])

if  st.button("Summarize", key="summarize", disabled=(uploaded_file is None)):
    summary_openai = None
    summary_azure = None

st.divider()

# create Streamlit columns for the two providers
col1, col2 = st.columns(2)

if uploaded_file is not None:
    fileContents = uploaded_file.read().decode("utf-8")

    # Split the text into chunks and create a document with the chunks
    texts = CharacterTextSplitter(".", chunk_size=4000, chunk_overlap=200).split_text(
        fileContents
    )
    docs = [Document(page_content=t) for t in texts]

    with col1:
        provider = "OpenAI"
        st.header(provider)

        if summary_openai is None:
            with st.spinner(f"Summarizing file with {provider}..."):
                # Get summary from OpenAI
                openai_chain = load_summarize_chain(
                    llm_openai,
                    chain_type="map_reduce",
                    return_intermediate_steps=False,
                    map_prompt=PROMPT,
                    combine_prompt=PROMPT,
                )

                start_time = datetime.now()
                if openai_enabled:
                    summary_openai = openai_chain.run(docs)
                else:
                    summary_openai = "lorum ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
                    time.sleep(1.2)

                end_time = datetime.now()
                summary_openai_time = (end_time - start_time).total_seconds()

                st.session_state['summary_openai'] = summary_openai
                st.session_state['summary_openai_time'] = summary_openai_time

        st.markdown(f":blue[Seconds:] {summary_openai_time}")
        st.subheader(f"Summary")

        if st.button(f"Copy to Clipboard", key="copy_openai"):
            copy_to_clipboard(summary_openai)

        st.markdown(summary_openai)

    with col2:
        provider = "Azure OpenAI"
        st.header(provider)

        if summary_azure is None:
            with st.spinner(f"Summarizing file with {provider}..."):
                # Get summary from Azure
                azure_chain = load_summarize_chain(
                    llm_azure,
                    chain_type="map_reduce",
                    return_intermediate_steps=False,
                    map_prompt=PROMPT,
                    combine_prompt=PROMPT,
                )

                start_time = datetime.now()
                if azure_enabled:
                    summary_azure = azure_chain.run(docs)
                else:
                    summary_azure = "lorum ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
                    time.sleep(1.3)
                    
                end_time = datetime.now()
                summary_azure_time = (end_time - start_time).total_seconds()

                st.session_state['summary_azure'] = summary_azure
                st.session_state['summary_azure_time'] = summary_azure_time

        st.markdown(f":blue[Seconds:] {summary_azure_time}")
        st.subheader(f"Summary")

        if st.button(f"Copy to Clipboard",key="copy_azure"):
            copy_to_clipboard(summary_azure)

        st.markdown(summary_azure)
