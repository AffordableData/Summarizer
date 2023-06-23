# Reference: https://python.langchain.com/docs/modules/chains/popular/summarize.html
# To run: streamlit run app.py

import streamlit as st

from langchain.llms import OpenAI, AzureOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain import PromptTemplate

from rich import print
from datetime import datetime
import time

# use env vars
from os import getenv
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# region: initialize variables
TEMPERATURE = 0.9
print(f"\nTEMPERATURE: {TEMPERATURE}")

# Prompt Template
prompt_template = """Write a concise summary of the following sermon:


{text}


CONCISE SUMMARY IN ENGLISH:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

# print start time
start_time = datetime.now()
print(f"Start time: {start_time}")

uploaded_file = None
# endregion

# Get filecontents from Streamlit or filesystem
if st.runtime.exists() == False:

    raise RuntimeError("This must be run from Streamlit. Use > streamlit run app.py")

# Initialize Streamlit and get the file
st.title("The Summarizer")
uploaded_file = st.file_uploader("Choose a text file", type=["txt"])
st.divider()


if uploaded_file is not None:

    # create 2 colums in Streamlit
    col1,col2 = st.columns(2)

    fileContents = uploaded_file.read().decode("utf-8")

    # Connect to OpenAI
    llm_openai = OpenAI(
        openai_api_key=getenv("OpenAI_API_KEY"),
        temperature=TEMPERATURE,
    )

    # Connect to AzureOpenAI
    llm_azure = AzureOpenAI(
        openai_api_type="azure",
        openai_api_key = getenv("AZURE_OPENAI_KEY"),
        openai_api_base = getenv("AZURE_OPENAI_ENDPOINT"),
        openai_api_version = getenv("AZURE_OPENAI_API_VERSION"),
        deployment_name=getenv("DEPLOYMENT_NAME"),
        model_name = getenv("MODEL_NAME"),
        temperature=TEMPERATURE,
    )
    
    # Split the text into chunks and create a document with the chunks
    texts = CharacterTextSplitter(".", chunk_size=4000, chunk_overlap=200).split_text(fileContents)
    docs = [Document(page_content=t) for t in texts]

    with st.spinner("Summarizing file with OpenAI..."):

        # Get summary from OpenAI
        openai_chain = load_summarize_chain(
            llm_openai, 
            chain_type="map_reduce", 
            return_intermediate_steps=False,
            map_prompt=PROMPT, 
            combine_prompt=PROMPT
            )
    
        openai_start_time = datetime.now()
        summary_openai = "lorum ipsum dolor sit amet"
        time.sleep(3)
        # summary_openai = openai_chain.run(docs)
        openai_end_time = datetime.now()
        col1.header("OpenAI")
        col1.markdown(f"**OpenAI elapsed time**: {openai_end_time - openai_start_time}")
        col1.markdown(f"**summary_openai:** {summary_openai}")

    with st.spinner("Summarizing file with Azure..."):
        # Get summary from Azure
        azure_chain = load_summarize_chain(
            llm_azure, 
            chain_type="map_reduce", 
            return_intermediate_steps=False,
            map_prompt=PROMPT, 
            combine_prompt=PROMPT
            )
        
        azure_start_time = datetime.now()
        summary_azure = "lorum ipsum dolor sit amet"
        time.sleep(3)
        # summary_azure = azure_chain.run(docs)
        azure_end_time = datetime.now()
        col2.header("Azure")
        col2.markdown(f"**Azure elapsed time**: {azure_end_time - azure_start_time}")
        col2.markdown(f"**summary_azure:** {summary_azure}")
    
    end_time = datetime.now()
    st.divider()
    col1, col2, col3 = st.columns(3)
    col2.markdown(f"**End time:** {end_time}",)
    col2.markdown(f"**Elapsed time**: {end_time - start_time}")

else:
    print("No file uploaded")
