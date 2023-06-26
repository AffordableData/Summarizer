from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain import PromptTemplate

from os import getenv
from dotenv import load_dotenv, find_dotenv

# Define the prompt template
prompt_template = """Write a concise summary of the following transcription:


{text}


CONCISE SUMMARY:"""

# Load the .env file into the environment
load_dotenv(find_dotenv())

PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

# Read the transcription file
filePath = "C:\\temp\\mp3_transcribed.txt"
with open(filePath, "r") as f:
    fileContents = f.read()

# Connection for OpenAI
llm_openai = OpenAI(
    openai_api_key=getenv("OpenAI_API_KEY"),
    temperature=.5
)

# Split the fileContents into chunks and create a document with the chunks
texts = CharacterTextSplitter(".", chunk_size=4000, chunk_overlap=200).split_text(fileContents)
docs = [Document(page_content=t) for t in texts]

# Construct the chain
openai_chain = load_summarize_chain(
    llm_openai,
    chain_type="map_reduce",
    return_intermediate_steps=False,
    map_prompt=PROMPT,
    combine_prompt=PROMPT,
)

# Get the summary
summary = openai_chain.run(docs)

print(summary)