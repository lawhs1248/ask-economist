import os

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import sqlite3
import streamlit as st 

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import AzureChatOpenAI
from langchain.vectorstores import Chroma
import chromadb
from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain


os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-12-01-preview"
os.environ["OPENAI_API_KEY"] = "c3569353e6854459a2b8576084f97a17"
os.environ["OPENAI_API_BASE"] = "https://mti-nerve-openai-us-east-2.openai.azure.com/"
embeddings = AzureOpenAIEmbeddings(deployment="text-embedding-ada-002",chunk_size=1)


def load_chunk_persist_pdf() -> Chroma:
    pdf_folder_path = "./input"
    documents = []
    for file in os.listdir(pdf_folder_path):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder_path, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    chunked_documents = text_splitter.split_documents(documents)
    client = chromadb.Client()
    if client.list_collections():
        consent_collection = client.create_collection("consent_collection")
    else:
        print("Collection already exists")
    vectordb = Chroma.from_documents(
        documents=chunked_documents,
        embedding=embeddings,
        persist_directory="./testing_space/chroma_store"
    )
    vectordb.persist()
    return vectordb

def create_agent_chain():
    llm = AzureChatOpenAI(temperature=0, 
        verbose=True, 
        deployment_name="gpt-4",
    )
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain

def get_llm_response(query):
    vectordb = load_chunk_persist_pdf()
    chain = create_agent_chain()
    matching_docs = vectordb.similarity_search(query)
    answer = chain.run(input_documents=matching_docs, question=query)
    return answer


# Streamlit UI
# ===============
st.set_page_config(page_title="Ask Economist", page_icon=":robot:")
st.header("Ask Economist")

form_input = st.text_input('Enter Query')
submit = st.button("Generate")

if submit:
    st.write(get_llm_response(form_input))