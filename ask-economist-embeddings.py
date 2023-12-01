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

embeddings = AzureOpenAIEmbeddings(deployment="text-embedding-ada-002",chunk_size=1)

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
    persist_directory="./chroma_store/"
)
vectordb.persist()