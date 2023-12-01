import os
import streamlit as st 
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import AzureChatOpenAI


openai_token = os.environ.get("OPENAI_TOKEN", "")
openai_endpoint = "https://mti-nerve-openai-us-east-2.openai.azure.com/"

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-12-01-preview"
os.environ["OPENAI_API_BASE"] = openai_endpoint
os.environ["OPENAI_API_KEY"] = openai_token    

class VectorDatabase:
    def __init__(self, path):
        self.path = path

def create_agent_chain():
    llm = AzureChatOpenAI(temperature=0, 
        verbose=True, 
        deployment_name="gpt-4",
    )
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain

def get_llm_response(query):
    vectordb = VectorDatabase("./chroma_store")
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