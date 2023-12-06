import os

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import sqlite3
import streamlit as st 

from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import AzureChatOpenAI
import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain


openai_token = os.environ.get("OPENAI_TOKEN", "")
openai_endpoint = "https://mti-nerve-openai-us-east-2.openai.azure.com/"

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-12-01-preview"
os.environ["OPENAI_API_BASE"] = openai_endpoint
os.environ["OPENAI_API_KEY"] = openai_token    

embeddings = AzureOpenAIEmbeddings(deployment="text-embedding-ada-002",chunk_size=1)

dir="./chroma_store/"
vectordb = Chroma(persist_directory=dir,embedding_function=embeddings)

def create_agent_chain():
    llm = AzureChatOpenAI(temperature=0, 
        verbose=True, 
        deployment_name="gpt-4",
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm, vectordb.as_retriever(), return_source_documents=True
        )
    #chain = load_qa_chain(llm, chain_type="stuff")
    return chain

def get_llm_response(query):
    matching_docs = vectordb.similarity_search(query)
    chain = create_agent_chain()
    answer = chain.run(input_documents=matching_docs, question=query)
    return answer


# Streamlit UI
# ===============
#st.set_page_config(page_title="Ask Economist", page_icon=":robot:")
#st.header("Ask Economist")

#form_input = st.text_input('Enter Query')
#submit = st.button("Generate")

#if submit:
#    st.write(get_llm_response(form_input))

st.title("Ask Economist")
# container for text box
container = st.container()
# container for chat history
response_container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        st.subheader("Enter Query")
        user_input = st.text_area("Enter Query", key='input', height=100, label_visibility="hidden")
        submit_button = st.form_submit_button(label='Generate')

    
    if submit_button:
        st.write(get_llm_response(user_input))

# The 'message' function is defined to display the messages in the conversation history.
if 'answers' in st.session_state:
    if st.session_state['answers']:
        with response_container:
            cols = st.columns(2)
            for i in range(len(st.session_state['answers'])):
                with cols[0]:
                    st.subheader("Enter Query")
                    st.write(st.session_state['past'][i])
                    st.subheader("Answer")
                    st.write(st.session_state['answers'][i])
                with cols[1]:
                    st.subheader("Sources: ")
                    for index, url in enumerate(st.session_state['sources'][i].split(" ")):
                        st.write(index+1, ". ", url)
                        st.text(" ")

def generate_response(prompt, conversation_chain):
    try:
        result = conversation_chain(prompt)
        return result["answer"], ' '.join(list(set([doc.metadata['source'] for doc in result['source_documents']])))
    except Exception as e:
        print(e)
        return "I am unable to get the response based on this question, please fine-tune it before retrying", ""