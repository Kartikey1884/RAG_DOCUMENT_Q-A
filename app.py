import streamlit as st
import os
from langchain_groq import ChatGroq
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
load_dotenv()

## load groq api key
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="llama-3.1-8b-instant"  # Updated!
)
prompt = ChatPromptTemplate.from_template(
    """
    answer the question based on the context only.
    please provide the most accurate response based on 
    the question
    <context>
    {context}
    <context>
    question:{input}
    """
)

# ✅ initialize session state
if "vectors" not in st.session_state:
    st.session_state.vectors = None

def create_vectors_embeddings():
    if st.session_state.vectors is None:
        st.session_state.loader = PyPDFDirectoryLoader("research_papers")  # data ingestion
        st.session_state.docs = st.session_state.loader.load()  # document loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, embeddings)

user_prompt = st.text_input("Enter your query from the research paper")

if st.button("Document Embedding"):
    create_vectors_embeddings()
    st.write("✅ Vector database is ready")

import time

if user_prompt:
    if st.session_state.vectors is None:
        st.warning("⚠️ Please create the vector database first by clicking 'Document Embedding'")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retriever_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retriever_chain.invoke({'input': user_prompt})  # ✅ fixed
        print(f"response time : {time.process_time() - start}")

        st.write(response['answer'])

        ## With a streamlit expander
        with st.expander("Document similarity Search"):
            for i, doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write('------------------------')
