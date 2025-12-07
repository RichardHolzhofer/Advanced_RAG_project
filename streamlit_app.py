import os
import sys
import time
import streamlit as st
import asyncio

# Assuming these modules are correctly implemented in your project structure
from src.exception.exception import RAGException
from src.config.config import Config
# from src.document_ingestion.document_loader import DocumentLoader
# from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder

st.set_page_config(
    page_title="Advanced RAG application for Neuroflow AI",
    page_icon="🤖🔎",
    layout="centered"
)

def init_session_state(): 
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'history' not in st.session_state:
        st.session_state.history = []


@st.cache_resource
def initialize_rag():

    try:
        llm=Config.get_llm_model()
        vs = VectorStore()
        vector_store = vs.load_vectorstore()
        retriever = vs.create_retriever()
        
        graph_builder = GraphBuilder(retriever=retriever, llm=llm).build_graph()
        
        return graph_builder
        
    except Exception as e:
        raise RAGException(e, sys)
    
    
def main():
    """
    Synchronous main function for the Streamlit application flow.
    """
    
    init_session_state()
    
    st.title("🔍 RAG Document Search")
    st.markdown("Ask questions about the loaded documents")
    
    if not st.session_state.initialized:
        with st.spinner("Loading system..."):
            rag_system = initialize_rag()
            if rag_system:
                st.session_state.rag_system = rag_system
                st.session_state.initialized = True
                st.success(f"System is ready to be used")

    st.markdown("---")
    
    with st.form("search_form"):
        question = st.text_input(
            "Enter your question:",
            placeholder="What would you like to know?"
        )
        submit = st.form_submit_button("🔍 Search")
        
        if submit and question:
            if st.session_state.rag_system:
                with st.spinner("Searching..."):
                    start_time = time.time()
                    
                    result = st.session_state.rag_system.invoke({'question':question})
                    
                    elapsed_time = time.time() - start_time
                    
                    st.session_state.history.append({
                    'question': question,
                    'answer': result['answer'],
                    'time': elapsed_time
                    })
                    
                    st.markdown("### 💡 Answer")
                    st.success(result['answer'])
                    
                    st.caption(f"⏱️ Response time: {elapsed_time:.2f} seconds")
                    
if __name__ == "__main__":
    main()