import os
import sys
import time
import streamlit as st
import asyncio
import uuid
from langchain_core.messages import HumanMessage, AIMessage # Import required for history tracking

# Assuming these modules are correctly implemented in your project structure
from src.exception.exception import RAGException
from src.config.config import Config
# from src.document_ingestion.document_loader import DocumentLoader
# from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import RAGVectorStore
from src.graph_builder.graph_builder import GraphBuilder



def init_session_state(): 
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'display_history' not in st.session_state: # Renamed 'history' for clarity
        st.session_state.display_history = []
    if 'chat_history_messages' not in st.session_state: # NEW: Stores BaseMessages for LangGraph
        st.session_state.chat_history_messages = [] 
    if 'thread_id' not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())


@st.cache_resource
def initialize_rag():

    try:
        llm=Config.get_llm_model()
        vs = RAGVectorStore()
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
    
    config = {'configurable': {'thread_id': st.session_state.thread_id}}
    
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
    
    # Display previous history outside the form
    for turn in st.session_state.display_history:
        st.markdown(f"**👤 You:** {turn['question']}")
        st.markdown(f"**🤖 Assistant:** {turn['answer']}")
        st.caption(f"⏱️ Response time: {turn['time']:.2f} seconds")
    st.markdown("---")


    with st.form("search_form"):
        question = st.text_input(
            "Enter your question:",
            placeholder="What would you like to know?"
        )
        submit = st.form_submit_button("🔍 Search")
        
        if submit and question:
            if st.session_state.rag_system:
                
                start_time = time.time()
                
                # 1. Prepare initial state for LangGraph
                initial_state = {
                    "question": question, 
                    # Pass the correct history object (list of BaseMessages)
                    "chat_history": st.session_state.chat_history_messages 
                }
                
                # 2. Setup visualization placeholders
                with st.container():
                    st.markdown("### 💡 Answer")
                    answer_placeholder = st.empty()
                    
                    st.markdown("### ⚙️ Graph Execution Trace (DEBUG)")
                    trace_placeholder = st.empty()
                
                full_answer = ""
                trace_updates = []

                # 3. Stream the execution using stream_mode="values"
                for state_update in st.session_state.rag_system.stream(
                    initial_state, 
                    config=config, 
                    stream_mode="values"
                ):
                    
                    # A. Collect Trace/Debug Information
                    debug_output = {}
                    for key, value in state_update.items():
                        # Only show relevant keys and format for readability
                        if key == "retrieved_docs" and value:
                            doc_contents = [doc.page_content[:40] + "..." for doc in value]
                            debug_output[key] = f"{len(value)} Docs retrieved: {doc_contents}"
                        elif key == "rewritten_query" and value:
                             debug_output[key] = str(value)
                        elif key == "answer" and value and value != full_answer:
                             # Track the final answer
                             debug_output[key] = str(value)
                        
                    # B. Update the Debug Trace display
                    if debug_output:
                        trace_updates.append(debug_output)
                        
                        trace_markdown = ""
                        for item in trace_updates:
                            trace_markdown += "\n\n**State Update:**\n"
                            for k, v in item.items():
                                trace_markdown += f"- **{k}:** `{v}`\n"
                        
                        trace_placeholder.markdown(trace_markdown)
                        
                        # C. Update the final answer if it exists in the stream
                        if "answer" in state_update:
                             full_answer = state_update["answer"]
                             answer_placeholder.success(full_answer)

                elapsed_time = time.time() - start_time
                
                # 4. Final Cleanup and History Update
                if full_answer:
                    # Update display history (for the simple view)
                    st.session_state.display_history.append({
                        'question': question,
                        'answer': full_answer,
                        'time': elapsed_time
                    })
                    
                    # Update LangGraph history (for conversational turns)
                    st.session_state.chat_history_messages.append(HumanMessage(content=question))
                    st.session_state.chat_history_messages.append(AIMessage(content=full_answer))

                    st.caption(f"⏱️ Response time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()