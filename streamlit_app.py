import os
import sys
import time
import streamlit as st
import asyncio
import uuid
import json
from langchain_core.messages import HumanMessage, AIMessage

from src.exception.exception import RAGException
from src.config.config import Config
from src.vectorstore.vectorstore import RAGVectorStore
from src.graph_builder.graph_builder import GraphBuilder
from src.state.state import RAGState, Decision, Router
from pydantic.json import pydantic_encoder


def init_session_state(): 
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'display_history' not in st.session_state:
        st.session_state.display_history = []
    if 'chat_history_messages' not in st.session_state:
        st.session_state.chat_history_messages = []
    if 'thread_id' not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())


@st.cache_resource
def initialize_rag():
    try:
        llm = Config.get_llm_model()
        vs = RAGVectorStore()
        vector_store = vs.load_vectorstore()
        retriever = vs.create_retriever()
        
        graph_builder = GraphBuilder(retriever=retriever, llm=llm).build_graph()
        return graph_builder

    except Exception as e:
        raise RAGException(e, sys)
    

def main():

    init_session_state()

    config = {'configurable': {'thread_id': st.session_state.thread_id}}

    st.title("🔍 RAG Document Search (Node-by-Node Debugger)")
    st.markdown("Ask anything and inspect each node's state updates in real time.")

    if not st.session_state.initialized:
        with st.spinner("Loading system..."):
            rag_system = initialize_rag()
            if rag_system:
                st.session_state.rag_system = rag_system
                st.session_state.initialized = True
                st.success("System is ready!")

    st.markdown("---")

    # Display history
    for turn in st.session_state.display_history:
        st.markdown(f"**👤 You:** {turn['question']}")
        st.markdown(f"**🤖 Assistant:** {turn['answer']}")
        st.caption(f"⏱️ {turn['time']:.2f} seconds")
    st.markdown("---")
    
    def serialize_for_json(obj):
        if isinstance(obj, dict):
            return {k: serialize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [serialize_for_json(v) for v in obj]
        elif hasattr(obj, "dict"):  # Pydantic model
            return serialize_for_json(obj.dict())
        elif hasattr(obj, "page_content"):  # LangChain Document
            return obj.page_content
        else:
            try:
                return str(obj)  # fallback to string representation
            except:
                return repr(obj)

    # Input form
    with st.form("search_form"):
        question = st.text_input("Your question:")
        submit = st.form_submit_button("🔍 Search")

        if submit and question:
            if st.session_state.rag_system:

                start_time = time.time()

                initial_state = {
                    "question": question,
                    "chat_history": st.session_state.chat_history_messages
                }

                # Display containers
                st.markdown("### 💡 Final Answer")
                answer_placeholder = st.empty()

                st.markdown("### ⚙️ Graph Execution Trace (Node Debug)")
                trace_placeholder = st.empty()

                final_answer = ""
                trace_blocks = []

                last_rewritten_query = question
                
                # STREAM GRAPH EXECUTION (values mode)
                for state_update in st.session_state.rag_system.stream(
                    initial_state,
                    config=config,
                    stream_mode="values"
                ):

                    # Identify which node produced this update
                    # LangGraph adds a special key: "__state__"
                    node_name = state_update.get("__state__", "UNKNOWN_NODE")

                    # Remove metadata keys if present
                    cleaned_state = {k: v for k, v in state_update.items() if not k.startswith("__")}
                    safe_state = serialize_for_json(cleaned_state)
                    
                    
                trace_placeholder.markdown(f"```json\n{json.dumps(safe_state, indent=2)}\n```")

                # Update answer if present
                if "answer" in cleaned_state:
                    final_answer = cleaned_state["answer"]
                    answer_placeholder.success(final_answer)

            elapsed = time.time() - start_time

            # Store history
            if final_answer:
                st.session_state.display_history.append({
                    "question": question,
                    "answer": final_answer,
                    "time": elapsed
                })
                st.session_state.chat_history_messages.append(HumanMessage(content=question))
                st.session_state.chat_history_messages.append(AIMessage(content=final_answer))

                st.caption(f"⏱️ {elapsed:.2f} seconds")
                
                
if __name__ == "__main__":
    main()