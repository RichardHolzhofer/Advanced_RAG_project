import os
import sys
import time
import streamlit as st
import uuid
from langchain_core.messages import HumanMessage, AIMessage

# Assuming these imports are correct based on your file structure
from src.exception.exception import RAGException
from src.config.config import Config
from src.vectorstore.vectorstore import RAGVectorStore
from src.graph_builder.graph_builder import GraphBuilder
# RAGState and pydantic_encoder are no longer strictly needed in this simplified script

# ----------------- Initialization and Session State -----------------

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
    """Initializes the RAG system and LangGraph instance."""
    try:
        # NOTE: Environment variables for LLM, TAVILY, and LANGSMITH 
        # must be set in your environment (e.g., in a .env file loaded globally)
        
        llm = Config.get_llm_model()
        vs = RAGVectorStore()
        vs.load_vectorstore()
        
        # We only need to call create_retriever which will handle loading the vector store internally
        retriever = vs.create_retriever()
        
        # Build the graph
        graph_builder = GraphBuilder(retriever=retriever, llm=llm).build_graph()
        return graph_builder

    except Exception as e:
        # Use RAGException logic for clearer logging if needed, or stick to simple st.error
        st.error(f"Error during RAG initialization: {e}")
        return None
    

# ----------------- Main Streamlit App -----------------

def main():
    # Load env variables for TAVILY if not globally set
    os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY") 
    init_session_state()

    # The config now only needs the thread_id for LangGraph/LangSmith tracing
    config = {'configurable': {'thread_id': st.session_state.thread_id}}

    st.title("🔍 RAG Document Search (LangSmith Traced)")
    st.markdown("Ask a question to trigger the graph. View the full execution trace and debug nodes on the **LangSmith** platform.")

    if not st.session_state.initialized:
        with st.spinner("Loading system..."):
            rag_system = initialize_rag()
            if rag_system:
                st.session_state.rag_system = rag_system
                st.session_state.initialized = True
                st.success("System is ready! Submit a query to begin.")
            else:
                st.warning("RAG System failed to initialize. Please check your console for errors.")

    st.markdown("---")

    # Display history
    for turn in st.session_state.display_history:
        st.markdown(f"**👤 You:** {turn['question']}")
        st.markdown(f"**🤖 Assistant:** {turn['answer']}")
        st.caption(f"⏱️ {turn['time']:.2f} seconds")
    st.markdown("---")
    
    
    # Input form
    with st.form("search_form"):
        question = st.text_input("Your question:")
        submit = st.form_submit_button("🔍 Search")

        if submit and question and st.session_state.rag_system:
            
            start_time = time.time()

            # Initialize state for the LangGraph run
            initial_state = {
                "question": question,
                "chat_history": st.session_state.chat_history_messages
            }

            st.markdown("### 💡 Final Answer")
            answer_placeholder = st.empty()
            
            final_answer = ""
            
            try:
                # Use synchronous invoke to execute the entire graph. 
                # LangSmith tracing is automatically enabled via environment variables.
                
                with st.spinner(f"Running graph for '{question}'... Check LangSmith for trace."):
                    
                    final_state = st.session_state.rag_system.invoke(
                        initial_state,
                        config=config,
                    )
                
                # Extract the final answer and display it
                if "answer" in final_state and final_state["answer"]:
                    final_answer = final_state["answer"]
                    answer_placeholder.success(final_answer)
                else:
                    final_answer = "The graph finished, but did not return a final answer."
                    answer_placeholder.error(final_answer)

            except Exception as e:
                final_answer = f"An error occurred during graph execution: {e}"
                st.error(final_answer)
                
            # END OF EXECUTION
            elapsed = time.time() - start_time

            # Store history
            if final_answer and not final_answer.startswith("An error occurred"):
                st.session_state.display_history.append({
                    "question": question,
                    "answer": final_answer,
                    "time": elapsed
                })
                # Append to messages history for context in next turn
                st.session_state.chat_history_messages.append(HumanMessage(content=question))
                st.session_state.chat_history_messages.append(AIMessage(content=final_answer))

                st.caption(f"⏱️ Total time: {elapsed:.2f} seconds")
                
            # Rerun Streamlit to show the updated history cleanly (Corrected function name)
            st.rerun()


if __name__ == "__main__":
    main()