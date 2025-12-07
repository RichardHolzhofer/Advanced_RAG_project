import os
import sys

from src.logger.logger import logging
from src.exception.exception import RAGException
from src.config.config import Config
from src.state.state import RAGState
from src.vectorstore.vectorstore import VectorStore
from langchain_core.documents import Document
from typing import List
import textwrap

class RAGNodes:
    
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        
    def retrieve_documents(self, state:RAGState) -> RAGState:
        docs = self.retriever.invoke(state["question"])
        
        
        return {"retrieved_docs":docs}
                
    def rerank_documents(self, state:RAGState):
        pass
    
    def generate_answer(self, state:RAGState) -> RAGState:
        context = "\n\n".join([doc.page_content for doc in state['retrieved_docs']])
        
        prompt = textwrap.dedent(
        f"""
        Answer the question based on the context.
        
        Context:
        {context}
        
        Question:
        {state['question']}
        
        """
        )
        
        response = self.llm.invoke(prompt)
        
        return {"answer": response.content}