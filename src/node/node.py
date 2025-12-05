import os
import sys

from src.logger.logger import logging
from src.exception.exception import RAGException
from src.state.state import RAGState
from langchain_community.retrievers import BM25Retriever
import textwrap

class RAGNodes:
    
    def __init__(self, dense_retriever, sparse_retriever, llm):
        
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.llm = llm
        
    def retrieve_docs_dense(self, state:RAGState):
        docs = self.dense_retriever.invoke(state.question)
        return RAGState(
            question=state.question,
            retrieved_docs=docs
        )
        
    def retrieve_docs_sparse(self, state:RAGState):
        docs = self.sparse_retriever.invoke(state.question)
        return RAGState(
            question=state.question,
            retrieved_docs=docs
        )
        
    def rerank_documents(self, state:RAGState):
        pass
    
    def generate_answer(self, state:RAGState):
        context = "\n\n".join([doc.page_content for doc in state.retrieved_docs])
        
        prompt = textwrap.dedent(
        f"""
        Answer the question based on the context.
        
        Context:
        {context}
        
        Question:
        {state.question}
        
        """
        )
        
        response = self.llm.invoke(prompt)
        
        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=response.content
        )