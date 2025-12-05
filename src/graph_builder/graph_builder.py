import os
import sys

from src.logger.logger import logging
from src.exception.exception import RAGException
from src.state.state import RAGState
from src.node.node import RAGNodes

from langgraph.graph import StateGraph, START, END

class GraphBuilder:
    def __init__(self, dense_retriever, sparse_retriever, llm):
        self.nodes = RAGNodes(dense_retriever=dense_retriever, sparse_retriever=sparse_retriever, llm=llm)
        self.graph = None
        
    def build_graph(self):
        graph_builder = StateGraph(RAGState)
        
        graph_builder.add_node("sparse_retriever", self.nodes.sparse_retriever)
        graph_builder.add_node("dense_retriever", self.nodes.dense_retriever)
        #graph_builder.add_node("reranker", self.nodes.rerank_documents)
        graph_builder.add_node("answer_generator", self.nodes.generate_answer)
        
        
        graph_builder.add_edge(START, "sparse_retriever")
        graph_builder.add_edge(START, "dense_retriever")
        
        #graph_builder.add_edge("sparse_retriever", "reranker")
        #graph_builder.add_edge("dense_retriever", "reranker")
        
        #graph_builder.add_edge("reranker", "answer_generator")
        
        graph_builder.add_edge("sparse_retriever", "answer_generator")
        graph_builder.add_edge("dense_retriever", "answer_generator")
        
        
        graph_builder.add_edge("answer_generator", END)
        
        self.graph = graph_builder.compile()
        
        return self.graph