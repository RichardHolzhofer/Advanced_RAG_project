import os
import sys

from src.logger.logger import logging
from src.exception.exception import RAGException
from src.config.config import Config
from src.state.state import RAGState
from src.node.node import RAGNodes

from langgraph.graph import StateGraph, START, END

class GraphBuilder:
    def __init__(self,retriever, llm):
        self.nodes = RAGNodes(retriever=retriever, llm=llm)
        self.graph = None
        
    def build_graph(self):
        graph_builder = StateGraph(RAGState)
        
        graph_builder.add_node("retriever", self.nodes.retrieve_documents)
        #graph_builder.add_node("reranker", self.nodes.rerank_documents)
        graph_builder.add_node("answer_generator", self.nodes.generate_answer)
        
        
        graph_builder.set_entry_point("retriever")

        
        #graph_builder.add_edge("retriever", "reranker")
        
        #graph_builder.add_edge("reranker", "answer_generator")
        
        graph_builder.add_edge("retriever", "answer_generator")

        
        
        graph_builder.add_edge("answer_generator", END)
        
        self.graph = graph_builder.compile()
        
        return self.graph
    
    def run(self, question:str) -> dict:
        if self.graph is None:
            self.build_graph()
        initialize_state = {
            'question':question,
            'retrieved_docs': [],
            'answer':''
            }
        return self.graph.invoke(initialize_state)