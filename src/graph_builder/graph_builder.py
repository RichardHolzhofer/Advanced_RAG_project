import os
import sys

from src.logger.logger import logging
from src.exception.exception import RAGException
from src.config.config import Config
from src.state.state import RAGState
from src.node.node import RAGNodes

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

class GraphBuilder:
    def __init__(self,retriever, llm):
        self.nodes = RAGNodes(retriever=retriever,llm=llm)
        self.graph = None
        self.memory = MemorySaver()
        
    def route_next_step(self, state: RAGState):
        router_obj = state["next_step"]
        
        return router_obj.route
        
    def build_graph(self):
        graph_builder = StateGraph(RAGState)
        
        graph_builder.add_node("rewriter_node", self.nodes.rewrite_query)
        graph_builder.add_node("router_node", self.nodes.route_selector)
        #graph_builder.add_node("web_search_node", self.nodes.web_search)
        #graph_builder.add_node("query_expander_node", self.nodes.expand_query)
        graph_builder.add_node("conversational_node", self.nodes.conversational_query)
        
        graph_builder.add_node("retriever_node", self.nodes.retrieve_documents)
        #graph_builder.add_node("reranker", self.nodes.rerank_documents)
        graph_builder.add_node("answer_generator_node", self.nodes.generate_answer)
        
        
        graph_builder.set_entry_point("rewriter_node")
        #graph_builder.add_edge("retriever", "reranker")
        
        graph_builder.add_edge("rewriter_node", "router_node")
        
        graph_builder.add_conditional_edges(
            source="router_node",
            path=self.route_next_step,
            path_map={
                "conversational": "conversational_node",
                #"web": "web_search_node",
                #"expand": "query_expander_node",
                "rag": "retriever_node"
            }
            )
        
        #graph_builder.add_edge("web_search_node", "answer_generator_node")
        #graph_builder.add_edge("query_expander_node", "answer_generator_node")
        graph_builder.add_edge("conversational_node", "answer_generator_node")
        graph_builder.add_edge("retriever_node", "answer_generator_node")

        graph_builder.add_edge("answer_generator_node", END)
        
        self.graph = graph_builder.compile(checkpointer=self.memory)
        
        return self.graph
    