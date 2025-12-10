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
        self.MAX_EXPANSION = 2
        
    def route_next_step(self, state: RAGState):
        router_obj = state["next_step"]
        return router_obj.route
    
    def route_after_grade(self, state:RAGState):
        return state["rating"].result
    
    def is_docs_found(self, state:RAGState):
        if state["retrieved_docs"] == []:
            return "invoke agent"
        return "generate answer"
    
    def track_query_expansion(self, state:RAGState):
        if state["expansion_counter"] >= self.MAX_EXPANSION:
            return "invoke agent"
        
        last_docs = set(state.get("last_retrieved_doc_ids", []))
        curr_docs = set(state.get("retrieved_doc_ids", []))
        all_docs = set(state.get("all_retrieval_doc_ids", []))
        
        
        if curr_docs.issubset(last_docs) or curr_docs.issubset(all_docs):
            return "invoke agent"
        
        return "retrieve documents"
        
    def build_graph(self):
        graph_builder = StateGraph(RAGState)
        
        graph_builder.add_node("rewriter_node", self.nodes.rewrite_query)
        graph_builder.add_node("router_node", self.nodes.route_selector)
        graph_builder.add_node("agent_node", self.nodes.invoke_agent)
        
        #graph_builder.add_node("web_search_node", self.nodes.web_search)
        graph_builder.add_node("query_expander_node", self.nodes.expand_query)
        
        
        graph_builder.add_node("retriever_node", self.nodes.retrieve_documents)
        #graph_builder.add_node("reranker", self.nodes.rerank_documents)
        graph_builder.add_node("answer_generator_node", self.nodes.generate_answer)
        graph_builder.add_node("rate_answer_node", self.nodes.rate_answer)
        
        
        graph_builder.set_entry_point("rewriter_node")
        #graph_builder.add_edge("retriever", "reranker")
        
        graph_builder.add_edge("rewriter_node", "router_node")
        
        graph_builder.add_conditional_edges(
            source="router_node",
            path=self.route_next_step,
            path_map={
                "agent": "agent_node",
                #"expand": "query_expander_node",
                "rag": "retriever_node"
            }
            )
        
        graph_builder.add_conditional_edges(
            source="retriever_node",
            path=self.is_docs_found,
            path_map={
                "invoke agent": "agent_node",
                "generate answer": "answer_generator_node"
            }
            
        )
        
        graph_builder.add_edge("answer_generator_node", "rate_answer_node")
        
        graph_builder.add_conditional_edges(
            source="rate_answer_node",
            path=self.route_after_grade,
            path_map={
                "pass": END,
                "fail": "query_expander_node"
            }
        )
        
        #graph_builder.add_edge("web_search_node", "answer_generator_node")
        graph_builder.add_edge("query_expander_node", "retriever_node")
        
        graph_builder.add_conditional_edges(
            source="query_expander_node",
            path=self.track_query_expansion,
            path_map={
                "retrieve documents": "retriever_node",
                "invoke agent": "agent_node"
            }
            )

        graph_builder.add_edge("agent_node", END)
        
        self.graph = graph_builder.compile(checkpointer=self.memory)
        
        return self.graph
    