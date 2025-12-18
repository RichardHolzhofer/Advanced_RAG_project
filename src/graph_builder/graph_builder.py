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
        self.MAX_EXPANSION= 1
        
    
    def route_query(self, state:RAGState):
        return state["route"]
    
    def doc_grader(self, state:RAGState):
        ids= state["relevant_ids"]
        
        if not ids:
            return "agent"
        return "rag"
    
    def hallucination_checker(self, state:RAGState):
        final_grade= state["final_grade"]
        expansion_counter = state["expansion_counter"]
        
        if final_grade == "relevant":
            return "end_conversation"
        if final_grade == "not_relevant":
            return "agent"
        if final_grade == "partially_relevant" and expansion_counter < self.MAX_EXPANSION:
            return "expander"
        return "agent"
            

        
    def build_graph(self):
        graph_builder = StateGraph(RAGState)
        
        
        #Nodes
        graph_builder.add_node("rewriter_node", self.nodes.rewrite_query)
        graph_builder.add_node("router_node", self.nodes.route_query)
        graph_builder.add_node("conversational_node", self.nodes.conversational_answer)
        graph_builder.add_node("retriever_node", self.nodes.retrieve_documents)
        graph_builder.add_node("agent_node", self.nodes.invoke_agent)
        graph_builder.add_node("grade_documents_node", self.nodes.grade_documents)
        graph_builder.add_node("answer_generator_node", self.nodes.generate_answer)
        graph_builder.add_node("hallucination_grader_node", self.nodes.grade_hallucination)
        graph_builder.add_node("query_expander_node", self.nodes.expand_query)
   
        
        
        #Edges
        graph_builder.set_entry_point("rewriter_node")      
        graph_builder.add_edge("rewriter_node", "router_node")
        
        
        graph_builder.add_conditional_edges(
            source="router_node",
            path=self.route_query,
            path_map={
                "rag": "retriever_node",
                "chat": "conversational_node",
                "agent": "agent_node"
            }
        )
        
        graph_builder.add_edge("conversational_node", END)
        graph_builder.add_edge("agent_node", END)
        
        
        
        graph_builder.add_edge("retriever_node", "grade_documents_node")
        
        graph_builder.add_conditional_edges(
            source="grade_documents_node",
            path=self.doc_grader,
            path_map={
                "rag": "answer_generator_node",
                "agent": "agent_node"
            }
        )
        
        graph_builder.add_edge("answer_generator_node", "hallucination_grader_node")
        
        graph_builder.add_conditional_edges(
            source="hallucination_grader_node",
            path=self.hallucination_checker,
            path_map={
                "end_conversation": END,
                "expander": "query_expander_node",
                "agent": "agent_node"
            }
            )

        graph_builder.add_edge("query_expander_node", "retriever_node")
        
        self.graph = graph_builder.compile(checkpointer=self.memory)
        
        return self.graph
    