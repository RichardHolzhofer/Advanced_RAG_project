import os
import sys

from src.logger.logger import logging
from src.exception.exception import RAGException
from src.config.config import Config
from src.vectorstore.vectorstore import RAGVectorStore
from src.state.state import RAGState
from src.node.node import RAGNodes

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


class GraphBuilder:
    def __init__(self,retriever, llm):
        """
        Initializes graph building object.
        """
        
        try:
            logging.info("Initiating graph building.")
            self.nodes = RAGNodes(retriever=retriever,llm=llm)
            self.graph = None
            self.memory = MemorySaver()
            self.MAX_EXPANSION= 1
            
        except Exception as e:
            raise RAGException(e, sys)
        
    def route_query(self, state:RAGState):
        """
        Routes query to the most appropriate execution path ('chat', 'rag' or 'agent').
        """
        try:
            logging.info(f"Query has been routed to {state['route']}")
            return state["route"]
        
        except Exception as e:
            raise RAGException(e, sys)
    
    def doc_grader(self, state:RAGState):
        """
        Checks if relevant document IDs were found and routes to query to either 'external_agent' or 'rag'.
        """
        
        try:
            
            ids= state["relevant_ids"]
            
            if not ids:
                logging.info("No relevant IDs found, query routed to 'external_agent'.")
                return "external_agent"
            logging.info("Relevant IDs have been found, query routed to 'rag'.")
            return "rag"
        
        except Exception as e:
            raise RAGException(e, sys)
    
    def hallucination_checker(self, state:RAGState):
        """
        Based on the grading of the generated answer to the question, it routes the execution path to the relevant node.
        """
        try:
            final_grade= state["final_grade"]
            expansion_counter = state["expansion_counter"]
            frozen_facts = state["frozen_rag_facts"]
            
            if final_grade == "relevant":
                logging.info("Answer is relevant, conversation to be ended.")
                return "end_conversation"
            
            if final_grade == "not_relevant":
                logging.info("Answer is not relevant, routed to 'external_agent'.")
                return "external_agent"
            
            #partially relevant
            if expansion_counter < self.MAX_EXPANSION:                
                logging.info("Answer is partially relevant, routed to query expansion.")
                return "expander"
            
            if frozen_facts:
                logging.info("Answer is only partially relevant after query expansion, routed to 'rag_agent'.")
                return "rag_agent"
            
            logging.info("Fallback to 'external_agent' for answering.")
            return "external_agent"
        
        except Exception as e:
            raise RAGException(e, sys)
        
    def build_graph(self):
        """
        Builds StateGraph using nodes and edges to define execution path.
        """
        try:
            logging.info("Building graph has been started.")
            graph_builder = StateGraph(RAGState)
            
            
            #Nodes
            graph_builder.add_node("reset_node", self.nodes.reset_turn_state)
            graph_builder.add_node("rewriter_node", self.nodes.rewrite_query)
            graph_builder.add_node("router_node", self.nodes.route_query)
            graph_builder.add_node("conversational_node", self.nodes.conversational_answer)
            graph_builder.add_node("retriever_node", self.nodes.retrieve_documents)
            graph_builder.add_node("external_agent_node", self.nodes.invoke_external_agent)
            graph_builder.add_node("rag_agent_node", self.nodes.invoke_rag_agent)
            graph_builder.add_node("grade_documents_node", self.nodes.grade_documents)
            graph_builder.add_node("answer_generator_node", self.nodes.generate_answer)
            graph_builder.add_node("hallucination_grader_node", self.nodes.grade_hallucination)
            graph_builder.add_node("expansion_counter_node", self.nodes.increment_expansion_counter)
            graph_builder.add_node("query_expander_node", self.nodes.expand_query)
    
            
            
            #Edges
            graph_builder.set_entry_point("reset_node")  
            graph_builder.add_edge("reset_node", "rewriter_node")    
            graph_builder.add_edge("rewriter_node", "router_node")
            
            #Router node logic
            graph_builder.add_conditional_edges(
                source="router_node",
                path=self.route_query,
                path_map={
                    "rag": "retriever_node",
                    "chat": "conversational_node",
                    "agent": "external_agent_node"
                }
            )
            
            graph_builder.add_edge("conversational_node", END)
            graph_builder.add_edge("external_agent_node", END)
            
            graph_builder.add_edge("retriever_node", "grade_documents_node")
            
            #Grader node logic
            graph_builder.add_conditional_edges(
                source="grade_documents_node",
                path=self.doc_grader,
                path_map={
                    "rag": "answer_generator_node",
                    "external_agent": "external_agent_node"
                }
            )
            
            graph_builder.add_edge("answer_generator_node", "hallucination_grader_node")
            
            #Hallucination grader node logic
            graph_builder.add_conditional_edges(
                source="hallucination_grader_node",
                path=self.hallucination_checker,
                path_map={
                    "end_conversation": END,
                    "expander": "expansion_counter_node",
                    "external_agent": "external_agent_node",
                    "rag_agent": "rag_agent_node"
                }
                )
            
            graph_builder.add_edge("expansion_counter_node", "query_expander_node")
            graph_builder.add_edge("query_expander_node", "retriever_node")
            graph_builder.add_edge("external_agent_node", END)
            graph_builder.add_edge("rag_agent_node", END)
            
            #Compiling and adding checkpointer for memory
            self.graph = graph_builder.compile(checkpointer=self.memory)
            logging.info("Graph has been built successfully.")
            
            return self.graph
        
        except Exception as e:
            raise RAGException(e, sys)
    

if __name__ == "__main__":
    vector_store = RAGVectorStore()
    vector_store.load_vectorstore()
    graph_builder = GraphBuilder(
        retriever=vector_store.create_retriever(),
        llm=Config.get_llm_model()
        )
    
    graph = graph_builder.build_graph()    

    graph_image = graph.get_graph().draw_mermaid_png()

    os.makedirs("./graph_image", exist_ok=True)
    with open("./graph_image/graph_image.png", 'wb') as f:
        f.write(graph_image)
        