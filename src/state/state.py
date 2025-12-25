import os
import sys

from src.logger.logger import logging
from src.exception.exception import RAGException

from typing import List, TypedDict, Annotated, Literal, Optional
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain.agents import AgentState
import uuid

class Validator(BaseModel):
    is_valid: Literal["yes", "no"] = Field(description="Checks if the expanded query is relevant to the original question.")    
class Router(BaseModel):
    next_step: Literal["rag", "agent", "chat"] = Field(description="Routes the query to the right execution past.")

class GraderDecision(BaseModel):
    relevant_ids: List[str] = Field(
        description="A list of the exact Document IDs that are relevant and useful for answering the question. If document_state is 'not_relevant', this list MUST be empty: []."
    )
class HallucinationGrade(BaseModel):
    grade: Literal["relevant", "partially_relevant", "not_relevant"] = Field(description="The grading of the answer's factual grounding againt the provided context.")

class ExpandedQueries(BaseModel):
    queries: List[str] = Field(description="List of 3 alternative search queries, each one distinct from the others.")
class RAGState(TypedDict):
    question: str
    chat_history: Annotated[List[BaseMessage], add_messages]
    
    #Context creation
    rewritten_query: str
    #Routing
    route: Router
    
    #Retrieval
    retrieved_docs: List[Document]
    retrieved_doc_ids: List[str]
    
    #Grading
    relevant_ids: Optional[List[str]]
 
    #Answer and context
    answer: Optional[str]
    answer_source: Literal["chat", "rag", "agent_external", "agent_rag"]
    context: Optional[str]
    
    #Hallucination check
    final_grade: Optional[Literal["relevant", "partially_relevant", "not_relevant"]] 
        
    #Query expansion tracking
    expansion_counter: Optional[int] = 0
    expanded_query_list: Optional[List[str]]
    frozen_rag_facts: Optional[str]
    

    
