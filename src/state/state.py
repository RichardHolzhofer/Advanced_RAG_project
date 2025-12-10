import os
import sys

from src.logger.logger import logging
from src.exception.exception import RAGException

from typing import List, TypedDict, Annotated, Literal
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
import uuid



class Decision(BaseModel):
    result: Literal["pass", "fail"] = Field(description="Rating of the generated answer's quality against the original question and the retrieved context.")
class Router(BaseModel):
    route: Literal['rag', 'agent'] = Field(description="The next step in the routing process.")

class RAGState(TypedDict):
    question: str
    next_step: Router
    rewritten_query: str
    
    #Retrieval tracking
    retrieved_docs: List[Document]
    retrieved_doc_ids: List[str]
    last_retrieved_doc_ids: List[str]
    all_retrieval_doc_ids: List[str]
    
    #Query expansion tracking
    expansion_counter: int = 0
    expanded_query: List[str]
    previous_queries: List[str]
    
    chat_history: Annotated[List[BaseMessage], add_messages]
    answer: str
    rating: Decision
    
    
