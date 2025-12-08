import os
import sys

from src.logger.logger import logging
from src.exception.exception import RAGException

from typing import List, TypedDict, Annotated, Literal
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class Router(BaseModel):
    route: Literal['rag', 'conversational'] = Field(description="The next step in the routing process.")

class RAGState(TypedDict):
    question: str
    next_step: Router
    rewritten_query: str
    retrieved_docs: List[Document] = []
    chat_history: Annotated[List[BaseMessage], add_messages] = []
    #web_search_results: str
    answer: str = ""
    
