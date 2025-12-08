import os
import sys

from src.logger.logger import logging
from src.exception.exception import RAGException

from typing import List, TypedDict, Annotated
from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class RAGState(TypedDict):
    question: str
    rewritten_query: str
    retrieved_docs: List[Document] = []
    chat_history: Annotated[List[BaseMessage], add_messages] = []
    answer: str = ""
    
    