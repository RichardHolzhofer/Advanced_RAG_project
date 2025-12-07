import os
import sys

from src.logger.logger import logging
from src.exception.exception import RAGException

from typing import List, TypedDict
from pydantic import BaseModel
from langchain_core.documents import Document

class RAGState(TypedDict):
    question: str
    retrieved_docs: list[Document] = []
    answer: str = ""
    
    