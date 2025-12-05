import os
import sys

from src.logger.logger import logging
from src.exception.exception import RAGException

from typing import List
from pydantic import BaseModel
from langchain_core.documents import Document

class RAGState(BaseModel):
    question: str
    retrieved_docs: List[Document] = []
    answer: str = ""
    
    