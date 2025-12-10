import os
import sys
from src.logger.logger import logging
from src.exception.exception import RAGException
from src.config.config import Config
from src.utils.utils import load_json
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
import uuid


def create_master_summary(document_path):
    llm = Config.get_llm_model()
    
    doc_info = load_json(document_path)
    summaries = "\n".join([doc["summary"] for doc in doc_info])
    
    prompt = PromptTemplate(template=
                            """
                            You are an expert assistant. You have the following summaries of all documents
                            in the knowledge base:

                            {all_summaries}

                            Produce a concise 3-5 sentence overview of what kind of information is contained
                            in the knowledge base. This will be used by a router to decide if a query
                            can be answered by RAG.
                            """
                            ,
                            input_variables=["all_summaries"]
                            )
        
    
    
    formatted_prompt = prompt.format(all_summaries=summaries)
    
    
    return llm.invoke(formatted_prompt).content
    
    
    