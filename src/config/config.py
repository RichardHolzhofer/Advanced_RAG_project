import os
import sys
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings

from src.logger.logger import logging
from src.exception.exception import RAGException

load_dotenv()

class Config:
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    LLM_MODEL = "openai:gpt-4o-mini"
    EMBEDDING_MODEL = "text-embedding-3-small"
    
    @classmethod
    def get_llm_model(cls):
        """
        Initialize and return the LLM model
        """
        try:
            logging.info(f"Loading LLM model: {cls.LLM_MODEL}")
            os.environ["OPENAI_API_KEY"] = cls.OPENAI_API_KEY
            
            logging.info("LLM has been loaded successfully.")
            return init_chat_model(cls.LLM_MODEL, temperature=0)
        except Exception as e:
            raise RAGException(e, sys)
    
    @classmethod
    def get_embedding_model(cls):
        """
        Initialize and return embedding model
        """
        try:
            logging.info(f"Loading Embedding Model: {cls.EMBEDDING_MODEL}")
            os.environ["OPENAI_API_KEY"] = cls.OPENAI_API_KEY
            
            logging.info("Embedding Model has been loaded successfully.")
            return OpenAIEmbeddings(model=cls.EMBEDDING_MODEL)
        except Exception as e:
            raise RAGException(e, sys)