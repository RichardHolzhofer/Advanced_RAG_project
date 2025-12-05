import os
import sys
from src.logger.logger import logging
from src.exception.exception import RAGException
from src.config.config import Config
from langchain_postgres import PGEngine, PGVectorStore
from langchain_community.retrievers import BM25Retriever


class VectorStore(Config):
    def __init__(self):
        self.llm = Config.get_llm_model()
        self.embeddings = Config.get_embedding_model()
        self.dense_store = None
        self.sparse_store = None
        self.connection_uri = None
        self.engine = None
        self.retriever = None
        
    def _create_conn_uri(self):
        
        try:
            logging.info("Creating connection URI.")
        
            POSTGRES_USER = os.getenv("SUPABASE_USER")
            POSTGRES_PASSWORD = os.getenv("SUPABASE_PASSWORD")
            POSTGRES_HOST = os.getenv("SUPABASE_HOST")
            POSTGRES_PORT = os.getenv("SUPABASE_PORT")
            POSTGRES_DB = os.getenv("SUPABASE_DB")
            
            conn_uri = (
                f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}"
                f":{POSTGRES_PORT}/{POSTGRES_DB}"
            )
            self.connection_uri = conn_uri
            
            logging.info("Connection URI has been created successfully.")
        except Exception as e:
            raise RAGException(e, sys)
        
    def _create_engine(self):
        try:
            logging.info("Creating DB Engine.")
            self.engine = PGEngine.from_connection_string(url=self.connection_uri)
            logging.info("DB Engine has been created successfully.")
        except Exception as e:
            raise RAGException(e, sys)
        
    async def _create_vectorstore(self):
        
        try:
            logging.info("Setting up vector store.")
            self.dense_store = await PGVectorStore.create(
                engine=self.engine,
                embedding_service=self.embeddings,
                table_name="chunks",
                id_column="chunk_id",
                content_column="content",
                embedding_column="embedding",
                metadata_columns=["document_id", "chunk_index", "source_type"],
                metadata_json_column="metadata"
            )
            logging.info("Vector store has been set up successfully.")
        except Exception as e:
            raise RAGException(e, sys)
        
    async def _add_documents(self, documents):
        try:
            logging.info("Uploading documents to the vector store")
            await self.dense_store.aadd_documents(documents=documents)
            logging.info("Documents have been uploaded successfully to the vector store.")
        except Exception as e:
            raise RAGException(e, sys)
        
    def get_dense_retriever(self, documents):
        try:
            logging.info("Setting up dense retriever.")
            self._create_conn_uri()
            self._create_engine()
            self._create_vectorstore()
            self._add_documents(documents=documents)
            self.retriever = self.dense_store.as_retriever()
            logging.info("Dense retriever has been set up successfully.")
        except Exception as e:
            raise RAGException(e, sys)    
        
    def get_sparse_retriever(self, documents):
        try:
            logging.info("Creating BM25 retriever")
            self.sparse_store = BM25Retriever.from_documents(documents=documents)
            logging.info("BM25 retriever is ready to use")
        except Exception as e:
            raise RAGException(e, sys)
        
        
