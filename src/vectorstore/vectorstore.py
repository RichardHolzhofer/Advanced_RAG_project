import os
import sys
from src.logger.logger import logging
from src.exception.exception import RAGException
from src.config.config import Config
from qdrant_client import QdrantClient, models
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from langchain_core.documents import Document
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams
from langchain_community.vectorstores import InMemoryVectorStore
import uuid


class RAGVectorStore:
    def __init__(self):
        self.llm = Config.get_llm_model()
        self.embeddings = Config.get_embedding_model()
        self.persist_path = "./qdrant_db"
        self.client = QdrantClient(path=self.persist_path)
        self.collection_name = "my_collection"
        self.vectorstore = None
        self.retriever = None
    
    def load_vectorstore(self):
        
        try:
            logging.info("Loading vector store.")
            self.vectorstore = QdrantVectorStore(
                client=self.client,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                vector_name="dense",
                retrieval_mode=RetrievalMode.HYBRID,
                distance=Distance.COSINE,
                sparse_vector_name="sparse",
                sparse_embedding=FastEmbedSparse(model_name="Qdrant/bm25")
            )
            
            logging.info("Vector store has been loaded successfully.")
            return self.vectorstore
            
        except Exception as e:
            raise RAGException(e, sys)
        
    def _add_documents(self, documents):
        try:
            logging.info("Uploading documents to the vector store")
            doc_ids = [doc.id for doc in documents]
            self.vectorstore.add_documents(documents=documents, ids=doc_ids)
            logging.info("Documents have been uploaded successfully to the vector store.")
        except Exception as e:
            raise RAGException(e, sys)
        
    def close_client(self):
        try:
            if self.client:
                self.client.close()
        except Exception as e:
            raise RAGException(e, sys)
        
    def create_vectorstore(self, documents):
        
        try:
            logging.info("Creating vector store")    
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={"dense": VectorParams(size=1536, distance=Distance.COSINE)},
                sparse_vectors_config={"sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))}
            )
            
            sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
            
            
            self.vectorstore = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embeddings,
                sparse_embedding=sparse_embeddings,
                retrieval_mode=RetrievalMode.HYBRID,
                vector_name="dense",
                sparse_vector_name="sparse"
            )
            
            self._add_documents(documents=documents)
            
            logging.info("Vector store has been created successfully.")
            return self.vectorstore
        
        except Exception as e:
            raise RAGException(e,sys)
    
    def create_retriever(self):
        try:
            logging.info("Creating retriever")
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k":7})
            logging.info("Retriever has been created successfully")
            return self.retriever
        except Exception as e:
            raise RAGException(e, sys)
