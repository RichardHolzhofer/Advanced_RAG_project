import os
import sys
import asyncio
import json
from src.logger.logger import logging
from src.exception.exception import RAGException
from src.config.config import Config
from src.document_ingestion.document_loader import DocumentLoader
from src.document_ingestion.document_processor import DocumentProcessor
from src.utils.utils import save_json
from src.vectorstore.vectorstore import VectorStore

class IngestionPipeline:
    def __init__(self):
        self.loader = DocumentLoader()
        self.processor = DocumentProcessor()
        self.vectorstore = VectorStore()
    
    def load_default_content(self):
        return self.loader.load_base_data()
    
    async def crawl_webpages(self, urls):
        crawled_pages = await self.processor.crawl_websites(urls=urls)
        return crawled_pages
    
    async def process_documents(self, documents):
        chunked_docs = await self.processor.process_documents(documents)
        return chunked_docs
    
    def create_vectorstore(self, documents):
        vs = self.vectorstore.create_vectorstore(documents=documents)
        return vs
    
    

    async def run_pipeline(self):
        file_paths, urls = self.load_default_content()
        crawled_pages = await self.crawl_webpages(urls=urls)
        
        to_process = file_paths + crawled_pages
        
        chunked_docs = await self.process_documents(documents=to_process)
        
        data = [doc.to_json() for doc in chunked_docs]
        
        save_json(
            path="./ingested_documents",
            filename="ingested_docs",
            file=data
        )
        vs = self.create_vectorstore(documents=chunked_docs)
        
        self.vectorstore.close_client()
        


if __name__ == "__main__":
    pipe = IngestionPipeline()
    asyncio.run(pipe.run_pipeline())