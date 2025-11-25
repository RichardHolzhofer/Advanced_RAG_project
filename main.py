import os
import sys
import asyncio
from src.logger.logger import logging
from src.exception.exception import RAGException
from src.document_ingestion.document_loader import DocumentLoader
from src.document_ingestion.document_processor import DocumentProcessor

class IngestionPipeline:
    def __init__(self):
        self.loader = DocumentLoader()
        self.processor = DocumentProcessor()
    
    def load_default_content(self):
        file_paths, urls_to_crawl = self.loader.load_base_data()
        return file_paths, urls_to_crawl
    
    def process_documents(self, file_paths):
        file_full_docs, file_chunks = self.processor.convert_documents(document_paths=file_paths)
        return file_full_docs, file_chunks
    
    async def process_websites(self, urls_to_crawl):
        web_full_docs, web_chunks =  await self.processor.crawl_websites(urls=urls_to_crawl)
        return web_full_docs, web_chunks
        
        
    async def run_pipeline(self):
        file_paths, urls_to_crawl = self.load_default_content()
        file_full_docs, file_chunks = self.process_documents(file_paths=file_paths)
        web_full_docs, web_chunks = await self.process_websites(urls_to_crawl=urls_to_crawl)
        


if __name__ == "__main__":
    pipe = IngestionPipeline()
    asyncio.run(pipe.run_pipeline())