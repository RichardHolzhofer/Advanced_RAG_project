import os
import sys
import asyncio
from src.logger.logger import logging
from src.exception.exception import RAGException
from src.config.config import Config
from src.document_ingestion.document_loader import DocumentLoader
from src.document_ingestion.document_processor import DocumentProcessor

class IngestionPipeline:
    def __init__(self):
        self.loader = DocumentLoader()
        self.processor = DocumentProcessor()
    
    def load_default_content(self):
        return self.loader.load_base_data()

        
    async def run_pipeline(self):
        file_paths, urls = self.load_default_content()
        
        crawled_pages = await self.processor.crawl_websites(urls=urls)
        
        to_process = file_paths + crawled_pages
        
        all_chunks = await self.processor.convert_documents(to_process)
        
        return all_chunks
        


if __name__ == "__main__":
    pipe = IngestionPipeline()
    asyncio.run(pipe.run_pipeline())