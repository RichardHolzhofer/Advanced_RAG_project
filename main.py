import os
import sys
import asyncio
from src.logger.logger import logging
from src.exception.exception import RAGException
from src.document_ingestion.document_loader import DocumentLoader
from src.document_ingestion.document_processor import DocumentProcessor

class IngestionPipeline:
    def __init__(self):
        pass
    
    def load_default_content(self):
        loader = DocumentLoader()
        loaded_docs, loaded_urls = loader.load_base_data()
        return loaded_docs, loaded_urls
    
        
        
    async def run_pipeline(self):
        loaded_docs, loaded_urls = self.load_default_content()
        
        processor = DocumentProcessor()
        
        for doc in loaded_docs:
            await processor.convert_to_markdown(file_or_url=doc)
            
        for url in loaded_urls:
            await processor.convert_to_markdown(file_or_url=url)
        








if __name__ == "__main__":
    pipe = IngestionPipeline()
    asyncio.run(pipe.run_pipeline())