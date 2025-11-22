import os
import sys
import asyncio
from src.logger.logger import logging
from src.exception.exception import RAGException
from docling.document_converter import DocumentConverter
from pathlib import Path
from crawl4ai import AsyncWebCrawler
from urllib.parse import urlparse
from src.utils.utils import validate_input

class DocumentProcessor: 
    def __init__(self):
        pass

    def convert_document(self, document_path):
        
        try:
            logging.info(f"Converting document {document_path} to markdown.")
            converter = DocumentConverter()
            converted_document = converter.convert(document_path).document.export_to_markdown()
            
            current_dir = Path(__file__).resolve().parent
            output_dir = current_dir / "converted_documents"
            output_dir.mkdir(exist_ok=True)
            
            with open(output_dir / f"{Path(document_path).stem}.md", "w", encoding="utf-8") as f:
                f.write(converted_document)
                
            logging.info("Document has been converted to markdown successfully.")
            return converted_document
        except Exception as e:
            raise RAGException(e, sys)
        
    async def convert_website(self, url):
        try:
            logging.info(f"Converting content from url: {url} to markdown.")
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(url=url)
                
                converted_url = result.markdown
                
                current_dir = Path(__file__).resolve().parent
                output_dir = current_dir / "converted_documents"
                output_dir.mkdir(exist_ok=True)
                
                hostname = urlparse(url).hostname.replace(".","-")
                    
                with open(output_dir / f"{hostname}_content.md", "w", encoding="utf-8") as f:
                    f.write(converted_url)
                
                logging.info("Url content has been successfully converted to markdown.")
        
        except Exception as e:
            raise RAGException(e, sys)
        
        
    async def convert_to_markdown(self, file_or_url):
        try:
            provided_input = validate_input(file_or_url=file_or_url)
            
            logging.info(f"Detected input type: {provided_input}")
            
            if provided_input == "file":
                return self.convert_document(document_path=file_or_url)
            elif provided_input == "url":
                return await self.convert_website(url=file_or_url)

        except Exception as e:
            raise RAGException(e, sys)
        
        
if __name__ == '__main__':
    doc_processor = DocumentProcessor()
    #asyncio.run(doc_processor.convert_website(url="https://crawl4ai.com"))
    #doc_processor.convert_document(document_path="./documents/q4-2024-business-review.pdf")
    asyncio.run(doc_processor.convert_to_markdown(file_or_url="./documents/dog.jpg"))
    