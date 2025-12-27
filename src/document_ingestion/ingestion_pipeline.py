import sys
import asyncio
from src.logger.logger import logging
from src.exception.exception import RAGException
from src.document_ingestion.document_loader import DocumentLoader
from src.document_ingestion.document_processor import DocumentProcessor
from src.utils.utils import save_json
from src.vectorstore.vectorstore import RAGVectorStore

class IngestionPipeline:
    def __init__(self):
        """
        Initializes ingestion pipeline.
        """
        try:
            logging.info("Initializing ingestion pipeline.")
            #Loading documents from a given folder and urls from a json file
            self.loader = DocumentLoader(docs_path="test_documents", url_path="data/urls.json") #Change these or their contents to ingest new documents
            self.processor = DocumentProcessor()
            self.vectorstore = RAGVectorStore()
            
        except Exception as e:
            raise RAGException(e, sys)
    
    def load_default_content(self):
        """
        Loads local files and specified urls.
        """
        try:
            logging.info("Loading base data.")
            
            return self.loader.load_base_data()
        except Exception as e:
            raise RAGException(e, sys)
    
    async def crawl_webpages(self, urls):
        """
        Crawls urls and returns their content as markdown.
        """
        try:
            logging.info("Crawling and converting webpages to markdown.")
            
            crawled_pages = await self.processor.crawl_websites(urls=urls)
            return crawled_pages
        except Exception as e:
            raise RAGException(e, sys)
    
    async def process_documents(self, documents):
        """
        Chunks documents and supplements them and their chunks with metadata for traceability.
        """
        try:
            logging.info("Document processing has been started.")
            
            document_infos, document_chunks = await self.processor.process_documents(documents)
            return document_infos, document_chunks
        except Exception as e:
            raise RAGException(e, sys)
    
    def create_vectorstore(self, documents):
        """
        Creating vector store and uploading provided documents to it.
        """
        try:
            logging.info("Creating vector store from provided documents.")
            
            vs = self.vectorstore.create_vectorstore(documents=documents)
            return vs
        except Exception as e:
            raise RAGException(e, sys)
    
    async def run_pipeline(self):
        """
        Runs the whole document processing pipeline.
        """
        try:
            file_paths, urls = self.load_default_content()
            crawled_pages = await self.crawl_webpages(urls=urls)
            
            to_process = file_paths + crawled_pages
            
            document_infos, document_chunks = await self.process_documents(documents=to_process)
            
            logging.info("Saving documents with metadata as JSON.")
            #Saving documents as JSON.
            save_json(
                path="./ingested_documents",
                filename="ingested_docs",
                file=document_infos
            )
            
            #Converting each chunk to JSON object
            chunks_data = [chunk.to_json() for chunk in document_chunks]
            
            logging.info("Saving chunks with metadata as JSON.")
            #Saving chunks as JSON
            save_json(
                path="./ingested_documents",
                filename="ingested_chunks",
                file=chunks_data
            )
            
            #Creating vector store
            self.create_vectorstore(documents=document_chunks)
            
            #Closing connection to vector db.
            self.vectorstore.close_client()
            
            logging.info("Document ingestion pipeline has run successfully.")
        
        except Exception as e:
            raise RAGException(e, sys)
        


if __name__ == "__main__":
    pipe = IngestionPipeline()
    asyncio.run(pipe.run_pipeline())