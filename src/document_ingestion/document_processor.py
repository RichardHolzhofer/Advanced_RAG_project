import os
import sys
import asyncio
from src.logger.logger import logging
from src.exception.exception import RAGException
from docling.document_converter import DocumentConverter
from pathlib import Path
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai import CrawlResult
from crawl4ai.models import CrawlResultContainer
from docling.chunking import HybridChunker
from transformers import AutoTokenizer
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from urllib.parse import urlparse
from langchain_core.documents import Document
from docling.datamodel.document import ConversionResult
from docling_core.types.doc import DoclingDocument
from docling.datamodel.base_models import InputFormat
import uuid


class DocumentProcessor: 
    EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
    MAX_TOKENS = 512
    tokenizer = HuggingFaceTokenizer(tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL_ID), max_tokens=MAX_TOKENS)
    
    def __init__(self):
        self.converter = DocumentConverter()
        self.chunker =HybridChunker(tokenizer=self.tokenizer, merge_peers=True)
        
        
    def _process_single_document(self, doc_obj):
        
        doc_uuid = str(uuid.uuid4())
        
        full_content = None
        document_source = None
        doc_type = "unknown"
        num_pages = 0
        docling_document = None
        
        if isinstance(doc_obj, ConversionResult):
            full_content = doc_obj.document.export_to_markdown()
            document_source = doc_obj.input.file.name
            doc_type = doc_obj.input.format.value
        
            try:
                num_pages = len(doc_obj.pages)
            except Exception:
                num_pages =len(doc_obj.document.export_to_dict().get("pages", {}))
                
            docling_document = doc_obj.document
        
        elif isinstance(doc_obj, CrawlResultContainer):
            
            full_content = doc_obj.markdown.raw_markdown
            document_source = doc_obj.url
            doc_type = "web_article"
            num_pages = 1
            
            docling_document = self.converter.convert_string(content=full_content, format=InputFormat.MD).document
            
        else:
            raise TypeError(f"Not supported document type: {type(doc_obj)}")
        
        full_doc = Document(
            page_content=full_content,
            metadata={
                "document_id": doc_uuid,
                "source": document_source,
                "file_type": doc_type,
                "no_of_pages": num_pages,
                "title": doc_obj.title if hasattr(doc_obj, 'title') else "Untitled"
            }
        )
            
        
        chunks = list(self.chunker.chunk(docling_document))
        final_chunks = []
        
        for chunk_id, chunk_content in enumerate(chunks):
            page_no = None
            try:
                page_no = chunk_content.export_json_dict()["meta"]["doc_items"][0]['prov'][0]["page_no"]
            except (IndexError, KeyError, AttributeError):
                page_no = 1 if doc_type == "web_article" else None
                
            final_chunks.append(
                Document(
                    page_content=chunk_content.text,
                    metadata={
                        "document_id": doc_uuid,
                        "chunk_id": chunk_id,
                        "source" : document_source,
                        "file_type": doc_type,
                        "page_no": page_no
                    }
                )
            )
            
        return full_doc, final_chunks
                
                
    def convert_documents(self, document_paths):
        
        final_documents = []
        final_chunks = []
        
        try:
            converted_docs = self.converter.convert_all(document_paths)
            
            for doc_content in converted_docs:
                
                full_doc, chunks = self._process_single_document(doc_obj=doc_content)
                final_documents.append(full_doc)
                final_chunks.extend(chunks)
                
                    
            with open("./final_documents/final_documents.txt", "w", encoding='utf-8') as f:
                f.write("\n\n".join([str(x) for x in final_documents]))
                
            with open("./final_documents/final_chunks.txt", "w", encoding='utf-8') as f:
                f.write("\n\n".join([str(x) for x in final_chunks]))
                
            return final_documents, final_chunks
                

        except Exception as e:
            raise RAGException(e, sys)
        
        
    async def crawl_websites(self, urls):
        
        final_documents = []
        final_chunks = []
        
        try:
            browser_config = BrowserConfig(headless=True, verbose=False)
            md_generator = DefaultMarkdownGenerator(
                options={
                    "ignore_links": True,
                    "escape_html": False,
                    "body_width": 0
                }
            )
            crawl_config = CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                stream=False,
                markdown_generator=md_generator,
                exclude_external_links=True,
                exclude_internal_links=True,
                exclude_external_images=True,
                exclude_all_images=True)           
            
            async with AsyncWebCrawler(config=browser_config) as crawler:
                    crawl_results = await crawler.arun_many(urls=urls, config=crawl_config)
                    
                    for crawl_cont in crawl_results:
                        
                        full_doc, chunks = self._process_single_document(crawl_cont)
                        final_documents.append(full_doc)
                        final_chunks.extend(chunks)
                        
            with open("./final_documents/web_documents.txt", "w", encoding='utf-8') as f:
                f.write("\n\n".join([str(x) for x in final_documents]))
                
            with open("./final_documents/web_chunks.txt", "w", encoding='utf-8') as f:
                f.write("\n\n".join([str(x) for x in final_chunks]))            
                        
            return final_documents, final_chunks

        except Exception as e:
            raise RAGException(e, sys)
        

        
        
