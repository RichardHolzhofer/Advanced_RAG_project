import sys
import asyncio
import uuid
import textwrap
from pathlib import Path

from src.logger.logger import logging
from src.exception.exception import RAGException
from src.config.config import Config

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.models import CrawlResultContainer
from docling.chunking import HybridChunker
from transformers import AutoTokenizer
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate


class DocumentProcessor(Config): 
    EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
    MAX_TOKENS = 512
    def __init__(self):
        self.llm = Config.get_llm_model()
        self.converter = DocumentConverter()
        self.tokenizer = HuggingFaceTokenizer(tokenizer=AutoTokenizer.from_pretrained(self.EMBED_MODEL_ID), max_tokens=self.MAX_TOKENS)
        self.chunker =HybridChunker(tokenizer=self.tokenizer, merge_peers=True)
        
    def _get_token_count(self, text):
        """
        Returns the number of tokens in a given text.
        """
        try:
            logging.info(f"Calculating token number for {text[:30]}...")
            return len(self.tokenizer.tokenizer.encode(text))
        except Exception as e:
            raise RAGException(e, sys)
    
    def _get_provenance(self, chunk):
        """
        Returns the provenance of a chunk.
        """
        try:
            logging.info(f"Creating provenance information for chunk: {chunk[:30]}...")
            return self.chunker.contextualize(chunk)
        except Exception:
            return "Unknown"
        
    def _get_summary(self, text):
        """
        Returns a short summary of a provided text.
        """
        try:
            logging.info(f"Creating a short summary of text: {text[:30]}...")
            summary = self.llm.invoke([
                ("system", "Summarize the provided text in maximum 3-5 sentences."),
                ("user", text)
                ]
                )
            return summary.content
        except Exception as e:
            raise RAGException(e, sys)
    
    def _create_context(self, full_doc, chunk):
        """
        Supplement a chunk with its context using it's origin for allowing contextual embeddings.
        """
        try:
        
            prompt = textwrap.dedent(
                """
                You are an expert document analysis and text enrichment engine. Your primary goal is to generate a concise,
                factual context prefix for a specific TEXT CHUNK, drawing upon the FULL DOCUMENT to establish its source and relevance.

                Your final output MUST be the full, enriched text: **[CONTEXT PREFIX] + [Original TEXT CHUNK]**.

                ### INSTRUCTIONS:

                1.  **Analyze the Full Document:** Review the document's title, headers, and overall content to identify the source, date, key entities, and main topic of the TEXT CHUNK.
                2.  **Generate the Context Prefix:** Create a single, highly descriptive prefix sentence that explains *where* the TEXT CHUNK comes from within the FULL DOCUMENT.
                3.  **Combine and Output:** Concatenate the generated context prefix and the original TEXT CHUNK.

                ### CONSTRAINTS:

                * **Output Format:** The final output MUST be *only* the single, combined string of the context prefix and the original text. Do not include any extra commentary, headers, or quotes.
                * **Context Length:** The context prefix MUST be highly concise, with a maximum length of **100 tokens**.
                * **Factual Basis:** The context prefix must only contain information found in the FULL DOCUMENT.
                * **Separation:** Use a clear separator (e.g., a period followed by a space) between the context prefix and the original TEXT CHUNK.

                ---

                ### INPUTS:

                **FULL DOCUMENT:**
                ---
                {full_doc}
                ---

                **TEXT CHUNK:**
                ---
                {chunk}
                ---

                ### EXAMPLE (DO NOT output this):

                **If FULL DOCUMENT is:** A Q4 2023 financial report for ACME Corp, Section 2 is 'Revenue Breakdown'.
                **If TEXT CHUNK is:** "...cloud services revenue grew by 18% year-over-year..."

                **EXPECTED OUTPUT:**
                This text is from the Q4 2023 financial report for ACME Corporation, detailing the Revenue Breakdown section. ...cloud services revenue grew by 18% year-over-year...

                ---

                ### FINAL ENRICHED TEXT:
                            
                """
            )
            
            logging.info("Contextualizing chunk for contextual embedding.")
            llm_chain = PromptTemplate.from_template(prompt) | self.llm
            return llm_chain.invoke({"full_doc": full_doc, "chunk": chunk}).content
        
        except Exception as e:
            raise RAGException(e, sys)
        
        
    async def _process_single_document(self, input_obj):
        """
        It chunks and processes markdown files supplementing them with additional metadata, preserving the original documents for traceability.
        Returns a list for documents and chunks as langchain Document objects.
        """
        
        try:
        
            doc_uuid = str(uuid.uuid4())
            
            
            if isinstance(input_obj, Path):
                local_doc = self.converter.convert(input_obj)
                
                full_content = local_doc.document.export_to_markdown()
                document_source = local_doc.input.file.name
                doc_type = local_doc.input.format.value
            
                try:
                    num_pages = len(local_doc.pages)
                except Exception:
                    num_pages = 1
                    
                docling_document = local_doc.document
            
            elif isinstance(input_obj, CrawlResultContainer):
                
                full_content = input_obj.markdown.raw_markdown
                document_source = input_obj.url
                doc_type = "web_article"
                num_pages = 1
                
                docling_document = self.converter.convert_string(content=full_content, format=InputFormat.MD).document
                
            else:
                raise TypeError(f"Not supported document type: {type(input_obj)}")
            
            logging.info("Creating metadata for document.")
            
            full_doc = {
                "document_id": doc_uuid,
                "source": document_source,
                "file_type": doc_type,
                "title": getattr(input_obj, "title", "Untitled"),
                "summary": self._get_summary(full_content),
                "full_content":full_content,
                "no_of_pages": num_pages
            }
            
            chunks = list(self.chunker.chunk(docling_document))
            logging.info("Chunking of the document has been completed successfully.")
            
            final_chunks = []
            
            for chunk_id, chunk_content in enumerate(chunks):
                
                chunk_uuid = str(uuid.uuid4())
                provenance = self._get_provenance(chunk=chunk_content)
                token_count = self._get_token_count(text=chunk_content.text)
                
                try:
                    page_no = chunk_content.export_json_dict()["meta"]["doc_items"][0]['prov'][0]["page_no"]
                except (IndexError, KeyError, AttributeError):
                    page_no = 1 if doc_type == "web_article" else None
                    
                
                logging.info("Converting chunk to langchain Document and supplementing it with metadata.")    
                final_chunks.append(
                    Document(
                        id=chunk_uuid,
                        page_content=self._create_context(full_doc=full_doc["full_content"], chunk=chunk_content.text),
                        metadata={
                            "document_id": doc_uuid,
                            "chunk_index": chunk_id,
                            "original_chunk": chunk_content.text,
                            "source" : document_source,
                            "source_type": doc_type,
                            "file_type": doc_type,
                            "page_no": page_no,
                            "token_count": token_count,
                            "provenance": provenance
                        }
                    )
                )
                
            return full_doc, final_chunks
        
        except Exception as e:
            raise RAGException(e, sys)
                

    async def crawl_websites(self, urls):
        """
        Crawls all websites asynchronously provided under 'urls' and returns their content as markdowns.
        """
        
        try:
            logging.info("Crawling process for urls has been started.")
            
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
                    
            logging.info("Websites has been crawled and converted to markdown successfully.")
                                  
            return crawl_results

        except Exception as e:
            raise RAGException(e, sys)
        

    async def process_documents(self, inputs):
        """
        Runs the end-to-end document processing pipeline.
        """
           
        try:
            doc_info_list = []
            all_chunks = []
            
            tasks = [self._process_single_document(doc) for doc in inputs]
            results = await asyncio.gather(*tasks)

            for doc_info, chunks in results:
                doc_info_list.append(doc_info)
                all_chunks.extend(chunks)
            
            return doc_info_list, all_chunks

        except Exception as e:
            raise RAGException(e, sys)    
        
