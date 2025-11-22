import os
import sys
from src.logger.logger import logging
from src.exception.exception import RAGException
from pathlib import Path
from urllib.parse import urlparse

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".pptx", ".md", ".csv", ".html", ".xhtml", ".mp3"}

def validate_input(file_or_url):
    
    try:
        logging.info(f"Validating input file or url: {file_or_url}")
        path = Path(file_or_url)
        
        if path.exists():
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise ValueError(f"Unsupported file format: {path.suffix}")
            logging.info("Supported document format has been provided.")
            return "file"
        
        parsed = urlparse(file_or_url)
        if parsed.scheme in ("http", "https") and parsed.netloc != "":
            logging.info("Valid URL has been provided.")
            return "url"
        
        raise FileNotFoundError(f"Unsupported path or invalid URL: {file_or_url}")
    
    except (ValueError, FileNotFoundError) as e:
        raise e
    except Exception as e:
        raise RAGException(e, sys)