import sys
from src.logger.logger import logging
from src.exception.exception import RAGException
from src.utils.utils import validate_input, build_final_url_list
from pathlib import Path
import json

class DocumentLoader:
    def __init__(self, docs_path="test_documents", url_path="data/urls.json"):
        logging.info("Initializing DocumentLoader for default file loading.")
        self.docs_path = Path(docs_path)
        self.url_path = Path(url_path)
        
    def load_documents(self):
        """
        Loads documents from a folder specified under 'docs_path' in the initialization method.
        """
        try:
            logging.info(f"Loading files from the following folder: {self.docs_path}")
            return [f for f in self.docs_path.glob("*") if validate_input(f) == 'file']
        except Exception as e:
            raise RAGException(e, sys)

    def load_urls(self):
        """
        Loads urls from a JSON file specified under 'url_path' in the initialization method.
        """
        try:
            with open(self.url_path, 'r') as f:
                urls = json.load(f)
            
            logging.info(f"Loading files from the following JSON file: {self.url_path}")
            
            
            return build_final_url_list(urls=urls)
        except Exception as e:
            raise RAGException(e, sys)
    
    def load_base_data(self):
        """
        Loads documents and urls.
        """
        try:
            return self.load_documents(), self.load_urls()
        except Exception as e:
            raise RAGException(e, sys)
