import os
import sys
from src.logger.logger import logging
from src.exception.exception import RAGException
from pathlib import Path
from urllib.parse import urlparse, urljoin
import requests
from xml.etree import ElementTree
import json

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".pptx", ".md", ".csv", ".html", ".xhtml", ".mp3"}

def validate_input(file_or_url):
    """
    Validates document formats and URLs.
    """
    
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
    
    
def categorize_site(url: str):
        """
        Categorizes a site based on available endpoint.
        """
        try:
            endpoints = [
                ("llm_text", "llms-full.txt"),
                ("sitemap", "sitemap.xml")
            ]

            for category, endpoint in endpoints:
                target = urljoin(url if url.endswith("/") else url + "/", endpoint)

                try:
                    resp = requests.get(target, timeout=5, allow_redirects=True)
                    if resp.status_code == 200:
                        return category, target
                except requests.RequestException:
                    pass

            return "basic", url
        
        except Exception as e:
            raise RAGException(e, sys)



def build_final_url_list(urls):
    """
    Builds the final url list, which in case of 'sitemap' extracts all the underlying URLs from the website.
    """
    try:
        logging.info("Building final URL list...")

        final_urls = []

        for url in urls:
            category, resolved = categorize_site(url)

            if category == "sitemap":
                try:
                    logging.info(f"Parsing sitemap: {resolved}")
                    resp = requests.get(resolved, timeout=10)

                    root = ElementTree.fromstring(resp.content)
                    ns = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}

                    extracted_urls = [loc.text for loc in root.findall(".//ns:loc", ns)]
                    logging.info(f"Sitemap contains {len(extracted_urls)} URLs")

                    final_urls.extend(extracted_urls)
                    continue

                except Exception as e:
                    logging.error(f"Failed to parse sitemap {resolved}: {e}")
                    final_urls.append(url)
                    continue

            final_urls.append(resolved)

        logging.info(f"Final URL count: {len(final_urls)}")
        return final_urls
    
    except Exception as e:
        raise RAGException(e, sys)

def save_json(path, filename, file):
    """
    Saves objects as JSON to a user provided path.
    """
    try:
        os.makedirs(f"./{path}", exist_ok=True)
        with open (f"./{path}/{filename}.json", "w", encoding='utf-8') as f:
                json.dump(file, f, ensure_ascii=False)
    
    except Exception as e:
        raise RAGException(e, sys)
            
def load_json(path):
    """
    Loads objects as JSON from a user provided path.
    """
    try:
        with open (f"{path}", "r", encoding='utf-8') as f:
                data = json.load(f)
                
        return data
    
    except Exception as e:
        raise RAGException(e, sys)