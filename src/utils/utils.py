import os
import sys
from src.logger.logger import logging
from src.exception.exception import RAGException
from pathlib import Path
from urllib.parse import urlparse, urljoin
import requests
from xml.etree import ElementTree
import asyncpg
import json

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
    
    
def categorize_site(url: str):
        """
        Categorize a site based on available endpoint.
        """
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



def build_final_url_list(urls):
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

async def insert_documents(document):
    POSTGRES_USER = os.getenv("SUPABASE_USER")
    POSTGRES_PASSWORD = os.getenv("SUPABASE_PASSWORD")
    POSTGRES_HOST = os.getenv("SUPABASE_HOST")
    POSTGRES_PORT = os.getenv("SUPABASE_PORT")
    POSTGRES_DB = os.getenv("SUPABASE_DB")
    
    conn_uri = (
        f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}"
        f":{POSTGRES_PORT}/{POSTGRES_DB}"
    )
    
    metadata = {
        "summary": document["summary"],
        "full_content": document["full_content"]
    }

    query = """
        INSERT INTO documents (
            document_id, source, title, file_type, no_of_pages, metadata
        )
        VALUES ($1, $2, $3, $4, $5, $6)
        ON CONFLICT (document_id) DO NOTHING;
    """
    conn = await asyncpg.connect(conn_uri)
    
    await conn.execute(
        query,
        document["document_id"],
        document["source"],
        document["title"],
        document["file_type"],
        document["no_of_pages"],
        json.dumps(metadata)
    )
    
    await conn.close()
