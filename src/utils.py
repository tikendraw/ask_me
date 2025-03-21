import requests
from bs4 import BeautifulSoup
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def validate_url(url: str) -> bool:
    """Validate if a URL is properly formatted and accessible"""
    try:
        response = requests.head(url, timeout=5)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Error validating URL {url}: {e}")
        return False

def extract_domain(url: str) -> Optional[str]:
    """Extract domain name from URL"""
    try:
        from urllib.parse import urlparse
        return urlparse(url).netloc
    except Exception as e:
        logger.error(f"Error extracting domain from {url}: {e}")
        return None
