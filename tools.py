from langchain.tools import tool 
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
import os 
import io
import re
from dotenv import load_dotenv
from rich import print
load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


@tool
def web_search(query : str) -> str:
    """Search the web for recent and reliable information on a topic . Returns Titles , URLs and snippets."""
    results = tavily.search(query=query,max_results=5)

    out = []

    for r in results['results']:
        out.append(
            f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['content'][:300]}\n"
        )
    
    return "\n----\n".join(out)


@tool
def scrape_url(url: str) -> str:
    """Scrape and return clean text content from a URL or search-result block."""
    
    try:
        url_matches = re.findall(r"https?://[^\s\)\"']+", str(url))
        if url_matches:
            url = url_matches[0].rstrip(".,")

        resp = requests.get(
            url,
            timeout=15,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0 Safari/537.36"
                ),
                "Accept": "text/html,application/pdf,text/plain;q=0.9,*/*;q=0.8",
            },
        )
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "").lower()
        looks_like_pdf = "application/pdf" in content_type or url.lower().split("?")[0].endswith(".pdf")

        if looks_like_pdf:
            try:
                from PyPDF2 import PdfReader

                reader = PdfReader(io.BytesIO(resp.content))
                text_parts = []
                for page in reader.pages[:6]:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text.strip())
                pdf_text = "\n\n".join(text_parts).strip()
                if pdf_text:
                    return pdf_text[:3000]
                return "PDF was downloaded, but no extractable text was found."
            except Exception as pdf_error:
                return f"Could not extract text from PDF: {pdf_error}"

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < 120:
            return (
                "Could not extract enough clean text from this page. "
                "The page may be access-restricted, script-rendered, or mostly non-text."
            )
        return text[:3000]
    except Exception as e:
        return f"Could not scrape URL: {str(e)}"
