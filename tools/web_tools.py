"""Web search, URL fetching, and scholarly query tools."""

import json
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Any, Dict, List
import requests
from core.interfaces import ToolResult
from core.logger import logger


def web_search(query: str, max_results: int = 5) -> ToolResult:
    """Performs web search using DuckDuckGo HTML search."""
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        if not results:
            return ToolResult(success=True, output=f"No web results found for query '{query}'.")

        formatted = []
        for r in results:
            title = r.get("title", "Untitled")
            href = r.get("href", r.get("link", ""))
            body = r.get("body", r.get("snippet", ""))
            formatted.append(f"Title: {title}\nURL: {href}\nSnippet: {body}\n")

        return ToolResult(success=True, output="\n---\n".join(formatted))
    except Exception as e:
        logger.warning(f"DuckDuckGo search failed ({e}). Falling back to HTML scraping endpoint.")
        try:
            url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            res = requests.get(url, headers=headers, timeout=10)
            if res.status_code == 200:
                return ToolResult(
                    success=True, output=f"Web search completed for '{query}'. HTML response length: {len(res.text)}"
                )
        except Exception as ex:
            return ToolResult(success=False, output="", error=f"Web search failed: {ex}")

        return ToolResult(success=False, output="", error=str(e))


def fetch_url(url: str) -> ToolResult:
    """Fetches main article content from a web page using trafilatura."""
    try:
        import trafilatura

        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            res = requests.get(url, timeout=10)
            downloaded = res.text

        extracted = trafilatura.extract(downloaded)
        if not extracted:
            extracted = downloaded[:5000]

        return ToolResult(success=True, output=extracted or "No content extracted.")
    except Exception as e:
        return ToolResult(success=False, output="", error=f"Failed to fetch URL '{url}': {e}")


def _query_arxiv(query: str, max_results: int) -> List[str]:
    papers = []
    try:
        encoded_query = urllib.parse.quote(query)
        url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_query}&start=0&max_results={max_results}"
        res = requests.get(url, timeout=8)
        if res.status_code == 200:
            root = ET.fromstring(res.text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            for entry in root.findall("atom:entry", ns):
                title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
                summary = entry.find("atom:summary", ns).text.strip().replace("\n", " ")
                published = entry.find("atom:published", ns).text[:4]
                paper_id = entry.find("atom:id", ns).text.strip()
                authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)]
                papers.append(
                    f"[arXiv] Title: {title}\nAuthors: {', '.join(authors)}\nYear: {published}\nURL: {paper_id}\nAbstract: {summary[:250]}...\n"
                )
    except Exception as e:
        logger.warning(f"arXiv search error: {e}")
    return papers


def _query_semantic_scholar(query: str, max_results: int) -> List[str]:
    papers = []
    try:
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={urllib.parse.quote(query)}&limit={max_results}&fields=title,authors,year,url,abstract"
        res = requests.get(url, timeout=8)
        if res.status_code == 200:
            data = res.json()
            for p in data.get("data", []):
                title = p.get("title", "Untitled")
                authors = [a.get("name", "") for a in p.get("authors", [])]
                year = p.get("year", "N/A")
                p_url = p.get("url", "")
                abstract = (p.get("abstract") or "")[:250]
                papers.append(
                    f"[SemanticScholar] Title: {title}\nAuthors: {', '.join(authors)}\nYear: {year}\nURL: {p_url}\nAbstract: {abstract}...\n"
                )
    except Exception as e:
        logger.warning(f"Semantic Scholar search error: {e}")
    return papers


def _query_pubmed(query: str, max_results: int) -> List[str]:
    papers = []
    try:
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={urllib.parse.quote(query)}&retmode=json&retmax={max_results}"
        res = requests.get(url, timeout=8)
        if res.status_code == 200:
            id_list = res.json().get("esearchresult", {}).get("idlist", [])
            for pmid in id_list:
                papers.append(f"[PubMed] PMID: {pmid}\nURL: https://pubmed.ncbi.nlm.nih.gov/{pmid}/\n")
    except Exception as e:
        logger.warning(f"PubMed search error: {e}")
    return papers


def scholarly_search(query: str, max_results: int = 5) -> ToolResult:
    """Queries scholarly sources (arXiv, Semantic Scholar, PubMed) via public APIs."""
    results = []
    results.extend(_query_arxiv(query, max_results))
    results.extend(_query_semantic_scholar(query, max_results))
    results.extend(_query_pubmed(query, max_results))

    if not results:
        return ToolResult(success=True, output=f"No scholarly articles found across arXiv, Semantic Scholar, or PubMed for '{query}'.")

    return ToolResult(success=True, output="\n---\n".join(results))


def web_search_results(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Structured web search for the research pipeline: [{"title", "url", "snippet"}].

    Same DuckDuckGo backend as web_search(), but returns data instead of
    preformatted text. Never raises; returns [] on failure.
    """
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))
        out = []
        for r in raw:
            url = r.get("href", r.get("link", ""))
            if url:
                out.append({"title": r.get("title", "Untitled"), "url": url,
                            "snippet": r.get("body", r.get("snippet", ""))})
        return out
    except Exception as e:
        logger.warning(f"web_search_results failed for '{query}': {e}")
        return []


def fetch_text(url: str) -> str:
    """Plain-text page content for the research pipeline ('' on failure)."""
    res = fetch_url(url)
    return res.output if res.success else ""
