from langchain_core.tools import tool
from duckduckgo_search import DDGS
import arxiv
import wikipedia


@tool
def duckduckgo_search(query: str) -> str:
    """Search the web using DuckDuckGo for current events and general questions."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=3))
        if not results:
            return "No results found."
        output = ""
        for r in results:
            output += f"Title: {r['title']}\n"
            output += f"URL: {r['href']}\n"
            output += f"Summary: {r['body']}\n\n"
        return output


@tool
def arxiv_search(query: str) -> str:
    """Search Arxiv for academic papers on scientific and research topics."""
    search = arxiv.Search(query=query, max_results=3)
    results = list(search.results())
    if not results:
        return "No papers found."
    output = ""
    for r in results:
        output += f"Title: {r.title}\n"
        output += f"Authors: {', '.join(str(a) for a in r.authors[:3])}\n"
        output += f"Summary: {r.summary[:300]}\n\n"
    return output


@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for factual and encyclopedic information."""
    try:
        page = wikipedia.page(query)
        return page.content[:2000]
    except wikipedia.DisambiguationError as e:
        return f"Disambiguation: {e.options[:5]}"
    except wikipedia.PageError:
        return "No Wikipedia page found."


def get_tools() -> list:
    """Return all research tools available to the agent."""
    return [duckduckgo_search, arxiv_search, wikipedia_search]
