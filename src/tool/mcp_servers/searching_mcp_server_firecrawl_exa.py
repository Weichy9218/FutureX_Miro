# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0
#
# MCP Server using Firecrawl (Web Scraping) + Exa (AI Search)
# Complete replacement for Serper + Jina solution

import os
import json
import requests
import datetime
import calendar
from typing import Optional, Dict, Any, List
from fastmcp import FastMCP
import asyncio
from src.logging.logger import setup_mcp_logging

# Import dateutil for flexible date parsing
try:
    from dateutil import parser as dateutil_parser
    DATEUTIL_AVAILABLE = True
except ImportError:
    DATEUTIL_AVAILABLE = False


# API Keys from environment
FIRECRAWL_API_KEY = os.environ.get("FIRECRAWL_API_KEY", "")
EXA_API_KEY = os.environ.get("EXA_API_KEY", "")

# Initialize FastMCP server
setup_mcp_logging(tool_name=os.path.basename(__file__))
mcp = FastMCP("searching-mcp-server-firecrawl-exa")


@mcp.tool()
async def exa_search(
    q: str,
    num_results: int = 5,  # 🚀 优化：从10降低到5
    search_type: str = "auto",
    use_autoprompt: bool = True,
    start_published_date: Optional[str] = None,
    end_published_date: Optional[str] = None,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
) -> str:
    """Perform intelligent web searches using Exa AI and retrieve rich results.
    Exa is an AI-native search engine that understands semantic meaning and context, not just keywords.
    This tool can retrieve search results with AI-powered relevance ranking, content snippets, and metadata.

    Args:
        q: Search query string (supports natural language queries).
        num_results: Number of results to return (default: 10, max: 100).
        search_type: Type of search to perform:
            - "auto": Automatically choose the best search method (recommended)
            - "neural": AI semantic search that understands intent and meaning
            - "keyword": Traditional keyword-based search (similar to Google)
        use_autoprompt: Whether to use AI to optimize the query for better results (recommended: True).
        start_published_date: Filter results published after this date (format: YYYY-MM-DD).
        end_published_date: Filter results published before this date (format: YYYY-MM-DD).
        include_domains: Only search within these domains (e.g., ["nytimes.com", "bbc.com"]).
        exclude_domains: Exclude these domains from search results.

    Returns:
        JSON string containing search results with titles, URLs, snippets, highlights, and metadata.
    """
    if not EXA_API_KEY:
        return "[ERROR]: EXA_API_KEY is not set, exa_search tool is not available."
    
    try:
        # Exa API endpoint
        url = "https://api.exa.ai/search"
        
        # Build request payload
        payload: Dict[str, Any] = {
            "query": q,
            "num_results": min(num_results, 8),  # Exa limit
            "type": search_type,
            "use_autoprompt": use_autoprompt,
            "contents": {
                "text": True,  # Return text content
                "highlights": True,  # Return highlighted snippets
            }
        }
        
        # Add optional parameters
        if start_published_date:
            payload["start_published_date"] = start_published_date
        if end_published_date:
            payload["end_published_date"] = end_published_date
        if include_domains:
            payload["include_domains"] = include_domains
        if exclude_domains:
            payload["exclude_domains"] = exclude_domains
        
        # Send request
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "x-api-key": EXA_API_KEY
        }
        
        response = await asyncio.to_thread(
            lambda: requests.post(url, json=payload, headers=headers, timeout=30)
        )
        response.raise_for_status()
        
        data = response.json()
        
        # Format results
        if "results" not in data or not data["results"]:
            return json.dumps({
                "message": f"No results found for query: {q}",
                "query": q,
                "results": []
            }, ensure_ascii=False, indent=2)
        
        # Structure results
        formatted_results = {
            "query": q,
            "autoprompt_string": data.get("autoprompt_string", q),
            "num_results": len(data["results"]),
            "results": []
        }
        
        for idx, result in enumerate(data["results"], 1):
            formatted_result = {
                "position": idx,
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "published_date": result.get("published_date", ""),
                "author": result.get("author", ""),
                "score": result.get("score", 0),
                "text": result.get("text", "")[:300],  # 🚀 优化：从500降低到300
                "highlights": result.get("highlights", [])[:2]  # 🚀 优化：从3降低到2
            }
            formatted_results["results"].append(formatted_result)
        
        return json.dumps(formatted_results, ensure_ascii=False, indent=2)
        
    except requests.exceptions.RequestException as e:
        return f"[ERROR]: Failed to connect to Exa API: {str(e)}"
    except Exception as e:
        return f"[ERROR]: Unexpected error in exa_search: {str(e)}"


@mcp.tool()
async def exa_find_similar(
    url: str,
    num_results: int = 5,  # 🚀 优化：从10降低到5
    exclude_source_domain: bool = True,
) -> str:
    """Find content similar to a given URL using Exa's AI-powered similarity search.
    This tool analyzes the content and context of the provided URL to find semantically similar web pages.

    Args:
        url: The URL to find similar content for.
        num_results: Number of similar results to return (default: 10, max: 100).
        exclude_source_domain: Whether to exclude results from the same domain as the source URL (default: True).

    Returns:
        JSON string containing similar content with titles, URLs, snippets, and similarity scores.
    """
    if not EXA_API_KEY:
        return "[ERROR]: EXA_API_KEY is not set, exa_find_similar tool is not available."
    
    try:
        api_url = "https://api.exa.ai/findSimilar"
        
        payload = {
            "url": url,
            "num_results": min(num_results, 8),
            "exclude_source_domain": exclude_source_domain,
            "contents": {
                "text": True,
                "highlights": True,
            }
        }
        
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "x-api-key": EXA_API_KEY
        }
        
        response = await asyncio.to_thread(
            lambda: requests.post(api_url, json=payload, headers=headers, timeout=30)
        )
        response.raise_for_status()
        
        data = response.json()
        
        # Format results
        formatted_results = {
            "source_url": url,
            "num_results": len(data.get("results", [])),
            "results": []
        }
        
        for idx, result in enumerate(data.get("results", []), 1):
            formatted_result = {
                "position": idx,
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "published_date": result.get("published_date", ""),
                "score": result.get("score", 0),
                "text": result.get("text", "")[:500],
                "highlights": result.get("highlights", [])[:3]
            }
            formatted_results["results"].append(formatted_result)
        
        return json.dumps(formatted_results, ensure_ascii=False, indent=2)
        
    except requests.exceptions.RequestException as e:
        return f"[ERROR]: Failed to connect to Exa API: {str(e)}"
    except Exception as e:
        return f"[ERROR]: Unexpected error in exa_find_similar: {str(e)}"


# @mcp.tool()
# async def scrape_website(url: str) -> str:
#     """Scrape a website and extract its content in LLM-friendly format using Firecrawl.
#     Firecrawl converts web pages to clean Markdown format, removing ads, navigation, and other noise.
#     This tool can also be used to get YouTube video non-visual information (e.g., titles, descriptions, subtitles, key moments),
#     though the information may be incomplete. Search engines are not supported by this tool.

#     Args:
#         url: The URL of the website to scrape.

#     Returns:
#         The scraped website content in Markdown format, including title, description, and main content.
#     """
#     if not FIRECRAWL_API_KEY:
#         return "[ERROR]: FIRECRAWL_API_KEY is not set, scrape_website tool is not available."
    
#     # Handle empty URL
#     if not url or not url.strip():
#         return "[ERROR]: Invalid URL: URL cannot be empty."
    
#     # Auto-add https:// if no protocol is specified
#     protocol_hint = ""
#     if not url.startswith(("http://", "https://")):
#         original_url = url
#         url = f"https://{url}"
#         protocol_hint = f"[NOTE]: Automatically added 'https://' to URL '{original_url}' -> '{url}'\n\n"
    
#     # Check for restricted domains
#     if "huggingface.co/datasets" in url or "huggingface.co/spaces" in url:
#         return "[ERROR]: You are trying to scrape a Hugging Face dataset for answers. Please do not use the scrape tool for this purpose."
    
#     # YouTube hint
#     youtube_hint = ""
#     if "youtube.com/watch" in url or "youtube.com/shorts" in url or "youtube.com/live" in url:
#         youtube_hint = "[NOTE]: If you need to get information about visual or audio content, please use tool 'visual_audio_youtube_analyzing' instead. This tool may not provide visual and audio content of a YouTube Video.\n\n"
    
#     try:
#         # Firecrawl API endpoint
#         api_url = "https://api.firecrawl.dev/v1/scrape"
        
#         payload = {
#             "url": url,
#             "formats": ["markdown"],
#         }
        
#         headers = {
#             "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
#             "Content-Type": "application/json"
#         }
        
#         response = await asyncio.to_thread(
#             lambda: requests.post(api_url, json=payload, headers=headers, timeout=120)
#         )
#         response.raise_for_status()
        
#         data = response.json()
        
#         # Check if successful
#         if not data.get("success"):
#             error_msg = data.get("error", "Unknown error")
#             return f"[ERROR]: Firecrawl scraping failed: {error_msg}"
        
#         # Extract content
#         result_data = data.get("data", {})
        
#         # Build return result
#         result_parts = [
#             f"URL: {url}",
#             f"Title: {result_data.get('metadata', {}).get('title', 'N/A')}",
#             f"Description: {result_data.get('metadata', {}).get('description', 'N/A')[:200]}",
#             ""
#         ]
        
#         # Add Markdown content
#         if "markdown" in result_data:
#             markdown_content = result_data["markdown"]
#             # Limit length to avoid excessive output
#             if len(markdown_content) > 15000:
#                 markdown_content = markdown_content[:15000] + "\n\n... (content truncated due to length)"
#             result_parts.append("Content (Markdown):")
#             result_parts.append(markdown_content)
#         else:
#             result_parts.append("[ERROR]: No content extracted from the webpage.")
        
#         return protocol_hint + youtube_hint + "\n".join(result_parts)
        
#     except requests.exceptions.Timeout:
#         return f"[ERROR]: Request timeout while scraping {url}. The website might be slow or unavailable."
#     except requests.exceptions.RequestException as e:
#         return f"[ERROR]: Failed to scrape {url} with Firecrawl: {str(e)}"
#     except Exception as e:
#         return f"[ERROR]: Unexpected error in scrape_website: {str(e)}"


@mcp.tool()
async def firecrawl_search_before(
    q: str,
    end_time: str,
    num_results: int = 5,
    floor_date: str = "1900-01-01",
    sources: Optional[List[str]] = None,
    country: Optional[str] = None,
    language: Optional[str] = None,
    scrape: bool = False
) -> str:
    """Search the web using Firecrawl with time constraints, returning only results published before end_time.
    This tool uses Google Custom Search API through Firecrawl with date filtering to ensure results
    are from before a specific cutoff date. Ideal for historical research and time-sensitive predictions.
    
    Args:
        q: Search query string.
        end_time: End date/time cutoff in "YYYY-MM-DD" or ISO8601 format (e.g., "2025-07-24" or "2025-10-11T15:30:00-07:00").
                  Only results published on or before this date will be returned.
        num_results: Number of results to return (default: 5, max: 20).
        floor_date: Earliest date boundary in "YYYY-MM-DD" format (default: "1900-01-01").
        sources: Optional list of domains to search within (e.g., ["nytimes.com", "reuters.com"]).
        country: Optional country code for localized search (e.g., "us", "jp", "uk").
        language: Optional language code (e.g., "en", "ja", "zh").
        scrape: If True, also scrape full content of each result (default: False, to save time).
    
    Returns:
        JSON string containing search results with titles, URLs, snippets, published times, and sources.
    """
    if not FIRECRAWL_API_KEY:
        return "[ERROR]: FIRECRAWL_API_KEY is not set, firecrawl_search_before is not available."
    
    def _to_mmddyyyy(date_str: str) -> str:
        """Convert date string to MM/DD/YYYY format for Google Custom Search tbs parameter."""
        try:
            # Try parsing with dateutil if available (more flexible)
            if DATEUTIL_AVAILABLE:
                parsed = dateutil_parser.parse(date_str)
                d = parsed.date()
            else:
                # Fallback: simple YYYY-MM-DD parsing
                if 'T' in date_str:
                    date_str = date_str.split('T')[0]
                parts = date_str.split('-')
                if len(parts) != 3:
                    raise ValueError(f"Invalid date format: {date_str}")
                year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                d = datetime.date(year, month, day)
            
            return f"{d.month}/{d.day}/{d.year}"
        except Exception as e:
            raise ValueError(f"Failed to parse date '{date_str}': {str(e)}")
    
    # Construct tbs (time-based search) parameter
    try:
        cd_max = _to_mmddyyyy(end_time)
        cd_min = _to_mmddyyyy(floor_date)
    except ValueError as e:
        return f"[ERROR]: {str(e)}. Use YYYY-MM-DD or ISO8601 format."
    
    tbs_param = f"cdr:1,cd_min:{cd_min},cd_max:{cd_max}"
    
    try:
        api_url = "https://api.firecrawl.dev/v1/search"
        
        # Build payload
        payload: Dict[str, Any] = {
            "query": q,
            "limit": max(1, min(num_results, 20)),
            "tbs": tbs_param,
            "scrape": scrape,
        }
        
        # Add optional parameters
        if sources:
            payload["sources"] = sources
        if country:
            payload["country"] = country
        if language:
            payload["language"] = language
        
        headers = {
            "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        
        response = await asyncio.to_thread(
            lambda: requests.post(api_url, json=payload, headers=headers, timeout=30)
        )
        response.raise_for_status()
        
        data = response.json()
        
        # Extract results (Firecrawl may use "data" or "results" key)
        results = data.get("data", []) or data.get("results", [])
        
        if not results:
            return json.dumps({
                "message": f"No results found for query: {q} (before {end_time})",
                "query": q,
                "end_time": end_time,
                "tbs": tbs_param,
                "results": []
            }, ensure_ascii=False, indent=2)
        
        # Format results
        formatted = {
            "query": q,
            "num_results": len(results),
            "tbs": tbs_param,
            "end_time": end_time,
            "floor_date": floor_date,
            "results": []
        }
        
        for i, r in enumerate(results):
            formatted_result = {
                "position": i + 1,
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": (r.get("snippet") or r.get("description") or "")[:300],
                "published_time": r.get("publishedTime") or r.get("date") or "",
                "source": r.get("source") or "",
            }
            
            # If scrape=True, include full content
            if scrape and "markdown" in r:
                formatted_result["content"] = r["markdown"][:5000]  # Limit content length
            
            formatted["results"].append(formatted_result)
        
        return json.dumps(formatted, ensure_ascii=False, indent=2)
        
    except requests.exceptions.RequestException as e:
        return f"[ERROR]: Failed to connect to Firecrawl Search API: {str(e)}"
    except Exception as e:
        return f"[ERROR]: Unexpected error in firecrawl_search_before: {str(e)}"


@mcp.tool()
async def wiki_get_page_content(entity: str, first_sentences: int = 5) -> str:  # 🚀 优化：从10降低到5
    """Get specific Wikipedia page content for a given entity (people, places, concepts, events) and return structured information.
    This tool searches Wikipedia for the given entity and returns either the first few sentences
    (which typically contain the summary/introduction) or full page content based on parameters.
    It handles disambiguation pages and provides clean, structured output.

    Args:
        entity: The entity to search for in Wikipedia (e.g., person name, place, concept, event).
        first_sentences: Number of first sentences to return from the page. Set to 0 to return full content. Defaults to 10.

    Returns:
        Formatted search results containing title, first sentences/full content, and URL.
        Returns error message if page not found or other issues occur.
    """
    try:
        import wikipedia
        
        # Try to get the Wikipedia page directly
        page = wikipedia.page(title=entity, auto_suggest=False)

        # Prepare the result
        result_parts = [f"Page Title: {page.title}"]

        if first_sentences > 0:
            # Get summary with specified number of sentences
            try:
                summary = wikipedia.summary(
                    entity, sentences=first_sentences, auto_suggest=False
                )
                result_parts.append(
                    f"First {first_sentences} sentences (introduction): {summary}"
                )
            except Exception:
                # Fallback to page summary if direct summary fails
                content_sentences = page.content.split(". ")[:first_sentences]
                summary = (
                    ". ".join(content_sentences) + "."
                    if content_sentences
                    else page.content[:5000] + "..."
                )
                result_parts.append(
                    f"First {first_sentences} sentences (introduction): {summary}"
                )
        else:
            # Return full content if first_sentences is 0
            result_parts.append(f"Content: {page.content}")

        result_parts.append(f"URL: {page.url}")

        return "\n\n".join(result_parts)

    except ImportError:
        return "[ERROR]: wikipedia package is not installed. Install it with: pip install wikipedia"

    except Exception as e:
        # Simplified error handling
        return f"[ERROR]: Failed to get Wikipedia content for '{entity}': {str(e)}\n\nTry using exa_search or scrape_website instead."


@mcp.tool()
async def search_wiki_revision(
    entity: str, year: int, month: int, max_revisions: int = 20  # 🚀 优化：从50降低到20
) -> str:
    """Search for an entity in Wikipedia and return the revision history for a specific month.
    This tool retrieves the edit history of a Wikipedia page for a given time period,
    including timestamps, revision IDs, and URLs to view specific versions.

    Args:
        entity: The entity to search for in Wikipedia (e.g., article title).
        year: The year of the revision (e.g., 2024). Values are auto-adjusted to valid range (2000-current year).
        month: The month of the revision (1-12). Values are auto-adjusted to valid range.
        max_revisions: Maximum number of revisions to return (default: 50, max: 500).

    Returns:
        Formatted revision history with timestamps, revision IDs, and URLs to view each revision.
        Returns error message if page not found or other issues occur.
    """
    # Auto-adjust date values and track changes
    adjustments = []
    original_year, original_month = year, month
    current_year = datetime.datetime.now().year

    # Adjust year to valid range
    if year < 2000:
        year = 2000
        adjustments.append(
            f"Year adjusted from {original_year} to 2000 (minimum supported)"
        )
    elif year > current_year:
        year = current_year
        adjustments.append(
            f"Year adjusted from {original_year} to {current_year} (current year)"
        )

    # Adjust month to valid range
    if month < 1:
        month = 1
        adjustments.append(f"Month adjusted from {original_month} to 1")
    elif month > 12:
        month = 12
        adjustments.append(f"Month adjusted from {original_month} to 12")

    # Prepare adjustment message if any changes were made
    if adjustments:
        adjustment_msg = (
            "Date auto-adjusted: "
            + "; ".join(adjustments)
            + f". Using {year}-{month:02d} instead.\n\n"
        )
    else:
        adjustment_msg = ""

    base_url = "https://en.wikipedia.org/w/api.php"

    try:
        # Construct the time range
        start_date = datetime.datetime(year, month, 1)
        last_day = calendar.monthrange(year, month)[1]
        end_date = datetime.datetime(year, month, last_day, 23, 59, 59)

        # Convert to ISO format (UTC time)
        start_iso = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_iso = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")

        # API parameters configuration
        params = {
            "action": "query",
            "format": "json",
            "titles": entity,
            "prop": "revisions",
            "rvlimit": min(max_revisions, 50),  # 🚀 优化：从500降低到50
            "rvstart": start_iso,
            "rvend": end_iso,
            "rvdir": "newer",
            "rvprop": "timestamp|ids",
        }

        response = await asyncio.to_thread(
            lambda: requests.get(base_url, params=params, timeout=30)
        )
        response.raise_for_status()
        data = response.json()

        # Check for API errors
        if "error" in data:
            return f"[ERROR]: Wikipedia API Error: {data['error'].get('info', 'Unknown error')}"

        # Process the response
        pages = (data.get("query") or {}).get("pages", {})

        if not pages:
            return f"[ERROR]: No results found for entity '{entity}'"

        # Check if page exists
        page_id = list(pages.keys())[0]
        if page_id == "-1":
            return f"[ERROR]: Page Not Found: No Wikipedia page found for '{entity}'"

        page_info = pages[page_id]
        page_title = page_info.get("title", entity)

        if "revisions" not in page_info or not page_info["revisions"]:
            return (
                adjustment_msg + f"Page Title: {page_title}\n\n"
                f"No revisions found for '{entity}' in {year}-{month:02d}.\n\n"
                f"The page may not have been edited during this time period."
            )

        # Format the results
        result_parts = [
            f"Page Title: {page_title}",
            f"Revision Period: {year}-{month:02d}",
            f"Total Revisions Found: {len(page_info['revisions'])}",
        ]

        # Add revision details
        revisions_details = []
        for i, rev in enumerate(page_info["revisions"], 1):
            revision_id = rev["revid"]
            timestamp = rev["timestamp"]

            # Format timestamp for better readability
            try:
                dt = datetime.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            except Exception:
                formatted_time = timestamp

            # Construct revision URL
            rev_url = f"https://en.wikipedia.org/w/index.php?title={entity}&oldid={revision_id}"

            revisions_details.append(
                f"{i}. Revision ID: {revision_id}\n"
                f"   Timestamp: {formatted_time}\n"
                f"   URL: {rev_url}"
            )

        if revisions_details:
            result_parts.append("Revisions:\n" + "\n\n".join(revisions_details))

        return (
            adjustment_msg
            + "\n\n".join(result_parts)
            + "\n\nHint: You can use the `scrape_website` tool to get the webpage content of a specific revision URL."
        )

    except requests.exceptions.Timeout:
        return f"[ERROR]: Network Error: Request timed out while fetching revision history for '{entity}'"

    except requests.exceptions.RequestException as e:
        return f"[ERROR]: Network Error: Failed to connect to Wikipedia: {str(e)}"

    except ValueError as e:
        return f"[ERROR]: Date Error: Invalid date values - {str(e)}"

    except Exception as e:
        return f"[ERROR]: Unexpected Error: An unexpected error occurred: {str(e)}"


@mcp.tool()
async def search_archived_webpage(url: str, year: int, month: int, day: int) -> str:
    """Search the Wayback Machine (archive.org) for archived versions of a webpage at a specific date.
    This tool queries the Internet Archive to find snapshots of web pages from the past.

    Args:
        url: The URL to search for in the Wayback Machine.
        year: The target year (e.g., 2023). Values are auto-adjusted to valid range (1995-current year).
        month: The target month (1-12). Values are auto-adjusted to valid range.
        day: The target day (1-31). Values are auto-adjusted to valid range for the given month.

    Returns:
        Formatted archive information including archived URL, timestamp, and availability status.
        Returns error message if URL not found in the archive or other issues occur.
    """
    # Handle empty URL
    if not url or not url.strip():
        return f"[ERROR]: Invalid URL: '{url}'. URL cannot be empty."

    # Auto-add https:// if no protocol is specified
    protocol_hint = ""
    if not url.startswith(("http://", "https://")):
        original_url = url
        url = f"https://{url}"
        protocol_hint = f"[NOTE]: Automatically added 'https://' to URL '{original_url}' -> '{url}'\n\n"

    hint_message = ""
    if ".wikipedia.org" in url:
        hint_message = "[NOTE]: You are searching for a Wikipedia page. You can also use the `search_wiki_revision` tool to get the revision content of a Wikipedia page.\n\n"

    # Check if specific date is requested
    date = ""
    adjustment_msg = ""
    if year > 0 and month > 0:
        # Auto-adjust date values and track changes
        adjustments = []
        original_year, original_month, original_day = year, month, day
        current_year = datetime.datetime.now().year

        # Adjust year to valid range
        if year < 1995:
            year = 1995
            adjustments.append(
                f"Year adjusted from {original_year} to 1995 (minimum supported)"
            )
        elif year > current_year:
            year = current_year
            adjustments.append(
                f"Year adjusted from {original_year} to {current_year} (current year)"
            )

        # Adjust month to valid range
        if month < 1:
            month = 1
            adjustments.append(f"Month adjusted from {original_month} to 1")
        elif month > 12:
            month = 12
            adjustments.append(f"Month adjusted from {original_month} to 12")

        # Adjust day to valid range for the given month/year
        max_day = calendar.monthrange(year, month)[1]
        if day < 1:
            day = 1
            adjustments.append(f"Day adjusted from {original_day} to 1")
        elif day > max_day:
            day = max_day
            adjustments.append(
                f"Day adjusted from {original_day} to {max_day} (max for {year}-{month:02d})"
            )

        # Update the date string with adjusted values
        date = f"{year:04d}{month:02d}{day:02d}"

        try:
            # Validate the final adjusted date
            datetime.datetime(year, month, day)
        except ValueError as e:
            return f"[ERROR]: Invalid date: {year}-{month:02d}-{day:02d}. {str(e)}"

        # Prepare adjustment message if any changes were made
        if adjustments:
            adjustment_msg = (
                "Date auto-adjusted: "
                + "; ".join(adjustments)
                + f". Using {date} instead.\n\n"
            )

    try:
        base_url = "https://archive.org/wayback/available"
        
        # Prepare parameters
        params = {"url": url}
        if date:
            params["timestamp"] = date
        
        # Make request with retry
        retry_count = 0
        max_retries = 5
        data = None
        
        while retry_count < max_retries:
            response = await asyncio.to_thread(
                lambda: requests.get(base_url, params=params, timeout=30)
            )
            response.raise_for_status()
            data = response.json()
            
            if "archived_snapshots" in data and "closest" in data["archived_snapshots"]:
                break
            
            retry_count += 1
            if retry_count < max_retries:
                await asyncio.sleep(min(2 ** retry_count, 60))

        if data and "archived_snapshots" in data and "closest" in data["archived_snapshots"]:
            closest = data["archived_snapshots"]["closest"]
            archived_url = closest["url"]
            archived_timestamp = closest["timestamp"]
            available = closest.get("available", True)

            if not available:
                if date:
                    return (
                        hint_message
                        + adjustment_msg
                        + protocol_hint
                        + (
                            f"Archive Status: Snapshot exists but is not available\n\n"
                            f"Original URL: {url}\n"
                            f"Requested Date: {year:04d}-{month:02d}-{day:02d}\n"
                            f"Closest Snapshot: {archived_timestamp}\n\n"
                            f"Try a different date or use the `scrape_website` tool for current content."
                        )
                    )
                else:
                    return (
                        hint_message
                        + protocol_hint
                        + (
                            f"Archive Status: Most recent snapshot exists but is not available\n\n"
                            f"Original URL: {url}\n"
                            f"Most Recent Snapshot: {archived_timestamp}\n\n"
                            f"The URL may have been archived but access is restricted."
                        )
                    )

            # Format timestamp for better readability
            try:
                dt = datetime.datetime.strptime(archived_timestamp, "%Y%m%d%H%M%S")
                formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            except Exception:
                formatted_time = archived_timestamp

            if date:
                return (
                    protocol_hint
                    + hint_message
                    + adjustment_msg
                    + (
                        f"Archive Found: Archived version located\n\n"
                        f"Original URL: {url}\n"
                        f"Requested Date: {year:04d}-{month:02d}-{day:02d}\n"
                        f"Archived URL: {archived_url}\n"
                        f"Archived Timestamp: {formatted_time}\n"
                    )
                    + "\n\nHint: You can use the `scrape_website` tool to get the content of the archived URL."
                )
            else:
                return (
                    protocol_hint
                    + hint_message
                    + (
                        f"Archive Found: Most recent archived version\n\n"
                        f"Original URL: {url}\n"
                        f"Archived URL: {archived_url}\n"
                        f"Archived Timestamp: {formatted_time}\n"
                    )
                    + "\n\nHint: You can use the `scrape_website` tool to get the content of the archived URL."
                )
        else:
            return (
                protocol_hint
                + hint_message
                + (
                    f"Archive Not Found: No archived versions available\n\n"
                    f"Original URL: {url}\n\n"
                    f"The URL '{url}' has not been archived by the Wayback Machine.\n"
                    f"You may want to:\n"
                    f"- Check if the URL is correct\n"
                    f"- Try a different date\n"
                    f"- Use the `scrape_website` tool to get current content\n"
                )
            )

    except requests.exceptions.Timeout:
        return f"[ERROR]: Network Error: Request timed out while querying Wayback Machine for '{url}'"

    except requests.exceptions.RequestException as e:
        return f"[ERROR]: Network Error: Failed to connect to Wayback Machine: {str(e)}"

    except ValueError as e:
        return f"[ERROR]: Data Error: Failed to parse response from Wayback Machine: {str(e)}"

    except Exception as e:
        return f"[ERROR]: Unexpected Error: An unexpected error occurred: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)
