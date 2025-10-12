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


# @mcp.tool()
# async def exa_find_similar(
#     url: str,
#     num_results: int = 5,  # 🚀 优化：从10降低到5
#     exclude_source_domain: bool = True,
# ) -> str:
#     """Find content similar to a given URL using Exa's AI-powered similarity search.
#     This tool analyzes the content and context of the provided URL to find semantically similar web pages.

#     Args:
#         url: The URL to find similar content for.
#         num_results: Number of similar results to return (default: 10, max: 100).
#         exclude_source_domain: Whether to exclude results from the same domain as the source URL (default: True).

#     Returns:
#         JSON string containing similar content with titles, URLs, snippets, and similarity scores.
#     """
#     if not EXA_API_KEY:
#         return "[ERROR]: EXA_API_KEY is not set, exa_find_similar tool is not available."
    
#     try:
#         api_url = "https://api.exa.ai/findSimilar"
        
#         payload = {
#             "url": url,
#             "num_results": min(num_results, 8),
#             "exclude_source_domain": exclude_source_domain,
#             "contents": {
#                 "text": True,
#                 "highlights": True,
#             }
#         }
        
#         headers = {
#             "accept": "application/json",
#             "content-type": "application/json",
#             "x-api-key": EXA_API_KEY
#         }
        
#         response = await asyncio.to_thread(
#             lambda: requests.post(api_url, json=payload, headers=headers, timeout=30)
#         )
#         response.raise_for_status()
        
#         data = response.json()
        
#         # Format results
#         formatted_results = {
#             "source_url": url,
#             "num_results": len(data.get("results", [])),
#             "results": []
#         }
        
#         for idx, result in enumerate(data.get("results", []), 1):
#             formatted_result = {
#                 "position": idx,
#                 "title": result.get("title", ""),
#                 "url": result.get("url", ""),
#                 "published_date": result.get("published_date", ""),
#                 "score": result.get("score", 0),
#                 "text": result.get("text", "")[:500],
#                 "highlights": result.get("highlights", [])[:3]
#             }
#             formatted_results["results"].append(formatted_result)
        
#         return json.dumps(formatted_results, ensure_ascii=False, indent=2)
        
#     except requests.exceptions.RequestException as e:
#         return f"[ERROR]: Failed to connect to Exa API: {str(e)}"
#     except Exception as e:
#         return f"[ERROR]: Unexpected error in exa_find_similar: {str(e)}"


@mcp.tool()
async def firecrawl_search(
    q: str,
    end_time: str,
    num_results: int = 5,
    floor_date: str = "2000-01-01",
    location: Optional[str] = None,
    language: Optional[str] = None,
    scrape: bool = False
) -> str:
    """🔍 Firecrawl v2 Search - Time-constrained web search with enhanced reliability.
    
    Search the web using Firecrawl v2 API with strict time filtering. Returns only results 
    published before end_time. Automatically handles retries and fallback to smaller result sets.
    
    Args:
        q: Search query string.
        end_time: End date cutoff in "YYYY-MM-DD" format (e.g., "2025-08-03").
                  Only results published on or before this date will be returned.
        num_results: Number of results to return (default: 5, recommended: 3-10, max: 20).
        floor_date: Earliest date boundary in "YYYY-MM-DD" format (default: "2000-01-01").
        location: Optional location for search (e.g., "us", "cn", "jp").
        language: Optional language code (e.g., "en", "zh-CN", "ja").
        scrape: If True, scrape full content of each result (slower, default: False).
    
    Returns:
        JSON string with search results including titles, URLs, snippets, and timestamps.
    
    Note: If this tool times out, try using exa_search(q='...', end_published_date='...') instead.
    """
    if not FIRECRAWL_API_KEY:
        return "[ERROR]: FIRECRAWL_API_KEY is not set, firecrawl_search is not available."
    
    def _to_mmddyyyy(date_str: str) -> str:
        """Convert YYYY-MM-DD to MM/DD/YYYY for Google Custom Search tbs parameter."""
        try:
            if DATEUTIL_AVAILABLE:
                d = dateutil_parser.parse(date_str).date()
            else:
                parts = date_str.split('T')[0].split('-')
                if len(parts) != 3:
                    raise ValueError(f"Invalid date format: {date_str}")
                d = datetime.date(int(parts[0]), int(parts[1]), int(parts[2]))
            return f"{d.month:02d}/{d.day:02d}/{d.year}"
        except Exception as e:
            raise ValueError(f"Failed to parse date '{date_str}': {str(e)}")
    
    # Construct tbs parameter for time-based search
    try:
        cd_max = _to_mmddyyyy(end_time)
        cd_min = _to_mmddyyyy(floor_date)
        tbs_param = f"cdr:1,cd_min:{cd_min},cd_max:{cd_max}"
    except ValueError as e:
        return f"[ERROR]: {str(e)}. Please use YYYY-MM-DD format (e.g., '2025-08-03')."
    
    # 🚀 自适应 limit：从请求的数量开始，失败时降级到 3
    limits_to_try = [min(num_results, 10), 5, 3] if num_results > 3 else [3]
    
    for attempt, limit in enumerate(limits_to_try):
        try:
            # Firecrawl v2 API endpoint
            api_url = "https://api.firecrawl.dev/v2/search"
            
            # Build v2 payload with enhanced options
            payload = {
                "query": q,
                "limit": limit,
                "sources": ["web"],  # v2: 必须指定 sources
                "tbs": tbs_param,
                "timeout": 25000,  # 顶层超时 25秒
                "ignoreInvalidURLs": True,  # 过滤无效URL，提高成功率
            }
            
            # Add optional parameters
            if location:
                payload["location"] = location
            if language:
                payload["lang"] = language
            
            # Scrape options (only if scrape=True)
            if scrape:
                payload["scrapeOptions"] = {
                    "formats": ["markdown"],
                    "timeout": 15000,  # 单页抓取超时 15秒
                    "waitFor": 0,  # 不等待 JS 渲染，加快速度
                    "onlyMainContent": True,  # 只抓取主要内容
                    "includeHtml": False,  # 不返回 HTML
                    "includeTags": [],  # 不返回特定标签
                    "excludeTags": ["nav", "footer", "aside"],  # 排除导航、页脚等
                }
            
            headers = {
                "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
                "Content-Type": "application/json",
            }
            
            # 发送请求（单次，不在此层重试）
            response = await asyncio.to_thread(
                lambda: requests.post(api_url, json=payload, headers=headers, timeout=30)
            )
            response.raise_for_status()
            data = response.json()
            
            # v2 API 返回格式：{"success": true, "data": [...]}
            if not data.get("success"):
                error_msg = data.get("error", "Unknown error")
                raise Exception(f"API returned success=false: {error_msg}")
            
            results = data.get("data", [])
            
            # 成功获取结果，格式化并返回
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
                "end_time": end_time,
                "floor_date": floor_date,
                "tbs": tbs_param,
                "results": []
            }
            
            for i, r in enumerate(results):
                formatted_result = {
                    "position": i + 1,
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": (r.get("description") or r.get("snippet") or "")[:400],
                    "published_time": r.get("publishedDate") or r.get("date") or "",
                }
                
                # Add scraped content if available
                if scrape and "markdown" in r:
                    formatted_result["content"] = r["markdown"][:8000]  # 限制长度
                
                formatted["results"].append(formatted_result)
            
            return json.dumps(formatted, ensure_ascii=False, indent=2)
            
        except requests.exceptions.Timeout:
            # 超时错误：尝试更小的 limit
            if attempt < len(limits_to_try) - 1:
                await asyncio.sleep(0.5)
                continue
            return (
                f"[ERROR]: Request timeout after trying limits {limits_to_try}.\n\n"
                f"⚠️  Alternative: Try using exa_search(q='{q}', end_published_date='{end_time}') instead."
            )
            
        except requests.exceptions.RequestException as e:
            # 网络/HTTP错误：尝试更小的 limit
            if attempt < len(limits_to_try) - 1:
                await asyncio.sleep(0.5)
                continue
            return (
                f"[ERROR]: Failed to connect to Firecrawl API: {str(e)}\n\n"
                f"⚠️  Alternative: Try using exa_search(q='{q}', end_published_date='{end_time}') instead."
            )
            
        except Exception as e:
            # 其他错误：尝试更小的 limit
            if attempt < len(limits_to_try) - 1:
                await asyncio.sleep(0.5)
                continue
            return f"[ERROR]: Unexpected error in firecrawl_search: {str(e)}"
    
    # 理论上不会到这里
    return (
        f"[ERROR]: All retry attempts failed.\n\n"
        f"⚠️  Alternative: Try using exa_search(q='{q}', end_published_date='{end_time}') instead."
    )


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


# ❌ 已移除的工具：
# - search_archived_webpage: archive.org 网络连接极不稳定，已删除
# - exa_find_similar: 使用频率低，已注释
# 
# ✅ 推荐使用的替代方案：
# 1. exa_search(q='...', end_published_date='YYYY-MM-DD') - 主要工具，最稳定
# 2. firecrawl_search(q='...', end_time='YYYY-MM-DD') - 备用工具，支持中文
# 3. search_wiki_revision(entity='...', year=YYYY, month=MM) - 维基百科历史版本


if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)
