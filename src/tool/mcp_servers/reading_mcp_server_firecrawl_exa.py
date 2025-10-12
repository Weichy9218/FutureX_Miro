# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0
#
# Reading MCP Server using Firecrawl (Web Scraping)
# Complete replacement for Jina solution

import argparse
import os
import tempfile
import aiohttp
import atexit
import requests

from fastmcp import FastMCP
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import asyncio

# Initialize FastMCP server
from src.logging.logger import setup_mcp_logging

setup_mcp_logging(tool_name=os.path.basename(__file__))
mcp = FastMCP("reading-mcp-server-firecrawl-exa")

# API Keys from environment
FIRECRAWL_API_KEY = os.environ.get("FIRECRAWL_API_KEY", "")
EXA_API_KEY = os.environ.get("EXA_API_KEY", "")


async def scrape_with_firecrawl(url: str) -> str:
    """Internal helper function to scrape webpage content using Firecrawl v2 API.
    
    Args:
        url: The URL of the webpage to scrape.
        
    Returns:
        The scraped webpage content in Markdown format.
    """
    if not FIRECRAWL_API_KEY:
        return "[ERROR]: FIRECRAWL_API_KEY is not set, cannot scrape website."
    
    try:
        # Firecrawl v2 API endpoint
        api_url = "https://api.firecrawl.dev/v2/scrape"
        
        payload = {
            "url": url,
            "formats": ["markdown"],
            "timeout": 30000,  # 顶层超时 60秒
            "onlyMainContent": True,  # 只提取主要内容
            "waitFor": 0,  # 不等待JS渲染，加快速度
            "mobile": False,  # 使用桌面版
            "skipTlsVerification": False,
            "removeBase64Images": True,  # 移除base64图片，减少内容大小
        }
        
        headers = {
            "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Use asyncio.to_thread to make synchronous requests non-blocking
        response = await asyncio.to_thread(
            lambda: requests.post(api_url, json=payload, headers=headers, timeout=70)
        )
        response.raise_for_status()
        
        data = response.json()
        
        # v2 API 返回格式：{"success": true, "data": {...}}
        if not data.get("success"):
            error_msg = data.get("error", "Unknown error")
            return f"[ERROR]: Firecrawl scraping failed: {error_msg}"
        
        # Extract Markdown content from v2 response
        result_data = data.get("data", {})
        markdown_content = result_data.get("markdown", "")
        
        if not markdown_content:
            return "[ERROR]: No content extracted from the webpage."
        
        # Return content (limit length to avoid token overflow)
        if len(markdown_content) > 12000:  # 提高到12000，平衡内容完整性和token消耗
            markdown_content = markdown_content[:12000] + "\n\n... (content truncated due to length)"
        
        return markdown_content
        
    except requests.exceptions.Timeout:
        return f"[ERROR]: Request timeout while scraping {url}. The website might be slow or unavailable."
    except requests.exceptions.RequestException as e:
        return f"[ERROR]: Failed to scrape {url} with Firecrawl: {str(e)}"
    except Exception as e:
        return f"[ERROR]: Unexpected error in scrape_with_firecrawl: {str(e)}"


@mcp.tool()
async def read_file(uri: str) -> str:
    """Read various types of resources (Doc, PPT, PDF, Excel, CSV, ZIP files, etc.) described by a file: or data: URI.
    This tool supports reading both local files and remote files (HTTP/HTTPS URLs).
    
    The tool can:
    1. Read local files using direct file paths or file:// URIs
    2. Download and convert remote files from HTTP/HTTPS URLs
    3. Automatically fall back to web scraping (using Firecrawl) if direct download fails
    4. Convert various file formats to LLM-friendly Markdown format
    
    Supported file types include:
    - Documents: PDF, DOCX, PPTX, TXT, MD
    - Spreadsheets: XLSX, CSV
    - Archives: ZIP (extracts and reads contents)
    - Images: PNG, JPG (extracts text via OCR if available)
    - Web content: HTML pages via URL
    
    Args:
        uri: The URI of the resource to read. Must start with 'file:', 'data:', 'http:', or 'https:' schemes,
             or be a valid local file path. Files from sandbox are not supported - use local paths instead.

    Returns:
        The content of the resource in Markdown format, or an error message if reading fails.
        
    Examples:
        - Local file: read_file("/path/to/document.pdf")
        - File URI: read_file("file:///path/to/document.pdf")
        - HTTP URL: read_file("https://example.com/document.pdf")
    """
    if not uri or not uri.strip():
        return "[ERROR]: URI parameter is required and cannot be empty."

    # Check if it's a sandbox file path (not supported)
    if "home/user" in uri:
        return "[ERROR]: The read_file tool cannot access sandbox files. Please use the local path provided by the original instruction."

    # Validate URI scheme
    valid_schemes = ["http:", "https:", "file:", "data:"]
    
    # If it's a local path that exists, automatically add file: prefix
    if not any(uri.lower().startswith(scheme) for scheme in valid_schemes) and os.path.exists(uri):
        uri = f"file:{os.path.abspath(uri)}"

    # Re-validate URI scheme
    if not any(uri.lower().startswith(scheme) for scheme in valid_schemes):
        return f"[ERROR]: Invalid URI scheme. Supported schemes are: {', '.join(valid_schemes)}. Got: {uri}"

    # Handle HTTP(S) URLs
    if uri.lower().startswith(("http://", "https://")):
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        retry_count = 0
        max_retries = 3
        data = None
        
        # Try to download the file directly
        while retry_count <= max_retries:
            try:
                async with aiohttp.ClientSession(headers=headers) as session:
                    async with session.get(uri, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                        resp.raise_for_status()
                        data = await resp.read()
                break  # Success, exit retry loop
            except Exception as e:
                retry_count += 1
                if retry_count > max_retries:
                    # Download failed, try using Firecrawl scraping as fallback
                    try:
                        scrape_result = await scrape_with_firecrawl(uri)
                        if scrape_result.startswith("[ERROR]"):
                            return f"[ERROR]: Failed to download {uri}: {e}\n\nAlso failed to scrape with Firecrawl: {scrape_result}"
                        else:
                            return f"[INFO]: Direct download failed, successfully used Firecrawl to scrape the webpage instead.\n\n{scrape_result}"
                    except Exception as scrape_error:
                        return f"[ERROR]: Failed to download {uri}: {e}\n\nAlso failed to scrape with Firecrawl: {scrape_error}"
                
                # Wait before retrying with exponential backoff
                await asyncio.sleep(2 ** retry_count)

        # If download succeeded, write to a temp file
        if data:
            suffix = os.path.splitext(uri)[1] or ""
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(data)
            tmp.flush()
            tmp.close()
            uri = f"file:{tmp.name}"

            # Ensure the temp file is deleted when the program exits
            def _cleanup_tempfile(path):
                try:
                    os.remove(path)
                except Exception:
                    pass

            atexit.register(_cleanup_tempfile, tmp.name)
        else:
            return "[ERROR]: Failed to download file after multiple retries."

    # Use markitdown-mcp to convert file to Markdown
    tool_name = "convert_to_markdown"
    arguments = {"uri": uri}

    server_params = StdioServerParameters(
        command="uv",
        args=["run", "--active", "--", "markitdown-mcp"],
    )

    result_content = ""
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write, sampling_callback=None) as session:
                await session.initialize()
                try:
                    tool_result = await session.call_tool(
                        tool_name, arguments=arguments
                    )
                    result_content = (
                        tool_result.content[-1].text if tool_result.content else ""
                    )
                    
                    # Add hint message
                    if result_content:
                        result_content += "\n\nNote: If the document contains instructions or important information, please review them thoroughly and ensure you follow all relevant guidance."
                    
                except Exception as tool_error:
                    return f"[ERROR]: Tool execution failed: {str(tool_error)}.\n\nHint: The reading tool cannot access sandbox files. Use the local path provided by the original instruction instead."
                    
    except Exception as session_error:
        return f"[ERROR]: Failed to connect to markitdown-mcp server: {str(session_error)}\n\nMake sure 'markitdown-mcp' is installed: uv add markitdown-mcp"

    if not result_content:
        return "[ERROR]: No content was extracted from the file. The file might be empty or in an unsupported format."

    return result_content


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Reading MCP Server (Firecrawl replacement)")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport method: 'stdio' or 'http' (default: stdio)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to use when running with HTTP transport (default: 8080)",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="/mcp",
        help="URL path to use when running with HTTP transport (default: /mcp)",
    )

    # Parse command line arguments
    args = parser.parse_args()

    # Run the server with the specified transport method
    if args.transport == "stdio":
        mcp.run(transport="stdio", show_banner=False)
    else:
        # For HTTP transport, include port and path options
        mcp.run(
            transport="streamable-http",
            port=args.port,
            path=args.path,
            show_banner=False,
        )
