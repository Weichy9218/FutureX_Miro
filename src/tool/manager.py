# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import functools
from typing import Any, Awaitable, Callable, Protocol, TypeVar

from mcp import ClientSession, StdioServerParameters  # (already imported in config.py)
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client

from src.logging.logger import bootstrap_logger
from .mcp_servers.browser_session import PlaywrightSession

import os

LOGGER_LEVEL = os.getenv("LOGGER_LEVEL", "INFO")
logger = bootstrap_logger(level=LOGGER_LEVEL)

R = TypeVar("R")


def update_server_params_with_context_var(
    server_params: StdioServerParameters,
) -> StdioServerParameters:
    """
    Update the server params with the context var.
    """
    from src.logging.logger import TASK_CONTEXT_VAR

    if TASK_CONTEXT_VAR.get() is not None:
        server_params.env["TASK_ID"] = TASK_CONTEXT_VAR.get()
    return server_params


def with_timeout(timeout_s: float = 300.0):
    """
    Decorator: wraps any *async* function in asyncio.wait_for().
    Usage:
        @with_timeout(20)
        async def create_message_foo(...): ...
    """

    def decorator(
        func: Callable[..., Awaitable[R]],
    ) -> Callable[..., Awaitable[R]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> R:
            return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout_s)

        return wrapper

    return decorator


class ToolManagerProtocol(Protocol):
    """this enables other kinds of tool manager."""

    async def get_all_tool_definitions(self) -> Any: ...
    async def execute_tool_call(
        self, *, server_name: str, tool_name: str, arguments: dict[str, Any]
    ) -> Any: ...


class ToolManager(ToolManagerProtocol):
    def __init__(self, server_configs, tool_blacklist=None):
        """
        Initialize ToolManager.
        :param server_configs: List returned by create_server_parameters()
        """
        self.server_configs = server_configs
        self.server_dict = {
            config["name"]: config["params"] for config in server_configs
        }
        self.browser_session = None
        self.tool_blacklist = tool_blacklist if tool_blacklist else set()
        
        # Task context for time constraints
        self.task_context = {}

        logger.info(
            f"ToolManager initialized, loaded servers: {list(self.server_dict.keys())}"
        )
    
    def set_task_context(self, context: dict):
        """
        Set the current task context (e.g., end_time for time constraints).
        :param context: Dictionary with task metadata including 'end_time'
        """
        self.task_context = context
        if "end_time" in context:
            logger.info(f"Task time constraint set: end_time={context['end_time']}")
    
    def clear_task_context(self):
        """Clear the task context."""
        self.task_context = {}
    
    def _apply_time_constraints(self, tool_name: str, arguments: dict) -> dict:
        """
        Apply time constraints to tool arguments based on task context.
        This is a HARD CONSTRAINT that automatically adds time filters.
        
        :param tool_name: Name of the tool being called
        :param arguments: Original tool arguments
        :return: Modified arguments with time constraints applied
        """
        if not self.task_context.get("end_time"):
            return arguments
        
        end_time = self.task_context["end_time"]
        end_date = end_time.split('T')[0]  # Extract YYYY-MM-DD
        
        # Create a copy to avoid modifying the original
        modified_args = arguments.copy() if arguments else {}
        
        # Apply constraints based on tool name
        if tool_name == "exa_search":
            # For exa_search, enforce end_published_date
            if "end_published_date" not in modified_args or not modified_args.get("end_published_date"):
                modified_args["end_published_date"] = end_date
                logger.info(
                    f"🔒 HARD CONSTRAINT: Automatically set end_published_date='{end_date}' for exa_search"
                )
            elif modified_args["end_published_date"] > end_date:
                logger.warning(
                    f"🔒 HARD CONSTRAINT: Overriding end_published_date from '{modified_args['end_published_date']}' to '{end_date}' (task time limit)"
                )
                modified_args["end_published_date"] = end_date
        
        elif tool_name == "google_search":
            # For google_search, we could add tbs (time-based search) parameter
            # Example: qdr:d (past day), qdr:w (past week), qdr:m (past month)
            # However, Google doesn't support exact date filtering via tbs
            # So we just log a warning
            logger.info(
                f"⚠️  Note: google_search doesn't support exact date filtering. End time: {end_time}"
            )
        
        return modified_args

    def _is_huggingface_dataset_or_space_url(self, url):
        """
        Check if the URL is a Hugging Face dataset or space URL.
        :param url: The URL to check
        :return: True if it's a HuggingFace dataset or space URL, False otherwise
        """
        if not url:
            return False
        return "huggingface.co/datasets" in url or "huggingface.co/spaces" in url

    def _should_block_hf_scraping(self, tool_name, arguments):
        """
        Check if we should block scraping of Hugging Face datasets/spaces.
        :param tool_name: The name of the tool being called
        :param arguments: The arguments passed to the tool
        :return: True if scraping should be blocked, False otherwise
        """
        return (
            tool_name == "scrape"
            and arguments.get("url")
            and self._is_huggingface_dataset_or_space_url(arguments["url"])
        )

    def get_server_params(self, server_name):
        """Get parameters for specified server"""
        return self.server_dict.get(server_name)

    async def _find_servers_with_tool(self, tool_name):
        """
        Find servers containing the specified tool name among all servers
        :param tool_name: Tool name to search for
        :return: List of server names containing the tool
        """
        servers_with_tool = []

        for config in self.server_configs:
            server_name = config["name"]
            server_params = config["params"]

            try:
                if isinstance(server_params, StdioServerParameters):
                    async with stdio_client(
                        update_server_params_with_context_var(server_params)
                    ) as (read, write):
                        async with ClientSession(
                            read, write, sampling_callback=None
                        ) as session:
                            await session.initialize()
                            tools_response = await session.list_tools()
                            # Follow the same blacklist logic as get_all_tool_definitions
                            for tool in tools_response.tools:
                                if (server_name, tool.name) in self.tool_blacklist:
                                    continue
                                if tool.name == tool_name:
                                    servers_with_tool.append(server_name)
                                    break
                elif isinstance(server_params, str) and server_params.startswith(
                    ("http://", "https://")
                ):
                    # SSE endpoint
                    async with sse_client(server_params) as (read, write):
                        async with ClientSession(
                            read, write, sampling_callback=None
                        ) as session:
                            await session.initialize()
                            tools_response = await session.list_tools()
                            for tool in tools_response.tools:
                                # Consistent with get_all_tool_definitions: SSE part has no blacklist processing
                                # Can add specific tool filtering logic here (if needed)
                                # if server_name == "tool-excel" and tool.name not in ["get_workbook_metadata", "read_data_from_excel"]:
                                #     continue
                                if tool.name == tool_name:
                                    servers_with_tool.append(server_name)
                                    break
                else:
                    logger.error(
                        f"Error: Unknown parameter type for server '{server_name}': {type(server_params)}"
                    )
                    # For unknown types, we skip rather than throw an exception, because this is a search function
                    continue
            except Exception as e:
                logger.error(
                    f"Error: Cannot connect or get tools from server '{server_name}' to find '{tool_name}': {e}"
                )
                continue

        return servers_with_tool

    async def get_all_tool_definitions(self):
        """
        Connect to all configured servers and get their tool definitions.
        Returns a list suitable for passing to Prompt generators.
        """
        all_servers_for_prompt = []
        # Handle remote server tools
        for config in self.server_configs:
            server_name = config["name"]
            server_params = config["params"]
            one_server_for_prompt = {"name": server_name, "tools": []}
            logger.info(f"Getting tool definitions for server '{server_name}'...")

            try:
                if isinstance(server_params, StdioServerParameters):
                    async with stdio_client(
                        update_server_params_with_context_var(server_params)
                    ) as (read, write):
                        async with ClientSession(
                            read, write, sampling_callback=None
                        ) as session:
                            await session.initialize()
                            tools_response = await session.list_tools()
                            # black list some tools
                            for tool in tools_response.tools:
                                if (server_name, tool.name) in self.tool_blacklist:
                                    logger.info(
                                        f"Tool '{tool.name}' in server '{server_name}' is blacklisted, skipping."
                                    )
                                    continue
                                one_server_for_prompt["tools"].append(
                                    {
                                        "name": tool.name,
                                        "description": tool.description,
                                        "schema": tool.inputSchema,
                                    }
                                )
                elif isinstance(server_params, str) and server_params.startswith(
                    ("http://", "https://")
                ):
                    # SSE endpoint
                    async with sse_client(server_params) as (read, write):
                        async with ClientSession(
                            read, write, sampling_callback=None
                        ) as session:
                            await session.initialize()
                            tools_response = await session.list_tools()
                            for tool in tools_response.tools:
                                # Can add specific tool filtering logic here (if needed)
                                # if server_name == "tool-excel" and tool.name not in ["get_workbook_metadata", "read_data_from_excel"]:
                                #     continue
                                one_server_for_prompt["tools"].append(
                                    {
                                        "name": tool.name,
                                        "description": tool.description,
                                        "schema": tool.inputSchema,
                                    }
                                )
                else:
                    logger.error(
                        f"Error: Unknown parameter type for server '{server_name}': {type(server_params)}"
                    )
                    raise TypeError(
                        f"Unknown server params type for {server_name}: {type(server_params)}"
                    )

                logger.info(
                    f"Successfully obtained {len(one_server_for_prompt['tools'])} tool definitions for server '{server_name}'."
                )
                all_servers_for_prompt.append(one_server_for_prompt)

            except Exception as e:
                logger.error(
                    f"Error: Cannot connect or get tools from server '{server_name}': {e}"
                )
                # Still add server entry, but mark tool list as empty or containing error information
                one_server_for_prompt["tools"] = [
                    {"error": f"Failed to fetch tools: {e}"}
                ]
                all_servers_for_prompt.append(one_server_for_prompt)

        return all_servers_for_prompt

    @with_timeout(900)
    async def execute_tool_call(self, server_name, tool_name, arguments) -> Any:
        """
        Execute a single tool call.
        :param server_name: Server name
        :param tool_name: Tool name
        :param arguments: Tool arguments dictionary
        :return: Dictionary containing result or error
        """

        # Original remote server call logic
        server_params = self.get_server_params(server_name)
        if not server_params:
            logger.error(
                f"Error: Attempting to call server '{server_name}' that was not found"
            )

            # Try to find the tool in all available servers
            suggested_servers = await self._find_servers_with_tool(tool_name)

            error_message = f"Server '{server_name}' not found."

            if len(suggested_servers) == 1:
                # Auto-correction: only one server contains the tool, try to auto-correct and execute
                correct_server = suggested_servers[0]
                logger.info(
                    f"Auto-correction: Server '{server_name}' not found, but found tool '{tool_name}' in '{correct_server}', trying to auto-correct and execute"
                )

                try:
                    # Recursive call, using the correct server name
                    corrected_result = await self.execute_tool_call(
                        correct_server, tool_name, arguments
                    )

                    # If auto-correction is successful, add a note in the result
                    if "result" in corrected_result:
                        # Add auto-correction note in the result, including the reason for the correction
                        correction_note = f"[Auto-corrected: Server '{server_name}' not found, but tool '{tool_name}' was found only in server '{correct_server}', so automatically used '{correct_server}' instead] "
                        corrected_result["result"] = correction_note + str(
                            corrected_result["result"]
                        )
                        return corrected_result
                    elif "error" in corrected_result:
                        # If there is an error after auto-correction, add a note in the error message
                        correction_note = f"[Auto-corrected: Server '{server_name}' not found, but tool '{tool_name}' was found only in server '{correct_server}', attempted auto-correction but still failed] "
                        corrected_result["error"] = correction_note + str(
                            corrected_result["error"]
                        )
                        return corrected_result

                except Exception as auto_correct_error:
                    logger.error(f"Auto-correction failed: {auto_correct_error}")
                    error_message += f" Found tool '{tool_name}' in server '{correct_server}' and attempted auto-correction, but it failed: {str(auto_correct_error)}"

            elif len(suggested_servers) > 1:
                error_message += f" However, found tool '{tool_name}' in these servers: {', '.join(suggested_servers)}. You may want to use one of these servers instead."
            else:
                error_message += (
                    " It is possible that the server_name and tool_name were confused or mixed up. "
                    "You should try again and carefully check the server name and tool name provided in the system prompt."
                )

            return {
                "server_name": server_name,
                "tool_name": tool_name,
                "error": error_message,
            }

        # Apply time constraints (硬约束)
        if self.task_context.get("end_time"):
            arguments = self._apply_time_constraints(tool_name, arguments)
        
        logger.info(
            f"Connecting to server '{server_name}' to call tool '{tool_name}'...call arguments: '{arguments}'..."
        )

        if server_name == "playwright":
            try:
                if self.browser_session is None:
                    self.browser_session = PlaywrightSession(server_params)
                    await self.browser_session.connect()
                tool_result = await self.browser_session.call_tool(
                    tool_name, arguments=arguments
                )

                # Check if result is empty and provide better feedback
                if tool_result is None or tool_result == "":
                    logger.error(
                        f"Tool '{tool_name}' returned empty result, this may be normal (such as delete operations) or the tool execution may have issues"
                    )
                    return {
                        "server_name": server_name,
                        "tool_name": tool_name,
                        "result": f"Tool '{tool_name}' returned empty result - this may be expected (e.g., delete operations) or indicate an issue with tool execution",
                    }

                return {
                    "server_name": server_name,
                    "tool_name": tool_name,
                    "result": tool_result,
                }
            except Exception as e:
                return {
                    "server_name": server_name,
                    "tool_name": tool_name,
                    "error": f"Tool call failed: {str(e)}",
                }
        else:
            try:
                result_content = None
                if isinstance(server_params, StdioServerParameters):
                    async with stdio_client(
                        update_server_params_with_context_var(server_params)
                    ) as (read, write):
                        async with ClientSession(
                            read, write, sampling_callback=None
                        ) as session:
                            await session.initialize()
                            try:
                                tool_result = await session.call_tool(
                                    tool_name, arguments=arguments
                                )
                                # Safely extract result content without changing original format
                                if tool_result.content and len(tool_result.content) > 0:
                                    text_content = tool_result.content[-1].text
                                    if (
                                        text_content is not None
                                        and text_content.strip()
                                    ):
                                        result_content = (
                                            text_content  # Preserve original format!
                                        )
                                    else:
                                        result_content = f"Tool '{tool_name}' completed but returned empty text - this may be expected or indicate an issue"
                                else:
                                    result_content = f"Tool '{tool_name}' completed but returned no content - this may be expected or indicate an issue"

                                # If result is empty, log warning
                                if not tool_result.content:
                                    logger.error(
                                        f"Tool '{tool_name}' returned empty content, tool_result.content: {tool_result.content}"
                                    )

                                # post hoc check for browsing agent reading answers from hf datsets
                                if self._should_block_hf_scraping(tool_name, arguments):
                                    result_content = "You are trying to scrape a Hugging Face dataset for answers, please do not use the scrape tool for this purpose."
                            except Exception as tool_error:
                                logger.error(f"Tool execution error: {tool_error}")
                                return {
                                    "server_name": server_name,
                                    "tool_name": tool_name,
                                    "error": f"Tool execution failed: {str(tool_error)}",
                                }
                elif isinstance(server_params, str) and server_params.startswith(
                    ("http://", "https://")
                ):
                    async with sse_client(server_params) as (read, write):
                        async with ClientSession(
                            read, write, sampling_callback=None
                        ) as session:
                            await session.initialize()
                            try:
                                tool_result = await session.call_tool(
                                    tool_name, arguments=arguments
                                )
                                # Safely extract result content without changing original format
                                if tool_result.content and len(tool_result.content) > 0:
                                    text_content = tool_result.content[-1].text
                                    if (
                                        text_content is not None
                                        and text_content.strip()
                                    ):
                                        result_content = (
                                            text_content  # Preserve original format!
                                        )
                                    else:
                                        result_content = f"Tool '{tool_name}' completed but returned empty text - this may be expected or indicate an issue"
                                else:
                                    result_content = f"Tool '{tool_name}' completed but returned no content - this may be expected or indicate an issue"

                                # If result is empty, log warning
                                if not tool_result.content:
                                    logger.error(
                                        f"Tool '{tool_name}' returned empty content, tool_result.content: {tool_result.content}"
                                    )

                                # post hoc check for browsing agent reading answers from hf datsets
                                if self._should_block_hf_scraping(tool_name, arguments):
                                    result_content = "You are trying to scrape a Hugging Face dataset for answers, please do not use the scrape tool for this purpose."
                            except Exception as tool_error:
                                logger.error(f"Tool execution error: {tool_error}")
                                return {
                                    "server_name": server_name,
                                    "tool_name": tool_name,
                                    "error": f"Tool execution failed: {str(tool_error)}",
                                }
                else:
                    raise TypeError(
                        f"Unknown server params type for {server_name}: {type(server_params)}"
                    )

                logger.info(
                    f"Tool '{tool_name}' (server: '{server_name}') called successfully."
                )

                if (
                    isinstance(result_content, str)
                    and "Unknown tool:" in result_content
                ):
                    suggested_servers = await self._find_servers_with_tool(tool_name)
                    if len(suggested_servers) == 1:
                        logger.info(
                            f"Auto-correction: Tool '{tool_name}' not found in '{server_name}', trying '{suggested_servers[0]}'"
                        )
                        return await self.execute_tool_call(
                            suggested_servers[0], tool_name, arguments
                        )

                return {
                    "server_name": server_name,
                    "tool_name": tool_name,
                    "result": result_content,  # Return extracted text content
                }

            except Exception as outer_e:  # Rename this to outer_e to avoid shadowing
                logger.error(
                    f"Error: Failed to call tool '{tool_name}' (server: '{server_name}'): {outer_e}"
                )
                # import traceback
                # traceback.print_exc() # Print detailed stack trace for debugging

                # Store the original error message for later use
                error_message = str(outer_e)

                if (
                    tool_name == "scrape"
                    and "unhandled errors" in error_message
                    and "url" in arguments
                    and arguments["url"] is not None
                ):
                    try:
                        logger.info("Attempting to use MarkItDown for fallback...")
                        from markitdown import MarkItDown

                        md = MarkItDown(
                            docintel_endpoint="<document_intelligence_endpoint>"
                        )
                        result = md.convert(arguments["url"])
                        logger.info("Successfully used MarkItDown")
                        return {
                            "server_name": server_name,
                            "tool_name": tool_name,
                            "result": result.text_content,  # Return extracted text content
                        }
                    except (
                        Exception
                    ) as inner_e:  # Use a different name to avoid shadowing
                        # Log the inner exception if needed
                        logger.error(f"Fallback also failed: {inner_e}")
                        # No need for pass here as we'll continue to the return statement

                # Always use the outer exception for the final error response
                return {
                    "server_name": server_name,
                    "tool_name": tool_name,
                    "error": f"Tool call failed: {error_message}",
                }
