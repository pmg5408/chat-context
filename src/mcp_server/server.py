"""
MCP Server for Chat History Search.

Exposes tools that allow AI assistants to search through past conversations.
"""
import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from ..config import load_config
from ..orchestrator import IngestionOrchestrator

app = Server("chat-history-mcp")

config = None
orchestrator = None


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools."""
    return [
        Tool(
            name="search_conversations",
            description="Search through the user's past AI conversations for relevant context. You can search specific conversations or all of them.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in the conversations"
                    },
                    "conversation_titles": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of conversation titles to search (leave empty to search all)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="list_available_conversations",
            description="List all conversations that are available for searching. Use this to see what context you have access to.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls from AI assistants."""
    
    if name == "search_conversations":
        # TODO: Implement search logic
        return [TextContent(
            type="text",
            text="Search tool not yet implemented"
        )]
    
    elif name == "list_available_conversations":
        # TODO: Implement list logic
        return [TextContent(
            type="text",
            text="List tool not yet implemented"
        )]
    
    else:
        raise ValueError(f"Unknown tool: {name}")


async def main():
    """Main entry point for MCP server."""
    global config, orchestrator
    
    # Load configuration
    config = load_config()
    orchestrator = IngestionOrchestrator(config)
    
    # Run server with stdio transport
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())