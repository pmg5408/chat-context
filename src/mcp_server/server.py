"""
MCP Server for Chat History Search.

Exposes tools that allow AI assistants to search through past conversations.
"""
import asyncio
import signal
import sys
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
        ),
        Tool(
            name="search_by_topic",
            description="Search through conversations filtered by a specific topic. Use this when you want to find messages related to a particular topic.",
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
                    "topic": {
                        "type": "string",
                        "description": "Topic to filter by (e.g., 'react', 'database', 'auth')"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 5)",
                        "default": 5
                    }
                },
                "required": ["query", "topic"]
            }
        ),
        Tool(
            name="get_conversation_topics",
            description="Get all topics associated with a conversation. Use this to see what topics a conversation covers. You can use these to filter when searching for context",
            inputSchema={
                "type": "object",
                "properties": {
                    "conversation_title": {
                        "type": "string",
                        "description": "Title of the conversation to get topics for"
                    }
                },
                "required": ["conversation_title"]
            }
        ),
        Tool(
            name="deprecate_message",
            description="Mark a message as deprecated/superseded. Deprecated messages won't appear in future searches but remain in the database. Use this when the user indicates that information in a previous message is outdated or incorrect.",
            inputSchema={
                "type": "object",
                "properties": {
                    "conversation_title": {
                        "type": "string",
                        "description": "Title of the conversation"
                    },
                    "message_index": {
                        "type": "integer",
                        "description": "Index of the message to deprecate (0-based)"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why this message is being deprecated (optional)"
                    }
                },
                "required": ["conversation_title", "message_index"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls from AI assistants."""
    
    if name == "search_conversations":
        query = arguments["query"]
        conversation_titles = arguments.get("conversation_titles", [])
        limit = arguments.get("limit", 5)
        
        try:
            conversations = orchestrator.list_conversations()
            
            if not conversations:
                return [TextContent(
                    type="text",
                    text="No conversations available"
                )]

            if conversation_titles:
                conversations = [
                    c for c in conversations
                    if c['original_title'] in conversation_titles
                ]
                
                if not conversations:
                    return [TextContent(
                        type="text",
                        text=f"None of the specified conversations were found: {conversation_titles}"
                    )]

            all_results = []
            
            for conv in conversations:
                collection_name = conv['collection_name']

                from ..storage.chroma_store import ChromaStore

                if config.embedding.provider == 'openai':
                    from ..embedders.openai_embedder import OpenAIEmbedder
                    embedder = OpenAIEmbedder(config.embedding.openai)
                
                storage = ChromaStore(
                    config.storage.chroma_db_path,
                    collection_name,
                    embedder
                )

                results = storage.search(query, n_results=limit)

                for i in range(len(results['documents'][0])):
                    all_results.append({
                        'conversation': conv['original_title'],
                        'platform': conv['platform'],
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i]
                    })

            all_results.sort(key=lambda x: x['distance'])

            all_results = all_results[:limit]
            
            if not all_results:
                return [TextContent(
                    type="text",
                    text=f"No results found for query: '{query}'"
                )]

            response = f"Found {len(all_results)} relevant messages for '{query}':\n\n"
            
            for i, result in enumerate(all_results, 1):
                similarity = 1 - result['distance']
                response += f"**Result {i}** (similarity: {similarity:.2f})\n"
                response += f"From: {result['conversation']} ({result['platform']})\n"
                response += f"Role: {result['metadata']['role']}\n"
                response += f"Content: {result['text']}\n\n"
                response += "-" * 80 + "\n\n"
            
            return [TextContent(type="text", text=response)]
            
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error searching conversations: {str(e)}"
            )]
    
    elif name == "search_by_topic":
        query = arguments["query"]
        topic = arguments["topic"]
        conversation_titles = arguments.get("conversation_titles", [])
        limit = arguments.get("limit", 5)
        
        try:
            conversations = orchestrator.list_conversations()
            
            if not conversations:
                return [TextContent(
                    type="text",
                    text="No conversations available"
                )]

            if conversation_titles:
                conversations = [
                    c for c in conversations
                    if c['original_title'] in conversation_titles
                ]
                
                if not conversations:
                    return [TextContent(
                        type="text",
                        text=f"None of the specified conversations were found: {conversation_titles}"
                    )]

            all_results = []
            
            for conv in conversations:
                collection_name = conv['collection_name']

                from ..storage.chroma_store import ChromaStore

                if config.embedding.provider == 'openai':
                    from ..embedders.openai_embedder import OpenAIEmbedder
                    embedder = OpenAIEmbedder(config.embedding.openai)
                
                storage = ChromaStore(
                    config.storage.chroma_db_path,
                    collection_name,
                    embedder
                )

                results = storage.search_by_topic(query, topic, n_results=limit)

                for i in range(len(results['documents'][0])):
                    all_results.append({
                        'conversation': conv['original_title'],
                        'platform': conv['platform'],
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i]
                    })

            all_results.sort(key=lambda x: x['distance'])

            all_results = all_results[:limit]
            
            if not all_results:
                return [TextContent(
                    type="text",
                    text=f"No results found for query: '{query}' with topic: '{topic}'"
                )]

            response = f"Found {len(all_results)} relevant messages for '{query}' with topic '{topic}':\n\n"
            
            for i, result in enumerate(all_results, 1):
                similarity = 1 - result['distance']
                topics = result['metadata'].get('topics', '')
                response += f"**Result {i}** (similarity: {similarity:.2f})\n"
                response += f"From: {result['conversation']} ({result['platform']})\n"
                response += f"Role: {result['metadata']['role']}\n"
                response += f"Topics: {topics}\n"
                response += f"Content: {result['text']}\n\n"
                response += "-" * 80 + "\n\n"
            
            return [TextContent(type="text", text=response)]
            
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error searching by topic: {str(e)}"
            )]
    
    elif name == "list_available_conversations":
        try:
            conversations = orchestrator.list_conversations()
            
            if not conversations:
                return [TextContent(
                    type="text",
                    text="No conversations available.\n\nAdd conversations using:\npython -m src.cli add <platform> <title>"
                )]

            response = f"Available conversations ({len(conversations)}):\n\n"
            
            for i, conv in enumerate(conversations, 1):
                response += f"{i}. **{conv['original_title']}**\n"
                response += f"   Platform: {conv['platform']}\n"
                response += f"   Messages: {conv['message_count']}\n"
                response += f"   Collection: {conv['collection_name']}\n"
                response += f"   Last updated: {conv['last_message_timestamp'][:10]}\n\n"
            
            return [TextContent(type="text", text=response)]
            
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error listing conversations: {str(e)}"
            )]
    
    elif name == "get_conversation_topics":
        conversation_title = arguments["conversation_title"]
        
        try:
            conversations = orchestrator.list_conversations()
            
            if not conversations:
                return [TextContent(
                    type="text",
                    text="No conversations available"
                )]

            # Find the conversation
            conv = None
            for c in conversations:
                if c['original_title'] == conversation_title:
                    conv = c
                    break
            
            if not conv:
                return [TextContent(
                    type="text",
                    text=f"Conversation '{conversation_title}' not found"
                )]

            # Get topics from ChromaDB
            from ..storage.chroma_store import ChromaStore
            
            if config.embedding.provider == 'openai':
                from ..embedders.openai_embedder import OpenAIEmbedder
                embedder = OpenAIEmbedder(config.embedding.openai)
            
            storage = ChromaStore(
                config.storage.chroma_db_path,
                conv['collection_name'],
                embedder
            )

            topics = storage.get_all_topics()
            
            if not topics:
                return [TextContent(
                    type="text",
                    text=f"No topics found for '{conversation_title}'. Topics are extracted when the conversation is added."
                )]

            response = f"Topics for '{conversation_title}' ({len(topics)} topics):\n\n"
            for topic in topics:
                response += f"- {topic}\n"
            
            return [TextContent(type="text", text=response)]
            
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error getting topics: {str(e)}"
            )]
    
    elif name == "deprecate_message":
        conversation_title = arguments["conversation_title"]
        message_index = arguments["message_index"]
        reason = arguments.get("reason", "")
        
        try:
            conversations = orchestrator.list_conversations()
            
            if not conversations:
                return [TextContent(
                    type="text",
                    text="No conversations available"
                )]

            # Find the conversation
            conv = None
            for c in conversations:
                if c['original_title'] == conversation_title:
                    conv = c
                    break
            
            if not conv:
                return [TextContent(
                    type="text",
                    text=f"Conversation '{conversation_title}' not found"
                )]

            # Deprecate the message in ChromaDB
            from ..storage.chroma_store import ChromaStore
            
            if config.embedding.provider == 'openai':
                from ..embedders.openai_embedder import OpenAIEmbedder
                embedder = OpenAIEmbedder(config.embedding.openai)
            
            storage = ChromaStore(
                config.storage.chroma_db_path,
                conv['collection_name'],
                embedder
            )

            result = storage.deprecate_document(message_index, reason)
            
            if result['success']:
                response = f"✓ {result['message']}\n\n"
                response += f"Conversation: {conversation_title}\n"
                response += f"Message index: {message_index}\n"
                if reason:
                    response += f"Reason: {reason}\n"
                response += "\nThis message will no longer appear in search results."
            else:
                response = f"✗ {result['message']}"
            
            return [TextContent(type="text", text=response)]
            
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error deprecating message: {str(e)}"
            )]
    
    else:
        raise ValueError(f"Unknown tool: {name}")


async def main():
    """Main entry point for MCP server."""
    global config, orchestrator

    def signal_handler(sig, frame):
        print("\nShutting down MCP server...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Load configuration
        print("Loading config...", file=sys.stderr)
        config = load_config()
        print("Config loaded successfully", file=sys.stderr)
        
        orchestrator = IngestionOrchestrator(config)
        print("Orchestrator initialized", file=sys.stderr)
        
        # Run server
        async with stdio_server() as (read_stream, write_stream):
            print("Server running...", file=sys.stderr)
            await app.run(read_stream, write_stream, app.create_initialization_options())
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    asyncio.run(main())
