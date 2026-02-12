"""
Command-line interface for Chat History MCP Server.
"""
import click
import sys
from pathlib import Path

from .config import load_config
from .orchestrator import IngestionOrchestrator


@click.group()
def cli():
    """Chat History MCP Server - Manage conversation contexts."""
    pass


@cli.command()
@click.argument('platform', type=click.Choice(['claude', 'chatgpt', 'cursor']))
@click.argument('title')
@click.option('--config', default='config.yaml', help='Path to config file')
def add(platform, title, config):
    """
    Add a conversation to the vector database.
    
    PLATFORM: claude, chatgpt, or cursor
    TITLE: Exact title of the conversation to add
    
    Example:
        python cli.py add claude "Database Design Discussion"
    """
    try:
        cfg = load_config(config)
        orchestrator = IngestionOrchestrator(cfg)

        click.echo(f"Adding conversation '{title}' from {platform}...")
        result = orchestrator.add_conversation(platform, title)

        click.echo(f"\n✅ Successfully added '{result['conversation_title']}'")
        click.echo(f"   Collection: {result['collection_name']}")
        click.echo(f"   Messages: {result['message_count']}")
        click.echo(f"   Platform: {result['platform']}")
        
    except FileNotFoundError as e:
        click.echo(f"❌ Error: {e}", err=True)
        click.echo(f"\nMake sure your export file exists at the path specified in {config}", err=True)
        sys.exit(1)
    except ValueError as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('platform', type=click.Choice(['claude', 'chatgpt', 'cursor']))
@click.argument('title')
@click.option('--config', default='config.yaml', help='Path to config file')
def update(platform, title, config):
    """
    Update an existing conversation with new messages.
    
    PLATFORM: claude, chatgpt, or cursor
    TITLE: Title of the conversation to update
    
    Example:
        python cli.py update claude "Database Design Discussion"
    """
    try:
        cfg = load_config(config)
        orchestrator = IngestionOrchestrator(cfg)

        click.echo(f"Updating conversation '{title}' from {platform}...")
        result = orchestrator.update_conversation(platform, title)

        if result['new_messages'] == 0:
            click.echo(f"\n✅ No new messages found for '{result['conversation_title']}'")
            click.echo(f"   Total messages: {result['total_messages']}")
        else:
            click.echo(f"\n✅ Successfully updated '{result['conversation_title']}'")
            click.echo(f"   New messages: {result['new_messages']}")
            click.echo(f"   Total messages: {result['total_messages']}")
        
    except ValueError as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('title')
@click.option('--config', default='config.yaml', help='Path to config file')
def remove(title, config):
    """
    Remove a conversation from the vector database.
    
    TITLE: Title of the conversation to remove
    
    Example:
        python cli.py remove "Database Design Discussion"
    """
    try:
        cfg = load_config(config)
        orchestrator = IngestionOrchestrator(cfg)

        conversations = orchestrator.list_conversations()
        collection_name = orchestrator._sanitize_collection_name(title)
        
        conv_meta = None
        for c in conversations:
            if c['collection_name'] == collection_name or c['original_title'] == title:
                conv_meta = c
                break
        
        if not conv_meta:
            click.echo(f"❌ Conversation '{title}' not found", err=True)
            sys.exit(1)
        
        # Confirm removal
        click.echo(f"\nConversation: {conv_meta['original_title']}")
        click.echo(f"Messages: {conv_meta['message_count']}")
        click.echo(f"Platform: {conv_meta['platform']}")
        
        if not click.confirm(f"\nRemove this conversation?"):
            click.echo("Cancelled.")
            return

        result = orchestrator.remove_conversation(title)

        click.echo(f"\n✅ Removed '{conv_meta['original_title']}'")
        click.echo(f"   Deleted {result['message_count']} messages")
        
    except ValueError as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--config', default='config.yaml', help='Path to config file')
def list(config):
    """
    List all available conversations.
    
    Example:
        python cli.py list
    """
    try:
        cfg = load_config(config)
        orchestrator = IngestionOrchestrator(cfg)

        conversations = orchestrator.list_conversations()
        
        if not conversations:
            click.echo("No conversations found.")
            click.echo("\nAdd a conversation with: python cli.py add <platform> <title>")
            return

        click.echo(f"\nAvailable conversations ({len(conversations)}):\n")
        click.echo("=" * 80)
        
        for i, conv in enumerate(conversations, 1):
            click.echo(f"{i}. {conv['original_title']}")
            click.echo(f"   Collection: {conv['collection_name']}")
            click.echo(f"   Platform: {conv['platform']}")
            click.echo(f"   Messages: {conv['message_count']}")
            click.echo(f"   Last updated: {conv['last_message_timestamp'][:10]}")
            click.echo()
        
        click.echo("=" * 80)
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.argument('query')
@click.argument('conversation_title')
@click.option('--limit', default=5, help='Number of results')
@click.option('--config', default='config.yaml')
def search(query, conversation_title, limit, config):
    """
    Search within a conversation.
    
    Example:
        python cli.py search "database design" "Migrating chat context"
    """
    try:
        cfg = load_config(config)
        
        # Get collection name
        orchestrator = IngestionOrchestrator(cfg)
        collection_name = orchestrator._sanitize_collection_name(conversation_title)
        
        # Initialize embedder
        if cfg.embedding.provider == 'openai':
            from .embedders.openai_embedder import OpenAIEmbedder
            embedder = OpenAIEmbedder(cfg.embedding.openai)
        
        # Search
        from .storage.chroma_store import ChromaStore
        storage = ChromaStore(cfg.storage.chroma_db_path, collection_name, embedder)
        
        results = storage.search(query, n_results=limit)
        
        # Display
        click.echo(f"\nSearching '{conversation_title}' for: '{query}'\n")
        click.echo("=" * 80)
        
        for i, doc in enumerate(results['documents'][0], 1):
            meta = results['metadatas'][0][i-1]
            distance = results['distances'][0][i-1]
            
            click.echo(f"\n{i}. [{meta['role']}] (similarity: {1-distance:.2f})")
            click.echo(f"   {doc[:200]}...")
        
        click.echo("\n" + "=" * 80)
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)

if __name__ == '__main__':
    cli()