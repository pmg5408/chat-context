"""
Message Chunking Strategy for Vector Database

This module defines how we prepare conversation messages for ChromaDB storage.

KEY DECISION: Message-level vs Conversation-level chunking
===========================================================

We use MESSAGE-LEVEL chunking with conversation context in metadata.

Each message becomes a separate document in ChromaDB because:
1. Semantic search works better on individual exchanges
2. User can find specific Q&A pairs quickly
3. Can retrieve just relevant parts of long conversations
4. More flexible for context reconstruction

CHUNKING STRATEGY
=================

Basic Unit: Single Message (User or Assistant)
-----------------------------------------------
- Each message is stored as ONE document
- Includes full context in metadata
- Large messages may be split (see below)

Metadata Attached to Each Message:
----------------------------------
{
    'platform': 'chatgpt' | 'claude' | 'cursor',
    'conversation_id': 'uuid-123',
    'conversation_title': 'Project Planning Discussion',
    'message_index': 5,  # Position in conversation
    'role': 'user' | 'assistant',
    'timestamp': 1678015311,
    'has_previous': True,  # Has messages before this
    'has_next': True,      # Has messages after this
    'total_messages': 20   # Total in conversation
}

HANDLING LONG MESSAGES
======================

For messages >1000 tokens, we split into chunks:
- Max chunk size: 1000 tokens
- Overlap: 100 tokens between chunks
- Each chunk gets same metadata + chunk_index

Example for a 2500-token assistant response:
Chunk 0: tokens 0-1000 (metadata: chunk_index=0, total_chunks=3)
Chunk 1: tokens 900-1900 (metadata: chunk_index=1, total_chunks=3)
Chunk 2: tokens 1800-2500 (metadata: chunk_index=2, total_chunks=3)

CONTEXT WINDOW STRATEGY
========================

When retrieving messages for MCP query:

1. Semantic Search Phase:
   - Search ChromaDB with user's query
   - Get top N relevant message chunks
   - Return ranked by similarity

2. Context Expansion Phase (optional):
   - For each relevant message, optionally fetch:
     - Previous message (for context)
     - Next message (for continuity)
   - Controlled by `expand_context` parameter

Example Query Flow:
-------------------
User asks MCP: "What did we decide about the database?"
↓
ChromaDB returns:
  - Message #15 (similarity: 0.92): "We decided to use PostgreSQL..."
  - Message #8 (similarity: 0.85): "The database requirements are..."
↓
MCP can expand to include:
  - Message #14: "Should we use PostgreSQL or MySQL?"
  - Message #15: "We decided to use PostgreSQL..." ← Original match
  - Message #16: "Great, I'll set that up today."

EMBEDDING STRATEGY
==================

Document Text Format for Embedding:
------------------------------------
For USER messages:
  "USER: {message_content}"

For ASSISTANT messages:
  "ASSISTANT: {message_content}"

This prefix helps embeddings capture the role context.

For multi-turn context (optional enhancement):
  "{previous_msg_role}: {previous_msg_content}
   {current_msg_role}: {current_msg_content}"

BENEFITS OF THIS APPROACH
==========================

1. Precision: Find exact relevant exchanges
2. Flexibility: Reconstruct any context window needed
3. Efficiency: Don't embed entire 50-message conversations
4. Scalability: Works with thousands of conversations
5. Query Power: Filter by conversation, date, platform, etc.

SEARCH EXAMPLES
===============

Example 1: Find specific topic
Query: "database schema design"
Filter: platform='chatgpt', conversation_title CONTAINS 'project'
Returns: All messages about database from project chats

Example 2: Reconstruct conversation
Query: conversation_id='uuid-123'
Order: message_index ASC
Returns: Full conversation in order

Example 3: Recent assistant suggestions
Query: "recommendation for frontend framework"
Filter: role='assistant', timestamp > last_week
Returns: Recent AI suggestions about frameworks
"""

from dataclasses import dataclass
from typing import List, Tuple
from .base import Message, Conversation


@dataclass
class VectorDocument:
    """A document ready for ChromaDB ingestion."""
    document_text: str  # Text to embed
    metadata: dict      # Searchable metadata
    doc_id: str        # Unique identifier


def prepare_message_for_vector_db(
    message: Message,
    conversation: Conversation,
    chunk_size: int = 1000,
    overlap: int = 100
) -> List[VectorDocument]:
    """
    Convert a message into one or more VectorDocuments for ChromaDB.
    
    Args:
        message: The message to prepare
        conversation: Parent conversation (for metadata)
        chunk_size: Maximum tokens per chunk
        overlap: Token overlap between chunks
    
    Returns:
        List of VectorDocuments (usually just one, unless message is very long)
    """
    # Format message text with role prefix
    formatted_text = f"{message.role.upper()}: {message.content}"
    
    # Base metadata shared by all chunks
    base_metadata = {
        'platform': conversation.platform,
        'conversation_id': conversation.conversation_id,
        'conversation_title': conversation.title,
        'message_index': message.message_index,
        'role': message.role,
        'timestamp': int(message.timestamp.timestamp()),
        'has_previous': message.message_index > 0,
        'has_next': message.message_index < len(conversation.messages) - 1,
        'total_messages': len(conversation.messages),
    }
    
    # Simple token estimation (rough: 1 token ≈ 4 characters)
    estimated_tokens = len(formatted_text) // 4
    
    if estimated_tokens <= chunk_size:
        # Message fits in one chunk
        doc_id = f"{conversation.conversation_id}_{message.message_index}"
        
        return [VectorDocument(
            document_text=formatted_text,
            metadata=base_metadata,
            doc_id=doc_id
        )]
    
    else:
        # Need to split into chunks
        chunks = []
        char_chunk_size = chunk_size * 4  # Rough character equivalent
        char_overlap = overlap * 4
        
        start = 0
        chunk_index = 0
        total_chunks = (len(formatted_text) + char_chunk_size - 1) // char_chunk_size
        
        while start < len(formatted_text):
            end = min(start + char_chunk_size, len(formatted_text))
            chunk_text = formatted_text[start:end]
            
            # Add chunk-specific metadata
            chunk_metadata = {
                **base_metadata,
                'chunk_index': chunk_index,
                'total_chunks': total_chunks,
                'is_chunked': True
            }
            
            doc_id = f"{conversation.conversation_id}_{message.message_index}_chunk{chunk_index}"
            
            chunks.append(VectorDocument(
                document_text=chunk_text,
                metadata=chunk_metadata,
                doc_id=doc_id
            ))
            
            # Move to next chunk with overlap
            start += char_chunk_size - char_overlap
            chunk_index += 1
        
        return chunks


def prepare_conversation_for_vector_db(
    conversation: Conversation,
    chunk_size: int = 1000,
    overlap: int = 100
) -> List[VectorDocument]:
    """
    Prepare all messages in a conversation for ChromaDB.
    
    Args:
        conversation: The conversation to prepare
        chunk_size: Maximum tokens per chunk
        overlap: Token overlap between chunks
    
    Returns:
        List of all VectorDocuments for this conversation
    """
    all_documents = []
    
    for message in conversation.messages:
        docs = prepare_message_for_vector_db(
            message=message,
            conversation=conversation,
            chunk_size=chunk_size,
            overlap=overlap
        )
        all_documents.extend(docs)
    
    return all_documents


# Example usage:
if __name__ == '__main__':
    from datetime import datetime
    
    # Create a sample conversation
    conv = Conversation(
        conversation_id='test-123',
        title='Sample Chat',
        platform='chatgpt',
        messages=[
            Message(
                role='user',
                content='What is machine learning?',
                timestamp=datetime.now(),
                message_index=0
            ),
            Message(
                role='assistant',
                content='Machine learning is a subset of AI that...',
                timestamp=datetime.now(),
                message_index=1
            )
        ],
        created_at=datetime.now()
    )
    
    # Prepare for vector DB
    docs = prepare_conversation_for_vector_db(conv)
    
    print(f"Created {len(docs)} documents")
    for doc in docs:
        print(f"\nDoc ID: {doc.doc_id}")
        print(f"Text: {doc.document_text[:100]}...")
        print(f"Metadata: {doc.metadata}")
