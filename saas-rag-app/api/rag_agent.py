"""
RAG CLI Agent with PostgreSQL/PGVector
=======================================
Text-based CLI agent that searches through knowledge base using semantic similarity
"""

import asyncio
import asyncpg
import json
import logging
import os
import sys
from typing import Any, List

from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Observability (Langfuse) - optional at runtime
try:
    from .utils.observability import (
        start_span,
        end_span,
        text_payload,
        store_prompts,
        store_responses,
    )
except ImportError:
    try:
        from api.utils.observability import (
            start_span,
            end_span,
            text_payload,
            store_prompts,
            store_responses,
        )
    except ImportError:
        from utils.observability import (  # type: ignore
            start_span,
            end_span,
            text_payload,
            store_prompts,
            store_responses,
        )

# Global database pool
db_pool = None


async def initialize_db():
    """Initialize database connection pool."""
    global db_pool
    if not db_pool:
        db_pool = await asyncpg.create_pool(
            os.getenv("DATABASE_URL"),
            min_size=2,
            max_size=10,
            command_timeout=60
        )
        logger.info("Database connection pool initialized")


async def close_db():
    """Close database connection pool."""
    global db_pool
    if db_pool:
        await db_pool.close()
        logger.info("Database connection pool closed")


async def retrieve_chunks(query: str, limit: int = 5) -> List[dict[str, Any]]:
    """
    Retrieve raw chunk matches for evaluation or debugging.

    This helper mirrors the retrieval steps without changing any existing agent flow.
    """
    try:
        # Ensure database is initialized
        if not db_pool:
            await initialize_db()

        # Generate embedding for query
        try:
            # When running as a package: `python -m api.rag_agent_file`
            from .file_data_ingestion.embedder import create_embedder  # type: ignore
        except ImportError:
            try:
                # When running from repo root: `python api/rag_agent_file.py`
                from api.file_data_ingestion.embedder import create_embedder
            except ImportError:
                # When running from inside `api/`: `python rag_agent_file.py`
                from file_data_ingestion.embedder import create_embedder

        embedder = create_embedder()
        query_embedding = await embedder.embed_query(query)

        # Convert to PostgreSQL vector format
        embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'

        async with db_pool.acquire() as conn:
            results = await conn.fetch(
                """
                SELECT * FROM match_chunks($1::vector, $2)
                """,
                embedding_str,
                limit
            )

        return [dict(row) for row in results]

    except Exception as e:
        logger.error(f"Raw chunk retrieval failed: {e}", exc_info=True)
        return []


async def search_knowledge_base(ctx: RunContext[None], query: str, limit: int = 5) -> str:
    """
    Search the knowledge base using semantic similarity.

    Args:
        query: The search query to find relevant information
        limit: Maximum number of results to return (default: 5)

    Returns:
        Formatted search results with source citations
    """
    span = start_span(
        "kb.search",
        input={"query": text_payload(query, store=store_prompts()), "limit": limit},
    )
    try:
        # Ensure database is initialized
        if not db_pool:
            await initialize_db()

        # Generate embedding for query
        embed_span = start_span("kb.embed_query", input={"query": text_payload(query, store=store_prompts())})
        try:
            # When running as a package: `python -m api.rag_agent_file`
            from .file_data_ingestion.embedder import create_embedder  # type: ignore
        except ImportError:
            try:
                # When running from repo root: `python api/rag_agent_file.py`
                from api.file_data_ingestion.embedder import create_embedder
            except ImportError:
                # When running from inside `api/`: `python rag_agent_file.py`
                from file_data_ingestion.embedder import create_embedder

        embedder = create_embedder()
        query_embedding = await embedder.embed_query(query)
        end_span(embed_span, metadata={"embedding_dim": len(query_embedding)})

        # Convert to PostgreSQL vector format
        embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'

        # Search using match_chunks function
        db_span = start_span("kb.match_chunks", input={"limit": limit})
        async with db_pool.acquire() as conn:
            results = await conn.fetch(
                """
                SELECT * FROM match_chunks($1::vector, $2)
                """,
                embedding_str,
                limit
            )
        end_span(db_span, metadata={"rows": len(results)})

        # Format results for response
        if not results:
            end_span(span, output={"results": 0})
            return "No relevant information found in the knowledge base for your query."

        # Build response with sources
        response_parts = []
        sources: list[dict[str, str]] = []
        for i, row in enumerate(results, 1):
            similarity = row['similarity']
            content = row['content']
            doc_title = row['document_title']
            doc_source = row['document_source']

            response_parts.append(
                f"[Source: {doc_title}]\n{content}\n"
            )
            if len(sources) < 20:
                sources.append({"title": str(doc_title), "source": str(doc_source)})

        if not response_parts:
            end_span(span, output={"results": 0})
            return "Found some results but they may not be directly relevant to your query. Please try rephrasing your question."

        output_text = f"Found {len(response_parts)} relevant results:\n\n" + "\n---\n".join(response_parts)
        end_span(
            span,
            output={"results": len(response_parts), "sources": sources, "output": text_payload(output_text, store=store_responses())},
        )
        return output_text

    except Exception as e:
        logger.error(f"Knowledge base search failed: {e}", exc_info=True)
        end_span(span, error=e)
        return f"I encountered an error searching the knowledge base: {str(e)}"


async def find_all_titles(ctx: RunContext[None]) -> List[str]:
    """
    Retrieve all list of all available document titles

    Returns:
    List[str]: List of available document titles
    """
    span = start_span("kb.find_all_titles")
    try:
        # Ensure database is initialized
        if not db_pool:
            await initialize_db()

        async with db_pool.acquire() as conn:
            results = await conn.fetch(
                """
                SELECT title
                FROM "documents"
                """,
            )

        titles = [r["title"] for r in results if r["title"] is not None]
        end_span(span, output={"count": len(titles), "sample": titles[:20]})
        return titles

    except Exception as e:
        logger.error(f"Fetching document titles failed: {e}", exc_info=True)
        end_span(span, error=e)
        return []


async def find_content_by_title(ctx: RunContext[None], title: str) -> str:
    """
    Retrieve the full content of a document with a specific title, by combining all it's chunks

    Args:
    ctx: The run context
    title: The title of the document you need full content for

    Returns:
    str: The complete document content with all chunks combined in order
    """
    span = start_span("kb.find_content_by_title", input={"title": title})
    try:
        # Ensure database is initialized
        if not db_pool:
            await initialize_db()

        async with db_pool.acquire() as conn:
            results = await conn.fetch(
                """
                SELECT chunk_index, content
                FROM chunks
                WHERE metadata->>'title' = $1
                ORDER BY chunk_index
                """,
                title,
            )

        full_content = "\n".join(
            r["content"] for r in sorted(results, key=lambda x: x["chunk_index"]) if r["content"]
        )
        end_span(
            span,
            output={
                "chunks": len(results),
                "content": text_payload(full_content, store=store_responses()),
            },
        )
        return full_content

    except Exception as e:
        logger.error(f"Fetching document content failed: {e}", exc_info=True)
        end_span(span, error=e)
        return ""


# Create the PydanticAI agent with the RAG tool
# agent = Agent(
#     'openai:gpt-4o-mini',
#     system_prompt="""You are an intelligent knowledge assistant with access to some documentation and information.
# Your role is to help users find accurate information from the knowledge base.
# You have a professional yet friendly demeanor.

# IMPORTANT: Always search the knowledge base before answering questions about specific information.
# If information isn't in the knowledge base, clearly state that and offer general guidance.
# Be concise but thorough in your responses.
# Ask clarifying questions if the user's query is ambiguous.
# When you find relevant information, synthesize it clearly and cite the source documents.""",
#     tools=[search_knowledge_base]
# )
agent = Agent(
    'openai:gpt-4o-mini',
    system_prompt="""You are an intelligent knowledge assistant with access to some documentation and information.
Your role is to help users find accurate information from the knowledge base.
You have a professional yet friendly demeanor.

IMPORTANT: 
1. Always search the knowledge base before answering questions about specific information.
2. When you first look at the documentation, always start with RAG (provided by the 'search_knowledge_base' tool)
3. If it helps, then you may also always check the list of titles of all available documents (provided by the 'find_all_titles' tool) and retrieve the content of a relevant document (using the 'find_content_by_title' tool).
If information isn't in the knowledge base, clearly state that and offer general guidance.
Be concise but thorough in your responses.
Ask clarifying questions if the user's query is ambiguous.
When you find relevant information, synthesize it clearly and cite the source documents.""",
    tools=[search_knowledge_base, find_all_titles, find_content_by_title]
)


async def run_cli():
    """Run the agent in an interactive CLI with streaming."""

    # Initialize database
    await initialize_db()

    print("=" * 60)
    print("RAG Knowledge Assistant")
    print("=" * 60)
    print("Ask me anything about the knowledge base!")
    print("Type 'quit', 'exit', or press Ctrl+C to exit.")
    print("=" * 60)
    print()

    message_history = []

    try:
        while True:
            # Get user input
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nAssistant: Thank you for using the knowledge assistant. Goodbye!")
                break

            print("Assistant: ", end="", flush=True)

            try:
                # Stream the response using run_stream
                async with agent.run_stream(
                    user_input,
                    message_history=message_history
                ) as result:
                    # Stream text as it comes in (delta=True for only new tokens)
                    async for text in result.stream_text(delta=True):
                        # Print only the new token
                        print(text, end="", flush=True)

                    print()  # New line after streaming completes

                    # Update message history for context
                    message_history = result.all_messages()

            except KeyboardInterrupt:
                print("\n\n[Interrupted]")
                break
            except Exception as e:
                print(f"\n\nError: {e}")
                logger.error(f"Agent error: {e}", exc_info=True)

            print()  # Extra line for readability

    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    finally:
        await close_db()


async def main():
    """Main entry point."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Check required environment variables
    if not os.getenv("DATABASE_URL"):
        logger.error("DATABASE_URL environment variable is required")
        sys.exit(1)

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable is required")
        sys.exit(1)

    # Run the CLI
    await run_cli()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nShutting down...")
