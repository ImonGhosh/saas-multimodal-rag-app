import os
import sys
import json
import asyncio
import requests
from xml.etree import ElementTree
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from pathlib import Path
from dotenv import load_dotenv
import re

def _is_windows_selector_event_loop(loop: asyncio.AbstractEventLoop) -> bool:
    # On Windows, SelectorEventLoop doesn't support subprocesses; Playwright needs subprocess.
    return sys.platform.startswith("win") and "SelectorEventLoop" in loop.__class__.__name__


async def _crawl_data_impl(url: str) -> str:
    urls = get_pydantic_ai_docs_urls(url)
    if not urls:
        raise ValueError("No URLs found to crawl. Please enter a valid URL.")

    print(f"Found {len(urls)} URLs to crawl")
    md_path = await crawl_parallel(urls)
    return str(md_path)


def _run_coroutine_in_new_proactor_loop(coro: "asyncio.Future[str] | asyncio.coroutines.Coroutine[Any, Any, str]") -> str:
    if not sys.platform.startswith("win") or not hasattr(asyncio, "ProactorEventLoop"):
        # Best-effort fallback; on non-Windows this should be fine.
        return asyncio.run(coro)  # type: ignore[arg-type]

    loop = asyncio.ProactorEventLoop()  # type: ignore[attr-defined]
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)  # type: ignore[arg-type]
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            loop.close()
            asyncio.set_event_loop(None)

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from openai import AsyncOpenAI
from supabase import create_client, Client

load_dotenv()

# Initialize OpenAI and Supabase clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks, paragraphs, and ignoring unnecessary metadata like URLs, images, and HTML elements."""
    chunks = []
    start = 0
    text_length = len(text)

    # Define regex patterns for unwanted elements like URLs, image links, and HTML-like tags.
    url_pattern = r"https?://[^\s]+"
    image_pattern = r"!\[.*?\]\(.*?\)"  # Matches image markdown syntax ![alt_text](url)
    html_tag_pattern = r"<[^>]+>"  # Matches any HTML-like tag
    unwanted_metadata_patterns = [
        url_pattern,    # Remove URLs
        image_pattern,  # Remove images
        html_tag_pattern # Remove HTML tags (if any)
    ]
    
    # Remove unwanted metadata (URLs, images, HTML tags)
    for pattern in unwanted_metadata_patterns:
        text = re.sub(pattern, '', text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a code block boundary first (```), ensuring we are respecting the chunk boundaries
        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # If no code block, try to break at a paragraph
        elif '\n\n' in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            # Find the last sentence break
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1

        # Extract chunk and clean it up (strip any remaining leading/trailing whitespaces)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = max(start + 1, end)

    return chunks

async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """Extract title and summary using GPT-4."""
    system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk.
    Keep both title and summary concise but informative."""
    
    try:
        response = await openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}  # Send first 1000 chars for context
            ],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error getting title and summary: {e}")
        return {"title": "Error processing title", "summary": "Error processing summary"}

async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

async def process_chunk(chunk: str, doc_name: str, topic_name: str, chunk_number: int, url: str) -> ProcessedChunk:
    """Process a single chunk of text."""
    # Get title and summary
    extracted = await get_title_and_summary(chunk, url)
    
    # Get embedding
    embedding = await get_embedding(chunk)
    
    # Create metadata
    metadata = {
        "source": doc_name, # changeable (Use an LLM to derive this name here itself.)
        "topic": topic_name,
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path
    }
    
    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=extracted['title'],
        summary=extracted['summary'],
        content=chunk,  # Store the original chunk content
        metadata=metadata,
        embedding=embedding
    )

async def insert_chunk(chunk: ProcessedChunk):
    """Insert a processed chunk into Supabase."""
    try:
        data = {
            "url": chunk.url,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding
        }
        
        result = supabase.table("website_pages").insert(data).execute()
        print(f"Inserted chunk {chunk.chunk_number} for {chunk.url}")
        return result
    except Exception as e:
        print(f"Error inserting chunk: {e}")
        return None

async def get_doc_name(chunks: str) -> Dict[str, str]:
    """Extract title and summary"""
    system_prompt = """You are a helpful assistant. 
    You can generate relevant a short document name and a short topic name from a given document snippet. 
    You always return only a JSON object with exactly 2 keys: 'doc_name' and 'topic_name', whose values should be the generated short document name and topic name respectively.
    The short document name should always and always be in the format 'xxx_xxx_doc" and strictly should not exceed 40 characters.
    The short topic name should always and always be a 1 to 10 word phrase summarizing the overall topic of the document.
    For example,
    If the content is about agentic ai, the short document name can be 'agentic_ai_doc' and short topic name can be 'agentic ai'.
    If the content is about the best football players of all time, short name can be 'best_football_player_doc' and short topic name can be 'best football players'.
    If the content is about the current political scenario in the USA, short name can be 'usa_political_scenario_doc' and short topic name can be 'political scenario in usa'.
    """
    
    # Prepare the prompt from joining 4 chunks
    prompt = "\n".join(chunks)

    try:
        response = await openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Based on the following text, generate a meaningful short name for it:\n\n{prompt}"}
            ],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error getting title and summary: {e}")
        return {"title": "Error processing title", "summary": "Error processing summary"}

async def process_and_store_document(url: str, markdown: str):
    # from docling.chunking import HybridChunker
    # from transformers import AutoTokenizer
    # from docling.document_converter import DocumentConverter
    """Process a document and store its chunks in parallel."""
    # Split into chunks
    chunks = chunk_text(markdown)
    
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i} length: {len(chunk)}")

    response = await get_doc_name(chunks[:4])

    # Get the response content
    doc_name = response['doc_name'].strip()
    topic_name = response['topic_name'].strip()
    
    # Process chunks in parallel
    tasks = [
        process_chunk(chunk, doc_name, topic_name, i, url) 
        for i, chunk in enumerate(chunks)
    ]
    processed_chunks = await asyncio.gather(*tasks)
    
    # Store chunks in parallel
    insert_tasks = [
        insert_chunk(chunk) 
        for chunk in processed_chunks
    ]
    await asyncio.gather(*insert_tasks)

async def crawl_parallel(urls: List[str], max_concurrent: int = 5) -> Path:
    """
    Crawl multiple URLs in parallel with a concurrency limit.

    Returns:
        Path to the newly created markdown file containing the crawl output.
    """
    documents_dir = Path(__file__).resolve().parent / "documents"
    documents_dir.mkdir(parents=True, exist_ok=True)

    # Create a single new markdown file per crawl run (predictable numbering).
    existing_max = 0
    for p in documents_dir.glob("markdown-*.md"):
        m = re.match(r"markdown-(\d+)\.md$", p.name)
        if m:
            existing_max = max(existing_max, int(m.group(1)))

    md_path = documents_dir / f"markdown-{existing_max + 1}.md"
    write_lock = asyncio.Lock()
    md_path.write_text("", encoding="utf-8")

    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    # Create the crawler instance
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_url(url: str):
            async with semaphore:
                result = await crawler.arun(
                    url=url,
                    config=crawl_config,
                    session_id=f"session-{hash(url)}"
                )
                if result.success:
                    print(f"Successfully crawled: {url}")
                    # crawl4ai versions differ: some return `result.markdown` as a str, others as an object with `.raw_markdown`.
                    markdown_obj = result.markdown
                    markdown_text = (
                        markdown_obj
                        if isinstance(markdown_obj, str)
                        else getattr(markdown_obj, "raw_markdown", str(markdown_obj))
                    )
                    block = f"\n\n<!-- Source: {url} -->\n\n{markdown_text}\n"
                    async with write_lock:
                        with md_path.open("a", encoding="utf-8") as out:
                            out.write(block)
                    print(f"Appended markdown to: {md_path}")
                else:
                    print(f"Failed: {url} - Error: {result.error_message}")
        
        # Process all URLs in parallel with limited concurrency
        await asyncio.gather(*[process_url(url) for url in urls])
    finally:
        await crawler.close()

    return md_path

def get_pydantic_ai_docs_urls(base_url: str):
    """
    Fetches all URLs from a website sitemap.
    Uses the sitemap URL formed by appending /sitemap.xml to the base URL to get these URLs.
    
    Returns:
        List[str]: List of URLs or the base URL if an error occurs
    """            
    # Build the sitemap URL by appending '/sitemap.xml' to the base URL
    sitemap_url = f"{base_url}/sitemap.xml"
    
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()

        # Parse the XML
        root = ElementTree.fromstring(response.content)

        # Dynamically extract the namespace if it is present
        namespaces = {}
        if '}' in root.tag:  # If the tag contains a namespace
            # Extract the namespace part from the root element tag
            namespace_uri = root.tag.split('}')[0].strip('{')
            namespaces['ns'] = namespace_uri  # Map it to the prefix 'ns'

        print(f'namespaces: {namespaces}')

        # Extract URLs based on the presence of the namespace
        if namespaces:
            urls = [loc.text for loc in root.findall('.//ns:loc', namespaces)]
        else:
            # No namespace, use the default search
            urls = [loc.text for loc in root.findall('.//loc')]

        print(f'urls: {urls}')

        return urls
    
    except Exception as e:
        print(f"Error fetching sitemap: {e}")
        # Return the base URL if any error occurs
        return [base_url]
    
async def crawl_data(url: str) -> str:
    """
    Ingest website docs into Supabase.

    On Windows, Playwright requires an event loop that supports subprocesses.
    Uvicorn/ASGI stacks sometimes run on SelectorEventLoop, which raises NotImplementedError.
    To be robust, we run ingestion on a dedicated ProactorEventLoop when needed.
    """
    if sys.platform.startswith("win"):
        try:
            loop = asyncio.get_running_loop()
            if _is_windows_selector_event_loop(loop):
                print(
                    f"Detected {loop.__class__.__name__} on Windows; running ingestion in a Proactor event loop thread for Playwright compatibility."
                )
                return await asyncio.to_thread(_run_coroutine_in_new_proactor_loop, _crawl_data_impl(url))
        except RuntimeError:
            # No running loop; fall back to direct execution.
            pass

    return await _crawl_data_impl(url)

# async def main():
#     # Get URLs from Pydantic AI docs
#     urls = get_pydantic_ai_docs_urls("https://medium.com/data-and-beyond/vector-databases-a-beginners-guide-b050cbbe9ca0") # changeable

#     if not urls:
#         print("No URLs found to crawl")
#         return
    
#     print(f"Found {len(urls)} URLs to crawl")
#     await crawl_parallel(urls)

# if __name__ == "__main__":
#     asyncio.run(main())
