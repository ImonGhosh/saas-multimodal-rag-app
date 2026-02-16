"""
Main ingestion script for processing markdown documents into vector DB and knowledge graph.
"""

import os
import asyncio
import logging
import json
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import argparse
from uuid import uuid4

import asyncpg
from dotenv import load_dotenv
from openai import AsyncOpenAI

from .chunker import ChunkingConfig, create_chunker, DocumentChunk
from .embedder import create_embedder

# Import utilities
try:
    from ..utils.db_utils import initialize_database, close_database, db_pool
    from ..utils.models import IngestionConfig, IngestionResult
except ImportError:
    # For direct execution or testing
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.db_utils import initialize_database, close_database, db_pool
    from utils.models import IngestionConfig, IngestionResult

# Load environment variables
load_dotenv()
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

logger = logging.getLogger(__name__)

# Observability (Langfuse) - optional at runtime
try:
    from ..utils.observability import (
        start_trace,
        end_trace,
        start_span,
        end_span,
        text_payload,
        store_prompts,
        store_responses,
    )
except ImportError:
    try:
        from api.utils.observability import (
            start_trace,
            end_trace,
            start_span,
            end_span,
            text_payload,
            store_prompts,
            store_responses,
        )
    except ImportError:
        from utils.observability import (  # type: ignore
            start_trace,
            end_trace,
            start_span,
            end_span,
            text_payload,
            store_prompts,
            store_responses,
        )


def _project_root() -> Path:
    # ingest.py -> file_data_ingestion -> api -> project root
    return Path(__file__).resolve().parents[2]


def _resolve_documents_folder(documents_folder: str) -> str:
    """
    Resolve a documents folder path robustly across execution modes.

    Resolution order for relative paths:
    1) Relative to current working directory
    2) Relative to the project root
    """
    folder_path = Path(documents_folder).expanduser()

    if folder_path.is_absolute():
        return str(folder_path)

    cwd_candidate = Path.cwd() / folder_path
    if cwd_candidate.exists():
        return str(cwd_candidate)

    return str(_project_root() / folder_path)

async def create_title(content: str) -> Dict[str, str]:
    """Create a short document title """
    system_prompt = """You are a helpful assistant. 
    You can generate relevant a short title from a given document snippet. 
    You always return only a JSON object with exactly a single key: 'title', whose value should be the generated document title.
    The short title should always be a short word phrase summarizing the overall topic of the document.
    """
    
    # Prepare the prompt from joining 4 chunks
    prompt = content

    span = start_span(
        "ingest.create_title",
        input=text_payload(prompt, store=store_prompts()),
        metadata={"model": os.getenv("LLM_MODEL", "gpt-4o-mini")},
    )
    try:
        response = await openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Based on the following text, generate a short meaningful title for it:\n\n{prompt}"}
            ],
            response_format={ "type": "json_object" }
        )
        result = json.loads(response.choices[0].message.content)
        end_span(span, output={"title": (result or {}).get("title")})
        return result
    except Exception as e:
        logger.error(f"Error creating title: {e}")
        end_span(span, error=e)
        return {"title": "Error processing title", "summary": "Error processing summary"}


class DocumentIngestionPipeline:
    """Pipeline for ingesting documents into vector DB and knowledge graph."""
    
    def __init__(
        self,
        config: IngestionConfig,
        clean_before_ingest: bool = True,
        is_image_enabled: bool = False,
    ):
        """
        Initialize ingestion pipeline.

        Args:
            config: Ingestion configuration
            clean_before_ingest: Whether to clean existing data before ingestion (default: True)
        """
        self.config = config
        self.documents_folder: str | None = None
        self.clean_before_ingest = clean_before_ingest
        self.is_image_enabled = is_image_enabled
        
        # Initialize components
        self.chunker_config = ChunkingConfig(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            max_chunk_size=config.max_chunk_size,
            use_semantic_splitting=config.use_semantic_chunking
        )
        
        self.chunker = create_chunker(self.chunker_config)
        self.embedder = create_embedder()
        
        self._initialized = False
    
    async def initialize(self):
        """Initialize database connections."""
        if self._initialized:
            return
        
        logger.info("Initializing ingestion pipeline...")
        
        # Initialize database connections
        await initialize_database()
        
        self._initialized = True
        logger.info("Ingestion pipeline initialized")
    
    async def close(self):
        """Close database connections."""
        if self._initialized:
            await close_database()
            self._initialized = False
    
    async def ingest_documents(
        self,
        document_path: Optional[str] = None,
        documents_folder: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ) -> List[IngestionResult]:
        """
        Ingest a single document (when `document_path` is provided) or all documents from a folder.
        
        Args:
            document_path: Path to a single document to ingest
            documents_folder: Folder to scan for documents when `document_path` is not provided
            progress_callback: Optional callback for progress updates
        
        Returns:
            List of ingestion results
        """
        if not self._initialized:
            await self.initialize()
        
        if document_path is not None:
            resolved_path = str(Path(document_path).expanduser().resolve())
            file_candidate = Path(resolved_path)
            if not file_candidate.exists() or not file_candidate.is_file():
                raise FileNotFoundError(f"Document file not found: {resolved_path}")
            self.documents_folder = str(file_candidate.parent)
            document_files = [resolved_path]
        else:
            default_documents_dir = _project_root() / "api" / "documents"
            folder = str(default_documents_dir) if documents_folder is None else str(documents_folder)
            self.documents_folder = _resolve_documents_folder(folder)
            document_files = self._find_document_files()

        if not document_files:
            logger.warning(f"No supported document files found in {self.documents_folder}")
            return []

        # Clean existing data if requested (only if we have documents to ingest)
        if self.clean_before_ingest:
            await self._clean_databases()

        logger.info(f"Found {len(document_files)} document files to process")

        results = []

        for i, file_path in enumerate(document_files):
            try:
                logger.info(f"Processing file {i+1}/{len(document_files)}: {file_path}")

                result = await self._ingest_single_document(file_path)
                results.append(result)

                if progress_callback:
                    progress_callback(i + 1, len(document_files))
                
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                results.append(IngestionResult(
                    document_id="",
                    title=os.path.basename(file_path),
                    chunks_created=0,
                    entities_extracted=0,
                    relationships_created=0,
                    processing_time_ms=0,
                    errors=[str(e)]
                ))
        
        # Log summary
        total_chunks = sum(r.chunks_created for r in results)
        total_errors = sum(len(r.errors) for r in results)
        
        logger.info(f"Ingestion complete: {len(results)} documents, {total_chunks} chunks, {total_errors} errors")
        
        return results
    
    async def _ingest_single_document(self, file_path: str) -> IngestionResult:
        """
        Ingest a single document.

        Args:
            file_path: Path to the document file

        Returns:
            Ingestion result
        """
        start_time = datetime.now()
        doc_span = start_span("ingest.document", input={"file_path": file_path})

        try:
            # Read document (returns tuple: content, docling_doc)
            read_span = start_span("ingest.read", input={"file_path": file_path})
            document_content, docling_doc = self._read_document(file_path)
            end_span(read_span, metadata={"content_len": len(document_content), "docling_doc": bool(docling_doc)})
            

            title_span = start_span("ingest.extract_title", input={"file_path": file_path})
            document_title = await self._extract_title(document_content, file_path)
            end_span(title_span, output={"title": document_title})

            documents_folder = self.documents_folder or str(Path(file_path).resolve().parent)
            document_source = os.path.relpath(file_path, documents_folder)

            # Extract metadata from content
            metadata_span = start_span("ingest.extract_metadata")
            document_metadata = self._extract_document_metadata(document_content, file_path)
            end_span(metadata_span, metadata={"metadata_keys": list(document_metadata.keys())})

            logger.info(f"Processing document: {document_title}")

            # Chunk the document - pass DoclingDocument for HybridChunker
            chunk_span = start_span(
                "ingest.chunk_document",
                input={"title": document_title, "source": document_source},
            )
            chunks = await self.chunker.chunk_document(
                content=document_content,
                title=document_title,
                source=document_source,
                metadata=document_metadata,
                docling_doc=docling_doc  # Pass DoclingDocument for HybridChunker
            )
            # chunks = None
            end_span(chunk_span, metadata={"chunks": len(chunks)})
            
            if not chunks:
                logger.warning(f"No chunks created for {document_title}")
                end_span(doc_span, output={"chunks": 0, "title": document_title})
                return IngestionResult(
                    document_id="",
                    title=document_title,
                    chunks_created=0,
                    entities_extracted=0,
                    relationships_created=0,
                    processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                    errors=["No chunks created"]
                )
            
            logger.info(f"Created {len(chunks)} chunks")
            
            # Entity extraction removed (graph-related functionality)
            entities_extracted = 0
            
            # Generate embeddings
            embed_span = start_span("ingest.embed_chunks", input={"chunks": len(chunks)})
            embedded_chunks = await self.embedder.embed_chunks(chunks)
            end_span(embed_span, metadata={"embedded_chunks": len(embedded_chunks)})
            logger.info(f"Generated embeddings for {len(embedded_chunks)} chunks")
            
            # Save to PostgreSQL
            save_span = start_span("ingest.save_postgres", input={"chunks": len(embedded_chunks)})
            document_id = await self._save_to_postgres(
                document_title,
                document_source,
                document_content,
                embedded_chunks,
                document_metadata
            )
            # document_id = "1234"
            end_span(save_span, output={"document_id": document_id})
            
            logger.info(f"Saved document to PostgreSQL with ID: {document_id}")
            
            # Knowledge graph functionality removed
            relationships_created = 0
            graph_errors = []
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            end_span(
                doc_span,
                output={
                    "document_id": document_id,
                    "title": document_title,
                    "chunks": len(chunks),
                    "content": text_payload(document_content, store=store_responses()),
                },
            )
            return IngestionResult(
                document_id=document_id,
                title=document_title,
                chunks_created=len(chunks),
                entities_extracted=entities_extracted,
                relationships_created=relationships_created,
                processing_time_ms=processing_time,
                errors=graph_errors
            )
        except Exception as e:
            end_span(doc_span, error=e)
            raise
    
    def _find_document_files(self) -> List[str]:
        """Find all supported document files in the documents folder."""
        if not self.documents_folder:
            return []

        documents_dir = Path(self.documents_folder)

        if not documents_dir.exists():
            logger.error(
                "Documents folder not found: %s (cwd=%s)",
                str(documents_dir),
                os.getcwd(),
            )
            return []

        # Supported file patterns - Docling + text formats + audio
        patterns = [
            "*.md", "*.markdown", "*.txt",  # Text formats
            "*.pdf",  # PDF
            "*.docx", "*.doc",  # Word
            "*.pptx", "*.ppt",  # PowerPoint
            "*.xlsx", "*.xls",  # Excel
            "*.html", "*.htm",  # HTML
            "*.mp3", "*.wav", "*.m4a", "*.flac",  # Audio formats
        ]
        files = []

        for pattern in patterns:
            files.extend(glob.glob(str(documents_dir / "**" / pattern), recursive=True))

        return sorted(files)
    
    def _read_document(self, file_path: str) -> tuple[str, Optional[Any]]:
        """
        Read document content from file - supports multiple formats via Docling.

        Returns:
            Tuple of (markdown_content, docling_document)
            docling_document is None for text files and audio files
        """
        file_ext = os.path.splitext(file_path)[1].lower()

        # Audio formats - transcribe with Whisper ASR
        audio_formats = ['.mp3', '.wav', '.m4a', '.flac']
        if file_ext in audio_formats:
            content = self._transcribe_audio(file_path)

            # Testing helper: persist transcription next to the source file
            try:
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                output_path = os.path.join(os.path.dirname(file_path), f"{base_name}-converted.md")
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(content)
                logger.info(f"Saved transcribed markdown: {output_path}")
            except Exception as e:
                logger.warning(f"Failed to save transcribed markdown for {file_path}: {e}")
            return (content, None)  # No DoclingDocument for audio

        # Docling-supported formats (convert to markdown)
        docling_formats = ['.pdf', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls', '.html', '.htm']

        if file_ext in docling_formats:
            try:
                if self.is_image_enabled and file_ext == ".pdf":
                    from docling.datamodel.pipeline_options import (
                        granite_picture_description,
                        smolvlm_picture_description,
                        PdfPipelineOptions,
                    )
                    from docling.document_converter import DocumentConverter, PdfFormatOption
                    from docling.datamodel.base_models import InputFormat

                    logger.info(
                        "Converting %s file using Docling with image extraction: %s",
                        file_ext,
                        os.path.basename(file_path),
                    )

                    pipeline_options = PdfPipelineOptions()
                    pipeline_options.do_picture_description = True
                    pipeline_options.picture_description_options = smolvlm_picture_description
                    pipeline_options.picture_description_options.prompt = (
                        "Describe the image in three sentences. Be concise and accurate."
                    )
                    pipeline_options.images_scale = 2.0
                    pipeline_options.generate_picture_images = True

                    converter = DocumentConverter(
                        format_options={
                            InputFormat.PDF: PdfFormatOption(
                                pipeline_options=pipeline_options,
                            )
                        }
                    )
                else:
                    from docling.document_converter import DocumentConverter

                    logger.info(f"Converting {file_ext} file using Docling: {os.path.basename(file_path)}")

                    converter = DocumentConverter()
                result = converter.convert(file_path)

                # Export to markdown for consistent processing
                markdown_content = result.document.export_to_markdown()

                # Testing helper: persist converted markdown next to the source file
                try:
                    base_name = os.path.splitext(os.path.basename(file_path))[0]
                    output_path = os.path.join(os.path.dirname(file_path), f"{base_name}-converted.md")
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(markdown_content)
                    logger.info(f"Saved converted markdown: {output_path}")
                except Exception as e:
                    logger.warning(f"Failed to save converted markdown for {file_path}: {e}")
                logger.info(f"Successfully converted {os.path.basename(file_path)} to markdown")

                # Return both markdown and DoclingDocument for HybridChunker
                return (markdown_content, result.document)

            except Exception as e:
                logger.error(f"Failed to convert {file_path} with Docling: {e}")
                # Fall back to raw text if Docling fails
                logger.warning(f"Falling back to raw text extraction for {file_path}")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return (f.read(), None)
                except:
                    return (f"[Error: Could not read file {os.path.basename(file_path)}]", None)

        # Text-based formats (read directly)
        else:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return (f.read(), None)
            except UnicodeDecodeError:
                # Try with different encoding
                with open(file_path, 'r', encoding='latin-1') as f:
                    return (f.read(), None)

    def _transcribe_audio(self, file_path: str) -> str:
        """Transcribe audio file using Whisper ASR via Docling."""
        try:
            from pathlib import Path
            from docling.document_converter import DocumentConverter, AudioFormatOption
            from docling.datamodel.pipeline_options import AsrPipelineOptions
            from docling.datamodel import asr_model_specs
            from docling.datamodel.base_models import InputFormat
            from docling.pipeline.asr_pipeline import AsrPipeline

            # Use Path object - Docling expects this
            audio_path = Path(file_path).resolve()
            logger.info(f"Transcribing audio file using Whisper Turbo: {audio_path.name}")
            logger.info(f"Audio file absolute path: {audio_path}")

            # Verify file exists
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            # Configure ASR pipeline with Whisper Turbo model
            pipeline_options = AsrPipelineOptions()
            pipeline_options.asr_options = asr_model_specs.WHISPER_TURBO

            converter = DocumentConverter(
                format_options={
                    InputFormat.AUDIO: AudioFormatOption(
                        pipeline_cls=AsrPipeline,
                        pipeline_options=pipeline_options,
                    )
                }
            )

            # Transcribe the audio file - pass Path object
            result = converter.convert(audio_path)

            # Export to markdown with timestamps
            markdown_content = result.document.export_to_markdown()
            logger.info(f"Successfully transcribed {os.path.basename(file_path)}")
            return markdown_content

        except Exception as e:
            logger.error(f"Failed to transcribe {file_path} with Whisper ASR: {e}")
            return f"[Error: Could not transcribe audio file {os.path.basename(file_path)}]"

    async def _extract_title(self, content: str, file_path: str) -> str:
        """Extract title from document content or filename.

        Priority (within first 70 lines, excluding fenced code blocks):
        1) First '# ' heading
        2) First '## ' heading
        3) First '### ' heading
        Fallback: create_title(...) then filename without extension
        """
        lines = content.splitlines()[:30]
        text = "\n".join(lines)

        def _non_code_lines(lines: list[str]) -> list[str]:
            """
            Return lines excluding those inside triple-backtick fenced code blocks.

            Notes:
            - Handles fences like ``` or ```python
            - Toggle on each fence line encountered.
            """
            in_fence = False
            kept: list[str] = []
            for line in lines:
                s = line.strip()
                if s.startswith("```"):
                    in_fence = not in_fence
                    continue  # don't consider the fence line itself
                if not in_fence:
                    kept.append(line)
            return kept

        filtered_lines = _non_code_lines(lines)

        # 1) First '# '
        for line in filtered_lines:
            s = line.strip()
            if s.startswith("# "):
                return s[2:].strip()

        # 2) First '## '
        for line in filtered_lines:
            s = line.strip()
            if s.startswith("## "):
                return s[3:].strip()

        # 3) First '### '
        for line in filtered_lines:
            s = line.strip()
            if s.startswith("### "):
                return s[4:].strip()

        logger.info(f"Text for create title: {text}")

        response = await create_title(text)

        logger.info(f"Response for create title: {response}")

        title = (response or {}).get("title")
        if isinstance(title, str) and title.strip():
            return title.strip()

        # 4) Fallback to filename
        return os.path.splitext(os.path.basename(file_path))[0]

    
    def _extract_document_metadata(self, content: str, file_path: str) -> Dict[str, Any]:
        """Extract metadata from document content."""
        metadata = {
            "file_path": file_path,
            "file_size": len(content),
            "ingestion_date": datetime.now().isoformat()
        }
        
        # Try to extract YAML frontmatter
        if content.startswith('---'):
            try:
                import yaml
                end_marker = content.find('\n---\n', 4)
                if end_marker != -1:
                    frontmatter = content[4:end_marker]
                    yaml_metadata = yaml.safe_load(frontmatter)
                    if isinstance(yaml_metadata, dict):
                        metadata.update(yaml_metadata)
            except ImportError:
                logger.warning("PyYAML not installed, skipping frontmatter extraction")
            except Exception as e:
                logger.warning(f"Failed to parse frontmatter: {e}")
        
        # Extract some basic metadata from content
        lines = content.split('\n')
        metadata['line_count'] = len(lines)
        metadata['word_count'] = len(content.split())
        
        return metadata
    
    async def _save_to_postgres(
        self,
        title: str,
        source: str,
        content: str,
        chunks: List[DocumentChunk],
        metadata: Dict[str, Any]
    ) -> str:
        """Save document and chunks to PostgreSQL."""
        async with db_pool.acquire() as conn:
            async with conn.transaction():
                # Insert document
                document_result = await conn.fetchrow(
                    """
                    INSERT INTO documents (title, source, content, metadata)
                    VALUES ($1, $2, $3, $4)
                    RETURNING id::text
                    """,
                    title,
                    source,
                    content,
                    json.dumps(metadata)
                )
                
                document_id = document_result["id"]
                
                # Insert chunks
                for chunk in chunks:
                    # Convert embedding to PostgreSQL vector string format
                    embedding_data = None
                    if hasattr(chunk, 'embedding') and chunk.embedding:
                        # PostgreSQL vector format: '[1.0,2.0,3.0]' (no spaces after commas)
                        # converts something like [0.12, -0.3, 1.5] to "[0.12,-0.3,1.5]"
                        # map(str, chunk.embedding) converts each number to a string, joins by comma and adds bracket at end
                        embedding_data = '[' + ','.join(map(str, chunk.embedding)) + ']'
                    
                    await conn.execute(
                        """
                        INSERT INTO chunks (document_id, content, embedding, chunk_index, metadata, token_count)
                        VALUES ($1::uuid, $2, $3::vector, $4, $5, $6)
                        """,
                        document_id,
                        chunk.content,
                        embedding_data,
                        chunk.index,
                        json.dumps(chunk.metadata),
                        chunk.token_count
                    )
                
                return document_id
    
    async def _clean_databases(self):
        """Clean existing data from databases."""
        logger.warning("Cleaning existing data from databases...")
        
        # Clean PostgreSQL
        async with db_pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("DELETE FROM chunks")
                await conn.execute("DELETE FROM documents")
        
        logger.info("Cleaned PostgreSQL database")

async def main():
    """Main function for running ingestion."""
    parser = argparse.ArgumentParser(description="Ingest documents into vector DB")
    default_documents_dir = _project_root() / "api" / "documents"
    parser.add_argument(
        "--documents",
        "-d",
        default=str(default_documents_dir),
        help="Documents folder path (default: api/documents)",
    )
    parser.add_argument("--no-clean", action="store_true", help="Skip cleaning existing data before ingestion (default: cleans automatically)")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size for splitting documents")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap size")
    parser.add_argument("--no-semantic", action="store_true", help="Disable semantic chunking")
    # Graph-related arguments removed
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    await run_ingestion(
        documents=str(args.documents),
        no_clean=args.no_clean,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        no_semantic=args.no_semantic,
        verbose=args.verbose,
    )


async def run_ingestion(
    document_path: Optional[str] = None,
    documents: Optional[str] = None,
    no_clean: bool = True,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    no_semantic: bool = False,
    verbose: bool = False,
    is_image_enabled: bool = False,
) -> List[IngestionResult]:
    """
    Programmatic entrypoint for running the ingestion pipeline (used by API and CLI).

    Keeps the same default behavior as `main()`:
    - documents defaults to `api/documents`
    - cleans existing data by default (unless no_clean=True)
    - semantic chunking enabled by default (unless no_semantic=True)
    """
    documents_dir: Optional[str]
    if document_path is None:
        default_documents_dir = _project_root() / "api" / "documents"
        documents_dir = str(default_documents_dir) if documents is None else str(documents)
    else:
        documents_dir = None

    # Configure logging only if no handlers exist (avoid overriding app-level config)
    log_level = logging.DEBUG if verbose else logging.INFO
    if not logging.getLogger().handlers:
        try:
            logging.basicConfig(
                level=log_level,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                force=True,  # avoid duplicate handlers in IDEs/reloaders
            )
        except TypeError:
            logging.basicConfig(
                level=log_level,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )

    # Create ingestion configuration
    config = IngestionConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        use_semantic_chunking=not no_semantic
    )

    # Create and run pipeline - clean by default unless --no-clean is specified
    pipeline = DocumentIngestionPipeline(
        config=config,
        clean_before_ingest=not no_clean,  # Clean by default
        is_image_enabled=is_image_enabled,
    )
    
    def progress_callback(current: int, total: int):
        print(f"Progress: {current}/{total} documents processed")
    
    try:
        trace_ctx = start_trace(
            name="ingest.run",
            session_id=uuid4().hex,
            input={
                "document_path": document_path,
                "documents_dir": documents_dir,
            },
            metadata={
                "clean_before_ingest": not no_clean,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "no_semantic": no_semantic,
                "is_image_enabled": is_image_enabled,
            },
            tags=["ingest"],
        )
        start_time = datetime.now()
        
        results = await pipeline.ingest_documents(
            document_path=document_path,
            documents_folder=documents_dir,
            progress_callback=progress_callback,
        )
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()

        end_trace(
            trace_ctx,
            output={
                "documents_processed": len(results),
                "document_titles": [r.title for r in results],
                "total_chunks": sum(r.chunks_created for r in results),
                "total_errors": sum(len(r.errors) for r in results),
                "total_time_seconds": total_time,
            },
        )
        
        # Log summary
        logger.info("=" * 50)
        logger.info("INGESTION SUMMARY")
        logger.info("=" * 50)
        logger.info("Documents processed: %s", len(results))
        logger.info("Document titles: %s", [r.title for r in results])
        logger.info("Total chunks created: %s", sum(r.chunks_created for r in results))
        # Graph-related stats removed
        logger.info("Total errors: %s", sum(len(r.errors) for r in results))
        logger.info("Total processing time: %.2f seconds", total_time)
        
        # Log individual results
        for result in results:
            status = "✓" if not result.errors else "✗"
            logger.info("%s %s: %s chunks", status, result.title, result.chunks_created)
            
            if result.errors:
                for error in result.errors:
                    logger.warning("Error: %s", error)
        
    except KeyboardInterrupt:
        print("\nIngestion interrupted by user")
    except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            end_trace(trace_ctx, error=e)
            raise
    finally:
        await pipeline.close()

    return results


if __name__ == "__main__":
    asyncio.run(main())
