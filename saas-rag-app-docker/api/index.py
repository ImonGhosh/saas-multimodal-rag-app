from unittest import result
from fastapi import FastAPI  # type: ignore
from fastapi import BackgroundTasks, HTTPException  # type: ignore
from fastapi import UploadFile, File, Form  # type: ignore
from fastapi import Depends  # type: ignore
from fastapi.responses import PlainTextResponse  # type: ignore
from fastapi.responses import JSONResponse  # type: ignore
from fastapi_clerk_auth import ClerkConfig, ClerkHTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel  # type: ignore
from openai import OpenAI  # type: ignore
from supabase import create_client, Client
from openai import AsyncOpenAI

from dotenv import load_dotenv
from pathlib import Path
import os
import asyncio
from uuid import uuid4
import logging
from fastapi.middleware.cors import CORSMiddleware

# from . import rag_agent_web
from . import rag_agent
from . import crawl_web
from .file_data_ingestion import ingest as file_data_ingest
from .utils.logging_config import init_logging
from .utils.observability import start_trace, end_trace, text_payload, store_prompts, store_responses

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")
init_logging()

clerk_config = ClerkConfig(jwks_url=os.getenv("CLERK_JWKS_URL"))
clerk_guard = ClerkHTTPBearer(clerk_config)

# api_key = os.getenv("OPENAI_API_KEY")
# client = OpenAI(api_key=api_key)

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = Client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Prepare dependencies
# DEPS = rag_agent_web.PydanticAIDeps(
#     supabase=supabase,
#     openai_client=openai_client
# )

app = FastAPI()

_INGEST_JOBS: dict[str, dict[str, object]] = {}
_INGEST_LOCK = asyncio.Lock()
logger = logging.getLogger(__name__)


def _extract_tool_names(messages) -> list[str]:
    try:
        from pydantic_ai.messages import ToolCallPart, BuiltinToolCallPart
    except Exception:
        return []

    tool_names: list[str] = []
    for message in messages:
        parts = getattr(message, "parts", None) or []
        for part in parts:
            if isinstance(part, (ToolCallPart, BuiltinToolCallPart)):
                tool_names.append(part.tool_name)
    return tool_names


def _unique_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_items: list[str] = []
    for item in items:
        if item not in seen:
            unique_items.append(item)
            seen.add(item)
    return unique_items


async def _run_ingest_file_job(job_id: str) -> None:
    job = _INGEST_JOBS.get(job_id)
    if job is None:
        return

    job["status"] = "running"

    try:
        async with _INGEST_LOCK:
            document_path = job.get("document_path")
            if not document_path:
                raise ValueError("Missing document_path for ingestion job")
            is_image_enabled = bool(job.get("is_image_enabled", False))
            await file_data_ingest.run_ingestion(
                document_path=str(document_path),
                is_image_enabled=is_image_enabled,
            )
        job["status"] = "succeeded"
    except Exception as err:
        job["status"] = "failed"
        job["error"] = str(err)


class IdeaRequest(BaseModel):
    text: str


class IngestRequest(BaseModel):
    url: str

# Prevents accidental overwrites by saving as name (1).ext, name (2).ext, etc. if the filename already exists.
def _unique_path(path: Path) -> Path:
    """Return a non-existing path by suffixing ' (n)' if needed."""
    if not path.exists():
        return path

    
    # Splits filename into: stem: name without extension (report) and suffix: extension (.pdf)
    # Tries: report (1).pdf, report (2).pdf… until it finds a name that doesn’t exist.
    stem, suffix = path.stem, path.suffix
    i = 1
    while True:
        candidate = path.with_name(f"{stem} ({i}){suffix}")
        if not candidate.exists():
            return candidate
        i += 1

# Configure CORS (allowed urls that can access this API)
origins = [
    "http://localhost:3000", "http://127.0.0.1:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,   # optional but commonly needed
    allow_methods=["*"],      # Allow all HTTP methods(GET, POST etc.)
    allow_headers=["*"],      # IMPORTANT for JSON requests (Content-Type)
)

@app.post("/api", response_class=PlainTextResponse)
async def idea(
    payload: IdeaRequest,
    creds: HTTPAuthorizationCredentials = Depends(clerk_guard),
):

    request_id = uuid4().hex
    trace_ctx = start_trace(
        name="rag.query",
        session_id=request_id,
        input=text_payload(payload.text, store=store_prompts()),
        metadata={"endpoint": "/api", "request_id": request_id},
        tags=["fastapi", "rag"],
    )

    try:
        response = await rag_agent.agent.run(payload.text)

        try:
            tool_names = _unique_preserve_order(_extract_tool_names(response.new_messages()))
            if tool_names:
                logger.info("Tools used for run: %s", ", ".join(tool_names))
            else:
                logger.info("Tools used for run: none")
        except Exception as err:
            tool_names = []
            logger.warning("Failed to extract tool usage: %s", err)

        if hasattr(response, "data"):
            output_text = str(response.data)
        else:
            output_text = None
            for attr in ("output", "output_text", "text"):
                if hasattr(response, attr):
                    output_text = str(getattr(response, attr))
                    break
            if output_text is None:
                output_text = str(response)

        end_trace(
            trace_ctx,
            output=text_payload(output_text, store=store_responses()),
            metadata={"tools_used": tool_names},
        )
        return output_text
    except Exception as err:
        end_trace(trace_ctx, error=err)
        raise


@app.post("/ingest", response_class=PlainTextResponse)
async def ingest(
    payload: IngestRequest,
    creds: HTTPAuthorizationCredentials = Depends(clerk_guard),
):
    try:
        md_path = await crawl_web.crawl_data(payload.url)

        # After crawl saves markdown into api/documents, run the file ingestion pipeline over those files.
        await file_data_ingest.run_ingestion(
            document_path=md_path,
            is_image_enabled=False,
        )

        return "Sucessfully ingested website data"
    except Exception as err:
        return f"Error: {err}"


@app.post("/ingest-file", response_class=JSONResponse)
async def ingest_file(
    background_tasks: BackgroundTasks,
    creds: HTTPAuthorizationCredentials = Depends(clerk_guard),
    file: UploadFile = File(...),
    isImageEnabled: bool = Form(False),
):
    """
    Receives an uploaded file and stores it under the repo-relative `api/documents/` folder.
    """
    documents_dir = ROOT_DIR / "api" / "documents"
    documents_dir.mkdir(parents=True, exist_ok=True)

    # Avoid path traversal (e.g. "..\\..\\foo") by keeping only the base name.
    filename = Path(file.filename or "uploaded_file").name
    dest_path = _unique_path(documents_dir / filename)

    try:
        with dest_path.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break
                out.write(chunk)
    finally:
        await file.close()

    job_id = uuid4().hex
    _INGEST_JOBS[job_id] = {
        "status": "queued",
        "document_path": str(dest_path),
        "is_image_enabled": isImageEnabled,
    }
    background_tasks.add_task(_run_ingest_file_job, job_id)

    return JSONResponse(status_code=202, content={"job_id": job_id})


@app.get("/ingest-file/status/{job_id}")
async def ingest_file_status(job_id: str):
    job = _INGEST_JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return JSONResponse(content=job)
    
