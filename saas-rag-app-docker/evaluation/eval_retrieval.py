import asyncio
import csv
import json
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    AnswerRelevancyMetric,
    FaithfulnessMetric
)
from deepeval.test_case import LLMTestCase
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from api import rag_agent


DATASET_PATH = "rag_eval_dataset_final.csv"
RETRIEVAL_LIMIT = int(os.getenv("RAG_EVAL_RETRIEVAL_LIMIT", "5"))
MAX_CONCURRENCY = int(os.getenv("RAG_EVAL_CONCURRENCY", "2"))
EMPTY_CONTEXT_SENTINEL = "NO_RETRIEVAL_CONTEXT"
RESULTS_DIR = Path(os.getenv("RAG_EVAL_RESULTS_DIR", "eval_results"))
TESTCASES_CACHE_PATH = Path(os.getenv("RAG_EVAL_TESTCASES_CACHE", "eval_results/test_cases_cache.json"))
USE_CACHED_TESTCASES = os.getenv("RAG_EVAL_USE_CACHED_TESTCASES", "false")
DEEPEVAL_MAX_CONCURRENT = int(os.getenv("DEEPEVAL_MAX_CONCURRENT", "1"))
DEEPEVAL_THROTTLE_SEC = float(os.getenv("DEEPEVAL_THROTTLE_SEC", "5.0"))
DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE = float(
    os.getenv("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "180")
)
DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE = float(
    os.getenv("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", "1800")
)
DEEPEVAL_TASK_GATHER_BUFFER_SECONDS_OVERRIDE = float(
    os.getenv("DEEPEVAL_TASK_GATHER_BUFFER_SECONDS_OVERRIDE", "120")
)
DEEPEVAL_BATCH_SIZE = int(os.getenv("DEEPEVAL_BATCH_SIZE", "5"))

EVAL_SYSTEM_PROMPT = """You are an intelligent knowledge assistant with access to some documentation and information.
Your role is to help users find accurate information from the knowledge base.
You have a professional yet friendly demeanor.

IMPORTANT: 
1. Always search the knowledge base before answering questions about specific information.
2. When you first look at the documentation, always start with RAG (provided by the 'search_knowledge_base' tool)
3. If it helps, then you may also always check the list of titles of all available documents (provided by the 'find_all_titles' tool) and retrieve the content of a relevant document (using the 'find_content_by_title' tool).
If information isn't in the knowledge base, clearly state that and offer general guidance.
Be concise but thorough in your responses.
Ask clarifying questions if the user's query is ambiguous.
When you find relevant information, synthesize it clearly and cite the source documents."""


def _extract_output_text(response: Any) -> str:
    if hasattr(response, "data"):
        return str(response.data)
    for attr in ("output", "output_text", "text"):
        if hasattr(response, attr):
            return str(getattr(response, attr))
    return str(response)


def _contexts_from_chunks(chunks: list[dict[str, Any]]) -> list[str]:
    contexts: list[str] = []
    for chunk in chunks:
        content = str(chunk.get("content", "")).strip()
        if not content:
            continue
        title = chunk.get("document_title")
        if title:
            contexts.append(f"[Source: {title}]\n{content}")
        else:
            contexts.append(content)
    return contexts


def _select_contexts(capture: dict[str, list[str]]) -> list[str]:
    if capture["search_contexts"]:
        return capture["search_contexts"]
    if capture["doc_contexts"]:
        return capture["doc_contexts"]
    return [EMPTY_CONTEXT_SENTINEL]


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _serialize_test_case(test_case: LLMTestCase) -> dict[str, Any]:
    return {
        "input": test_case.input,
        "actual_output": test_case.actual_output,
        "expected_output": test_case.expected_output,
        "retrieval_context": test_case.retrieval_context,
    }


def _deserialize_test_case(data: dict[str, Any]) -> LLMTestCase:
    return LLMTestCase(
        input=data["input"],
        actual_output=data["actual_output"],
        expected_output=data["expected_output"],
        retrieval_context=data["retrieval_context"],
    )


def _is_retriable_error(e: Exception) -> bool:
    try:
        from openai import (
            APIConnectionError,
            APIError,
            APITimeoutError,
            InternalServerError,
            RateLimitError,
        )
        if isinstance(
            e,
            (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError, APIError),
        ):
            return True
    except Exception:
        pass

    msg = str(e).lower()
    if "rate limit" in msg or "429" in msg:
        return True
    if "timeout" in msg or "timed out" in msg:
        return True
    if "server error" in msg or "502" in msg or "503" in msg or "504" in msg:
        return True

    return False


@retry(
    retry=retry_if_exception(_is_retriable_error),
    stop=stop_after_attempt(6),
    wait=wait_random_exponential(min=1, max=30),
    reraise=True,
)
async def _run_with_retry(agent: Agent, prompt: str) -> Any:
    return await agent.run(prompt)


def _build_eval_agent(capture: dict[str, list[str]]) -> Agent:
    async def search_eval(ctx: RunContext[None], query: str, limit: int = RETRIEVAL_LIMIT) -> str:
        chunks = await rag_agent.retrieve_chunks(query, limit)
        capture["search_contexts"].extend(_contexts_from_chunks(chunks))
        return await rag_agent.search_knowledge_base(ctx, query, limit)

    async def find_all_titles_eval(ctx: RunContext[None]) -> list[str]:
        return await rag_agent.find_all_titles(ctx)

    async def find_content_eval(ctx: RunContext[None], title: str) -> str:
        result = await rag_agent.find_content_by_title(ctx, title)
        if result:
            capture["doc_contexts"].append(result)
        return result

    return Agent(
        "openai:gpt-4o-mini",
        system_prompt=EVAL_SYSTEM_PROMPT,
        tools=[search_eval, find_all_titles_eval, find_content_eval],
        model_settings={"temperature": 0},
    )


class _TeeStream:
    def __init__(self, primary, secondary) -> None:
        self._primary = primary
        self._secondary = secondary

    def write(self, data: str) -> int:
        self._primary.write(data)
        return self._secondary.write(data)

    def flush(self) -> None:
        self._primary.flush()
        self._secondary.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._primary, "isatty", lambda: False)())


async def _build_test_case(row: dict[str, str], semaphore: asyncio.Semaphore) -> tuple[int, LLMTestCase]:
    async with semaphore:
        capture = {"search_contexts": [], "doc_contexts": []}
        eval_agent = _build_eval_agent(capture)

        response = await _run_with_retry(eval_agent, row["input"])
        actual_output = _extract_output_text(response)
        retrieval_context = _select_contexts(capture)

        test_case = LLMTestCase(
            input=row["input"],
            actual_output=actual_output,
            expected_output=row["expected_output"],
            retrieval_context=retrieval_context,
        )
        return int(row["index"]), test_case


async def run_evaluation() -> None:
    load_dotenv()

    await rag_agent.initialize_db()
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    use_cached = _parse_bool(USE_CACHED_TESTCASES)
    if use_cached and TESTCASES_CACHE_PATH.exists():
        cached = json.loads(TESTCASES_CACHE_PATH.read_text(encoding="utf-8"))
        test_cases = [_deserialize_test_case(item) for item in cached]
    else:
        tasks: list[asyncio.Task[tuple[int, LLMTestCase]]] = []
        with open(DATASET_PATH, newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            for row in reader:
                tasks.append(asyncio.create_task(_build_test_case(row, semaphore)))

        results = await asyncio.gather(*tasks)
        results.sort(key=lambda item: item[0])
        test_cases = [item[1] for item in results]

        TESTCASES_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        serialized = [_serialize_test_case(tc) for tc in test_cases]
        TESTCASES_CACHE_PATH.write_text(
            json.dumps(serialized, indent=2, default=str),
            encoding="utf-8",
        )

    def _build_metrics() -> list:
        return [
            ContextualPrecisionMetric(model="gpt-4o-mini"),
            ContextualRecallMetric(model="gpt-4o-mini"),
            # ContextualRelevancyMetric(model="gpt-4o-mini"),
            AnswerRelevancyMetric(model="gpt-4o-mini"),
            FaithfulnessMetric(model="gpt-4o-mini"),
        ]

    metric_names = [metric.__class__.__name__ for metric in _build_metrics()]

    async_config = AsyncConfig(
        run_async=False,
        max_concurrent=DEEPEVAL_MAX_CONCURRENT,
        throttle_value=DEEPEVAL_THROTTLE_SEC,
    )

    os.environ.setdefault(
        "DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE",
        str(DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE),
    )
    os.environ.setdefault(
        "DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE",
        str(DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE),
    )
    os.environ.setdefault(
        "DEEPEVAL_TASK_GATHER_BUFFER_SECONDS_OVERRIDE",
        str(DEEPEVAL_TASK_GATHER_BUFFER_SECONDS_OVERRIDE),
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_path = RESULTS_DIR / f"retrieval_eval_{timestamp}.log"
    console_path = RESULTS_DIR / f"retrieval_eval_{timestamp}_console.log"

    # all_results = []
    # with console_path.open("w", encoding="utf-8") as console_file:
    #     stdout_tee = _TeeStream(sys.stdout, console_file)
    #     stderr_tee = _TeeStream(sys.stderr, console_file)
    #     with redirect_stdout(stdout_tee), redirect_stderr(stderr_tee):
    #         for start in range(0, len(test_cases), DEEPEVAL_BATCH_SIZE):
    #             batch = test_cases[start : start + DEEPEVAL_BATCH_SIZE]
    #             batch_results = evaluate(
    #                 test_cases=batch,
    #                 metrics=_build_metrics(),
    #                 async_config=async_config,
    #             )
    #             all_results.append(batch_results)

    with console_path.open("w", encoding="utf-8") as console_file:
        stdout_tee = _TeeStream(sys.stdout, console_file)
        stderr_tee = _TeeStream(sys.stderr, console_file)
        with redirect_stdout(stdout_tee), redirect_stderr(stderr_tee):
            results = evaluate(
                test_cases=test_cases,
                metrics=_build_metrics(),
                async_config=async_config,
            )

    log_lines = [
        f"timestamp_utc={timestamp}",
        f"dataset_path={DATASET_PATH}",
        f"retrieval_limit={RETRIEVAL_LIMIT}",
        f"max_concurrency={MAX_CONCURRENCY}",
        f"use_cached_testcases={use_cached}",
        f"testcases_cache_path={TESTCASES_CACHE_PATH}",
        f"deepeval_max_concurrent={DEEPEVAL_MAX_CONCURRENT}",
        f"deepeval_throttle_sec={DEEPEVAL_THROTTLE_SEC}",
        f"deepeval_batch_size={DEEPEVAL_BATCH_SIZE}",
        f"metrics={', '.join(metric_names)}",
        f"console_log_path={console_path}",
        f"results={results}",
    ]
    results_path.write_text("\n".join(log_lines), encoding="utf-8")

    await rag_agent.close_db()


if __name__ == "__main__":
    asyncio.run(run_evaluation())
