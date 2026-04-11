from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import requests
from openai import APITimeoutError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent_tools import EvidenceCollector, get_default_query_templates, retrieve_local_evidence
from src.commercial_api import COMMERCIAL_MODELS, call_commercial_llm
from src.config import build_embeddings, build_llm, build_reranker, get_settings
from src.rag_pipeline import (
    ANSWER_PROMPT,
    NO_DOCUMENT_MESSAGE,
    NO_REGULATION_MESSAGE,
    ProcurementRAGPipeline,
)
from src.vectorstore import ProcurementVectorStore


logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"
DEFAULT_DATASET_NAME = "qa_20_public_procurement_spec_compliance.csv"
DEFAULT_DATASET_CANDIDATES = (
    EVAL_DIR / DEFAULT_DATASET_NAME,
    EVAL_DIR / "datasets" / DEFAULT_DATASET_NAME,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate procurement-review answers for a QA dataset.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Optional CSV path. Defaults to the bundled QA dataset.",
    )
    parser.add_argument(
        "--api-mode",
        choices=tuple(COMMERCIAL_MODELS),
        default=None,
        help="Use a commercial API for the final answer step while keeping retrieval local.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help=(
            "Local model to run when --api-mode is not set. "
            "With --api-mode, this only overrides the output label."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of questions to process.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore any existing answers file and regenerate from scratch.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts for timeout failures.",
    )
    parser.add_argument(
        "--reranker-key",
        default=None,
        help="Optional reranker profile key. Defaults to the configured default.",
    )
    return parser.parse_args()


def resolve_dataset_path(explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        if not explicit_path.exists():
            raise FileNotFoundError(f"Dataset not found: {explicit_path}")
        return explicit_path

    for candidate in DEFAULT_DATASET_CANDIDATES:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find the QA dataset. "
        f"Tried: {', '.join(str(path) for path in DEFAULT_DATASET_CANDIDATES)}"
    )


def derive_model_label(
    explicit_label: str | None,
    api_mode: str | None,
    local_model_name: str,
) -> str:
    if explicit_label and api_mode:
        return explicit_label
    if api_mode:
        return api_mode
    return local_model_name


def slugify_model_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-") or "model"


def resolve_answers_path(model_label: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / f"answers_{slugify_model_label(model_label)}.json"


def load_dataset_rows(dataset_path: Path, limit: int | None = None) -> list[dict[str, str]]:
    with dataset_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    if limit is not None:
        return rows[:limit]
    return rows


def load_existing_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return payload


def save_results(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(rows, file, ensure_ascii=False, indent=2)


def is_timeout_error(exc: Exception) -> bool:
    timeout_type_names = {
        "APITimeoutError",
        "ReadTimeout",
        "ConnectTimeout",
        "TimeoutException",
        "TimeoutError",
    }
    if isinstance(exc, requests.Timeout | TimeoutError | APITimeoutError):
        return True
    if exc.__class__.__name__ in timeout_type_names:
        return True
    return "timed out" in str(exc).lower()


def run_with_retries(
    operation: Callable[[], dict[str, Any]],
    *,
    description: str,
    max_retries: int,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return operation()
        except Exception as exc:
            last_error = exc
            if not is_timeout_error(exc) or attempt >= max_retries:
                raise
            backoff_seconds = min(5, attempt * 2)
            logger.warning(
                "Timeout during %s (attempt %s/%s): %s. Retrying in %ss.",
                description,
                attempt,
                max_retries,
                exc,
                backoff_seconds,
            )
            time.sleep(backoff_seconds)
    assert last_error is not None
    raise last_error


def build_vectorstores(settings: Any) -> tuple[ProcurementVectorStore, ProcurementVectorStore]:
    embeddings = build_embeddings()
    document_store = ProcurementVectorStore(
        settings=settings,
        embeddings=embeddings,
        collection_name=settings.chroma_collection_name,
    )
    regulations_store = ProcurementVectorStore(
        settings=settings,
        embeddings=embeddings,
        collection_name=settings.regulations_collection_name,
    )
    return document_store, regulations_store


def build_context_only_response(
    *,
    question: str,
    document_store: ProcurementVectorStore,
    regulations_store: ProcurementVectorStore,
    reranker: Any,
    settings: Any,
    api_mode: str,
) -> tuple[str, list[Any], list[Any]]:
    helper = ProcurementRAGPipeline.__new__(ProcurementRAGPipeline)
    normalized_question = question.strip()

    if document_store.get_stats()["chunk_count"] == 0:
        return NO_DOCUMENT_MESSAGE, [], []
    if regulations_store.get_stats()["chunk_count"] == 0 and not settings.use_mcp:
        return NO_REGULATION_MESSAGE, [], []

    collector = EvidenceCollector()
    retrieve_local_evidence(
        query=normalized_question,
        scope="all",
        top_k=settings.rerank_top_k,
        document_store=document_store,
        regulations_store=regulations_store,
        reranker=reranker,
        collector=collector,
        settings=settings,
        query_templates=get_default_query_templates(),
    )

    if not collector.document_sources:
        return NO_DOCUMENT_MESSAGE, list(collector.regulation_sources), []

    prompt = ANSWER_PROMPT.format(
        question=normalized_question,
        regulations_context=helper._format_context(
            list(collector.regulation_sources),
            empty_message="관련 규정 근거 없음",
        ),
        document_context=helper._format_context(
            list(collector.document_sources),
            empty_message="관련 조달 문서 근거 없음",
        ),
    )
    raw_answer = call_commercial_llm(
        api_mode=api_mode,
        prompt=prompt,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.request_timeout,
        temperature=settings.llm_temperature,
    )
    answer = helper._sanitize_final_answer(raw_answer)
    return answer, list(collector.regulation_sources), list(collector.document_sources)


def build_record_from_pipeline_response(response: Any, ground_truth: str) -> dict[str, Any]:
    contexts = [
        document.page_content
        for document in [*response.regulation_sources, *response.document_sources]
    ]
    return {
        "question": response.question,
        "answer": response.answer,
        "contexts": contexts,
        "ground_truth": ground_truth,
    }


def build_record_from_commercial_answer(
    *,
    question: str,
    answer: str,
    regulation_sources: list[Any],
    document_sources: list[Any],
    ground_truth: str,
) -> dict[str, Any]:
    contexts = [
        document.page_content for document in [*regulation_sources, *document_sources]
    ]
    return {
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "ground_truth": ground_truth,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = parse_args()
    dataset_path = resolve_dataset_path(args.dataset_path)
    dataset_rows = load_dataset_rows(dataset_path, limit=args.limit)
    settings = get_settings()
    local_model_name = args.model_name or settings.llm_model_name
    runtime_settings = replace(
        settings,
        llm_model_name=local_model_name,
        use_mcp=False,
    )
    reranker_key = args.reranker_key or runtime_settings.default_reranker_key
    model_label = derive_model_label(args.model_name, args.api_mode, local_model_name)
    answers_path = resolve_answers_path(model_label)
    existing_results = [] if args.overwrite else load_existing_results(answers_path)
    existing_by_question = {
        str(item.get("question", "")).strip(): item for item in existing_results
    }

    document_store, regulations_store = build_vectorstores(runtime_settings)
    reranker = build_reranker(reranker_key)
    local_pipeline = None
    if args.api_mode is None:
        local_pipeline = ProcurementRAGPipeline(
            document_store=document_store,
            regulations_store=regulations_store,
            reranker=reranker,
            llm=build_llm(local_model_name),
            settings=runtime_settings,
        )

    ordered_results: list[dict[str, Any]] = []
    total_questions = len(dataset_rows)
    logger.info(
        "Starting answer generation | dataset=%s | total=%s | model_label=%s | local_model=%s | api_mode=%s",
        dataset_path,
        total_questions,
        model_label,
        local_model_name if args.api_mode is None else "-",
        args.api_mode or "local",
    )

    for index, row in enumerate(dataset_rows, start=1):
        question = row["question"].strip()
        ground_truth = row["ground_truth"].strip()
        if question in existing_by_question:
            cached_record = dict(existing_by_question[question])
            cached_record["ground_truth"] = ground_truth
            ordered_results.append(cached_record)
            logger.info("[%s/%s] Skipping cached question.", index, total_questions)
            continue

        logger.info("[%s/%s] Generating answer.", index, total_questions)
        started_at = time.perf_counter()

        def operation() -> dict[str, Any]:
            if local_pipeline is not None:
                response = local_pipeline.invoke(question)
                return build_record_from_pipeline_response(response, ground_truth)

            answer, regulation_sources, document_sources = build_context_only_response(
                question=question,
                document_store=document_store,
                regulations_store=regulations_store,
                reranker=reranker,
                settings=runtime_settings,
                api_mode=args.api_mode,
            )
            return build_record_from_commercial_answer(
                question=question,
                answer=answer,
                regulation_sources=regulation_sources,
                document_sources=document_sources,
                ground_truth=ground_truth,
            )

        record = run_with_retries(
            operation,
            description=f"question {index}/{total_questions}",
            max_retries=args.max_retries,
        )
        ordered_results.append(record)
        save_results(answers_path, ordered_results)
        elapsed = time.perf_counter() - started_at
        logger.info(
            "[%s/%s] Saved answer to %s (elapsed %.1fs).",
            index,
            total_questions,
            answers_path,
            elapsed,
        )

    logger.info("Finished answer generation: %s", answers_path)
    print(f"answers_path={answers_path}")
    print(f"model_label={model_label}")


if __name__ == "__main__":
    main()
