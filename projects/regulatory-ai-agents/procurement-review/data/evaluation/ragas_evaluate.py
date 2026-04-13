from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings


logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"
DEFAULT_RESULTS_CSV = RESULTS_DIR / "ragas_results.csv"
DEFAULT_DATASET_NAME = "qa_20_public_procurement_spec_compliance.csv"
DEFAULT_DATASET_CANDIDATES = (
    EVAL_DIR / DEFAULT_DATASET_NAME,
    EVAL_DIR / "datasets" / DEFAULT_DATASET_NAME,
)
EVALUATOR_LLM_MODEL = "gpt-4o"
EVALUATOR_EMBEDDING_MODEL = "text-embedding-3-small"
RAGAS_EVALUATOR_MAX_WORKERS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RAGAS metrics for a generated answers JSON file.",
    )
    parser.add_argument(
        "--answers-path",
        type=Path,
        default=None,
        help="Optional answers JSON path. Defaults to results/answers_{model_name}.json.",
    )
    parser.add_argument(
        "--api-mode",
        choices=("openai", "cohere", "anthropic"),
        default=None,
        help="Used for deriving the default model label when --model-name is not provided.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Optional model label override.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Optional QA dataset path used to inject missing ground_truth values.",
    )
    return parser.parse_args()


def derive_model_label(explicit_label: str | None, api_mode: str | None) -> str:
    if explicit_label:
        return explicit_label
    if api_mode:
        return api_mode
    return (
        os.getenv("LLM_MODEL")
        or os.getenv("LLM_MODEL_NAME")
        or get_settings().llm_model_name
    )


def slugify_model_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-") or "model"


def resolve_answers_path(model_label: str) -> Path:
    return RESULTS_DIR / f"answers_{slugify_model_label(model_label)}.json"


def resolve_per_case_path(answers_path: Path) -> Path:
    return answers_path.with_name(f"{answers_path.stem}_per_case.csv")


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


def load_answers(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Answers file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return payload


def save_answers(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(rows, file, ensure_ascii=False, indent=2)


def load_ground_truth_lookup(dataset_path: Path) -> dict[str, str]:
    with dataset_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        lookup: dict[str, str] = {}
        for row in reader:
            question = str(row.get("question", "")).strip()
            ground_truth = str(row.get("ground_truth", "")).strip()
            if question and ground_truth:
                lookup[question] = ground_truth
    return lookup


def ensure_ground_truth(
    rows: list[dict[str, Any]],
    *,
    answers_path: Path,
    dataset_path: Path | None,
) -> list[dict[str, Any]]:
    missing_indexes = [
        index for index, row in enumerate(rows) if not str(row.get("ground_truth", "")).strip()
    ]
    if not missing_indexes:
        return rows

    resolved_dataset_path = resolve_dataset_path(dataset_path)
    ground_truth_lookup = load_ground_truth_lookup(resolved_dataset_path)
    injected = 0

    for index in missing_indexes:
        question = str(rows[index].get("question", "")).strip()
        ground_truth = ground_truth_lookup.get(question, "").strip()
        if not ground_truth:
            raise ValueError(
                "Missing ground_truth and could not map question from dataset: "
                f"{question}"
            )
        rows[index]["ground_truth"] = ground_truth
        injected += 1

    save_answers(answers_path, rows)
    logger.info(
        "Injected missing ground_truth values into %s using dataset %s (count=%s).",
        answers_path,
        resolved_dataset_path,
        injected,
    )
    return rows


def build_evaluation_dataset(rows: list[dict[str, Any]]) -> EvaluationDataset:
    dataset_rows: list[dict[str, Any]] = []
    for row in rows:
        dataset_rows.append(
            {
                "user_input": row["question"],
                "response": row["answer"],
                "retrieved_contexts": list(row.get("contexts", [])),
                "reference": row["ground_truth"],
            }
        )
    return EvaluationDataset.from_list(dataset_rows)


def average_metric(scores: list[dict[str, Any]], metric_name: str) -> float:
    values: list[float] = []
    for row in scores:
        value = row.get(metric_name)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(numeric):
            continue
        values.append(numeric)
    if not values:
        return float("nan")
    return sum(values) / len(values)


def upsert_results_csv(path: Path, model_name: str, summary: dict[str, float]) -> None:
    fieldnames = [
        "model_name",
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]
    existing_rows: list[dict[str, str]] = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as file:
            existing_rows = list(csv.DictReader(file))

    updated = False
    serialized = {
        "model_name": model_name,
        "faithfulness": f"{summary['faithfulness']:.6f}",
        "answer_relevancy": f"{summary['answer_relevancy']:.6f}",
        "context_precision": f"{summary['context_precision']:.6f}",
        "context_recall": f"{summary['context_recall']:.6f}",
    }
    output_rows: list[dict[str, str]] = []
    for row in existing_rows:
        if row.get("model_name") == model_name:
            output_rows.append(serialized)
            updated = True
        else:
            output_rows.append({field: row.get(field, "") for field in fieldnames})
    if not updated:
        output_rows.append(serialized)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)


def save_per_case_csv(path: Path, result: Any) -> None:
    dataframe = result.to_pandas()
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False, encoding="utf-8-sig")


def print_summary_table(model_name: str, summary: dict[str, float]) -> None:
    rows = [
        ("model_name", model_name),
        ("faithfulness", f"{summary['faithfulness']:.4f}"),
        ("answer_relevancy", f"{summary['answer_relevancy']:.4f}"),
        ("context_precision", f"{summary['context_precision']:.4f}"),
        ("context_recall", f"{summary['context_recall']:.4f}"),
    ]
    key_width = max(len(key) for key, _ in rows)
    value_width = max(len(value) for _, value in rows)
    border = f"+-{'-' * key_width}-+-{'-' * value_width}-+"
    print(border)
    for key, value in rows:
        print(f"| {key.ljust(key_width)} | {value.ljust(value_width)} |")
    print(border)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = parse_args()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("WARNING: OPENAI_API_KEY is not set. Skipping RAGAS evaluation.")
        return

    model_label = derive_model_label(args.model_name, args.api_mode)
    answers_path = args.answers_path or resolve_answers_path(model_label)
    answers = load_answers(answers_path)
    answers = ensure_ground_truth(
        answers,
        answers_path=answers_path,
        dataset_path=args.dataset_path,
    )
    evaluation_dataset = build_evaluation_dataset(answers)

    evaluator_llm = ChatOpenAI(
        model=EVALUATOR_LLM_MODEL,
        api_key=openai_api_key,
        base_url="https://api.openai.com/v1",
        temperature=0,
        max_tokens=2048,
        request_timeout=120,
        max_retries=2,
        use_responses_api=False,
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model=EVALUATOR_EMBEDDING_MODEL,
            api_key=openai_api_key,
            base_url="https://api.openai.com/v1",
            request_timeout=120,
            max_retries=2,
            tiktoken_enabled=False,
            check_embedding_ctx_length=False,
        )
    )

    logger.info(
        "Running RAGAS evaluation for %s | evaluator_llm=%s | embedding=%s",
        answers_path,
        EVALUATOR_LLM_MODEL,
        EVALUATOR_EMBEDDING_MODEL,
    )
    result = evaluate(
        dataset=evaluation_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=LangchainLLMWrapper(evaluator_llm),
        embeddings=evaluator_embeddings,
        run_config=RunConfig(
            timeout=180,
            max_retries=3,
            max_wait=30,
            max_workers=RAGAS_EVALUATOR_MAX_WORKERS,
            exception_types=(Exception,),
            log_tenacity=True,
        ),
        raise_exceptions=False,
        show_progress=True,
    )

    summary = {
        "faithfulness": average_metric(result.scores, "faithfulness"),
        "answer_relevancy": average_metric(result.scores, "answer_relevancy"),
        "context_precision": average_metric(result.scores, "context_precision"),
        "context_recall": average_metric(result.scores, "context_recall"),
    }
    per_case_path = resolve_per_case_path(answers_path)
    save_per_case_csv(per_case_path, result)
    upsert_results_csv(DEFAULT_RESULTS_CSV, model_label, summary)
    print_summary_table(model_label, summary)
    print(f"results_csv={DEFAULT_RESULTS_CSV}")
    print(f"per_case_csv={per_case_path}")


if __name__ == "__main__":
    main()
