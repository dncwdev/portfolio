from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path


EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"
RAGAS_RESULTS_CSV = RESULTS_DIR / "ragas_results.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run answer generation and RAGAS evaluation for the procurement-review pipeline.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Optional QA dataset path.",
    )
    parser.add_argument(
        "--api-mode",
        choices=("openai", "cohere", "anthropic"),
        default=None,
        help="Use a commercial API for answer generation.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Optional model label override.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of QA rows to process.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate answers instead of resuming from an existing JSON file.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts for timeout failures during answer generation.",
    )
    parser.add_argument(
        "--reranker-key",
        default=None,
        help="Optional reranker profile key.",
    )
    return parser.parse_args()


def derive_model_label(explicit_label: str | None, api_mode: str | None) -> str:
    if explicit_label:
        return explicit_label
    if api_mode:
        return api_mode
    return os.getenv("LLM_MODEL") or os.getenv("LLM_MODEL_NAME") or "local_model"


def slugify_model_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-") or "model"


def resolve_answers_path(model_label: str) -> Path:
    return RESULTS_DIR / f"answers_{slugify_model_label(model_label)}.json"


def run_script(script_name: str, extra_args: list[str]) -> None:
    script_path = EVAL_DIR / script_name
    command = [sys.executable, str(script_path), *extra_args]
    subprocess.run(command, check=True)


def print_summary(model_label: str) -> None:
    answers_path = resolve_answers_path(model_label)
    print(f"answers_path={answers_path}")

    if not RAGAS_RESULTS_CSV.exists():
        print("RAGAS summary unavailable: ragas_results.csv was not created.")
        return

    with RAGAS_RESULTS_CSV.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    summary_row = None
    for row in rows:
        if row.get("model_name") == model_label:
            summary_row = row
            break

    if summary_row is None:
        print("RAGAS evaluation was skipped or no matching result row was found.")
        return

    table_rows = [
        ("model_name", summary_row["model_name"]),
        ("faithfulness", summary_row["faithfulness"]),
        ("answer_relevancy", summary_row["answer_relevancy"]),
        ("context_precision", summary_row["context_precision"]),
        ("context_recall", summary_row["context_recall"]),
    ]
    key_width = max(len(key) for key, _ in table_rows)
    value_width = max(len(value) for _, value in table_rows)
    border = f"+-{'-' * key_width}-+-{'-' * value_width}-+"
    print(border)
    for key, value in table_rows:
        print(f"| {key.ljust(key_width)} | {value.ljust(value_width)} |")
    print(border)


def main() -> None:
    args = parse_args()
    model_label = derive_model_label(args.model_name, args.api_mode)

    generate_args: list[str] = []
    if args.dataset_path is not None:
        generate_args.extend(["--dataset-path", str(args.dataset_path)])
    if args.api_mode is not None:
        generate_args.extend(["--api-mode", args.api_mode])
    if args.model_name is not None:
        generate_args.extend(["--model-name", args.model_name])
    if args.limit is not None:
        generate_args.extend(["--limit", str(args.limit)])
    if args.overwrite:
        generate_args.append("--overwrite")
    if args.max_retries != 3:
        generate_args.extend(["--max-retries", str(args.max_retries)])
    if args.reranker_key is not None:
        generate_args.extend(["--reranker-key", args.reranker_key])

    run_script("generate_answers.py", generate_args)

    if os.getenv("OPENAI_API_KEY"):
        ragas_args: list[str] = []
        if args.api_mode is not None:
            ragas_args.extend(["--api-mode", args.api_mode])
        if args.model_name is not None:
            ragas_args.extend(["--model-name", args.model_name])
        run_script("ragas_evaluate.py", ragas_args)
    else:
        print("WARNING: OPENAI_API_KEY is not set. Skipping ragas_evaluate.py.")

    print_summary(model_label)


if __name__ == "__main__":
    main()
