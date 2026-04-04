from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent_tools import EvidenceCollector, collect_chunk_ids, retrieve_local_evidence
from src.config import build_embeddings, build_reranker, get_settings
from src.vectorstore import ProcurementVectorStore


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the same retrieval query multiple times and check chunk ID stability.",
    )
    parser.add_argument("--query", required=True, help="Review question to test.")
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of repeated retrieval runs.",
    )
    parser.add_argument(
        "--scope",
        choices=("all", "documents", "regulations"),
        default="all",
        help="Which collection scope to test.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Maximum number of reranked chunks to keep per scope.",
    )
    parser.add_argument(
        "--reranker-key",
        default=None,
        help="Optional reranker profile key. Defaults to the configured default.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
    )
    args = parse_args()
    settings = get_settings()
    reranker_key = args.reranker_key or settings.default_reranker_key
    top_k = args.top_k or settings.rerank_top_k

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
    reranker = build_reranker(reranker_key)

    baseline_documents: list[str] | None = None
    baseline_regulations: list[str] | None = None
    variance_detected = False

    for run_index in range(1, args.runs + 1):
        collector = EvidenceCollector()
        regulation_docs, document_docs = retrieve_local_evidence(
            query=args.query,
            scope=args.scope,
            top_k=top_k,
            document_store=document_store,
            regulations_store=regulations_store,
            reranker=reranker,
            collector=collector,
            settings=settings,
        )
        document_ids = collect_chunk_ids(document_docs)
        regulation_ids = collect_chunk_ids(regulation_docs)

        logger.info(
            "run=%s scope=%s reranker=%s documents=%s regulations=%s",
            run_index,
            args.scope,
            reranker_key,
            document_ids,
            regulation_ids,
        )

        if baseline_documents is None:
            baseline_documents = document_ids
            baseline_regulations = regulation_ids
            continue

        if document_ids != baseline_documents:
            variance_detected = True
            logger.warning(
                "Document chunk variance detected on run %s. baseline=%s current=%s",
                run_index,
                baseline_documents,
                document_ids,
            )

        if regulation_ids != baseline_regulations:
            variance_detected = True
            logger.warning(
                "Regulation chunk variance detected on run %s. baseline=%s current=%s",
                run_index,
                baseline_regulations,
                regulation_ids,
            )

    if variance_detected:
        raise SystemExit(1)

    logger.info("Retrieval determinism check passed for %s runs.", args.runs)


if __name__ == "__main__":
    main()
