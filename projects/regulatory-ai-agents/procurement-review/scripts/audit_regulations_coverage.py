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
from src.domain_queries import get_domain_query_templates
from src.vectorstore import ProcurementVectorStore


logger = logging.getLogger(__name__)

POST_CONTRACT_KEYWORDS = (
    "계약이행",
    "이행",
    "납품",
    "착수",
    "검사",
    "검수",
    "대가지급",
    "지체상금",
    "하자",
    "하자보수",
    "운영",
    "유지보수",
    "준공",
    "사후",
)

PRE_BID_KEYWORDS = (
    "입찰",
    "공고",
    "구매규격서",
    "규격서",
    "참가자격",
    "허가",
    "면허",
    "등록",
    "인증",
    "사전",
    "평가",
    "낙찰자",
    "제안서",
    "법령",
    "고시",
    "계약방법",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit regulations chunks for pre-bidding coverage and post-contract leakage.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top chunks to collect per domain query template.",
    )
    parser.add_argument(
        "--reranker-key",
        default=None,
        help="Optional reranker profile key. Defaults to the configured default.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=20,
        help="How many flagged chunks to print per category.",
    )
    return parser.parse_args()


def classify_chunk(text: str) -> tuple[str, list[str], list[str]]:
    pre_bid_hits = [keyword for keyword in PRE_BID_KEYWORDS if keyword in text]
    post_contract_hits = [keyword for keyword in POST_CONTRACT_KEYWORDS if keyword in text]

    if pre_bid_hits and post_contract_hits:
        return "mixed", pre_bid_hits, post_contract_hits
    if post_contract_hits:
        return "post_contract_only", pre_bid_hits, post_contract_hits
    if pre_bid_hits:
        return "pre_bid_only", pre_bid_hits, post_contract_hits
    return "unknown", pre_bid_hits, post_contract_hits


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
    )
    args = parse_args()
    settings = get_settings()
    reranker_key = args.reranker_key or settings.default_reranker_key

    embeddings = build_embeddings()
    regulations_store = ProcurementVectorStore(
        settings=settings,
        embeddings=embeddings,
        collection_name=settings.regulations_collection_name,
    )
    reranker = build_reranker(reranker_key)

    covered_chunk_ids: set[str] = set()
    coverage_by_template: dict[str, list[str]] = {}
    audit_question = "구매규격서 작성 단계에서 입찰 전 법령상 필수 기재사항을 점검한다."

    for query_template in get_domain_query_templates():
        collector = EvidenceCollector()
        regulation_docs, _ = retrieve_local_evidence(
            query=audit_question,
            scope="regulations",
            top_k=args.top_k,
            document_store=regulations_store,
            regulations_store=regulations_store,
            reranker=reranker,
            collector=collector,
            settings=settings,
            query_templates=(query_template,),
        )
        chunk_ids = collect_chunk_ids(regulation_docs)
        coverage_by_template[query_template] = chunk_ids
        covered_chunk_ids.update(chunk_ids)

    raw = regulations_store.collection.get(
        limit=regulations_store.collection.count(),
        include=["documents", "metadatas"],
    )
    documents = raw.get("documents", [])
    metadatas = raw.get("metadatas", [])

    classifications: dict[str, list[dict[str, object]]] = {
        "pre_bid_only": [],
        "post_contract_only": [],
        "mixed": [],
        "unknown": [],
    }

    for text, metadata in zip(documents, metadatas):
        normalized_text = (text or "").strip()
        item_metadata = dict(metadata or {})
        chunk_id = str(item_metadata.get("chunk_id", "unknown"))
        classification, pre_bid_hits, post_contract_hits = classify_chunk(normalized_text)
        classifications[classification].append(
            {
                "chunk_id": chunk_id,
                "source": item_metadata.get("source", "unknown"),
                "page": item_metadata.get("page", "-"),
                "covered_by_templates": chunk_id in covered_chunk_ids,
                "pre_bid_hits": pre_bid_hits,
                "post_contract_hits": post_contract_hits,
                "snippet": normalized_text[:240].replace("\n", " "),
            }
        )

    logger.info("Regulations coverage summary")
    logger.info("total_chunks=%s", len(documents))
    logger.info("covered_chunks=%s", len(covered_chunk_ids))
    for query_template, chunk_ids in coverage_by_template.items():
        logger.info("template=%s covered=%s", query_template, len(chunk_ids))

    for classification, rows in classifications.items():
        logger.info("%s=%s", classification, len(rows))

    for classification in ("post_contract_only", "mixed"):
        logger.info("Sample %s chunks", classification)
        for row in classifications[classification][: args.sample_limit]:
            logger.info(
                "chunk_id=%s source=%s page=%s covered=%s pre_bid_hits=%s post_contract_hits=%s snippet=%s",
                row["chunk_id"],
                row["source"],
                row["page"],
                row["covered_by_templates"],
                row["pre_bid_hits"],
                row["post_contract_hits"],
                row["snippet"],
            )


if __name__ == "__main__":
    main()
