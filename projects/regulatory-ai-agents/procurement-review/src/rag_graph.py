from __future__ import annotations

import logging
import re
from typing import Any

from .config import build_llm, get_settings
from .graph_indexer import (
    GraphIndexer,
    GraphIndexerError,
    build_default_vectorstores,
    coerce_message_text,
    get_graphrag_extract_model_name,
)
from .vectorstore import ProcurementVectorStore


logger = logging.getLogger(__name__)

GRAPH_NO_CONTEXT_MESSAGE = (
    "GraphRAG와 조달문서 벡터 검색에서 관련 컨텍스트를 찾지 못해 규정 준수 여부를 판단하기 어렵습니다."
)

GRAPH_ANSWER_PROMPT = """
당신은 공공조달 구매규격서의 법령 준수 여부를 검토하는 분석가입니다.
반드시 제공된 컨텍스트만 근거로 답하세요.

근거가 부족하면 단정하지 말고 '추가 확인 필요'로 판단하세요.
계약 이후 이행 여부가 아니라 규격서나 제안요청서에 해당 요건이 포함되어 있는지만 판단하세요.
답변 형식은 아래 구조를 그대로 지키세요.

## 준수 판단
- 준수 / 불충분 / 추가 확인 필요 중 하나

## 핵심 근거
- 핵심 근거를 2~4개 bullet로 작성

## 판단 근거
- 왜 그렇게 판단했는지 1~3문단으로 설명

## 추가 확인 필요사항
- 추가 확인이 필요 없으면 '없음'

질문:
{question}

질문 핵심 엔티티:
{question_entities}

=== 법령 관계 컨텍스트 (GraphRAG) ===
{graph_contexts}

=== 규격서 컨텍스트 (조달문서) ===
{procurement_contexts}

위 두 컨텍스트를 모두 참고하여 질문에 답하세요.
""".strip()


def retrieve_and_answer(question: str, model_name: str) -> dict[str, Any]:
    normalized_question = question.strip()
    if not normalized_question:
        return {
            "question": question,
            "answer": "질문이 비어 있습니다.",
            "contexts": [],
        }

    settings = get_settings()
    document_store, regulations_store = build_default_vectorstores()
    procurement_store = _get_procurement_store(settings, document_store)

    graph_entities: list[dict[str, Any]] = []
    graph_communities: list[dict[str, Any]] = []
    graph_chunks: list[dict[str, Any]] = []
    procurement_chunks: list[dict[str, Any]] = []
    graph_error: str | None = None
    procurement_error: str | None = None

    try:
        with GraphIndexer() as indexer:
            with indexer.driver.session() as session:
                meta_state = indexer._get_graph_meta_state(session) or {}

            graph_is_usable = int(meta_state.get("chunk_count", 0) or 0) > 0
            graph_status = str(meta_state.get("status", "") or "").strip().lower()

            if not graph_is_usable:
                indexer.ensure_index_from_vectorstores(document_store, regulations_store)
            elif graph_status != "ready":
                logger.info(
                    "GraphRAG index is not finalized yet (status=%s, chunk_count=%s). "
                    "Using the current graph snapshot for retrieval.",
                    graph_status or "unknown",
                    int(meta_state.get("chunk_count", 0) or 0),
                )

            graph_entities = indexer.extract_entities(normalized_question)
            graph_result = _collect_graph_context(
                indexer,
                question_entities=graph_entities,
                question=normalized_question,
            )
            graph_communities = graph_result["communities"]
            graph_chunks = graph_result["chunks"]
    except GraphIndexerError as exc:
        graph_error = f"GraphRAG 처리 실패: {exc}"
    except Exception as exc:  # pragma: no cover - depends on runtime graph/LLM state
        logger.exception("GraphRAG retrieval failed.")
        graph_error = f"GraphRAG 처리 중 오류가 발생했습니다: {exc}"

    try:
        procurement_chunks = _search_procurement_contexts(
            normalized_question,
            procurement_store,
        )
    except Exception as exc:  # pragma: no cover - depends on runtime vector/embedding state
        logger.exception("Procurement vector search failed.")
        procurement_error = f"조달문서 벡터 검색 중 오류가 발생했습니다: {exc}"

    graph_contexts = [
        *_serialize_community_contexts(graph_communities),
        *_serialize_chunk_contexts(graph_chunks),
    ]
    procurement_contexts = _serialize_chunk_contexts(procurement_chunks)
    contexts = [*graph_contexts, *procurement_contexts]

    if graph_error and procurement_error:
        return {
            "question": normalized_question,
            "answer": f"{graph_error}; {procurement_error}",
            "contexts": [],
        }

    if not contexts:
        return {
            "question": normalized_question,
            "answer": GRAPH_NO_CONTEXT_MESSAGE,
            "contexts": [],
        }

    answer_model = _resolve_answer_model(model_name)
    answer_llm = build_llm(answer_model)
    prompt = GRAPH_ANSWER_PROMPT.format(
        question=normalized_question,
        question_entities=", ".join(entity["name"] for entity in graph_entities[:12]) or "없음",
        graph_contexts=_compose_graph_prompt_context(graph_communities, graph_chunks),
        procurement_contexts=_format_chunk_context(
            procurement_chunks,
            empty_message="관련 조달문서 청크 없음",
        ),
    )

    try:
        answer = coerce_message_text(answer_llm.invoke(prompt)).strip()
    except Exception as exc:  # pragma: no cover - depends on runtime LLM state
        logger.exception("GraphRAG answer generation failed.")
        answer = f"GraphRAG 답변 생성 중 오류가 발생했습니다: {exc}"

    return {
        "question": normalized_question,
        "answer": answer or GRAPH_NO_CONTEXT_MESSAGE,
        "contexts": contexts,
    }


def _resolve_answer_model(model_name: str) -> str:
    normalized = (model_name or "").strip()
    if not normalized or normalized.lower() == "graphrag":
        return get_graphrag_extract_model_name()
    return normalized


def _get_procurement_store(
    settings,
    document_store,
) -> ProcurementVectorStore:
    if getattr(document_store, "collection_name", "") == "procurement_regulations":
        return document_store

    return ProcurementVectorStore(
        settings=settings,
        embeddings=document_store.embeddings,
        collection_name="procurement_regulations",
    )


def _extract_terms(question_entities: list[dict[str, str]], question: str) -> list[str]:
    terms = [entity["name"].strip().lower() for entity in question_entities if entity.get("name")]
    if terms:
        return list(dict.fromkeys(term for term in terms if term))

    tokens = re.findall(r"[A-Za-z0-9가-힣().,%/-]{2,}", question)
    lowered_tokens = [token.strip().lower() for token in tokens if token.strip()]
    return list(dict.fromkeys(lowered_tokens[:12]))


def _collect_graph_context(
    indexer: GraphIndexer,
    *,
    question_entities: list[dict[str, str]],
    question: str,
) -> dict[str, list[dict[str, Any]]]:
    terms = _extract_terms(question_entities, question)
    if not terms:
        return {"entities": [], "communities": [], "chunks": []}

    with indexer.driver.session() as session:
        seed_record = session.run(
            """
            MATCH (entity:Entity)
            WHERE any(term IN $terms WHERE toLower(entity.name) CONTAINS term)
            RETURN collect(DISTINCT entity.entity_key)[..20] AS seed_keys
            """,
            terms=terms,
        ).single()
        seed_keys = [str(value) for value in (seed_record["seed_keys"] or [])] if seed_record else []
        if not seed_keys:
            return {"entities": [], "communities": [], "chunks": []}

        entity_rows = [
            {
                "entity_key": str(record["entity_key"]),
                "name": str(record["name"]),
                "type": str(record["type"]),
                "community_id": int(record["community_id"]) if record["community_id"] is not None else 0,
            }
            for record in session.run(
                """
                MATCH (seed:Entity)
                WHERE seed.entity_key IN $seed_keys
                OPTIONAL MATCH (seed)-[:RELATES_TO*0..2]-(neighbor:Entity)
                WITH collect(DISTINCT seed) + collect(DISTINCT neighbor) AS candidate_nodes
                UNWIND candidate_nodes AS node
                WITH DISTINCT node
                WHERE node IS NOT NULL
                RETURN node.entity_key AS entity_key,
                       node.name AS name,
                       node.type AS type,
                       coalesce(node.community_id, 0) AS community_id
                ORDER BY node.name
                LIMIT 40
                """,
                seed_keys=seed_keys,
            )
        ]
        if not entity_rows:
            return {"entities": [], "communities": [], "chunks": []}

        entity_keys = [row["entity_key"] for row in entity_rows]
        community_ids = list(
            dict.fromkeys(
                row["community_id"] for row in entity_rows if row["community_id"] is not None
            )
        )

        communities = [
            {
                "community_id": int(record["community_id"]),
                "summary": str(record["summary"] or ""),
                "algorithm": str(record["algorithm"] or "unknown"),
                "entity_count": int(record["entity_count"] or 0),
                "chunk_count": int(record["chunk_count"] or 0),
            }
            for record in session.run(
                """
                MATCH (community:Community)
                WHERE community.community_id IN $community_ids
                RETURN community.community_id AS community_id,
                       community.summary AS summary,
                       community.algorithm AS algorithm,
                       community.entity_count AS entity_count,
                       community.chunk_count AS chunk_count
                ORDER BY community.chunk_count DESC, community.entity_count DESC, community.community_id ASC
                LIMIT 6
                """,
                community_ids=community_ids,
            )
        ]

        chunks = [
            {
                "chunk_id": str(record["chunk_id"]),
                "source": str(record["source"] or "unknown"),
                "page": str(record["page"] or "-"),
                "source_group": str(record["source_group"] or "graph"),
                "text": str(record["text"] or ""),
            }
            for record in session.run(
                """
                MATCH (chunk:Chunk)-[:MENTIONS]->(entity:Entity)
                WHERE entity.entity_key IN $entity_keys
                RETURN DISTINCT chunk.chunk_id AS chunk_id,
                                chunk.source AS source,
                                chunk.page AS page,
                                chunk.source_group AS source_group,
                                chunk.text AS text
                ORDER BY chunk.source ASC, chunk.page ASC
                LIMIT 10
                """,
                entity_keys=entity_keys,
            )
        ]

    return {
        "entities": entity_rows,
        "communities": communities,
        "chunks": chunks,
    }


def _search_procurement_contexts(
    question: str,
    procurement_store,
) -> list[dict[str, str]]:
    procurement_documents = procurement_store.similarity_search(question, k=5)
    chunks: list[dict[str, str]] = []
    for document in procurement_documents[:5]:
        metadata = document.metadata or {}
        chunks.append(
            {
                "chunk_id": str(metadata.get("chunk_id", "procurement")),
                "source": str(metadata.get("source", "unknown")),
                "page": str(metadata.get("page", "-")),
                "source_group": str(metadata.get("source_group", "procurement")),
                "text": document.page_content,
            }
        )
    return chunks


def _serialize_community_contexts(communities: list[dict[str, Any]]) -> list[str]:
    serialized: list[str] = []
    for community in communities:
        serialized.append(
            "[source=graph/community/{community_id} page=- group=community]\n{summary}".format(
                community_id=community["community_id"],
                summary=community["summary"] or "커뮤니티 요약 없음",
            )
        )
    return serialized


def _serialize_chunk_contexts(chunks: list[dict[str, Any]]) -> list[str]:
    serialized: list[str] = []
    for chunk in chunks:
        serialized.append(
            "[source={source} page={page} group={group}]\n{text}".format(
                source=chunk.get("source", "unknown"),
                page=chunk.get("page", "-"),
                group=chunk.get("source_group", "graph"),
                text=chunk.get("text", ""),
            )
        )
    return serialized


def _format_graph_context(communities: list[dict[str, Any]]) -> str:
    if not communities:
        return "관련 커뮤니티 요약 없음"
    return "\n\n".join(
        "- community={community_id} algorithm={algorithm} entity_count={entity_count} chunk_count={chunk_count}\n{summary}".format(
            community_id=community["community_id"],
            algorithm=community["algorithm"],
            entity_count=community["entity_count"],
            chunk_count=community["chunk_count"],
            summary=community["summary"] or "요약 없음",
        )
        for community in communities
    )


def _format_chunk_context(
    chunks: list[dict[str, Any]],
    *,
    empty_message: str = "관련 청크 없음",
) -> str:
    if not chunks:
        return empty_message
    return "\n\n".join(
        "[source={source} page={page} group={group}]\n{text}".format(
            source=chunk.get("source", "unknown"),
            page=chunk.get("page", "-"),
            group=chunk.get("source_group", "graph"),
            text=chunk.get("text", ""),
        )
        for chunk in chunks
    )


def _compose_graph_prompt_context(
    communities: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> str:
    return "\n\n".join(
        [
            "[커뮤니티 요약]\n{content}".format(
                content=_format_graph_context(communities)
            ),
            "[관련 법령 청크]\n{content}".format(
                content=_format_chunk_context(chunks, empty_message="관련 법령 청크 없음")
            ),
        ]
    )
