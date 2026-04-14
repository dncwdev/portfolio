from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from typing import Any

from neo4j import Driver, GraphDatabase

from .config import build_embeddings, build_llm, get_settings
from .vectorstore import ProcurementVectorStore


logger = logging.getLogger(__name__)

GRAPH_META_NAME = "procurement_graphrag"
GRAPH_PROJECTION_NAME = "procurement_graphrag_entities"
DEFAULT_GRAPHRAG_EXTRACT_MODEL = "qwen3.5-35b-a3b"
DEFAULT_INDEX_BATCH_SIZE = 25
DEFAULT_EXTRACT_MAX_WORKERS = max(
    1,
    int((os.getenv("GRAPHRAG_EXTRACT_WORKERS", "4") or "4").strip() or "4"),
)
GRAPH_SOURCE_SCOPE = "regulations_only"
JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", flags=re.DOTALL)

ENTITY_EXTRACTION_PROMPT = """
다음 텍스트에서 법령명, 조문번호, 기관명, 규정 요건을 추출하고,
이들 간의 참조/적용/위임 관계를 JSON 형태로 반환하세요.

형식:
{{
  "entities": [{{"name": "엔티티명", "type": "law|article|organization|requirement"}}],
  "relations": [{{"source": "엔티티명", "target": "엔티티명", "type": "reference|applies_to|delegates_to"}}]
}}

규칙:
- JSON 외 텍스트를 출력하지 마세요.
- 근거가 약한 추정은 제외하세요.
- 중복 엔티티는 제거하세요.
- relation source/target은 entities.name과 동일한 문자열을 사용하세요.

텍스트:
{text}
""".strip()

COMMUNITY_SUMMARY_PROMPT = """
다음은 공공조달 규격 검토용 GraphRAG 커뮤니티의 핵심 엔티티와 관련 청크입니다.
이 커뮤니티가 어떤 법령/조문/기관/요건 묶음인지 한국어로 4문장 이내로 요약하세요.

- 법령과 조문 간 연결, 적용 대상 기관, 핵심 규정 요건을 우선 설명하세요.
- 요약은 검색용 컨텍스트이므로 압축적으로 작성하세요.
- 확인되지 않은 내용은 추정하지 마세요.

엔티티:
{entities}

청크 발췌:
{chunks}
""".strip()


class GraphIndexerError(RuntimeError):
    """Raised when GraphRAG indexing cannot proceed safely."""


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    text: str
    source: str
    page: str
    source_group: str
    collection_name: str


def get_graphrag_extract_model_name() -> str:
    value = os.getenv("GRAPHRAG_EXTRACT_MODEL", DEFAULT_GRAPHRAG_EXTRACT_MODEL).strip()
    return value or DEFAULT_GRAPHRAG_EXTRACT_MODEL


def coerce_message_text(message: Any) -> str:
    if hasattr(message, "content"):
        content = message.content
    else:
        content = message

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()

    return str(content or "")


def build_default_vectorstores() -> tuple[ProcurementVectorStore, ProcurementVectorStore]:
    settings = get_settings()
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


class GraphIndexer:
    def __init__(
        self,
        *,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        extract_model_name: str | None = None,
    ) -> None:
        self.settings = get_settings()
        self.uri = (uri or os.getenv("NEO4J_URI", "")).strip()
        self.user = (user or os.getenv("NEO4J_USER", "")).strip()
        self.password = (password or os.getenv("NEO4J_PASSWORD", "")).strip()
        if not self.uri or not self.user or not self.password:
            raise GraphIndexerError(
                "Missing Neo4j configuration. Set NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD."
            )

        self.extract_model_name = extract_model_name or get_graphrag_extract_model_name()
        try:
            self.driver: Driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
            )
        except Exception as exc:  # pragma: no cover - driver creation depends on runtime env
            raise GraphIndexerError(f"Failed to create Neo4j driver: {exc}") from exc

        self.extract_llm = build_llm(self.extract_model_name)
        self._thread_local = threading.local()

    def __enter__(self) -> GraphIndexer:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self.driver.close()

    def extract_entities(self, text: str) -> list[dict[str, str]]:
        payload = self.extract_graph_payload(text)
        return payload["entities"]

    def extract_graph_payload(self, text: str) -> dict[str, list[dict[str, str]]]:
        clipped_text = text.strip()[:5000]
        if not clipped_text:
            return {"entities": [], "relations": []}

        raw_response = coerce_message_text(
            self._get_extract_llm().invoke(ENTITY_EXTRACTION_PROMPT.format(text=clipped_text))
        )
        payload = self._safe_json_loads(raw_response)
        return self._normalize_graph_payload(payload)

    def ensure_index_from_vectorstores(
        self,
        document_store: ProcurementVectorStore,
        regulations_store: ProcurementVectorStore,
        *,
        force: bool = False,
        batch_size: int = DEFAULT_INDEX_BATCH_SIZE,
    ) -> dict[str, Any]:
        batch_size = max(1, int(batch_size or DEFAULT_INDEX_BATCH_SIZE))

        with self.driver.session() as session:
            self._verify_connection(session)
            self._ensure_constraints(session)
            meta_state = self._get_graph_meta_state(session)
            if (
                not force
                and meta_state
                and str(meta_state.get("status", "")).strip().lower() == "ready"
            ):
                logger.info("GraphRAG index already exists in Neo4j. Skipping indexing.")
                return {
                    "status": str(meta_state.get("status", "skipped")),
                    "chunk_count": int(meta_state.get("chunk_count", 0) or 0),
                    "entity_count": int(meta_state.get("entity_count", 0) or 0),
                    "community_count": int(meta_state.get("community_count", 0) or 0),
                    "algorithm": str(meta_state.get("algorithm", "unknown")),
                }

        chunk_records = self._load_chunks(document_store, regulations_store)
        total_chunks = len(chunk_records)
        if not chunk_records:
            logger.warning("No Chroma chunks were found for GraphRAG indexing.")
            with self.driver.session() as session:
                self._write_meta(
                    session,
                    status="empty",
                    chunk_count=0,
                    entity_count=0,
                    community_count=0,
                    algorithm="none",
                    total_chunks=0,
                    processed_chunks=0,
                    last_error=None,
                )
            return {
                "status": "empty",
                "chunk_count": 0,
                "entity_count": 0,
                "community_count": 0,
                "algorithm": "none",
            }

        with self.driver.session() as session:
            if force:
                self._clear_existing_graph(session)
            self._write_meta(
                session,
                status="in_progress",
                chunk_count=0,
                entity_count=0,
                community_count=0,
                algorithm="pending",
                total_chunks=total_chunks,
                processed_chunks=0,
                last_error=None,
            )

        with self.driver.session() as session:
            existing_chunk_ids = set() if force else self._load_existing_chunk_ids(session)
            existing_counts = self._get_graph_counts(session)
            self._write_meta(
                session,
                status="in_progress",
                chunk_count=existing_counts["chunk_count"],
                entity_count=existing_counts["entity_count"],
                community_count=0,
                algorithm="pending",
                total_chunks=total_chunks,
                processed_chunks=existing_counts["chunk_count"],
                last_error=None,
            )

        pending_chunks = [
            chunk for chunk in chunk_records if chunk.chunk_id not in existing_chunk_ids
        ]
        logger.info(
            "GraphRAG batch indexing starting | total=%s | existing=%s | pending=%s | batch_size=%s | workers=%s",
            total_chunks,
            len(existing_chunk_ids),
            len(pending_chunks),
            batch_size,
            DEFAULT_EXTRACT_MAX_WORKERS,
        )

        try:
            for batch_start in range(0, len(pending_chunks), batch_size):
                batch = pending_chunks[batch_start : batch_start + batch_size]
                extracted_payloads = self._extract_batch_payloads(batch)

                with self.driver.session() as session:
                    for chunk, payload in extracted_payloads:
                        self._upsert_chunk_graph(session, chunk, payload)

                    counts = self._get_graph_counts(session)
                    self._write_meta(
                        session,
                        status="in_progress",
                        chunk_count=counts["chunk_count"],
                        entity_count=counts["entity_count"],
                        community_count=0,
                        algorithm="pending",
                        total_chunks=total_chunks,
                        processed_chunks=counts["chunk_count"],
                        last_error=None,
                    )

                logger.info(
                    "GraphRAG batch indexed | persisted_chunks=%s/%s | entities=%s",
                    counts["chunk_count"],
                    total_chunks,
                    counts["entity_count"],
                )

            with self.driver.session() as session:
                algorithm = self._run_community_detection(session)
                community_count = self._write_community_summaries(session, algorithm)
                counts = self._get_graph_counts(session)
                chunk_count = counts["chunk_count"]
                entity_count = counts["entity_count"]
                self._write_meta(
                    session,
                    status="ready",
                    chunk_count=chunk_count,
                    entity_count=entity_count,
                    community_count=community_count,
                    algorithm=algorithm,
                    total_chunks=total_chunks,
                    processed_chunks=chunk_count,
                    last_error=None,
                )
        except Exception as exc:
            with self.driver.session() as session:
                counts = self._get_graph_counts(session)
                self._write_meta(
                    session,
                    status="failed",
                    chunk_count=counts["chunk_count"],
                    entity_count=counts["entity_count"],
                    community_count=counts["community_count"],
                    algorithm="failed",
                    total_chunks=total_chunks,
                    processed_chunks=counts["chunk_count"],
                    last_error=str(exc),
                )
            raise

        logger.info(
            "GraphRAG indexing complete | chunks=%s | entities=%s | communities=%s | algorithm=%s",
            chunk_count,
            entity_count,
            community_count,
            algorithm,
        )
        return {
            "status": "ready",
            "chunk_count": chunk_count,
            "entity_count": entity_count,
            "community_count": community_count,
            "algorithm": algorithm,
        }

    def _load_chunks(
        self,
        document_store: ProcurementVectorStore,
        regulations_store: ProcurementVectorStore,
    ) -> list[ChunkRecord]:
        logger.info(
            "GraphRAG source scope is '%s'; skipping document collection '%s' and indexing only regulations collection '%s'.",
            GRAPH_SOURCE_SCOPE,
            document_store.collection_name,
            regulations_store.collection_name,
        )
        return self._read_collection_chunks(regulations_store, source_group="regulation")

    def _read_collection_chunks(
        self,
        store: ProcurementVectorStore,
        *,
        source_group: str,
        batch_size: int = 256,
    ) -> list[ChunkRecord]:
        total = store.collection.count()
        records: list[ChunkRecord] = []
        for offset in range(0, total, batch_size):
            batch = store.collection.get(
                limit=batch_size,
                offset=offset,
                include=["documents", "metadatas"],
            )
            documents = batch.get("documents", []) or []
            metadatas = batch.get("metadatas", []) or []

            for document, metadata in zip(documents, metadatas):
                metadata = metadata or {}
                text = str(document or "").strip()
                if not text:
                    continue
                page = str(metadata.get("page", "-"))
                records.append(
                    ChunkRecord(
                        chunk_id=str(
                            metadata.get("chunk_id")
                            or f"{store.collection_name}:{offset + len(records) + 1}"
                        ),
                        text=text,
                        source=str(metadata.get("source", "unknown")),
                        page=page,
                        source_group=str(metadata.get("source_group", source_group)),
                        collection_name=str(
                            metadata.get("collection_name", store.collection_name)
                        ),
                    )
                )
        return records

    def _verify_connection(self, session) -> None:
        try:
            session.run("RETURN 1 AS ok").single()
        except Exception as exc:  # pragma: no cover - depends on external Neo4j runtime
            raise GraphIndexerError(f"Failed to connect to Neo4j: {exc}") from exc

    def _get_graph_meta_state(self, session) -> dict[str, Any] | None:
        record = session.run(
            """
            MATCH (meta:GraphIndexMeta {name: $name})
            RETURN meta.status AS status,
                   coalesce(meta.chunk_count, 0) AS chunk_count,
                   coalesce(meta.entity_count, 0) AS entity_count,
                   coalesce(meta.community_count, 0) AS community_count,
                   coalesce(meta.algorithm, 'unknown') AS algorithm,
                   coalesce(meta.total_chunks, 0) AS total_chunks,
                   coalesce(meta.processed_chunks, 0) AS processed_chunks,
                   meta.last_error AS last_error
            LIMIT 1
            """,
            name=GRAPH_META_NAME,
        ).single()
        if not record:
            return None

        return {
            "status": record["status"],
            "chunk_count": int(record["chunk_count"] or 0),
            "entity_count": int(record["entity_count"] or 0),
            "community_count": int(record["community_count"] or 0),
            "algorithm": record["algorithm"],
            "total_chunks": int(record["total_chunks"] or 0),
            "processed_chunks": int(record["processed_chunks"] or 0),
            "last_error": record["last_error"],
        }

    def _load_existing_chunk_ids(self, session) -> set[str]:
        return {
            str(record["chunk_id"])
            for record in session.run("MATCH (chunk:Chunk) RETURN chunk.chunk_id AS chunk_id")
            if record["chunk_id"] is not None
        }

    def _get_graph_counts(self, session) -> dict[str, int]:
        record = session.run(
            """
            OPTIONAL MATCH (chunk:Chunk)
            WITH count(chunk) AS chunk_count
            OPTIONAL MATCH (entity:Entity)
            WITH chunk_count, count(entity) AS entity_count
            OPTIONAL MATCH (community:Community)
            RETURN chunk_count, entity_count, count(community) AS community_count
            """
        ).single()
        return {
            "chunk_count": int(record["chunk_count"] or 0) if record else 0,
            "entity_count": int(record["entity_count"] or 0) if record else 0,
            "community_count": int(record["community_count"] or 0) if record else 0,
        }

    def _clear_existing_graph(self, session) -> None:
        session.run("MATCH (meta:GraphIndexMeta {name: $name}) DETACH DELETE meta", name=GRAPH_META_NAME).consume()
        session.run("MATCH (community:Community) DETACH DELETE community").consume()
        session.run("MATCH (chunk:Chunk) DETACH DELETE chunk").consume()
        session.run("MATCH (entity:Entity) DETACH DELETE entity").consume()

    def _ensure_constraints(self, session) -> None:
        queries = (
            "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (chunk:Chunk) REQUIRE chunk.chunk_id IS UNIQUE",
            "CREATE CONSTRAINT entity_key_unique IF NOT EXISTS FOR (entity:Entity) REQUIRE entity.entity_key IS UNIQUE",
            "CREATE CONSTRAINT community_id_unique IF NOT EXISTS FOR (community:Community) REQUIRE community.community_id IS UNIQUE",
            "CREATE CONSTRAINT graph_meta_unique IF NOT EXISTS FOR (meta:GraphIndexMeta) REQUIRE meta.name IS UNIQUE",
        )
        for query in queries:
            session.run(query).consume()

    def _upsert_chunk_graph(
        self,
        session,
        chunk: ChunkRecord,
        payload: dict[str, list[dict[str, str]]],
    ) -> None:
        entities, relations = self._normalize_entities_and_relations(payload)

        session.run(
            """
            MERGE (chunk:Chunk {chunk_id: $chunk_id})
            SET chunk.text = $text,
                chunk.source = $source,
                chunk.page = $page,
                chunk.source_group = $source_group,
                chunk.collection_name = $collection_name,
                chunk.updated_at = datetime()
            """,
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            source=chunk.source,
            page=chunk.page,
            source_group=chunk.source_group,
            collection_name=chunk.collection_name,
        ).consume()

        if entities:
            session.run(
                """
                UNWIND $entities AS entity
                MERGE (graph_entity:Entity {entity_key: entity.entity_key})
                ON CREATE SET graph_entity.name = entity.name,
                              graph_entity.type = entity.type,
                              graph_entity.created_at = datetime()
                ON MATCH SET graph_entity.name = entity.name,
                             graph_entity.type = entity.type,
                             graph_entity.updated_at = datetime()
                WITH graph_entity
                MATCH (chunk:Chunk {chunk_id: $chunk_id})
                MERGE (chunk)-[:MENTIONS]->(graph_entity)
                """,
                entities=entities,
                chunk_id=chunk.chunk_id,
            ).consume()

        if relations:
            session.run(
                """
                UNWIND $relations AS relation
                MERGE (source:Entity {entity_key: relation.source_key})
                ON CREATE SET source.name = relation.source,
                              source.type = relation.source_type,
                              source.created_at = datetime()
                MERGE (target:Entity {entity_key: relation.target_key})
                ON CREATE SET target.name = relation.target,
                              target.type = relation.target_type,
                              target.created_at = datetime()
                MERGE (source)-[edge:RELATES_TO {type: relation.type}]->(target)
                ON CREATE SET edge.created_at = datetime()
                """,
                relations=relations,
            ).consume()

    def _normalize_entities_and_relations(
        self,
        payload: dict[str, list[dict[str, str]]],
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        entity_map: dict[str, dict[str, str]] = {}
        for entity in payload.get("entities", []):
            name = str(entity.get("name", "")).strip()
            entity_type = str(entity.get("type", "unknown")).strip() or "unknown"
            if not name:
                continue
            key = self._entity_key(name, entity_type)
            entity_map[key] = {
                "entity_key": key,
                "name": name,
                "type": entity_type,
            }

        relations: list[dict[str, str]] = []
        for relation in payload.get("relations", []):
            source = str(relation.get("source", "")).strip()
            target = str(relation.get("target", "")).strip()
            relation_type = self._normalize_relation_type(relation.get("type", "related"))
            if not source or not target:
                continue

            source_key = self._find_entity_key(source, entity_map) or self._entity_key(source, "unknown")
            target_key = self._find_entity_key(target, entity_map) or self._entity_key(target, "unknown")
            if source_key == target_key:
                continue

            source_entity = entity_map.setdefault(
                source_key,
                {"entity_key": source_key, "name": source, "type": "unknown"},
            )
            target_entity = entity_map.setdefault(
                target_key,
                {"entity_key": target_key, "name": target, "type": "unknown"},
            )
            relations.append(
                {
                    "source": source_entity["name"],
                    "target": target_entity["name"],
                    "source_key": source_key,
                    "target_key": target_key,
                    "source_type": source_entity["type"],
                    "target_type": target_entity["type"],
                    "type": relation_type,
                }
            )

        return list(entity_map.values()), relations

    def _run_community_detection(self, session) -> str:
        record = session.run("MATCH (entity:Entity) RETURN count(entity) AS entity_count").single()
        entity_count = int(record["entity_count"] or 0) if record else 0
        if entity_count == 0:
            return "none"

        try:
            session.run(
                """
                CALL gds.graph.drop($graph_name, false)
                YIELD graphName
                RETURN graphName
                """,
                graph_name=GRAPH_PROJECTION_NAME,
            ).consume()
        except Exception:
            pass

        try:
            session.run(
                """
                CALL gds.graph.project(
                    $graph_name,
                    'Entity',
                    {RELATES_TO: {orientation: 'UNDIRECTED'}}
                )
                YIELD graphName, nodeCount, relationshipCount
                RETURN graphName, nodeCount, relationshipCount
                """,
                graph_name=GRAPH_PROJECTION_NAME,
            ).consume()
            session.run(
                """
                CALL gds.leiden.write(
                    $graph_name,
                    {writeProperty: 'community_id'}
                )
                YIELD communityCount, modularities
                RETURN communityCount, modularities
                """,
                graph_name=GRAPH_PROJECTION_NAME,
            ).consume()
            algorithm = "leiden"
        except Exception as exc:
            logger.warning(
                "Neo4j GDS Leiden is unavailable. Falling back to connected components. %s",
                exc,
            )
            self._assign_connected_components(session)
            algorithm = "connected_components"
        finally:
            try:
                session.run(
                    """
                    CALL gds.graph.drop($graph_name, false)
                    YIELD graphName
                    RETURN graphName
                    """,
                    graph_name=GRAPH_PROJECTION_NAME,
                ).consume()
            except Exception:
                pass

        return algorithm

    def _assign_connected_components(self, session) -> None:
        entity_keys = [
            str(record["entity_key"])
            for record in session.run(
                "MATCH (entity:Entity) RETURN entity.entity_key AS entity_key ORDER BY entity.entity_key"
            )
        ]
        adjacency = {entity_key: set() for entity_key in entity_keys}

        for record in session.run(
            """
            MATCH (source:Entity)-[:RELATES_TO]-(target:Entity)
            RETURN source.entity_key AS source_key, target.entity_key AS target_key
            """
        ):
            source_key = str(record["source_key"])
            target_key = str(record["target_key"])
            adjacency.setdefault(source_key, set()).add(target_key)
            adjacency.setdefault(target_key, set()).add(source_key)

        visited: set[str] = set()
        assignments: list[dict[str, Any]] = []
        community_id = 0

        for entity_key in entity_keys:
            if entity_key in visited:
                continue
            stack = [entity_key]
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                assignments.append({"entity_key": current, "community_id": community_id})
                stack.extend(sorted(adjacency.get(current, set()) - visited))
            community_id += 1

        session.run(
            """
            UNWIND $assignments AS assignment
            MATCH (entity:Entity {entity_key: assignment.entity_key})
            SET entity.community_id = assignment.community_id
            """,
            assignments=assignments,
        ).consume()

    def _write_community_summaries(self, session, algorithm: str) -> int:
        session.run("MATCH (community:Community) DETACH DELETE community").consume()

        community_ids = [
            int(record["community_id"])
            for record in session.run(
                """
                MATCH (entity:Entity)
                RETURN DISTINCT coalesce(entity.community_id, 0) AS community_id
                ORDER BY community_id
                """
            )
        ]

        for community_id in community_ids:
            record = session.run(
                """
                MATCH (entity:Entity {community_id: $community_id})
                OPTIONAL MATCH (chunk:Chunk)-[:MENTIONS]->(entity)
                RETURN collect(DISTINCT entity.name)[..20] AS entity_names,
                       collect(
                         DISTINCT {
                           chunk_id: chunk.chunk_id,
                           source: chunk.source,
                           page: chunk.page,
                           source_group: chunk.source_group,
                           text: chunk.text
                         }
                       )[..6] AS chunks,
                       count(DISTINCT entity) AS entity_count,
                       count(DISTINCT chunk) AS chunk_count
                """,
                community_id=community_id,
            ).single()

            entity_names = [str(name) for name in (record["entity_names"] or []) if name]
            chunks = [chunk for chunk in (record["chunks"] or []) if chunk and chunk.get("text")]
            summary = self._summarize_community(entity_names, chunks)

            session.run(
                """
                MERGE (community:Community {community_id: $community_id})
                SET community.summary = $summary,
                    community.algorithm = $algorithm,
                    community.entity_count = $entity_count,
                    community.chunk_count = $chunk_count,
                    community.updated_at = datetime()
                WITH community
                MATCH (entity:Entity {community_id: $community_id})
                MERGE (entity)-[:IN_COMMUNITY]->(community)
                WITH community
                MATCH (chunk:Chunk)-[:MENTIONS]->(:Entity {community_id: $community_id})
                MERGE (chunk)-[:IN_COMMUNITY]->(community)
                """,
                community_id=community_id,
                summary=summary,
                algorithm=algorithm,
                entity_count=int(record["entity_count"] or 0),
                chunk_count=int(record["chunk_count"] or 0),
            ).consume()

        return len(community_ids)

    def _summarize_community(
        self,
        entity_names: list[str],
        chunks: list[dict[str, Any]],
    ) -> str:
        if not entity_names and not chunks:
            return "커뮤니티 요약을 생성할 수 있는 정보가 없습니다."

        clipped_chunks: list[str] = []
        for chunk in chunks[:4]:
            source = chunk.get("source", "unknown")
            page = chunk.get("page", "-")
            text = str(chunk.get("text", "")).strip()[:700]
            if text:
                clipped_chunks.append(f"[{source} p.{page}] {text}")

        prompt = COMMUNITY_SUMMARY_PROMPT.format(
            entities=", ".join(entity_names[:20]) if entity_names else "없음",
            chunks="\n\n".join(clipped_chunks) if clipped_chunks else "없음",
        )
        summary = coerce_message_text(self.extract_llm.invoke(prompt)).strip()
        return summary or "커뮤니티 요약을 생성하지 못했습니다."

    def _write_meta(
        self,
        session,
        *,
        status: str,
        chunk_count: int,
        entity_count: int,
        community_count: int,
        algorithm: str,
        total_chunks: int | None = None,
        processed_chunks: int | None = None,
        last_error: str | None = None,
    ) -> None:
        session.run(
            """
            MERGE (meta:GraphIndexMeta {name: $name})
            ON CREATE SET meta.created_at = datetime()
            SET meta.status = $status,
                meta.chunk_count = $chunk_count,
                meta.entity_count = $entity_count,
                meta.community_count = $community_count,
                meta.algorithm = $algorithm,
                meta.extract_model = $extract_model,
                meta.source_scope = $source_scope,
                meta.total_chunks = $total_chunks,
                meta.processed_chunks = $processed_chunks,
                meta.last_error = $last_error,
                meta.updated_at = datetime()
            """,
            name=GRAPH_META_NAME,
            status=status,
            chunk_count=chunk_count,
            entity_count=entity_count,
            community_count=community_count,
            algorithm=algorithm,
            extract_model=self.extract_model_name,
            source_scope=GRAPH_SOURCE_SCOPE,
            total_chunks=total_chunks,
            processed_chunks=processed_chunks,
            last_error=last_error,
        ).consume()

    def _safe_json_loads(self, raw_text: str) -> dict[str, Any]:
        text = raw_text.strip()
        if not text:
            return {}

        fence_match = JSON_BLOCK_RE.search(text)
        if fence_match:
            text = fence_match.group(1).strip()
        else:
            first_brace = text.find("{")
            last_brace = text.rfind("}")
            if first_brace >= 0 and last_brace > first_brace:
                text = text[first_brace : last_brace + 1]

        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            logger.warning("Failed to decode GraphRAG extraction payload. Raw text: %s", raw_text)
            return {}

    def _normalize_graph_payload(self, payload: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
        entities: list[dict[str, str]] = []
        seen_entities: set[tuple[str, str]] = set()
        for item in payload.get("entities", []) or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            entity_type = str(item.get("type", "unknown")).strip() or "unknown"
            if not name:
                continue
            marker = (name, entity_type)
            if marker in seen_entities:
                continue
            seen_entities.add(marker)
            entities.append({"name": name, "type": entity_type})

        relations: list[dict[str, str]] = []
        seen_relations: set[tuple[str, str, str]] = set()
        for item in payload.get("relations", []) or []:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source", "")).strip()
            target = str(item.get("target", "")).strip()
            relation_type = self._normalize_relation_type(item.get("type", "related"))
            if not source or not target:
                continue
            marker = (source, target, relation_type)
            if marker in seen_relations:
                continue
            seen_relations.add(marker)
            relations.append(
                {
                    "source": source,
                    "target": target,
                    "type": relation_type,
                }
            )

        return {
            "entities": entities,
            "relations": relations,
        }

    @staticmethod
    def _entity_key(name: str, entity_type: str) -> str:
        return f"{entity_type.strip().lower()}::{name.strip().lower()}"

    @staticmethod
    def _find_entity_key(name: str, entity_map: dict[str, dict[str, str]]) -> str | None:
        lowered_name = name.strip().lower()
        for entity_key, entity in entity_map.items():
            if entity["name"].strip().lower() == lowered_name:
                return entity_key
        return None

    @staticmethod
    def _normalize_relation_type(value: Any) -> str:
        relation_type = str(value or "related").strip().lower()
        relation_type = re.sub(r"[^a-z0-9_]+", "_", relation_type)
        return relation_type.strip("_") or "related"

    def _get_extract_llm(self):
        llm = getattr(self._thread_local, "extract_llm", None)
        if llm is None:
            if threading.current_thread() is threading.main_thread():
                llm = self.extract_llm
            else:
                llm = build_llm(self.extract_model_name)
            self._thread_local.extract_llm = llm
        return llm

    def _extract_batch_payloads(
        self,
        batch: list[ChunkRecord],
    ) -> list[tuple[ChunkRecord, dict[str, list[dict[str, str]]]]]:
        if not batch:
            return []

        max_workers = max(1, min(DEFAULT_EXTRACT_MAX_WORKERS, len(batch)))
        if max_workers == 1:
            return [(chunk, self.extract_graph_payload(chunk.text)) for chunk in batch]

        results: list[tuple[ChunkRecord, dict[str, list[dict[str, str]]]] | None] = [None] * len(batch)
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="graphrag-extract") as executor:
            future_map = {
                executor.submit(self.extract_graph_payload, chunk.text): index
                for index, chunk in enumerate(batch)
            }
            for future in as_completed(future_map):
                index = future_map[future]
                chunk = batch[index]
                results[index] = (chunk, future.result())

        return [result for result in results if result is not None]


def ensure_graph_index(
    *,
    document_store: ProcurementVectorStore | None = None,
    regulations_store: ProcurementVectorStore | None = None,
    extract_model_name: str | None = None,
    force: bool = False,
    batch_size: int = DEFAULT_INDEX_BATCH_SIZE,
) -> dict[str, Any]:
    if document_store is None or regulations_store is None:
        document_store, regulations_store = build_default_vectorstores()

    with GraphIndexer(extract_model_name=extract_model_name) as indexer:
        return indexer.ensure_index_from_vectorstores(
            document_store,
            regulations_store,
            force=force,
            batch_size=batch_size,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GraphRAG index in Neo4j.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing GraphRAG nodes and rebuild the graph index.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_INDEX_BATCH_SIZE,
        help=f"Number of chunks to extract and persist per batch (default: {DEFAULT_INDEX_BATCH_SIZE}).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = parse_args()
    result = ensure_graph_index(force=args.force, batch_size=args.batch_size)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
