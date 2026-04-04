from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Literal

from langchain_core.documents import Document
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from .config import Settings, get_settings
from .domain_queries import (
    DomainQueryTemplate,
    build_domain_query,
    get_domain_rerank_query,
    get_domain_query_templates,
    normalize_domain_query_templates,
)
from .mcp_client import KoreanLawMCPClient, KoreanLawSearchType
from .reranker import BaseReranker
from .vectorstore import ProcurementVectorStore


logger = logging.getLogger(__name__)

LocalSearchScope = Literal["all", "documents", "regulations"]


@dataclass
class EvidenceCollector:
    regulation_sources: list[Document] = field(default_factory=list)
    document_sources: list[Document] = field(default_factory=list)
    _regulation_keys: set[str] = field(default_factory=set, init=False, repr=False)
    _document_keys: set[str] = field(default_factory=set, init=False, repr=False)
    _regulation_counter: int = field(default=0, init=False, repr=False)
    _document_counter: int = field(default=0, init=False, repr=False)

    def add_regulation_documents(self, documents: Sequence[Document]) -> list[Document]:
        return self._add_documents(documents, group="regulation")

    def add_document_documents(self, documents: Sequence[Document]) -> list[Document]:
        return self._add_documents(documents, group="document")

    def _add_documents(
        self,
        documents: Sequence[Document],
        *,
        group: Literal["regulation", "document"],
    ) -> list[Document]:
        output: list[Document] = []

        for document in documents:
            key = self._build_key(document, group=group)
            if group == "regulation":
                if key in self._regulation_keys:
                    existing = self._find_existing(
                        document,
                        self.regulation_sources,
                        key,
                        group,
                    )
                    if existing is not None:
                        output.append(existing)
                    continue
                self._regulation_keys.add(key)
                self._regulation_counter += 1
                citation = f"R{self._regulation_counter}"
                target_list = self.regulation_sources
            else:
                if key in self._document_keys:
                    existing = self._find_existing(
                        document,
                        self.document_sources,
                        key,
                        group,
                    )
                    if existing is not None:
                        output.append(existing)
                    continue
                self._document_keys.add(key)
                self._document_counter += 1
                citation = f"D{self._document_counter}"
                target_list = self.document_sources

            metadata = dict(document.metadata)
            metadata["citation"] = citation
            metadata["source_group"] = group
            normalized = Document(page_content=document.page_content, metadata=metadata)
            target_list.append(normalized)
            output.append(normalized)

        return output

    def _build_key(
        self,
        document: Document,
        *,
        group: Literal["regulation", "document"],
    ) -> str:
        metadata = document.metadata
        payload = "|".join(
            [
                group,
                str(metadata.get("origin", "local")),
                str(metadata.get("source", "unknown")),
                str(metadata.get("page", "-")),
                str(metadata.get("mcp_tool", "")),
                document.page_content,
            ]
        )
        return sha256(payload.encode("utf-8")).hexdigest()

    def _find_existing(
        self,
        document: Document,
        candidates: Sequence[Document],
        key: str,
        group: Literal["regulation", "document"],
    ) -> Document | None:
        for candidate in candidates:
            if self._build_key(candidate, group=group) == key:
                return candidate
        return None


class LocalSearchInput(BaseModel):
    query_template: DomainQueryTemplate = Field(
        ...,
        description=(
            "Choose one predefined procurement-review domain query template. "
            "Do not invent a free-form query."
        ),
    )
    scope: LocalSearchScope = Field(
        default="all",
        description="Search scope: all, documents, or regulations.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of reranked items per scope.",
    )


class KoreanLawMCPInput(BaseModel):
    search_type: KoreanLawSearchType = Field(
        default="law",
        description="One of: law, precedent, interpretation, all.",
    )
    query: str | None = Field(
        default=None,
        description="Natural-language legal query or law name.",
    )
    article: str | None = Field(
        default=None,
        description="Optional law article such as 제8조.",
    )
    document_id: str | None = Field(
        default=None,
        description="Optional MST/lawId/serial number returned by a prior MCP result.",
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of search results to request.",
    )


def build_agent_tools(
    *,
    document_store: ProcurementVectorStore,
    regulations_store: ProcurementVectorStore,
    reranker: BaseReranker,
    collector: EvidenceCollector,
    settings: Settings | None = None,
    question_context: str = "",
) -> list[BaseTool]:
    runtime_settings = settings or get_settings()
    mcp_client = KoreanLawMCPClient(settings=runtime_settings)

    @tool("search_local_procurement_context", args_schema=LocalSearchInput)
    def search_local_procurement_context(
        query_template: DomainQueryTemplate,
        scope: LocalSearchScope = "all",
        top_k: int = 5,
    ) -> str:
        """Search local ChromaDB with one predefined procurement-review domain query template."""
        regulation_sources, document_sources = retrieve_local_evidence(
            query=question_context,
            scope=scope,
            top_k=top_k,
            document_store=document_store,
            regulations_store=regulations_store,
            reranker=reranker,
            collector=collector,
            settings=runtime_settings,
            query_templates=(query_template,),
        )
        return format_local_evidence(
            review_question=question_context,
            query_templates=(query_template,),
            regulation_sources=regulation_sources,
            document_sources=document_sources,
            use_mcp=runtime_settings.use_mcp,
        )

    tools: list[BaseTool] = [search_local_procurement_context]

    if runtime_settings.use_mcp:

        @tool("search_korean_law_mcp", args_schema=KoreanLawMCPInput)
        def search_korean_law_mcp(
            search_type: KoreanLawSearchType = "law",
            query: str | None = None,
            article: str | None = None,
            document_id: str | None = None,
            limit: int = 5,
        ) -> str:
            """Search the korean-law-mcp server for laws, articles, precedents, or interpretations."""
            try:
                documents = mcp_client.search_documents(
                    search_type=search_type,
                    query=query,
                    article=article,
                    document_id=document_id,
                    limit=limit,
                )
            except Exception as exc:
                return f"MCP 법령 검색을 실행하지 못했습니다. 오류: {exc}"

            regulation_sources = collector.add_regulation_documents(documents)
            return format_regulation_evidence(
                review_question=query or document_id or "",
                query_templates=(),
                documents=regulation_sources,
                empty_message="MCP 법령 검색 결과가 없습니다.",
            )

        tools.append(search_korean_law_mcp)

    return tools


def retrieve_local_evidence(
    *,
    query: str,
    scope: LocalSearchScope,
    top_k: int,
    document_store: ProcurementVectorStore,
    regulations_store: ProcurementVectorStore,
    reranker: BaseReranker,
    collector: EvidenceCollector,
    settings: Settings | None = None,
    query_templates: Sequence[str] | None = None,
) -> tuple[list[Document], list[Document]]:
    runtime_settings = settings or get_settings()
    rerank_limit = max(1, min(top_k, runtime_settings.rerank_top_k))
    selected_templates = normalize_domain_query_templates(
        list(query_templates) if query_templates is not None else None
    )

    regulation_sources: list[Document] = []
    document_sources: list[Document] = []

    if scope in {"all", "regulations"}:
        local_regulations = _retrieve_template_bundle(
            vectorstore=regulations_store,
            reranker=reranker,
            review_question=query,
            query_templates=selected_templates,
            retrieval_top_k=runtime_settings.retrieval_top_k,
            rerank_top_k=rerank_limit,
            rerank_relative_threshold=runtime_settings.rerank_relative_threshold,
        )
        regulation_sources = collector.add_regulation_documents(local_regulations)

    if scope in {"all", "documents"}:
        local_documents = _retrieve_template_bundle(
            vectorstore=document_store,
            reranker=reranker,
            review_question=query,
            query_templates=selected_templates,
            retrieval_top_k=runtime_settings.retrieval_top_k,
            rerank_top_k=rerank_limit,
            rerank_relative_threshold=runtime_settings.rerank_relative_threshold,
        )
        document_sources = collector.add_document_documents(local_documents)

    return regulation_sources, document_sources


def format_local_evidence(
    *,
    review_question: str,
    query_templates: Sequence[str],
    regulation_sources: Sequence[Document],
    document_sources: Sequence[Document],
    use_mcp: bool,
) -> str:
    selected_templates = normalize_domain_query_templates(
        list(query_templates) if query_templates else None
    )
    sections = [f"Review question: {review_question}"]
    sections.append(
        "Selected domain query templates: "
        + ", ".join(selected_templates)
    )

    sections.append(
        format_regulation_evidence(
            review_question=review_question,
            query_templates=selected_templates,
            documents=regulation_sources,
            empty_message=(
                "No local regulation evidence found."
                if use_mcp
                else "No local regulation evidence found. MCP is disabled."
            ),
        )
    )
    sections.append(
        _format_document_block(
            title="Document Evidence",
            documents=document_sources,
            empty_message="No matching uploaded procurement document evidence found.",
        )
    )
    return "\n\n".join(section for section in sections if section.strip())


def format_regulation_evidence(
    *,
    review_question: str,
    query_templates: Sequence[str],
    documents: Sequence[Document],
    empty_message: str,
) -> str:
    if query_templates:
        selected_templates = normalize_domain_query_templates(list(query_templates))
        heading = (
            "Regulation Evidence "
            f"for templates: {', '.join(selected_templates)}"
        )
    elif review_question:
        heading = f"Regulation Evidence for question: {review_question}"
    else:
        heading = "Regulation Evidence"
    return _format_document_block(
        title=heading,
        documents=documents,
        empty_message=empty_message,
    )


def _format_document_block(
    *,
    title: str,
    documents: Sequence[Document],
    empty_message: str,
) -> str:
    if not documents:
        return f"{title}\n{empty_message}"

    chunks: list[str] = [title]
    for document in documents:
        citation = document.metadata.get("citation", "S?")
        source = document.metadata.get("source", "unknown")
        page = document.metadata.get("page", "-")
        retrieval_score = document.metadata.get("retrieval_score")
        rerank_score = document.metadata.get("rerank_score")

        details = [f"[{citation}] source={source}", f"page={page}"]
        if retrieval_score is not None:
            details.append(f"retrieve={float(retrieval_score):.4f}")
        if rerank_score is not None:
            details.append(f"rerank={float(rerank_score):.4f}")

        chunks.append(" | ".join(details))
        chunks.append(document.page_content)

    return "\n".join(chunks)


def _retrieve_template_bundle(
    *,
    vectorstore: ProcurementVectorStore,
    reranker: BaseReranker,
    review_question: str,
    query_templates: Sequence[DomainQueryTemplate],
    retrieval_top_k: int,
    rerank_top_k: int,
    rerank_relative_threshold: float,
) -> list[Document]:
    best_documents: dict[str, Document] = {}

    for query_template in query_templates:
        retrieval_query = build_domain_query(review_question, query_template)
        # The reranker works better with a targeted domain template string than the full review question.
        rerank_query = get_domain_rerank_query(query_template)
        reranked_documents = _retrieve_and_rerank(
            vectorstore=vectorstore,
            reranker=reranker,
            retrieval_query=retrieval_query,
            rerank_query=rerank_query,
            query_template=query_template,
            retrieval_top_k=retrieval_top_k,
            rerank_top_k=rerank_top_k,
            rerank_relative_threshold=rerank_relative_threshold,
        )

        for document in reranked_documents:
            key = _document_result_key(document)
            existing = best_documents.get(key)
            templates = set(existing.metadata.get("matched_query_templates", [])) if existing else set()
            templates.add(query_template)

            if existing is None or float(document.metadata.get("rerank_score", 0.0)) > float(
                existing.metadata.get("rerank_score", 0.0)
            ):
                metadata = dict(document.metadata)
                metadata["matched_query_templates"] = sorted(templates)
                best_documents[key] = Document(
                    page_content=document.page_content,
                    metadata=metadata,
                )
                continue

            metadata = dict(existing.metadata)
            metadata["matched_query_templates"] = sorted(templates)
            best_documents[key] = Document(
                page_content=existing.page_content,
                metadata=metadata,
            )

    ranked_bundle = sorted(
        best_documents.values(),
        key=lambda item: float(item.metadata.get("rerank_score", 0.0)),
        reverse=True,
    )

    output: list[Document] = []
    for index, document in enumerate(ranked_bundle[:rerank_top_k], start=1):
        metadata = dict(document.metadata)
        metadata["rerank_rank"] = index
        output.append(Document(page_content=document.page_content, metadata=metadata))
    return output


def _retrieve_and_rerank(
    *,
    vectorstore: ProcurementVectorStore,
    reranker: BaseReranker,
    retrieval_query: str,
    rerank_query: str,
    query_template: DomainQueryTemplate,
    retrieval_top_k: int,
    rerank_top_k: int,
    rerank_relative_threshold: float,
) -> list[Document]:
    retrieved = vectorstore.similarity_search(retrieval_query, k=retrieval_top_k)
    if not retrieved:
        return []

    reranked = reranker.rerank(rerank_query, retrieved, top_n=retrieval_top_k)
    if not reranked:
        return []

    max_score = max(float(document.metadata.get("rerank_score", 0.0)) for document in reranked)
    relative_cutoff = max_score * rerank_relative_threshold
    kept_documents: list[Document] = []

    for document in reranked:
        score = float(document.metadata.get("rerank_score", 0.0))
        metadata = dict(document.metadata)
        metadata["retrieval_query"] = retrieval_query
        metadata["rerank_query"] = rerank_query
        metadata["rerank_max_score"] = max_score
        metadata["rerank_relative_cutoff"] = relative_cutoff
        normalized_document = Document(
            page_content=document.page_content,
            metadata=metadata,
        )

        if score < relative_cutoff:
            logger.info(
                "Excluded chunk below relative rerank threshold %.2f | collection=%s | template=%s | rerank_query=%s | max_score=%.4f | cutoff=%.4f | chunk_id=%s | source=%s | page=%s | score=%.4f",
                rerank_relative_threshold,
                vectorstore.collection_name,
                query_template,
                rerank_query,
                max_score,
                relative_cutoff,
                metadata.get("chunk_id", "unknown"),
                metadata.get("source", "unknown"),
                metadata.get("page", "-"),
                score,
            )
            continue
        kept_documents.append(normalized_document)

    if not kept_documents:
        logger.warning(
            "No chunks survived relative rerank threshold %.2f | collection=%s | template=%s | rerank_query=%s | max_score=%.4f | cutoff=%.4f",
            rerank_relative_threshold,
            vectorstore.collection_name,
            query_template,
            rerank_query,
            max_score,
            relative_cutoff,
        )

    return kept_documents[:rerank_top_k]


def _document_result_key(document: Document) -> str:
    metadata = document.metadata
    chunk_id = metadata.get("chunk_id")
    if chunk_id:
        return str(chunk_id)

    payload = "|".join(
        [
            str(metadata.get("source", "unknown")),
            str(metadata.get("page", "-")),
            document.page_content,
        ]
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def collect_chunk_ids(documents: Sequence[Document]) -> list[str]:
    return [_document_result_key(document) for document in documents]


def get_default_query_templates() -> tuple[DomainQueryTemplate, ...]:
    return get_domain_query_templates()
