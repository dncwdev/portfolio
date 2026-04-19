from __future__ import annotations

import asyncio
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.documents import Document
from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp import ClientSession

from .config import Settings, get_settings


KoreanLawSearchType = Literal["law", "precedent", "interpretation", "all"]


@dataclass(frozen=True)
class MCPDocumentResult:
    tool_name: str
    title: str
    text: str

    def to_document(self) -> Document:
        return Document(
            page_content=self.text,
            metadata={
                "source": self.title,
                "page": "-",
                "origin": "mcp",
                "mcp_tool": self.tool_name,
                "collection_name": "korean-law-mcp",
            },
        )


class KoreanLawMCPClient:
    SERVER_NAME = "korean-law"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def search_documents(
        self,
        *,
        search_type: KoreanLawSearchType,
        query: str | None = None,
        article: str | None = None,
        document_id: str | None = None,
        limit: int = 5,
    ) -> list[Document]:
        try:
            results = self._run_sync(
                self._asearch_documents(
                    search_type=search_type,
                    query=query,
                    article=article,
                    document_id=document_id,
                    limit=limit,
                )
            )
        except BaseException as exc:
            normalized = self._unwrap_exception(exc)
            if normalized is exc:
                raise
            raise normalized from exc
        return [result.to_document() for result in results]

    async def _asearch_documents(
        self,
        *,
        search_type: KoreanLawSearchType,
        query: str | None = None,
        article: str | None = None,
        document_id: str | None = None,
        limit: int = 5,
    ) -> list[MCPDocumentResult]:
        client = MultiServerMCPClient({self.SERVER_NAME: self._build_connection()})
        async with client.session(self.SERVER_NAME) as session:
            if search_type == "law":
                return await self._search_law(
                    session=session,
                    query=query,
                    article=article,
                    document_id=document_id,
                    limit=limit,
                )
            if search_type == "precedent":
                return await self._search_precedent(
                    session=session,
                    query=query,
                    document_id=document_id,
                    limit=limit,
                )
            if search_type == "interpretation":
                return await self._search_interpretation(
                    session=session,
                    query=query,
                    document_id=document_id,
                    limit=limit,
                )
            if search_type == "all":
                if not query:
                    raise ValueError("search_type='all' requires a query.")
                output = await self._call_text(
                    session,
                    "search_all",
                    {"query": query},
                )
                return [
                    MCPDocumentResult(
                        tool_name="search_all",
                        title="korean-law-mcp/search_all",
                        text=output,
                    )
                ]

        raise ValueError(f"Unsupported search type: {search_type}")

    async def _search_law(
        self,
        *,
        session: ClientSession,
        query: str | None,
        article: str | None,
        document_id: str | None,
        limit: int,
    ) -> list[MCPDocumentResult]:
        results: list[MCPDocumentResult] = []

        law_identifier = self._normalize_law_identifier(document_id)
        derived_query, derived_article = self._normalize_law_query(query)
        effective_query = derived_query or query
        effective_article = article or derived_article
        if article and not query and not law_identifier:
            raise ValueError("Law article lookup requires a query or document_id.")

        if effective_query:
            search_text = await self._call_text(
                session,
                "search_law",
                {"query": effective_query, "display": limit},
            )
            results.append(
                MCPDocumentResult(
                    tool_name="search_law",
                    title="korean-law-mcp/search_law",
                    text=search_text,
                )
            )
            if law_identifier is None:
                law_identifier = self._extract_first_law_identifier(search_text)

        if effective_article or law_identifier is not None:
            arguments: dict[str, Any] = {}
            if law_identifier is not None:
                arguments.update(law_identifier)
            if effective_article:
                arguments["jo"] = effective_article

            detail_text = await self._call_text(session, "get_law_text", arguments)
            results.append(
                MCPDocumentResult(
                    tool_name="get_law_text",
                    title="korean-law-mcp/get_law_text",
                    text=detail_text,
                )
            )

        if not results and law_identifier is not None:
            detail_text = await self._call_text(session, "get_law_text", law_identifier)
            results.append(
                MCPDocumentResult(
                    tool_name="get_law_text",
                    title="korean-law-mcp/get_law_text",
                    text=detail_text,
                )
            )

        if not results:
            raise ValueError("Law search requires a query or document_id.")

        return results

    @staticmethod
    def _normalize_law_query(query: str | None) -> tuple[str | None, str | None]:
        if not query:
            return None, None

        normalized = re.sub(r"\s+", " ", query).strip()
        if not normalized:
            return None, None

        article_match = re.search(
            r"(제\s*\d+\s*조(?:\s*의\s*\d+)?(?:\s*제\s*\d+\s*항)?(?:\s*제\s*\d+\s*호)?)",
            normalized,
        )
        if not article_match:
            return normalized, None

        article = re.sub(r"\s+", "", article_match.group(1))
        law_query = normalized[: article_match.start()].strip(" ,:;")
        if not law_query:
            law_query = normalized
        return law_query, article

    async def _search_precedent(
        self,
        *,
        session: ClientSession,
        query: str | None,
        document_id: str | None,
        limit: int,
    ) -> list[MCPDocumentResult]:
        if document_id:
            output = await self._call_text(
                session,
                "get_precedent_text",
                {"id": document_id},
            )
            return [
                MCPDocumentResult(
                    tool_name="get_precedent_text",
                    title="korean-law-mcp/get_precedent_text",
                    text=output,
                )
            ]

        if not query:
            raise ValueError("Precedent search requires a query or document_id.")

        output = await self._call_text(
            session,
            "search_precedents",
            {"query": query, "display": limit},
        )
        return [
            MCPDocumentResult(
                tool_name="search_precedents",
                title="korean-law-mcp/search_precedents",
                text=output,
            )
        ]

    async def _search_interpretation(
        self,
        *,
        session: ClientSession,
        query: str | None,
        document_id: str | None,
        limit: int,
    ) -> list[MCPDocumentResult]:
        if document_id:
            output = await self._call_text(
                session,
                "get_interpretation_text",
                {"id": document_id},
            )
            return [
                MCPDocumentResult(
                    tool_name="get_interpretation_text",
                    title="korean-law-mcp/get_interpretation_text",
                    text=output,
                )
            ]

        if not query:
            raise ValueError("Interpretation search requires a query or document_id.")

        output = await self._call_text(
            session,
            "search_interpretations",
            {"query": query, "display": limit},
        )
        return [
            MCPDocumentResult(
                tool_name="search_interpretations",
                title="korean-law-mcp/search_interpretations",
                text=output,
            )
        ]

    async def _call_text(
        self,
        session: ClientSession,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        result = await session.call_tool(tool_name, arguments)
        output_parts: list[str] = []
        for item in result.content:
            item_type = getattr(item, "type", None)
            if item_type == "text":
                output_parts.append(getattr(item, "text", ""))
            elif item_type == "resource":
                resource = getattr(item, "resource", None)
                resource_text = getattr(resource, "text", None)
                if resource_text:
                    output_parts.append(resource_text)

        output = "\n\n".join(part.strip() for part in output_parts if part and part.strip())
        if result.isError:
            raise RuntimeError(output or f"{tool_name} returned an error.")
        if not output:
            raise RuntimeError(f"{tool_name} returned no text content.")
        return output

    def _build_connection(self) -> dict[str, Any]:
        transport = self.settings.korean_law_mcp_transport.strip().lower()
        if transport == "stdio":
            if not self.settings.law_oc:
                raise ValueError(
                    "LAW_OC is required when USE_MCP=true and "
                    "KOREAN_LAW_MCP_TRANSPORT=stdio."
                )
            return {
                "transport": "stdio",
                "command": self.settings.korean_law_mcp_command,
                "args": list(self.settings.korean_law_mcp_args),
                "env": {**dict(os.environ), "LAW_OC": self.settings.law_oc},
            }

        if transport in {"http", "streamable_http", "streamable-http"}:
            if not self.settings.korean_law_mcp_url:
                raise ValueError(
                    "KOREAN_LAW_MCP_URL is required when using HTTP MCP transport."
                )
            return {
                "transport": "http",
                "url": self.settings.korean_law_mcp_url,
            }

        if transport == "sse":
            if not self.settings.korean_law_mcp_url:
                raise ValueError(
                    "KOREAN_LAW_MCP_URL is required when using SSE MCP transport."
                )
            return {
                "transport": "sse",
                "url": self.settings.korean_law_mcp_url,
            }

        raise ValueError(
            "Unsupported KOREAN_LAW_MCP_TRANSPORT. "
            "Use one of: stdio, http, streamable_http, sse."
        )

    @staticmethod
    def _normalize_law_identifier(document_id: str | None) -> dict[str, str] | None:
        if not document_id:
            return None

        trimmed = document_id.strip()
        if not trimmed:
            return None
        if trimmed.isdigit():
            return {"mst": trimmed}
        return {"lawId": trimmed}

    @staticmethod
    def _extract_first_law_identifier(search_text: str) -> dict[str, str]:
        mst_match = re.search(r"\bMST:\s*([A-Za-z0-9_-]+)", search_text)
        if mst_match:
            return {"mst": mst_match.group(1)}

        law_id_match = re.search(r"lawId:\s*([A-Za-z0-9_-]+)", search_text)
        if law_id_match:
            return {"lawId": law_id_match.group(1)}

        korean_law_id_match = re.search(r"법령ID:\s*([A-Za-z0-9_-]+)", search_text)
        if korean_law_id_match:
            return {"lawId": korean_law_id_match.group(1)}

        raise ValueError(
            "search_law returned no MST/lawId. Refine the query or provide document_id."
        )

    @staticmethod
    def _run_sync(coro: Any):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()

    @classmethod
    def _unwrap_exception(cls, exc: BaseException) -> BaseException:
        nested = getattr(exc, "exceptions", None)
        if isinstance(nested, tuple) and len(nested) == 1:
            return cls._unwrap_exception(nested[0])
        return exc
