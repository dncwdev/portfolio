from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.agent_tools as agent_tools_module
import src.rag_pipeline as rag_pipeline_module
from src.agent_tools import EvidenceCollector, retrieve_local_evidence
from src.domain_queries import get_domain_query_templates
from src.rag_pipeline import ProcurementRAGPipeline


class _FakeStore:
    def __init__(self, chunk_count: int) -> None:
        self._chunk_count = chunk_count
        self.collection_name = "fake"

    def get_stats(self) -> dict[str, int]:
        return {"chunk_count": self._chunk_count}

    def similarity_search(self, query: str, k: int | None = None) -> list[Document]:
        del query, k
        return [
            Document(
                page_content="retrieved chunk",
                metadata={"source": "fake", "page": "1", "chunk_id": "chunk-1"},
            )
        ]


class _FakeProfile:
    key = "test"
    display_name = "Test"
    engine = "stub"
    base_url = "http://stub"


class _FakeReranker:
    profile = _FakeProfile()

    def rerank(self, query: str, documents: list[Document], top_n: int) -> list[Document]:
        del query, top_n
        output: list[Document] = []
        for document in documents:
            metadata = dict(document.metadata)
            metadata["rerank_score"] = 0.9
            output.append(Document(page_content=document.page_content, metadata=metadata))
        return output


class _FakeSettings:
    use_mcp = True
    rerank_top_k = 5
    retrieval_top_k = 5
    rerank_relative_threshold = 0.15
    llm_model_name = "qwen3.5-35b-a3b"

    @staticmethod
    def is_commercial_model() -> bool:
        return False


def _local_document(text: str = "local document evidence") -> Document:
    return Document(
        page_content=text,
        metadata={"source": "local-doc", "page": "1"},
    )


def _local_regulation(text: str = "local regulation evidence") -> Document:
    return Document(
        page_content=text,
        metadata={"source": "local-law", "page": "2"},
    )


def _mcp_regulation(text: str = "mcp regulation evidence") -> Document:
    return Document(
        page_content=text,
        metadata={
            "source": "mcp-law",
            "page": "-",
            "origin": "mcp",
            "mcp_tool": "search_law",
        },
    )


class MCPPreflightRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.traces: list[dict[str, object]] = []
        self.pipeline = ProcurementRAGPipeline(
            document_store=_FakeStore(chunk_count=10),
            regulations_store=_FakeStore(chunk_count=10),
            reranker=_FakeReranker(),
            llm=RunnableLambda(lambda _: "stub answer"),
            settings=_FakeSettings(),
        )
        self.pipeline.answer_chain = RunnableLambda(lambda _: "stub answer")

    def _capture_trace(self, trace: dict[str, object]) -> None:
        self.traces.append(copy.deepcopy(trace))

    def _make_local_retriever(
        self,
        *,
        regulation_docs: list[Document] | None = None,
        document_docs: list[Document] | None = None,
    ):
        regulation_docs = regulation_docs or []
        document_docs = document_docs or [_local_document()]

        def _fake_retrieve_local_evidence(**kwargs):
            collector = kwargs["collector"]
            trigger = kwargs.get("trigger", "pipeline")
            scope = kwargs.get("scope", "all")
            query_templates = list(kwargs.get("query_templates", []))
            collector.add_regulation_documents(regulation_docs)
            collector.add_document_documents(document_docs)
            collector.record_local_retrieval(
                trigger=trigger,
                scope=scope,
                query_templates=query_templates,
                query_count=max(1, len(query_templates)),
                regulation_count=len(regulation_docs),
                document_count=len(document_docs),
            )
            return regulation_docs, document_docs

        return _fake_retrieve_local_evidence

    def test_law_question_forces_normalized_preflight_and_skips_agent_loop(self) -> None:
        class _FakeMCPClient:
            calls: list[dict[str, object]] = []

            def __init__(self, settings) -> None:
                del settings

            def search_documents(self, **kwargs):
                type(self).calls.append(kwargs)
                if len(type(self).calls) == 1:
                    raise RuntimeError("no result for title-only query")
                return [_mcp_regulation()]

        def _unexpected_agent_loop(self, question, collector):
            del self, question, collector
            raise AssertionError("Agent loop should not run for forced MCP questions.")

        with (
            patch.object(rag_pipeline_module, "_append_trace", self._capture_trace),
            patch.object(rag_pipeline_module, "KoreanLawMCPClient", _FakeMCPClient),
            patch.object(
                rag_pipeline_module.ProcurementRAGPipeline,
                "_gather_agent_evidence",
                _unexpected_agent_loop,
            ),
            patch.object(
                rag_pipeline_module,
                "retrieve_local_evidence",
                self._make_local_retriever(document_docs=[_local_document()]),
            ),
        ):
            response = self.pipeline.invoke(
                "「정보시스템 감리기준」 제5조제3항 전단에 따른 요건이 규격서에 포함되어 있는가?"
            )

        self.assertEqual(response.answer, "stub answer")
        self.assertEqual(len(_FakeMCPClient.calls), 2)
        self.assertEqual(
            [call["query"] for call in _FakeMCPClient.calls],
            ["정보시스템 감리기준", "정보시스템 감리기준 제5조"],
        )
        trace = self.traces[-1]
        self.assertTrue(trace["forced_mcp_preflight"])
        self.assertEqual(trace["forced_mcp_reason"], "law_name_and_article_pattern")
        self.assertEqual(trace["forced_mcp_query"], "정보시스템 감리기준")
        self.assertEqual(trace["mcp_preflight_attempt_count"], 2)
        self.assertEqual(
            trace["mcp_preflight_queries"],
            ["정보시스템 감리기준", "정보시스템 감리기준 제5조"],
        )
        self.assertFalse(trace["agent_loop_entered"])
        self.assertEqual(trace["forced_mcp_result_origin"], "mcp")
        self.assertEqual(trace["final_chat_completion_calls"], 1)

    def test_failed_preflight_preserves_local_fallback(self) -> None:
        class _FailingMCPClient:
            calls = 0

            def __init__(self, settings) -> None:
                del settings

            def search_documents(self, **kwargs):
                del kwargs
                type(self).calls += 1
                raise RuntimeError("forced failure")

        def _unexpected_agent_loop(self, question, collector):
            del self, question, collector
            raise AssertionError("Agent loop should not run for forced MCP questions.")

        with (
            patch.object(rag_pipeline_module, "_append_trace", self._capture_trace),
            patch.object(rag_pipeline_module, "KoreanLawMCPClient", _FailingMCPClient),
            patch.object(
                rag_pipeline_module.ProcurementRAGPipeline,
                "_gather_agent_evidence",
                _unexpected_agent_loop,
            ),
            patch.object(
                rag_pipeline_module,
                "retrieve_local_evidence",
                self._make_local_retriever(
                    regulation_docs=[_local_regulation()],
                    document_docs=[_local_document()],
                ),
            ),
        ):
            response = self.pipeline.invoke(
                "「소프트웨어진흥법」 제52조에 따른 제안서 보상 적용 여부가 규격서에 포함되어 있는가?"
            )

        self.assertEqual(response.answer, "stub answer")
        self.assertEqual(_FailingMCPClient.calls, 2)
        trace = self.traces[-1]
        self.assertTrue(trace["forced_mcp_preflight"])
        self.assertTrue(trace["fallback_to_local"])
        self.assertFalse(trace["mcp_succeeded"])
        self.assertIsNone(trace["forced_mcp_result_origin"])
        self.assertEqual(trace["mcp_preflight_attempt_count"], 2)
        self.assertEqual(trace["final_document_origins"], ["local"])
        self.assertEqual(trace["final_regulation_origins"], ["local"])
        self.assertEqual(trace["final_chat_completion_calls"], 1)

    def test_question_without_law_pattern_keeps_existing_behavior(self) -> None:
        calls = {"agent": 0}

        def _fake_agent_loop(self, question, collector):
            del self, question
            calls["agent"] += 1
            collector.record_agent_started(available_tools=["search_local_procurement_context"])
            collector.trace["agent_loop_entered"] = True

        with (
            patch.object(rag_pipeline_module, "_append_trace", self._capture_trace),
            patch.object(
                rag_pipeline_module.ProcurementRAGPipeline,
                "_gather_agent_evidence",
                _fake_agent_loop,
            ),
            patch.object(
                rag_pipeline_module,
                "retrieve_local_evidence",
                self._make_local_retriever(document_docs=[_local_document()]),
            ),
        ):
            response = self.pipeline.invoke("사업 범위와 산출물 일정이 규격서에 포함되어 있는가?")

        self.assertEqual(response.answer, "stub answer")
        self.assertEqual(calls["agent"], 1)
        trace = self.traces[-1]
        self.assertFalse(trace["forced_mcp_preflight"])
        self.assertTrue(trace["agent_loop_entered"])
        self.assertFalse(trace["mcp_attempted"])

    def test_mcp_mode_local_retrieval_uses_single_query_only(self) -> None:
        collector = EvidenceCollector()
        collector.init_trace(question="질문", use_mcp=True)
        document_store = _FakeStore(chunk_count=10)
        regulations_store = _FakeStore(chunk_count=10)
        settings = _FakeSettings()

        with patch.object(
            agent_tools_module,
            "_retrieve_with_multi_query",
            side_effect=AssertionError("multi-query should not run in MCP mode"),
        ):
            regulation_sources, document_sources = retrieve_local_evidence(
                query="「정보시스템 감리기준」 제5조제3항 전단에 따른 요건이 규격서에 포함되어 있는가?",
                scope="all",
                top_k=5,
                document_store=document_store,
                regulations_store=regulations_store,
                reranker=_FakeReranker(),
                collector=collector,
                settings=settings,
                query_templates=(get_domain_query_templates()[0],),
                trigger="pipeline_fallback",
            )

        self.assertTrue(regulation_sources)
        self.assertTrue(document_sources)
        self.assertEqual(collector.trace["local_retrieval_query_count"], 2)


if __name__ == "__main__":
    unittest.main()
