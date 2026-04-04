from __future__ import annotations

from dataclasses import dataclass

from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .agent_tools import (
    EvidenceCollector,
    build_agent_tools,
    get_default_query_templates,
    retrieve_local_evidence,
)
from .config import Settings, build_llm, get_settings
from .domain_queries import format_domain_query_template_guide
from .reranker import BaseReranker
from .vectorstore import ProcurementVectorStore


NO_DOCUMENT_MESSAGE = (
    "조달 문서 근거를 찾지 못해 준수 여부를 판단할 수 없습니다. "
    "문서를 업로드하고 다시 시도해 주세요."
)

NO_REGULATION_MESSAGE = (
    "규정 근거가 없습니다. 로컬 규정 DB를 채우거나 USE_MCP=true로 설정한 뒤 다시 시도해 주세요."
)

AGENT_SYSTEM_PROMPT = """당신은 공공조달 구매규격서 검토를 위한 증거 수집 에이전트입니다.

반드시 다음 원칙을 따르세요.
- 로컬 ChromaDB 검색이 필요하면 자유로운 검색어를 만들지 말고, 아래의 고정 템플릿 중 하나를 `query_template`로 선택해 `search_local_procurement_context`를 호출하세요.
{query_template_guide}
- 질문과 가장 관련 있는 템플릿을 1개 이상 선택할 수 있지만, 불필요하게 같은 템플릿을 반복 호출하지 마세요.
- {mcp_instruction}
- 도구가 반환한 근거만 사용하세요. 사전지식으로 근거를 꾸며내지 마세요.
- 최종 응답은 짧아도 되지만, 어떤 템플릿과 도구를 사용했는지 드러나게 하세요.
"""

ANSWER_PROMPT = """당신은 공공조달 구매규격서 초안의 법령 준수 여부를 검토하는 AI 분석가입니다.

아래에 제공된 규정 근거와 문서 근거만 사용해서 판단하세요.
- 검토 범위는 규격서 작성 단계에서 법령이 요구하는 조항의 포함 여부입니다. 계약 이후 이행 여부는 판단 대상이 아닙니다.
- 낙찰 후 제출 서류, 계약 이행, 검사·검수, 대금 지급, 하자보수, 운영 단계 확인사항은 `추가 확인 필요사항`에 포함하지 마세요.
- 근거가 부족하거나 상충하면 반드시 `추가 확인 필요`로 판단하세요.
- 규정 근거는 [R1], [R2] 형식으로, 문서 근거는 [D1], [D2] 형식으로 인용하세요.
- 답변은 아래 형식을 그대로 지키세요.

## 준수 판단
- 준수 / 위반 가능성 / 추가 확인 필요 중 하나

## 핵심 근거
- 규정 근거와 문서 근거를 함께 묶어 핵심만 요약

## 판단 근거
- 왜 그렇게 판단했는지 설명

## 추가 확인 필요사항
- 입찰 전 구매규격서에 반영되어야 하지만 현재 근거가 부족한 항목만 적기

질문:
{question}

규정 근거:
{regulations_context}

조달 문서 근거:
{document_context}
"""


@dataclass(frozen=True)
class RAGResponse:
    question: str
    answer: str
    regulation_sources: list[Document]
    document_sources: list[Document]
    used_mcp: bool = False
    reranker_key: str = "default"
    reranker_name: str = ""
    reranker_engine: str = ""
    reranker_base_url: str = ""


class ProcurementRAGPipeline:
    def __init__(
        self,
        document_store: ProcurementVectorStore,
        regulations_store: ProcurementVectorStore,
        reranker: BaseReranker,
        llm: ChatOpenAI | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.document_store = document_store
        self.regulations_store = regulations_store
        self.reranker = reranker
        self.llm = llm or build_llm()
        self.answer_chain = (
            ChatPromptTemplate.from_template(ANSWER_PROMPT)
            | self.llm
            | StrOutputParser()
        )

    def invoke(self, question: str) -> RAGResponse:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("Question must not be empty.")

        if self.document_store.get_stats()["chunk_count"] == 0:
            return self._build_response(
                question=normalized_question,
                answer=NO_DOCUMENT_MESSAGE,
                collector=EvidenceCollector(),
            )

        if (
            self.regulations_store.get_stats()["chunk_count"] == 0
            and not self.settings.use_mcp
        ):
            return self._build_response(
                question=normalized_question,
                answer=NO_REGULATION_MESSAGE,
                collector=EvidenceCollector(),
            )

        collector = EvidenceCollector()
        if self.settings.use_mcp:
            self._gather_agent_evidence(normalized_question, collector)

        if (
            not self.settings.use_mcp
            or not collector.document_sources
            or (
                not collector.regulation_sources
                and self.regulations_store.get_stats()["chunk_count"] > 0
            )
        ):
            retrieve_local_evidence(
                query=normalized_question,
                scope="all",
                top_k=self.settings.rerank_top_k,
                document_store=self.document_store,
                regulations_store=self.regulations_store,
                reranker=self.reranker,
                collector=collector,
                settings=self.settings,
                query_templates=get_default_query_templates(),
            )

        if not collector.document_sources:
            return self._build_response(
                question=normalized_question,
                answer=NO_DOCUMENT_MESSAGE,
                collector=collector,
            )

        answer = self.answer_chain.invoke(
            {
                "question": normalized_question,
                "regulations_context": self._format_context(
                    collector.regulation_sources,
                    empty_message="관련 규정 근거 없음",
                ),
                "document_context": self._format_context(
                    collector.document_sources,
                    empty_message="관련 조달 문서 근거 없음",
                ),
            }
        )

        return self._build_response(
            question=normalized_question,
            answer=answer,
            collector=collector,
        )

    def _gather_agent_evidence(
        self,
        question: str,
        collector: EvidenceCollector,
    ) -> None:
        tools = build_agent_tools(
            document_store=self.document_store,
            regulations_store=self.regulations_store,
            reranker=self.reranker,
            collector=collector,
            settings=self.settings,
            question_context=question,
        )
        agent = create_agent(
            model=self.llm,
            tools=tools,
            system_prompt=self._build_agent_system_prompt(),
        )

        try:
            agent.invoke({"messages": [{"role": "user", "content": question}]})
        except Exception:
            # If tool calling is not fully supported by the runtime model, the
            # pipeline falls back to deterministic local retrieval below.
            return

    def _build_agent_system_prompt(self) -> str:
        if self.settings.use_mcp:
            mcp_instruction = (
                "`search_korean_law_mcp`는 필요할 때만 호출하고, "
                "로컬 검색은 반드시 고정 템플릿으로 수행하세요."
            )
        else:
            mcp_instruction = (
                "현재 MCP 법령 검색은 비활성화되어 있으므로 로컬 ChromaDB와 고정 템플릿만 사용하세요."
            )
        return AGENT_SYSTEM_PROMPT.format(
            mcp_instruction=mcp_instruction,
            query_template_guide=format_domain_query_template_guide(),
        )

    def _format_context(self, documents: list[Document], empty_message: str) -> str:
        if not documents:
            return empty_message

        chunks: list[str] = []
        for document in documents:
            citation = document.metadata.get("citation", "S?")
            source = document.metadata.get("source", "unknown")
            page = document.metadata.get("page", "-")
            matched_templates = document.metadata.get("matched_query_templates", [])
            template_text = (
                f" templates={','.join(matched_templates)}"
                if matched_templates
                else ""
            )
            chunks.append(
                f"[{citation}] source={source} page={page}{template_text}\n{document.page_content}"
            )
        return "\n\n".join(chunks)

    def _build_response(
        self,
        *,
        question: str,
        answer: str,
        collector: EvidenceCollector,
    ) -> RAGResponse:
        used_mcp = any(
            source.metadata.get("origin") == "mcp"
            for source in collector.regulation_sources
        )
        return RAGResponse(
            question=question,
            answer=answer,
            regulation_sources=list(collector.regulation_sources),
            document_sources=list(collector.document_sources),
            used_mcp=used_mcp,
            reranker_key=self.reranker.profile.key,
            reranker_name=self.reranker.profile.display_name,
            reranker_engine=self.reranker.profile.engine,
            reranker_base_url=self.reranker.profile.base_url,
        )
