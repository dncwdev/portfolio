from __future__ import annotations

from dataclasses import dataclass
from operator import itemgetter

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

from .config import Settings, build_llm, get_settings
from .reranker import VLLMReranker
from .vectorstore import ProcurementVectorStore


RAG_PROMPT = """당신은 공공조달 문서를 규정과 대조해 준수 여부를 검토하는 AI 분석가입니다.
반드시 제공된 규정 문맥과 문서 문맥 안에서만 판단하세요.
문맥이 불충분하면 '추가 검토 필요'로 판단하고 부족한 점을 명시하세요.

답변 형식:
## 준수 판단
- 준수 / 위반 가능성 / 추가 검토 필요 중 하나
## 핵심 근거
- 규정 근거와 문서 근거를 함께 요약
## 판단 근거
- 왜 그런 판단에 도달했는지 설명
## 추가 확인 필요사항
- 문맥 밖에서 추가로 확인할 항목

규정 근거는 [R1], [R2], 문서 근거는 [D1], [D2] 형식으로 인용하세요.

사용자 질문:
{question}

관련 규정 문맥:
{regulations_context}

조달 문서 문맥:
{document_context}
"""


@dataclass(frozen=True)
class RAGResponse:
    question: str
    answer: str
    regulation_sources: list[Document]
    document_sources: list[Document]


class ProcurementRAGPipeline:
    def __init__(
        self,
        document_store: ProcurementVectorStore,
        regulations_store: ProcurementVectorStore,
        reranker: VLLMReranker,
        llm: ChatOpenAI | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.document_store = document_store
        self.regulations_store = regulations_store
        self.reranker = reranker
        self.llm = llm or build_llm()
        self.prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

        answer_chain = (
            {
                "question": itemgetter("question"),
                "regulations_context": itemgetter("regulations_context"),
                "document_context": itemgetter("document_context"),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        generate_answer = RunnableBranch(
            (
                lambda payload: not payload["document_sources"],
                RunnableLambda(
                    lambda _: (
                        "조달 문서 근거를 찾지 못해 준수 여부를 검토할 수 없습니다. "
                        "문서를 업로드하고 인덱싱한 뒤 다시 시도해 주세요."
                    )
                ),
            ),
            answer_chain,
        )

        self.chain = (
            RunnablePassthrough.assign(
                regulations_query=RunnableLambda(
                    lambda payload: self._build_regulations_query(payload["question"])
                )
            )
            .assign(
                regulation_sources=RunnableLambda(
                    lambda payload: self._retrieve_and_rerank(
                        self.regulations_store,
                        payload["regulations_query"],
                        citation_prefix="R",
                        source_group="regulation",
                    )
                )
            )
            .assign(
                document_sources=RunnableLambda(
                    lambda payload: self._retrieve_and_rerank(
                        self.document_store,
                        payload["question"],
                        citation_prefix="D",
                        source_group="document",
                    )
                )
            )
            .assign(
                regulations_context=RunnableLambda(
                    lambda payload: self._format_context(
                        payload["regulation_sources"],
                        empty_message="관련 규정 근거 없음",
                    )
                )
            )
            .assign(
                document_context=RunnableLambda(
                    lambda payload: self._format_context(
                        payload["document_sources"],
                        empty_message="관련 조달 문서 근거 없음",
                    )
                )
            )
            .assign(answer=generate_answer)
            | RunnableLambda(self._to_response)
        )

    def invoke(self, question: str) -> RAGResponse:
        return self.chain.invoke({"question": question})

    def _build_regulations_query(self, question: str) -> str:
        procurement_excerpt = self.document_store.get_collection_excerpt()
        if not procurement_excerpt:
            return question
        return (
            f"검토 질문:\n{question}\n\n"
            "조달 문서 발췌:\n"
            f"{procurement_excerpt}"
        )

    def _retrieve_and_rerank(
        self,
        vectorstore: ProcurementVectorStore,
        query: str,
        citation_prefix: str,
        source_group: str,
    ) -> list[Document]:
        retrieved = vectorstore.similarity_search(
            query,
            k=self.settings.retrieval_top_k,
        )
        reranked = self.reranker.rerank(
            query,
            retrieved,
            top_n=self.settings.rerank_top_k,
        )
        return self._attach_citations(
            reranked,
            prefix=citation_prefix,
            source_group=source_group,
        )

    def _attach_citations(
        self,
        documents: list[Document],
        prefix: str,
        source_group: str,
    ) -> list[Document]:
        output: list[Document] = []
        for index, document in enumerate(documents, start=1):
            metadata = dict(document.metadata)
            metadata["citation"] = f"{prefix}{index}"
            metadata["source_group"] = source_group
            output.append(Document(page_content=document.page_content, metadata=metadata))
        return output

    def _format_context(self, documents: list[Document], empty_message: str) -> str:
        if not documents:
            return empty_message

        chunks: list[str] = []
        for document in documents:
            citation = document.metadata.get("citation", "S?")
            source = document.metadata.get("source", "unknown")
            page = document.metadata.get("page", "-")
            chunks.append(
                f"[{citation}] source={source} page={page}\n{document.page_content}"
            )
        return "\n\n".join(chunks)

    def _to_response(self, payload: dict[str, object]) -> RAGResponse:
        return RAGResponse(
            question=payload["question"],
            answer=payload["answer"],
            regulation_sources=payload["regulation_sources"],
            document_sources=payload["document_sources"],
        )
