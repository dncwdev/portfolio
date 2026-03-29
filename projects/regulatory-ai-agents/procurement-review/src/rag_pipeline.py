from __future__ import annotations

from dataclasses import dataclass
from operator import itemgetter

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
)
from langchain_openai import ChatOpenAI

from .config import Settings, build_llm, get_settings
from .reranker import VLLMReranker
from .vectorstore import ProcurementVectorStore


RAG_PROMPT = """당신은 공공조달 및 규제 문서를 검토하는 AI 분석가입니다.
반드시 제공된 문맥 안에서만 답변하세요.
문맥만으로 답을 확정할 수 없으면 부족한 점을 분명히 설명하세요.
답변은 한국어로 작성하고, 핵심 근거 뒤에 [S1], [S2] 같은 출처 표기를 붙이세요.

질문:
{question}

문맥:
{context}
"""


@dataclass(frozen=True)
class RAGResponse:
    question: str
    answer: str
    retrieved_documents: list[Document]
    sources: list[Document]


class ProcurementRAGPipeline:
    def __init__(
        self,
        vectorstore: ProcurementVectorStore,
        reranker: VLLMReranker,
        llm: ChatOpenAI | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.vectorstore = vectorstore
        self.reranker = reranker
        self.llm = llm or build_llm()
        self.prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

        answer_chain = (
            {
                "question": itemgetter("question"),
                "context": itemgetter("context"),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        generate_answer = RunnableBranch(
            (
                lambda payload: not payload["sources"],
                RunnableLambda(
                    lambda _: (
                        "검색된 근거 문서가 없어 답변을 생성할 수 없습니다. "
                        "문서를 업로드하거나 질문을 더 구체화해 주세요."
                    )
                ),
            ),
            answer_chain,
        )

        self.chain = (
            RunnablePassthrough.assign(
                retrieved_documents=RunnableLambda(
                    lambda payload: self.vectorstore.similarity_search(
                        payload["question"],
                        k=self.settings.retrieval_top_k,
                    )
                )
            )
            .assign(
                sources=RunnableLambda(
                    lambda payload: self._attach_citations(
                        self.reranker.rerank(
                            payload["question"],
                            payload["retrieved_documents"],
                            top_n=self.settings.rerank_top_k,
                        )
                    )
                )
            )
            .assign(
                context=RunnableLambda(
                    lambda payload: self._format_context(payload["sources"])
                )
            )
            .assign(answer=generate_answer)
            | RunnableLambda(self._to_response)
        )

    def invoke(self, question: str) -> RAGResponse:
        return self.chain.invoke({"question": question})

    def _attach_citations(self, documents: list[Document]) -> list[Document]:
        output: list[Document] = []
        for index, document in enumerate(documents, start=1):
            metadata = dict(document.metadata)
            metadata["citation"] = f"S{index}"
            output.append(Document(page_content=document.page_content, metadata=metadata))
        return output

    def _format_context(self, documents: list[Document]) -> str:
        if not documents:
            return "검색된 근거 없음"

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
            retrieved_documents=payload["retrieved_documents"],
            sources=payload["sources"],
        )
