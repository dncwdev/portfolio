from __future__ import annotations

import streamlit as st

from src.config import build_embeddings, build_llm, build_reranker, get_settings
from src.rag_pipeline import ProcurementRAGPipeline
from src.vectorstore import ProcurementVectorStore, UploadFilePayload


st.set_page_config(page_title="Procurement Review Agent", layout="wide")


@st.cache_resource(show_spinner=False)
def get_runtime() -> tuple[ProcurementVectorStore, ProcurementRAGPipeline]:
    settings = get_settings()
    vectorstore = ProcurementVectorStore(
        settings=settings,
        embeddings=build_embeddings(),
    )
    pipeline = ProcurementRAGPipeline(
        vectorstore=vectorstore,
        reranker=build_reranker(),
        llm=build_llm(),
        settings=settings,
    )
    return vectorstore, pipeline


def render_sidebar(vectorstore: ProcurementVectorStore) -> None:
    settings = get_settings()
    stats = vectorstore.get_stats()

    st.sidebar.title("Runtime")
    st.sidebar.write(f"Collection: `{stats['collection_name']}`")
    st.sidebar.write(f"Stored chunks: `{stats['chunk_count']}`")
    st.sidebar.write(
        f"Retrieve / Rerank: `{settings.retrieval_top_k}` / `{settings.rerank_top_k}`"
    )
    st.sidebar.write(f"LLM: `{settings.llm_model_name}`")
    st.sidebar.write(f"Embedding: `{settings.embedding_model_name}`")
    st.sidebar.write(f"Reranker: `{settings.reranker_model_name}`")


def handle_ingestion(vectorstore: ProcurementVectorStore) -> None:
    st.subheader("Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload procurement documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="PDF, TXT, MD files are chunked and stored in local ChromaDB.",
    )

    if st.button("Ingest Documents", use_container_width=True, disabled=not uploaded_files):
        payloads = [
            UploadFilePayload(name=uploaded_file.name, content=uploaded_file.getvalue())
            for uploaded_file in uploaded_files
        ]
        try:
            with st.spinner("Chunking, embedding, and indexing documents..."):
                result = vectorstore.ingest_files(payloads)
            st.session_state.pop("last_response", None)
            st.success(
                f"Indexed {result.files_indexed} file(s) into {result.chunks_indexed} chunk(s)."
            )
        except Exception as exc:  # pragma: no cover - UI feedback
            st.error(f"Ingestion failed: {exc}")


def handle_query(pipeline: ProcurementRAGPipeline, vectorstore: ProcurementVectorStore) -> None:
    st.subheader("Ask Questions")
    question = st.text_area(
        "Review question",
        placeholder="예: 계약 기간과 납품 기한 관련 조항을 요약해 주세요.",
        height=120,
    )

    has_documents = vectorstore.get_stats()["chunk_count"] > 0
    disabled = not has_documents or not question.strip()

    if not has_documents:
        st.info("문서를 먼저 업로드하고 인덱싱해야 질의를 실행할 수 있습니다.")

    if st.button("Run RAG Review", type="primary", use_container_width=True, disabled=disabled):
        try:
            with st.spinner("Retrieving evidence, reranking, and generating an answer..."):
                st.session_state["last_response"] = pipeline.invoke(question.strip())
        except Exception as exc:  # pragma: no cover - UI feedback
            st.error(f"Query failed: {exc}")

    response = st.session_state.get("last_response")
    if not response:
        return

    st.markdown("### Answer")
    st.write(response.answer)

    st.markdown("### Evidence")
    if not response.sources:
        st.warning("관련 근거를 찾지 못했습니다.")
        return

    for source in response.sources:
        label = source.metadata.get("citation", "S?")
        file_name = source.metadata.get("source", "unknown")
        page = source.metadata.get("page", "-")
        retrieval_score = float(source.metadata.get("retrieval_score", 0.0))
        rerank_score = float(source.metadata.get("rerank_score", 0.0))
        title = (
            f"[{label}] {file_name} | page {page} | "
            f"retrieve={retrieval_score:.4f} | rerank={rerank_score:.4f}"
        )
        with st.expander(title):
            st.write(source.page_content)


def main() -> None:
    st.title("Procurement Review Agent")
    st.caption("LangChain LCEL + ChromaDB + vLLM OpenAI API + Streamlit")

    vectorstore, pipeline = get_runtime()
    render_sidebar(vectorstore)

    left, right = st.columns([1, 1.3], gap="large")
    with left:
        handle_ingestion(vectorstore)
    with right:
        handle_query(pipeline, vectorstore)


if __name__ == "__main__":
    main()
