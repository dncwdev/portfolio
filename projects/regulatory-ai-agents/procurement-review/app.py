from __future__ import annotations

import streamlit as st

from src.config import build_embeddings, build_llm, build_reranker, get_settings
from src.rag_pipeline import ProcurementRAGPipeline
from src.vectorstore import ProcurementVectorStore, UploadFilePayload


st.set_page_config(page_title="Procurement Review Agent", layout="wide")


@st.cache_resource(show_spinner=False)
def get_vectorstores() -> tuple[ProcurementVectorStore, ProcurementVectorStore]:
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


def select_llm_model() -> str:
    settings = get_settings()
    options = list(settings.available_models)
    default_model = (
        settings.llm_model_name if settings.llm_model_name in options else options[0]
    )

    if "selected_llm_model" not in st.session_state:
        st.session_state["selected_llm_model"] = default_model
    elif st.session_state["selected_llm_model"] not in options:
        st.session_state["selected_llm_model"] = default_model

    return st.sidebar.selectbox(
        "LLM Model",
        options=options,
        index=options.index(st.session_state["selected_llm_model"]),
        key="selected_llm_model",
    )


def render_sidebar(
    selected_model: str,
    document_store: ProcurementVectorStore,
    regulations_store: ProcurementVectorStore,
) -> None:
    settings = get_settings()
    document_stats = document_store.get_stats()
    regulations_stats = regulations_store.get_stats()

    st.sidebar.title("Runtime")
    st.sidebar.write(f"Collection: `{document_stats['collection_name']}`")
    st.sidebar.write(
        "Regulations DB: "
        f"`{regulations_stats['collection_name']} ({regulations_stats['chunk_count']} chunks)`"
    )
    st.sidebar.write(f"Stored chunks: `{document_stats['chunk_count']}`")
    st.sidebar.write(
        f"Retrieve / Rerank: `{settings.retrieval_top_k}` / `{settings.rerank_top_k}`"
    )
    st.sidebar.write(f"LLM: `{selected_model}`")
    st.sidebar.write(f"Embedding: `{settings.embedding_display_name}`")
    st.sidebar.write(f"Reranker: `{settings.reranker_display_name}`")


def build_pipeline(
    selected_model: str,
    document_store: ProcurementVectorStore,
    regulations_store: ProcurementVectorStore,
) -> ProcurementRAGPipeline:
    settings = get_settings()
    return ProcurementRAGPipeline(
        document_store=document_store,
        regulations_store=regulations_store,
        reranker=build_reranker(),
        llm=build_llm(selected_model),
        settings=settings,
    )


def handle_ingestion(document_store: ProcurementVectorStore) -> None:
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
                result = document_store.ingest_files(payloads)
            st.session_state.pop("last_response", None)
            st.success(
                f"Indexed {result.files_indexed} file(s) into {result.chunks_indexed} chunk(s)."
            )
        except Exception as exc:  # pragma: no cover - UI feedback
            st.error(f"Ingestion failed: {exc}")


def render_source_section(title: str, sources: list) -> None:
    st.markdown(f"### {title}")
    if not sources:
        st.info("관련 근거를 찾지 못했습니다.")
        return

    for source in sources:
        label = source.metadata.get("citation", "S?")
        file_name = source.metadata.get("source", "unknown")
        page = source.metadata.get("page", "-")
        retrieval_score = float(source.metadata.get("retrieval_score", 0.0))
        rerank_score = float(source.metadata.get("rerank_score", 0.0))
        collection_name = source.metadata.get("collection_name", "unknown")
        title_text = (
            f"[{label}] {file_name} | page {page} | db={collection_name} | "
            f"retrieve={retrieval_score:.4f} | rerank={rerank_score:.4f}"
        )
        with st.expander(title_text):
            st.write(source.page_content)


def handle_query(
    pipeline: ProcurementRAGPipeline,
    document_store: ProcurementVectorStore,
    regulations_store: ProcurementVectorStore,
) -> None:
    st.subheader("Ask Questions")
    question = st.text_area(
        "Review question",
        placeholder="예: 이 구매규격서가 관련 규정을 준수하는지 검토해 주세요.",
        height=120,
    )

    has_documents = document_store.get_stats()["chunk_count"] > 0
    has_regulations = regulations_store.get_stats()["chunk_count"] > 0
    disabled = not has_documents or not has_regulations or not question.strip()

    if not has_documents:
        st.info("문서를 먼저 업로드하고 인덱싱해야 질의를 실행할 수 있습니다.")
    elif not has_regulations:
        st.info(
            "규정 DB가 비어 있습니다. `python scripts/ingest_regulations.py`를 먼저 실행해 주세요."
        )

    if st.button("Run RAG Review", type="primary", use_container_width=True, disabled=disabled):
        try:
            with st.spinner("Retrieving regulations, reviewing document evidence, and generating a compliance judgment..."):
                st.session_state["last_response"] = pipeline.invoke(question.strip())
        except Exception as exc:  # pragma: no cover - UI feedback
            st.error(f"Query failed: {exc}")

    response = st.session_state.get("last_response")
    if not response:
        return

    st.markdown("### Compliance Review")
    st.write(response.answer)

    render_source_section("Regulation Evidence", response.regulation_sources)
    render_source_section("Document Evidence", response.document_sources)


def main() -> None:
    st.title("Procurement Review Agent")
    st.caption("LangChain LCEL + ChromaDB + vLLM + Streamlit")

    selected_model = select_llm_model()
    document_store, regulations_store = get_vectorstores()
    pipeline = build_pipeline(selected_model, document_store, regulations_store)
    render_sidebar(selected_model, document_store, regulations_store)

    left, right = st.columns([1, 1.3], gap="large")
    with left:
        handle_ingestion(document_store)
    with right:
        handle_query(pipeline, document_store, regulations_store)


if __name__ == "__main__":
    main()
