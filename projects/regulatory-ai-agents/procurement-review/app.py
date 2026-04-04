from __future__ import annotations

from dataclasses import replace
from urllib.parse import urlparse

import streamlit as st

from src.config import (
    ENV_PATH,
    build_embeddings,
    build_llm,
    build_reranker,
    clear_config_caches,
    get_settings,
)
from src.rag_pipeline import ProcurementRAGPipeline
from src.vectorstore import ProcurementVectorStore, UploadFilePayload


st.set_page_config(page_title="Procurement Review Agent", layout="wide")


def format_reranker_label(reranker_key: str) -> str:
  profile = get_settings().get_reranker_profile(reranker_key)
  parsed = urlparse(profile.base_url)
  endpoint = parsed.netloc or parsed.path or profile.base_url
  return f"{profile.display_name} [{profile.engine}] @ {endpoint}"


def format_mode_label(mode_key: str) -> str:
  return "local + MCP" if mode_key == "local_mcp" else "local only"


def refresh_runtime_config() -> None:
  env_mtime = ENV_PATH.stat().st_mtime_ns if ENV_PATH.exists() else None
  last_mtime = st.session_state.get("_env_mtime_ns")

  if last_mtime is None:
    st.session_state["_env_mtime_ns"] = env_mtime
    return

  if env_mtime != last_mtime:
    clear_config_caches()
    get_vectorstores.clear()
    st.session_state.pop("selected_llm_model", None)
    st.session_state.pop("selected_reranker_key", None)
    st.session_state.pop("selected_runtime_mode", None)
    st.session_state["_env_mtime_ns"] = env_mtime


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


def select_runtime_mode() -> bool:
  settings = get_settings()
  options = ["local_only", "local_mcp"]
  default_mode = "local_mcp" if settings.use_mcp else "local_only"
  previous_mode = st.session_state.get("selected_runtime_mode")

  if "selected_runtime_mode" not in st.session_state:
    st.session_state["selected_runtime_mode"] = default_mode
  elif st.session_state["selected_runtime_mode"] not in options:
    st.session_state["selected_runtime_mode"] = default_mode

  selected_mode = st.sidebar.selectbox(
      "Mode",
      options=options,
      index=options.index(st.session_state["selected_runtime_mode"]),
      key="selected_runtime_mode",
      format_func=format_mode_label,
      help="The default comes from USE_MCP in .env, but you can override it here.",
  )

  if previous_mode is not None and previous_mode != selected_mode:
    st.session_state.pop("last_response", None)

  return selected_mode == "local_mcp"


def build_runtime_settings(use_mcp: bool):
  settings = get_settings()
  if settings.use_mcp == use_mcp:
    return settings
  return replace(settings, use_mcp=use_mcp)


def render_sidebar(
    runtime_settings,
    selected_model: str,
    selected_reranker_key: str,
    document_store: ProcurementVectorStore,
    regulations_store: ProcurementVectorStore,
) -> None:
  reranker_profile = runtime_settings.get_reranker_profile(selected_reranker_key)
  document_stats = document_store.get_stats()
  regulations_stats = regulations_store.get_stats()

  st.sidebar.title("Runtime")
  st.sidebar.write(f"Collection: `{document_stats['collection_name']}`")
  st.sidebar.write(
      "Regulations DB: "
      f"`{regulations_stats['collection_name']} ({regulations_stats['chunk_count']} chunks)`"
  )
  st.sidebar.write(f"Stored chunks: `{document_stats['chunk_count']}`")
  if st.sidebar.button("Clear Document DB", use_container_width=True):
    st.session_state["confirm_clear_document_db"] = True

  if st.session_state.get("confirm_clear_document_db"):
    st.sidebar.warning(
        "This clears only the uploaded document database. Regulations DB will be kept."
    )
    confirm_col, cancel_col = st.sidebar.columns(2)
    if confirm_col.button("Confirm Clear", use_container_width=True):
      document_store.clear_collection()
      st.session_state["confirm_clear_document_db"] = False
      st.session_state["document_db_cleared"] = True
      st.session_state.pop("last_response", None)
      st.rerun()
    if cancel_col.button("Cancel", use_container_width=True):
      st.session_state["confirm_clear_document_db"] = False
      st.rerun()

  if st.session_state.pop("document_db_cleared", False):
    st.sidebar.success("Document DB cleared.")

  st.sidebar.write(
      f"Retrieve / Rerank: `{runtime_settings.retrieval_top_k}` / `{runtime_settings.rerank_top_k}`"
  )
  st.sidebar.write(
      f"Mode: `{format_mode_label(st.session_state['selected_runtime_mode'])}`"
  )
  st.sidebar.caption(
      f".env default: `{format_mode_label('local_mcp' if get_settings().use_mcp else 'local_only')}`"
  )
  if runtime_settings.use_mcp:
    st.sidebar.write(f"MCP Transport: `{runtime_settings.korean_law_mcp_transport}`")
  st.sidebar.write(f"LLM: `{selected_model}`")
  st.sidebar.write(f"Embedding: `{runtime_settings.embedding_display_name}`")
  st.sidebar.write(f"Reranker: `{format_reranker_label(selected_reranker_key)}`")


def select_reranker() -> str:
  settings = get_settings()
  options = list(settings.available_reranker_keys)
  default_key = (
      settings.default_reranker_key
      if settings.default_reranker_key in options
      else options[0]
  )
  previous_key = st.session_state.get("selected_reranker_key")

  if "selected_reranker_key" not in st.session_state:
    st.session_state["selected_reranker_key"] = default_key
  elif st.session_state["selected_reranker_key"] not in options:
    st.session_state["selected_reranker_key"] = default_key

  selected_key = st.sidebar.selectbox(
      "Reranker",
      options=options,
      index=options.index(st.session_state["selected_reranker_key"]),
      key="selected_reranker_key",
      format_func=format_reranker_label,
  )

  if previous_key is not None and previous_key != selected_key:
    st.session_state.pop("last_response", None)

  return selected_key


def build_pipeline(
    runtime_settings,
    selected_model: str,
    selected_reranker_key: str,
    document_store: ProcurementVectorStore,
    regulations_store: ProcurementVectorStore,
) -> ProcurementRAGPipeline:
  return ProcurementRAGPipeline(
      document_store=document_store,
      regulations_store=regulations_store,
      reranker=build_reranker(selected_reranker_key),
      llm=build_llm(selected_model),
      settings=runtime_settings,
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
        UploadFilePayload(name=uploaded_file.name,
                          content=uploaded_file.getvalue())
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
    details = [f"[{label}] {file_name}", f"page {page}"]
    if "retrieval_score" in source.metadata:
      details.append(f"retrieve={float(source.metadata['retrieval_score']):.4f}")
    if "rerank_score" in source.metadata:
      details.append(f"rerank={float(source.metadata['rerank_score']):.4f}")
    title_text = " | ".join(details)
    with st.expander(title_text):
      st.write(source.page_content)


def handle_query(
    pipeline: ProcurementRAGPipeline,
    runtime_settings,
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
  disabled = (
      not has_documents
      or (not has_regulations and not runtime_settings.use_mcp)
      or not question.strip()
  )

  if not has_documents:
    st.info("문서를 먼저 업로드하고 인덱싱해야 질의를 실행할 수 있습니다.")
  elif not has_regulations:
    if runtime_settings.use_mcp:
      st.info("로컬 규정 DB는 비어 있지만 USE_MCP=true라서 korean-law-mcp 검색을 함께 사용할 수 있습니다.")
    else:
      st.info(
          "규정 DB가 비어 있습니다. `python scripts/ingest_regulations.py`를 먼저 실행해 주세요."
      )

  if st.button("Run Compliance Review", type="primary", use_container_width=True, disabled=disabled):
    try:
      with st.spinner(
          "Running the evidence-gathering agent with "
          f"{format_reranker_label(st.session_state['selected_reranker_key'])}..."
      ):
        st.session_state["last_response"] = pipeline.invoke(question.strip())
    except Exception as exc:  # pragma: no cover - UI feedback
      st.error(f"Query failed: {exc}")

  response = st.session_state.get("last_response")
  if not response:
    return

  st.markdown("### Compliance Review")
  if response.used_mcp:
    st.caption("This review used korean-law-mcp evidence in addition to local ChromaDB evidence.")
  st.write(response.answer)

  render_source_section("Regulation Evidence", response.regulation_sources)
  render_source_section("Document Evidence", response.document_sources)


def main() -> None:
  refresh_runtime_config()
  st.title("Procurement Review Agent")
  st.caption("LangChain Agent + ChromaDB + optional korean-law-mcp + vLLM + Streamlit")

  selected_model = select_llm_model()
  selected_reranker_key = select_reranker()
  runtime_settings = build_runtime_settings(select_runtime_mode())
  document_store, regulations_store = get_vectorstores()
  pipeline = build_pipeline(
      runtime_settings,
      selected_model,
      selected_reranker_key,
      document_store,
      regulations_store,
  )
  render_sidebar(
      runtime_settings,
      selected_model,
      selected_reranker_key,
      document_store,
      regulations_store,
  )

  left, right = st.columns([1, 1.3], gap="large")
  with left:
    handle_ingestion(document_store)
  with right:
    handle_query(
        pipeline,
        runtime_settings,
        document_store,
        regulations_store,
    )


if __name__ == "__main__":
  main()
