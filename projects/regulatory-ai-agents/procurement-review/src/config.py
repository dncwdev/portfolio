from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH)


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        raise ValueError(f"Missing required environment variable: {name}")
    return value.strip()


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value and value.strip() else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value and value.strip() else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_path(path_str: str) -> Path:
    candidate = Path(path_str)
    return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def _normalize_openai_base_url(base_url: str) -> str:
    trimmed = base_url.rstrip("/")
    return trimmed if trimmed.endswith("/v1") else f"{trimmed}/v1"


@dataclass(frozen=True)
class Settings:
    llm_base_url: str
    llm_api_base_url: str
    llm_model_name: str
    llm_api_key: str
    embedding_base_url: str
    embedding_api_base_url: str
    embedding_model_name: str
    embedding_api_key: str
    reranker_base_url: str
    reranker_score_url: str
    reranker_model_name: str
    reranker_api_key: str
    chroma_persist_dir: Path
    chroma_collection_name: str
    chunk_size: int
    chunk_overlap: int
    retrieval_top_k: int
    rerank_top_k: int
    request_timeout: float
    llm_temperature: float
    llm_max_tokens: int
    llm_include_reasoning: bool


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    llm_base_url = _require_env("LLM_BASE_URL")
    embedding_base_url = _require_env("EMBEDDING_BASE_URL")
    reranker_base_url = _require_env("RERANKER_BASE_URL")

    return Settings(
        llm_base_url=llm_base_url.rstrip("/"),
        llm_api_base_url=_normalize_openai_base_url(llm_base_url),
        llm_model_name=_require_env("LLM_MODEL_NAME"),
        llm_api_key=_require_env("LLM_API_KEY"),
        embedding_base_url=embedding_base_url.rstrip("/"),
        embedding_api_base_url=_normalize_openai_base_url(embedding_base_url),
        embedding_model_name=_require_env("EMBEDDING_MODEL_NAME"),
        embedding_api_key=_require_env("EMBEDDING_API_KEY"),
        reranker_base_url=reranker_base_url.rstrip("/"),
        reranker_score_url=f"{reranker_base_url.rstrip('/')}/v1/score",
        reranker_model_name=_require_env("RERANKER_MODEL_NAME"),
        reranker_api_key=_require_env("RERANKER_API_KEY"),
        chroma_persist_dir=_resolve_path(_env_str("CHROMA_PERSIST_DIR", ".chroma")),
        chroma_collection_name=_env_str(
            "CHROMA_COLLECTION_NAME",
            "procurement_regulations",
        ),
        chunk_size=_env_int("CHUNK_SIZE", 1200),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 200),
        retrieval_top_k=_env_int("RETRIEVAL_TOP_K", 20),
        rerank_top_k=_env_int("RERANK_TOP_K", 5),
        request_timeout=_env_float("REQUEST_TIMEOUT", 60.0),
        llm_temperature=_env_float("LLM_TEMPERATURE", 0.0),
        llm_max_tokens=_env_int("LLM_MAX_TOKENS", 2048),
        llm_include_reasoning=_env_bool("LLM_INCLUDE_REASONING", False),
    )


@lru_cache(maxsize=1)
def build_llm() -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        model=settings.llm_model_name,
        api_key=settings.llm_api_key,
        base_url=settings.llm_api_base_url,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        request_timeout=settings.request_timeout,
        max_retries=2,
        use_responses_api=False,
        extra_body={"include_reasoning": settings.llm_include_reasoning},
    )


@lru_cache(maxsize=1)
def build_embeddings() -> OpenAIEmbeddings:
    settings = get_settings()
    return OpenAIEmbeddings(
        model=settings.embedding_model_name,
        api_key=settings.embedding_api_key,
        base_url=settings.embedding_api_base_url,
        request_timeout=settings.request_timeout,
        max_retries=2,
        tiktoken_enabled=False,
        check_embedding_ctx_length=False,
    )


@lru_cache(maxsize=1)
def build_reranker():
    from .reranker import VLLMReranker

    return VLLMReranker(settings=get_settings())
