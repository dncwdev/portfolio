from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH)


@lru_cache(maxsize=1)
def _read_env_file_values() -> dict[str, str]:
    if not ENV_PATH.exists():
        return {}

    values: dict[str, str] = {}
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].lstrip()
        if "=" not in stripped:
            continue

        key, raw_value = stripped.split("=", 1)
        key = key.strip()
        value = raw_value.strip()
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            value = value[1:-1]
        if key:
            values[key] = value

    return values


def _env_lookup(name: str) -> str | None:
    file_value = _read_env_file_values().get(name)
    if file_value is not None and file_value.strip():
        return file_value.strip()

    value = os.getenv(name)
    if value is not None and value.strip():
        return value.strip()
    return None


def _require_env(name: str) -> str:
    value = _env_lookup(name)
    if value is None:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _env_str(name: str, default: str) -> str:
    value = _env_lookup(name)
    return value if value is not None else default


def _env_int(name: str, default: int) -> int:
    value = _env_lookup(name)
    return int(value) if value is not None else default


def _env_float(name: str, default: float) -> float:
    value = _env_lookup(name)
    return float(value) if value is not None else default


def _env_bool(name: str, default: bool) -> bool:
    value = _env_lookup(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_csv(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = _env_lookup(name)
    if value is None:
        return default
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    return items or default


def _env_shell_tokens(name: str, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    value = _env_lookup(name)
    if value is None:
        return default

    posix_mode = os.name != "nt"
    tokens = tuple(token for token in shlex.split(value, posix=posix_mode) if token)
    return tokens or default


def _resolve_path(path_str: str) -> Path:
    candidate = Path(path_str)
    return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def _normalize_openai_base_url(base_url: str) -> str:
    trimmed = base_url.rstrip("/")
    return trimmed if trimmed.endswith("/v1") else f"{trimmed}/v1"


@dataclass(frozen=True)
class RerankerProfile:
    key: str
    base_url: str
    model_name: str
    display_name: str
    api_key: str
    engine: str

    def get_score_url(self, engine: str | None = None) -> str:
        runtime_engine = (engine or self.engine).lower()
        if runtime_engine == "infinity":
            return f"{self.base_url}/rerank"
        return f"{self.base_url}/v1/score"


@dataclass(frozen=True)
class Settings:
    llm_base_url: str | None
    llm_model_name: str
    llm_display_name: str
    available_models: tuple[str, ...]
    llm_model_urls: dict[str, str]
    llm_api_key: str
    embedding_base_url: str
    embedding_api_base_url: str
    embedding_model_name: str
    embedding_display_name: str
    embedding_api_key: str
    available_reranker_keys: tuple[str, ...]
    reranker_profiles: dict[str, RerankerProfile]
    default_reranker_key: str
    chroma_persist_dir: Path
    chroma_collection_name: str
    regulations_collection_name: str
    regulations_data_dir: Path
    chunk_size: int
    chunk_overlap: int
    regulations_chunk_size: int
    regulations_chunk_overlap: int
    retrieval_top_k: int
    rerank_top_k: int
    rerank_relative_threshold: float
    request_timeout: float
    llm_temperature: float
    llm_max_tokens: int
    llm_include_reasoning: bool
    use_mcp: bool
    korean_law_mcp_transport: str
    korean_law_mcp_command: str
    korean_law_mcp_args: tuple[str, ...]
    korean_law_mcp_url: str | None
    law_oc: str | None

    def get_llm_base_url(self, model_name: str | None = None) -> str:
        selected_model = model_name or self.llm_model_name
        if selected_model in self.llm_model_urls:
            return self.llm_model_urls[selected_model]
        if self.llm_base_url:
            return self.llm_base_url
        raise ValueError(
            f"Missing MODEL_URL_{selected_model} and LLM_BASE_URL fallback."
        )

    def get_llm_api_base_url(self, model_name: str | None = None) -> str:
        return _normalize_openai_base_url(self.get_llm_base_url(model_name))

    def get_reranker_profile(
        self,
        reranker_key: str | None = None,
    ) -> RerankerProfile:
        selected_key = reranker_key or self.default_reranker_key
        if selected_key in self.reranker_profiles:
            return self.reranker_profiles[selected_key]
        raise ValueError(
            f"Unknown reranker profile: {selected_key}. "
            f"Available: {', '.join(self.available_reranker_keys)}"
        )

    @property
    def reranker_base_url(self) -> str:
        return self.get_reranker_profile().base_url

    @property
    def reranker_score_url(self) -> str:
        return self.get_reranker_profile().get_score_url()

    @property
    def reranker_model_name(self) -> str:
        return self.get_reranker_profile().model_name

    @property
    def reranker_display_name(self) -> str:
        return self.get_reranker_profile().display_name

    @property
    def reranker_api_key(self) -> str:
        return self.get_reranker_profile().api_key

    @property
    def reranker_engine(self) -> str:
        return self.get_reranker_profile().engine


def _discover_reranker_suffixes() -> tuple[str, ...]:
    suffixes: set[str] = set()
    pattern = re.compile(r"^RERANKER_BASE_URL(\d*)$")

    for key in _read_env_file_values():
        match = pattern.match(key)
        if match:
            suffixes.add(match.group(1))

    for key, value in os.environ.items():
        if not value or not value.strip():
            continue
        match = pattern.match(key)
        if match:
            suffixes.add(match.group(1))

    if not suffixes:
        return ("",)

    return tuple(
        sorted(
            suffixes,
            key=lambda item: (0, 0) if item == "" else (1, int(item or "0")),
        )
    )


def _build_reranker_profiles() -> tuple[tuple[str, ...], dict[str, RerankerProfile]]:
    available_keys: list[str] = []
    profiles: dict[str, RerankerProfile] = {}

    for suffix in _discover_reranker_suffixes():
        base_url_name = f"RERANKER_BASE_URL{suffix}"
        if _env_lookup(base_url_name) is None:
            continue

        key = "default" if not suffix else suffix
        engine = _env_str(f"RERANKER_ENGINE{suffix}", "auto").lower()
        if engine not in {"auto", "vllm", "infinity"}:
            raise ValueError(
                f"Unsupported reranker engine '{engine}' for profile '{key}'. "
                "Expected one of: auto, vllm, infinity."
            )

        model_name = _require_env(f"RERANKER_MODEL_NAME{suffix}")
        profiles[key] = RerankerProfile(
            key=key,
            base_url=_require_env(base_url_name).rstrip("/"),
            model_name=model_name,
            display_name=_env_str(
                f"RERANKER_DISPLAY_NAME{suffix}",
                model_name,
            ),
            api_key=_env_str(f"RERANKER_API_KEY{suffix}", ""),
            engine=engine,
        )
        available_keys.append(key)

    if not available_keys:
        raise ValueError("Missing required environment variable: RERANKER_BASE_URL")

    return tuple(available_keys), profiles


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    llm_model_name = _require_env("LLM_MODEL_NAME")
    llm_base_url = _env_lookup("LLM_BASE_URL")
    embedding_base_url = _require_env("EMBEDDING_BASE_URL")
    available_reranker_keys, reranker_profiles = _build_reranker_profiles()
    available_models = _env_csv("AVAILABLE_MODELS", (llm_model_name,))
    if llm_model_name not in available_models:
        available_models = (llm_model_name, *available_models)
    llm_model_urls = {
        model_name: model_url.rstrip("/")
        for model_name in available_models
        if (model_url := _env_lookup(f"MODEL_URL_{model_name}"))
    }
    if not llm_base_url and llm_model_name not in llm_model_urls:
        raise ValueError(
            f"Missing MODEL_URL_{llm_model_name} and LLM_BASE_URL fallback."
        )

    korean_law_mcp_url = _env_lookup("KOREAN_LAW_MCP_URL")
    default_reranker_key = _env_str("DEFAULT_RERANKER_KEY", available_reranker_keys[0])
    if default_reranker_key not in reranker_profiles:
        raise ValueError(
            f"Unknown DEFAULT_RERANKER_KEY: {default_reranker_key}. "
            f"Available: {', '.join(available_reranker_keys)}"
        )

    return Settings(
        llm_base_url=llm_base_url.rstrip("/") if llm_base_url else None,
        llm_model_name=llm_model_name,
        llm_display_name=_env_str("LLM_DISPLAY_NAME", llm_model_name),
        available_models=available_models,
        llm_model_urls=llm_model_urls,
        llm_api_key=_require_env("LLM_API_KEY"),
        embedding_base_url=embedding_base_url.rstrip("/"),
        embedding_api_base_url=_normalize_openai_base_url(embedding_base_url),
        embedding_model_name=_require_env("EMBEDDING_MODEL_NAME"),
        embedding_display_name=_env_str(
            "EMBEDDING_DISPLAY_NAME",
            _require_env("EMBEDDING_MODEL_NAME"),
        ),
        embedding_api_key=_require_env("EMBEDDING_API_KEY"),
        available_reranker_keys=available_reranker_keys,
        reranker_profiles=reranker_profiles,
        default_reranker_key=default_reranker_key,
        chroma_persist_dir=_resolve_path(_env_str("CHROMA_PERSIST_DIR", ".chroma")),
        chroma_collection_name=_env_str(
            "CHROMA_COLLECTION_NAME",
            "procurement_regulations",
        ),
        regulations_collection_name=_env_str(
            "REGULATIONS_COLLECTION_NAME",
            "regulations",
        ),
        regulations_data_dir=_resolve_path(
            _env_str("REGULATIONS_DATA_DIR", "data/regulations")
        ),
        chunk_size=_env_int("CHUNK_SIZE", 1200),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 200),
        regulations_chunk_size=_env_int("REGULATIONS_CHUNK_SIZE", 500),
        regulations_chunk_overlap=_env_int("REGULATIONS_CHUNK_OVERLAP", 50),
        retrieval_top_k=_env_int("RETRIEVAL_TOP_K", 20),
        rerank_top_k=_env_int("RERANK_TOP_K", 5),
        rerank_relative_threshold=_env_float("RERANK_RELATIVE_THRESHOLD", 0.15),
        request_timeout=_env_float("REQUEST_TIMEOUT", 600.0),
        llm_temperature=_env_float("LLM_TEMPERATURE", 0.0),
        llm_max_tokens=_env_int("LLM_MAX_TOKENS", 2048),
        llm_include_reasoning=_env_bool("LLM_INCLUDE_REASONING", False),
        use_mcp=_env_bool("USE_MCP", False),
        korean_law_mcp_transport=_env_str("KOREAN_LAW_MCP_TRANSPORT", "stdio"),
        korean_law_mcp_command=_env_str(
            "KOREAN_LAW_MCP_COMMAND",
            "korean-law-mcp",
        ),
        korean_law_mcp_args=_env_shell_tokens("KOREAN_LAW_MCP_ARGS"),
        korean_law_mcp_url=korean_law_mcp_url.rstrip("/") if korean_law_mcp_url else None,
        law_oc=_env_lookup("LAW_OC"),
    )


@lru_cache(maxsize=16)
def build_llm(model_name: str | None = None) -> ChatOpenAI:
    settings = get_settings()
    selected_model = model_name or settings.llm_model_name
    return ChatOpenAI(
        model=selected_model,
        api_key=settings.llm_api_key,
        base_url=settings.get_llm_api_base_url(selected_model),
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


@lru_cache(maxsize=16)
def build_reranker(reranker_key: str | None = None):
    from .reranker import build_reranker_client

    settings = get_settings()
    return build_reranker_client(
        settings=settings,
        profile=settings.get_reranker_profile(reranker_key),
    )


def clear_config_caches() -> None:
    _read_env_file_values.cache_clear()
    get_settings.cache_clear()
    build_llm.cache_clear()
    build_embeddings.cache_clear()
    build_reranker.cache_clear()
