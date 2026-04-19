from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re

try:
  from langchain.agents import create_agent
except ImportError:  # pragma: no cover - depends on installed langchain version
  create_agent = None
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .agent_tools import (
    EvidenceCollector,
    build_agent_tools,
    get_default_query_templates,
    retrieve_local_evidence,
)
from .config import Settings, build_llm, get_settings
from .domain_queries import format_domain_query_template_guide
try:
  from .mcp_client import KoreanLawMCPClient
except ImportError:  # pragma: no cover - depends on optional MCP dependencies
  KoreanLawMCPClient = None
from .reranker import BaseReranker
from .vectorstore import ProcurementVectorStore


logger = logging.getLogger(__name__)
TRACE_LOG_PATH = Path(__file__).resolve().parents[1] / "logs" / "mcp_trace.jsonl"


THINK_TAG_CLOSE_RE = re.compile(r"<\s*/\s*think\s*>", flags=re.IGNORECASE)
THINKING_LIKE_START_RE = re.compile(
    r"^\s*(?:<\s*think\b|thinking(?:\s+process)?\s*:|reasoning\s*:|analysis\s*:)",
    flags=re.IGNORECASE,
)
THINKING_PROCESS_LINE_RE = re.compile(
    r"(?im)^\s*Thinking Process:\s*$"
)
LAW_NAME_PATTERN_RE = re.compile(
    r"[가-힣A-Za-z0-9·ㆍ()\-\s]{1,120}(?:법|시행령|시행규칙|고시|예규|훈령|지침|세칙|기준)"
)
ARTICLE_PATTERN_RE = re.compile(
    r"제\s*\d+\s*조(?:\s*의\s*\d+)?(?:\s*제\s*\d+\s*항)?(?:\s*제\s*\d+\s*호)?"
    r"|제\s*\d+\s*항"
    r"|제\s*\d+\s*호"
)
ARTICLE_WITH_CONTEXT_PATTERN_RE = re.compile(
    r"제\s*\d+\s*조(?:\s*의\s*\d+)?(?:\s*제\s*\d+\s*항)?(?:\s*제\s*\d+\s*호)?"
)
LAW_QUOTE_RE = re.compile(r"[「」『』]")
SPACE_RE = re.compile(r"\s+")


NO_DOCUMENT_MESSAGE = (
    "조달 문서 근거를 찾지 못해 준수 여부를 판단할 수 없습니다. "
    "문서를 업로드하고 다시 시도해 주세요."
)

NO_REGULATION_MESSAGE = (
    "규정 근거가 없습니다. 로컬 규정 DB를 채우거나 USE_MCP=true로 설정한 뒤 다시 시도해 주세요."
)
MODEL_RESPONSE_ERROR_MESSAGE = (
    "모델 응답 생성 중 오류가 발생했습니다. 다시 시도해 주세요."
)

AGENT_SYSTEM_PROMPT = """당신은 공공조달 구매규격서 검토를 위한 증거 수집 에이전트입니다.

반드시 다음 원칙을 따르세요.
- 로컬 ChromaDB 검색이 필요하면 자유로운 검색어를 만들지 말고, 아래의 고정 템플릿 중 하나를 `query_template`로 선택해 `search_local_procurement_context`를 호출하세요.
{query_template_guide}
- 질문과 가장 관련 있는 템플릿을 1개 이상 선택할 수 있지만, 불필요하게 같은 템플릿을 반복 호출하지 마세요.
- {mcp_instruction}
- 도구가 반환한 근거만 사용하세요. 사전지식으로 근거를 꾸며내지 마세요.
- 최종 응답은 짧아도 되지만, 어떤 템플릿과 도구를 사용했는지 드러나게 하세요.
- 모든 최종 응답은 반드시 한국어로 작성하세요
"""

ANSWER_PROMPT = """당신은 공공조달 구매규격서 초안의 법령 준수 여부를 검토하는 AI 분석가입니다.
반드시 한국어로 답변하세요.
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


def _document_origin(document: Document) -> str:
  origin = document.metadata.get("origin")
  if origin:
    return str(origin)
  return "local"


def _serialize_sources(documents: list[Document]) -> list[dict[str, str]]:
  serialized: list[dict[str, str]] = []
  for document in documents:
    serialized.append(
        {
            "citation": str(document.metadata.get("citation", "")),
            "source": str(document.metadata.get("source", "unknown")),
            "page": str(document.metadata.get("page", "-")),
            "origin": _document_origin(document),
            "mcp_tool": str(document.metadata.get("mcp_tool", "")),
        }
    )
  return serialized


def _append_trace(trace: dict[str, object]) -> None:
  try:
    TRACE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRACE_LOG_PATH.open("a", encoding="utf-8") as handle:
      handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
  except Exception as exc:  # pragma: no cover - tracing must not break pipeline
    logger.warning("Failed to append MCP trace log: %s", exc)


def _detect_forced_mcp_preflight(
    question: str,
) -> tuple[bool, str | None, str | None]:
  normalized = question.strip()
  if not normalized:
    return False, None, None

  has_law_name = bool(LAW_NAME_PATTERN_RE.search(normalized))
  has_article = bool(ARTICLE_PATTERN_RE.search(normalized))

  if has_law_name and has_article:
    return True, "law_name_and_article_pattern", normalized
  if has_law_name:
    return True, "law_name_pattern", normalized
  if has_article:
    return True, "article_pattern", normalized
  return False, None, None


def _normalize_legal_question(question: str) -> str:
  without_quotes = LAW_QUOTE_RE.sub("", question)
  normalized = SPACE_RE.sub(" ", without_quotes).strip()
  return normalized


def _extract_law_title(question: str) -> str | None:
  normalized = _normalize_legal_question(question)
  match = LAW_NAME_PATTERN_RE.search(normalized)
  if not match:
    return None
  law_title = match.group(0).strip(" ,.:;")
  return law_title or None


def _normalize_article_token(token: str) -> str:
  article = SPACE_RE.sub("", token)
  article = article.replace("의", "의")
  article = re.sub(r"제(\d+)조", r"제\1조", article)
  article = re.sub(r"제(\d+)항", r" 제\1항", article)
  article = re.sub(r"제(\d+)호", r" 제\1호", article)
  return SPACE_RE.sub(" ", article).strip()


def _extract_article_candidates(question: str) -> tuple[str | None, str | None]:
  normalized = _normalize_legal_question(question)
  match = ARTICLE_WITH_CONTEXT_PATTERN_RE.search(normalized)
  if not match:
    fallback = ARTICLE_PATTERN_RE.search(normalized)
    if not fallback:
      return None, None
    article = _normalize_article_token(fallback.group(0))
    return article, article

  full_article = _normalize_article_token(match.group(0))
  jo_match = re.match(r"제\s*\d+\s*조(?:\s*의\s*\d+)?", full_article)
  if not jo_match:
    return full_article, full_article
  return jo_match.group(0).strip(), full_article


def _build_mcp_preflight_queries(question: str) -> list[str]:
  normalized = _normalize_legal_question(question)
  law_title = _extract_law_title(normalized)
  article_jo, article_full = _extract_article_candidates(normalized)

  candidates: list[str] = []

  def append_candidate(value: str | None) -> None:
    if not value:
      return
    normalized_value = SPACE_RE.sub(" ", value).strip(" ,.:;")
    if not normalized_value or normalized_value in candidates:
      return
    candidates.append(normalized_value)

  if law_title:
    append_candidate(law_title)
    if article_jo:
      append_candidate(f"{law_title} {article_jo}")
    if article_full and article_full != article_jo:
      append_candidate(f"{law_title} {article_full}")
  else:
    append_candidate(normalized)
    if article_full:
      append_candidate(article_full)

  return candidates[:3]


class ProcurementRAGPipeline:
  def __init__(
      self,
      document_store: ProcurementVectorStore,
      regulations_store: ProcurementVectorStore,
      reranker: BaseReranker,
      llm: BaseChatModel | None = None,
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

    mcp_enabled = self.settings.use_mcp and not self.settings.is_commercial_model()
    document_chunk_count = self.document_store.get_stats()["chunk_count"]
    regulation_chunk_count = self.regulations_store.get_stats()["chunk_count"]

    collector = EvidenceCollector()
    collector.init_trace(question=normalized_question, use_mcp=mcp_enabled)
    collector.trace["document_chunk_count"] = document_chunk_count
    collector.trace["regulation_chunk_count"] = regulation_chunk_count

    if document_chunk_count == 0:
      return self._build_response(
          question=normalized_question,
          answer=NO_DOCUMENT_MESSAGE,
          collector=collector,
      )

    if (
        regulation_chunk_count == 0
        and not mcp_enabled
    ):
      return self._build_response(
          question=normalized_question,
          answer=NO_REGULATION_MESSAGE,
          collector=collector,
      )

    self._run_forced_mcp_preflight(
        question=normalized_question,
        collector=collector,
        mcp_enabled=mcp_enabled,
    )

    forced_preflight = bool(collector.trace.get("forced_mcp_preflight"))

    if mcp_enabled and not forced_preflight:
      self._gather_agent_evidence(normalized_question, collector)

    fallback_to_local = (
        not mcp_enabled
        or not collector.document_sources
        or (
            not collector.regulation_sources and regulation_chunk_count > 0
        )
    )
    collector.record_fallback_to_local(fallback_to_local)
    if fallback_to_local:
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
          trigger="pipeline_fallback",
      )

    if not collector.document_sources:
      return self._build_response(
          question=normalized_question,
          answer=NO_DOCUMENT_MESSAGE,
          collector=collector,
      )

    regulations_context = self._format_context(
        collector.regulation_sources,
        empty_message="관련 규정 근거 없음",
    )
    if (
        forced_preflight
        and collector.trace.get("mcp_attempted")
        and not collector.trace.get("mcp_succeeded")
    ):
      regulations_context = (
          "외부 법령 MCP 조회를 시도했으나 검색되지 않아 로컬 문서 기준으로 판단했다.\n\n"
          f"{regulations_context}"
      )

    collector.trace["final_chat_completion_calls"] = (
        int(collector.trace.get("final_chat_completion_calls", 0)) + 1
    )
    answer = self.answer_chain.invoke(
        {
            "question": normalized_question,
            "regulations_context": regulations_context,
            "document_context": self._format_context(
                collector.document_sources,
                empty_message="관련 조달 문서 근거 없음",
            ),
        }
    )
    answer = self._sanitize_final_answer(answer)

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
    if create_agent is None:
      collector.record_agent_unavailable("create_agent unavailable")
      return

    tools = build_agent_tools(
        document_store=self.document_store,
        regulations_store=self.regulations_store,
        reranker=self.reranker,
        collector=collector,
        settings=self.settings,
        question_context=question,
    )
    collector.record_agent_started(
        available_tools=[tool.name for tool in tools],
    )
    collector.trace["agent_loop_entered"] = True
    agent = create_agent(
        model=self.llm,
        tools=tools,
        system_prompt=self._build_agent_system_prompt(),
    )

    try:
      agent.invoke({"messages": [{"role": "user", "content": question}]})
    except Exception as exc:
      # If tool calling is not fully supported by the runtime model, the
      # pipeline falls back to deterministic local retrieval below.
      collector.record_agent_error(exc)
      return

  def _run_forced_mcp_preflight(
      self,
      *,
      question: str,
      collector: EvidenceCollector,
      mcp_enabled: bool,
  ) -> None:
    forced, reason, query = _detect_forced_mcp_preflight(question)
    preflight_queries = _build_mcp_preflight_queries(question) if forced else []
    collector.trace["forced_mcp_preflight"] = forced
    collector.trace["forced_mcp_reason"] = reason
    collector.trace["forced_mcp_query"] = preflight_queries[0] if preflight_queries else query
    collector.trace["forced_mcp_result_origin"] = None

    if not forced or not mcp_enabled or KoreanLawMCPClient is None or not preflight_queries:
      return

    limit = max(1, min(self.settings.rerank_top_k or 5, 5))
    client = KoreanLawMCPClient(self.settings)
    for candidate in preflight_queries[:3]:
      collector.record_mcp_attempt(
          tool_name="forced_mcp_preflight",
          search_type="law",
          query=candidate,
          article=None,
          document_id=None,
          limit=limit,
      )
      try:
        documents = client.search_documents(
            search_type="law",
            query=candidate,
            limit=limit,
        )
      except Exception as exc:
        collector.record_mcp_error(exc)
        logger.warning(
            "Forced MCP preflight failed for query=%r: %s",
            candidate,
            exc,
        )
        continue

      collector.record_mcp_success(documents=documents)
      if not documents:
        continue

      collector.add_regulation_documents(documents)
      collector.trace["forced_mcp_result_origin"] = "mcp"
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
    if collector.trace:
      collector.trace["final_used_mcp"] = used_mcp
      collector.trace["final_regulation_source_count"] = len(
          collector.regulation_sources
      )
      collector.trace["final_document_source_count"] = len(
          collector.document_sources
      )
      collector.trace["final_regulation_origins"] = sorted(
          {_document_origin(source) for source in collector.regulation_sources}
      )
      collector.trace["final_document_origins"] = sorted(
          {_document_origin(source) for source in collector.document_sources}
      )
      collector.trace["final_regulation_sources"] = _serialize_sources(
          collector.regulation_sources
      )
      collector.trace["final_document_sources"] = _serialize_sources(
          collector.document_sources
      )
      collector.trace["answer_preview"] = answer[:300]
      _append_trace(collector.trace)

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

  def _sanitize_final_answer(self, answer: str) -> str:
    original = answer.strip()
    closing_matches = list(THINK_TAG_CLOSE_RE.finditer(answer))
    if closing_matches:
      sanitized = answer[closing_matches[-1].end() :]
    else:
      if THINKING_LIKE_START_RE.match(answer):
        return MODEL_RESPONSE_ERROR_MESSAGE
      sanitized = THINKING_PROCESS_LINE_RE.sub("", answer)
    sanitized = sanitized.strip()
    return sanitized or original
