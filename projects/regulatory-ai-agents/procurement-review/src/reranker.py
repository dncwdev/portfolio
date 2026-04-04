from __future__ import annotations

from collections.abc import Sequence

import requests
from langchain_core.documents import Document

from .config import RerankerProfile, Settings, get_settings


class BaseReranker:
    def __init__(
        self,
        settings: Settings | None = None,
        profile: RerankerProfile | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.profile = profile or self.settings.get_reranker_profile()
        self.session = session or requests.Session()

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        raise NotImplementedError

    def rerank(
        self,
        query: str,
        documents: Sequence[Document],
        top_n: int | None = None,
    ) -> list[Document]:
        if not documents:
            return []

        scores = self.score(query, [document.page_content for document in documents])
        ranked_documents: list[Document] = []

        for rank, (document, score) in enumerate(
            sorted(zip(documents, scores), key=lambda item: item[1], reverse=True),
            start=1,
        ):
            metadata = dict(document.metadata)
            metadata["rerank_rank"] = rank
            metadata["rerank_score"] = score
            metadata["reranker_name"] = self.profile.display_name
            metadata["reranker_engine"] = self.profile.engine
            ranked_documents.append(
                Document(page_content=document.page_content, metadata=metadata)
            )

        return ranked_documents[: top_n or self.settings.rerank_top_k]

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.profile.api_key:
            headers["Authorization"] = f"Bearer {self.profile.api_key}"
        return headers


class VLLMReranker(BaseReranker):
    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []

        response = self.session.post(
            self.profile.get_score_url("vllm"),
            headers=self._headers(),
            json={
                "model": self.profile.model_name,
                "text_1": query,
                "text_2": list(passages),
            },
            timeout=self.settings.request_timeout,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data", [])

        if len(data) != len(passages):
            raise ValueError(
                "Reranker returned a mismatched number of scores. "
                f"Expected {len(passages)}, got {len(data)}."
            )

        return [float(item["score"]) for item in data]


class InfinityReranker(BaseReranker):
    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []

        response = self.session.post(
            self.profile.get_score_url("infinity"),
            headers=self._headers(),
            json={
                "model": self.profile.model_name,
                "query": query,
                "documents": list(passages),
                "top_n": len(passages),
            },
            timeout=self.settings.request_timeout,
        )
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results", [])

        scores: list[float | None] = [None] * len(passages)
        for item in results:
            index = int(item["index"])
            scores[index] = float(item["relevance_score"])

        if any(score is None for score in scores):
            raise ValueError(
                "Infinity reranker returned incomplete scores. "
                f"Expected {len(passages)}, got {len(results)}."
            )

        return [float(score) for score in scores]


class AutoReranker(BaseReranker):
    def __init__(
        self,
        settings: Settings | None = None,
        profile: RerankerProfile | None = None,
    ) -> None:
        super().__init__(settings=settings, profile=profile)
        self._vllm = VLLMReranker(
            settings=self.settings,
            profile=self.profile,
            session=self.session,
        )
        self._infinity = InfinityReranker(
            settings=self.settings,
            profile=self.profile,
            session=self.session,
        )
        self._resolved_engine: str | None = None

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []

        if self._resolved_engine == "vllm":
            return self._vllm.score(query, passages)
        if self._resolved_engine == "infinity":
            return self._infinity.score(query, passages)

        try:
            scores = self._vllm.score(query, passages)
            self._resolved_engine = "vllm"
            self.profile = RerankerProfile(
                key=self.profile.key,
                base_url=self.profile.base_url,
                model_name=self.profile.model_name,
                display_name=self.profile.display_name,
                api_key=self.profile.api_key,
                engine="vllm",
            )
            return scores
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code not in {404, 405}:
                raise

        scores = self._infinity.score(query, passages)
        self._resolved_engine = "infinity"
        self.profile = RerankerProfile(
            key=self.profile.key,
            base_url=self.profile.base_url,
            model_name=self.profile.model_name,
            display_name=self.profile.display_name,
            api_key=self.profile.api_key,
            engine="infinity",
        )
        return scores


def build_reranker_client(
    *,
    settings: Settings | None = None,
    profile: RerankerProfile | None = None,
) -> BaseReranker:
    runtime_settings = settings or get_settings()
    runtime_profile = profile or runtime_settings.get_reranker_profile()
    engine = runtime_profile.engine.lower()

    if engine == "vllm":
        return VLLMReranker(settings=runtime_settings, profile=runtime_profile)
    if engine == "infinity":
        return InfinityReranker(settings=runtime_settings, profile=runtime_profile)
    return AutoReranker(settings=runtime_settings, profile=runtime_profile)
