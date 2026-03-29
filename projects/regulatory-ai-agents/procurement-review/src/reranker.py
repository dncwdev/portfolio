from __future__ import annotations

from typing import Sequence

import requests
from langchain_core.documents import Document

from .config import Settings, get_settings


class VLLMReranker:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.session = requests.Session()

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []

        response = self.session.post(
            self.settings.reranker_score_url,
            headers=self._headers(),
            json={
                "model": self.settings.reranker_model_name,
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
            ranked_documents.append(
                Document(page_content=document.page_content, metadata=metadata)
            )

        return ranked_documents[: top_n or self.settings.rerank_top_k]

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.settings.reranker_api_key:
            headers["Authorization"] = f"Bearer {self.settings.reranker_api_key}"
        return headers
