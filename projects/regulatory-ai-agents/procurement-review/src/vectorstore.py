from __future__ import annotations

import io
from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable, Sequence

import chromadb
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from .config import Settings, build_embeddings, get_settings


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


@dataclass(frozen=True)
class UploadFilePayload:
    name: str
    content: bytes


@dataclass(frozen=True)
class IngestionResult:
    files_indexed: int
    chunks_indexed: int


class ProcurementVectorStore:
    def __init__(
        self,
        settings: Settings | None = None,
        embeddings: OpenAIEmbeddings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embeddings = embeddings or build_embeddings()
        self.settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.settings.chroma_persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=self.settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            add_start_index=True,
        )

    def get_stats(self) -> dict[str, int | str]:
        return {
            "collection_name": self.settings.chroma_collection_name,
            "chunk_count": self.collection.count(),
        }

    def ingest_files(self, files: Sequence[UploadFilePayload]) -> IngestionResult:
        all_chunks: list[Document] = []

        for file_payload in files:
            all_chunks.extend(self._chunk_file(file_payload))

        if not all_chunks:
            raise ValueError("No indexable text was extracted from the provided files.")

        ids = [chunk.metadata["chunk_id"] for chunk in all_chunks]
        texts = [chunk.page_content for chunk in all_chunks]
        metadatas = [dict(chunk.metadata) for chunk in all_chunks]

        for batch_ids, batch_texts, batch_metadatas in self._iterate_batches(
            ids,
            texts,
            metadatas,
            batch_size=64,
        ):
            batch_embeddings = self.embeddings.embed_documents(batch_texts)
            self.collection.upsert(
                ids=batch_ids,
                documents=batch_texts,
                metadatas=batch_metadatas,
                embeddings=batch_embeddings,
            )

        return IngestionResult(
            files_indexed=len(files),
            chunks_indexed=len(all_chunks),
        )

    def similarity_search(self, query: str, k: int | None = None) -> list[Document]:
        if self.collection.count() == 0:
            return []

        query_embedding = self.embeddings.embed_query(query)
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k or self.settings.retrieval_top_k,
            include=["documents", "metadatas", "distances"],
        )

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        output: list[Document] = []
        for index, (text, metadata, distance) in enumerate(
            zip(documents, metadatas, distances),
            start=1,
        ):
            item_metadata = dict(metadata or {})
            item_metadata["retrieval_rank"] = index
            item_metadata["distance"] = float(distance)
            item_metadata["retrieval_score"] = 1.0 - float(distance)
            output.append(Document(page_content=text or "", metadata=item_metadata))

        return output

    def _chunk_file(self, file_payload: UploadFilePayload) -> list[Document]:
        documents = self._load_file(file_payload)
        chunks = self.splitter.split_documents(documents)

        if not chunks:
            return []

        file_hash = sha256(file_payload.content).hexdigest()
        for index, chunk in enumerate(chunks, start=1):
            chunk.metadata["chunk_index"] = index
            chunk.metadata["chunk_id"] = f"{file_hash}:{index}"
        return chunks

    def _load_file(self, file_payload: UploadFilePayload) -> list[Document]:
        extension = self._get_extension(file_payload.name)
        if extension not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {file_payload.name}. "
                f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        if extension == ".pdf":
            return self._load_pdf(file_payload)
        return self._load_text(file_payload)

    def _load_pdf(self, file_payload: UploadFilePayload) -> list[Document]:
        reader = PdfReader(io.BytesIO(file_payload.content))
        documents: list[Document] = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": file_payload.name,
                        "page": page_number,
                    },
                )
            )

        return documents

    def _load_text(self, file_payload: UploadFilePayload) -> list[Document]:
        text = file_payload.content.decode("utf-8-sig", errors="ignore").strip()
        if not text:
            return []

        return [
            Document(
                page_content=text,
                metadata={
                    "source": file_payload.name,
                    "page": 1,
                },
            )
        ]

    def _iterate_batches(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        metadatas: Sequence[dict[str, object]],
        batch_size: int,
    ) -> Iterable[tuple[list[str], list[str], list[dict[str, object]]]]:
        for start in range(0, len(ids), batch_size):
            stop = start + batch_size
            yield (
                list(ids[start:stop]),
                list(texts[start:stop]),
                list(metadatas[start:stop]),
            )

    @staticmethod
    def _get_extension(file_name: str) -> str:
        dot_index = file_name.rfind(".")
        return file_name[dot_index:].lower() if dot_index >= 0 else ""
