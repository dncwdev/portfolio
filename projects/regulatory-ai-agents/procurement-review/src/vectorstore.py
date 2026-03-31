from __future__ import annotations

import io
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
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
        collection_name: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embeddings = embeddings or build_embeddings()
        self.collection_name = collection_name or self.settings.chroma_collection_name
        self.settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.settings.chroma_persist_dir))
        self.collection = self._get_or_create_collection()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size or self.settings.chunk_size,
            chunk_overlap=chunk_overlap or self.settings.chunk_overlap,
            add_start_index=True,
        )

    def get_stats(self) -> dict[str, int | str]:
        return {
            "collection_name": self.collection_name,
            "chunk_count": self.collection.count(),
        }

    def has_source(self, source: str) -> bool:
        result = self.collection.get(where={"source": source}, limit=1, include=["metadatas"])
        return bool(result.get("ids"))

    def clear_collection(self) -> None:
        self.client.delete_collection(name=self.collection_name)
        self.collection = self._get_or_create_collection()

    def get_collection_excerpt(self, limit: int = 5, max_chars: int = 4000) -> str:
        if self.collection.count() == 0:
            return ""

        result = self.collection.get(limit=limit, include=["documents"])
        documents = result.get("documents", [])
        chunks: list[str] = []
        total_chars = 0

        for document in documents:
            if not document:
                continue
            remaining = max_chars - total_chars
            if remaining <= 0:
                break
            snippet = document[:remaining].strip()
            if snippet:
                chunks.append(snippet)
                total_chars += len(snippet)

        return "\n\n".join(chunks)

    def ingest_files(self, files: Sequence[UploadFilePayload]) -> IngestionResult:
        all_chunks: list[Document] = []

        for file_payload in files:
            documents = self.load_documents_from_bytes(
                file_name=file_payload.name,
                file_bytes=file_payload.content,
                source_name=file_payload.name,
            )
            all_chunks.extend(
                self._chunk_documents(documents, sha256(file_payload.content).hexdigest())
            )

        if not all_chunks:
            raise ValueError("No indexable text was extracted from the provided files.")

        self._upsert_chunks(all_chunks)
        return IngestionResult(
            files_indexed=len(files),
            chunks_indexed=len(all_chunks),
        )

    def ingest_documents(
        self,
        documents: Sequence[Document],
        fingerprint_key: str | None = None,
    ) -> IngestionResult:
        if not documents:
            raise ValueError("No documents were provided for ingestion.")

        chunks = self._chunk_documents(
            documents,
            fingerprint_key or self._fingerprint_documents(documents),
        )
        if not chunks:
            raise ValueError("No indexable text was extracted from the provided documents.")

        self._upsert_chunks(chunks)
        source_count = len({document.metadata.get("source", "unknown") for document in documents})
        return IngestionResult(files_indexed=source_count, chunks_indexed=len(chunks))

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
            item_metadata["collection_name"] = self.collection_name
            output.append(Document(page_content=text or "", metadata=item_metadata))

        return output

    def load_documents_from_path(
        self,
        file_path: Path,
        source_name: str | None = None,
    ) -> list[Document]:
        return self.load_documents_from_bytes(
            file_name=file_path.name,
            file_bytes=file_path.read_bytes(),
            source_name=source_name or file_path.name,
        )

    def load_documents_from_bytes(
        self,
        file_name: str,
        file_bytes: bytes,
        source_name: str | None = None,
    ) -> list[Document]:
        extension = self._get_extension(file_name)
        source = source_name or file_name
        if extension not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {file_name}. "
                f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        if extension == ".pdf":
            return self._load_pdf(file_bytes, source)
        return self._load_text(file_bytes, source)

    def _chunk_documents(
        self,
        documents: Sequence[Document],
        fingerprint: str,
    ) -> list[Document]:
        chunks = self.splitter.split_documents(list(documents))
        for index, chunk in enumerate(chunks, start=1):
            chunk.metadata["chunk_index"] = index
            chunk.metadata["chunk_id"] = f"{fingerprint}:{index}"
            chunk.metadata["collection_name"] = self.collection_name
        return chunks

    def _upsert_chunks(self, chunks: Sequence[Document]) -> None:
        ids = [chunk.metadata["chunk_id"] for chunk in chunks]
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [dict(chunk.metadata) for chunk in chunks]

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

    def _load_pdf(self, file_bytes: bytes, source: str) -> list[Document]:
        reader = PdfReader(io.BytesIO(file_bytes))
        documents: list[Document] = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": source,
                        "page": page_number,
                    },
                )
            )

        return documents

    def _load_text(self, file_bytes: bytes, source: str) -> list[Document]:
        text = file_bytes.decode("utf-8-sig", errors="ignore").strip()
        if not text:
            return []

        return [
            Document(
                page_content=text,
                metadata={
                    "source": source,
                    "page": 1,
                },
            )
        ]

    def _fingerprint_documents(self, documents: Sequence[Document]) -> str:
        payload = "\n".join(
            f"{document.metadata.get('source', 'unknown')}|{document.page_content}"
            for document in documents
        )
        return sha256(payload.encode("utf-8")).hexdigest()

    def _get_or_create_collection(self):
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

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
