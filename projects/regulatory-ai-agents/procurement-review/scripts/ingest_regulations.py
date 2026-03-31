from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import build_embeddings, get_settings
from src.vectorstore import ProcurementVectorStore


SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def iter_regulation_files(root_dir: Path) -> list[Path]:
    files: list[Path] = []
    for file_path in root_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(file_path)
    return sorted(files)


def main() -> None:
    settings = get_settings()
    regulations_dir = settings.regulations_data_dir
    if not regulations_dir.exists():
        raise SystemExit(f"Regulations directory does not exist: {regulations_dir}")

    store = ProcurementVectorStore(
        settings=settings,
        embeddings=build_embeddings(),
        collection_name=settings.regulations_collection_name,
        chunk_size=settings.regulations_chunk_size,
        chunk_overlap=settings.regulations_chunk_overlap,
    )

    files = iter_regulation_files(regulations_dir)
    if not files:
        print(f"No PDF/TXT files found under {regulations_dir}")
        return

    ingested_files = 0
    ingested_chunks = 0
    skipped_files = 0

    for file_path in files:
        source_name = file_path.relative_to(regulations_dir).as_posix()
        if store.has_source(source_name):
            skipped_files += 1
            print(f"skip_existing\t{source_name}")
            continue

        documents = store.load_documents_from_path(file_path, source_name=source_name)
        if not documents:
            skipped_files += 1
            print(f"skip_empty\t{source_name}")
            continue

        result = store.ingest_documents(documents, fingerprint_key=source_name)
        ingested_files += result.files_indexed
        ingested_chunks += result.chunks_indexed
        print(f"ingested\t{source_name}\tchunks={result.chunks_indexed}")

    stats = store.get_stats()
    print("\nSummary")
    print(f"collection={stats['collection_name']}")
    print(f"stored_chunks={stats['chunk_count']}")
    print(f"ingested_files={ingested_files}")
    print(f"ingested_chunks={ingested_chunks}")
    print(f"skipped_files={skipped_files}")


if __name__ == "__main__":
    main()
