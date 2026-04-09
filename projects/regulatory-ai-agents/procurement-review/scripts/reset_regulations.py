from __future__ import annotations
from src.config import get_settings
import chromadb
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))


settings = get_settings()
client = chromadb.PersistentClient(path=str(settings.chroma_db_path))

collections = client.list_collections()
print("현재 컬렉션:")
for col in collections:
  print(f"  - {col.name}")

client.delete_collection(settings.regulations_collection_name)
print(f"\n삭제 완료: {settings.regulations_collection_name}")
