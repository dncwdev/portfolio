# Procurement Review Agent

Phase 1 implementation of a procurement document review agent built with:

- LangChain
- ChromaDB
- Streamlit
- vLLM OpenAI-compatible endpoints for LLM, embeddings, and reranking

## Phase 1 Scope

This phase implements the baseline RAG flow:

1. Upload procurement documents in the Streamlit UI.
2. Chunk and embed documents into a local persistent ChromaDB collection.
3. Retrieve top 20 candidate chunks for a user query.
4. Rerank the retrieved chunks through the vLLM `/v1/score` endpoint.
5. Generate a cited answer from the top 5 reranked chunks.

## Implemented Components

- `src/config.py`
  - Loads `.env`
  - Builds LangChain LLM and embedding clients
  - Exposes the reranker client configuration
- `src/vectorstore.py`
  - Initializes persistent ChromaDB
  - Ingests `pdf`, `txt`, and `md`
  - Performs similarity search
- `src/reranker.py`
  - Calls the reranker `/v1/score` endpoint directly
- `src/rag_pipeline.py`
  - Composes the retrieval, reranking, and answer generation flow with LCEL
- `app.py`
  - Streamlit UI for ingestion, querying, and evidence review

## Environment Variables

Create a local `.env` file with these values:

```dotenv
LLM_BASE_URL=http://192.168.1.149:58888
LLM_MODEL_NAME=gpt-oss-120b
LLM_API_KEY=empty

EMBEDDING_BASE_URL=http://192.168.1.166:58001
EMBEDDING_MODEL_NAME=/model
EMBEDDING_API_KEY=empty

RERANKER_BASE_URL=http://192.168.1.166:58002
RERANKER_MODEL_NAME=/model
RERANKER_API_KEY=empty

CHROMA_PERSIST_DIR=.chroma
CHROMA_COLLECTION_NAME=procurement_regulations
CHUNK_SIZE=1200
CHUNK_OVERLAP=200
RETRIEVAL_TOP_K=20
RERANK_TOP_K=5
REQUEST_TIMEOUT=60
LLM_TEMPERATURE=0
LLM_MAX_TOKENS=2048
LLM_INCLUDE_REASONING=false
```

## Run

```powershell
.\.venv\Scripts\activate
streamlit run app.py
```

## Validation Status

- Offline validation completed
  - `python -m compileall app.py src`
  - Mocked end-to-end RAG smoke test
- Live backend validation is still pending
  - Run after the vLLM, embedding, and reranker services are back online
