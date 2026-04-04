# Procurement Review Agent

Procurement compliance review agent built with:

- LangChain agent runtime
- ChromaDB dual collections (`regulations` + uploaded procurement documents)
- vLLM OpenAI-compatible endpoints for LLM and embeddings
- Selectable rerankers via vLLM `/v1/score` or Infinity `/rerank`
- Streamlit UI
- Optional `korean-law-mcp` integration for external law lookup

## Architecture

The current flow is agent-driven rather than a single LCEL RAG chain:

1. The agent must call the local ChromaDB search tool first.
2. The agent can optionally call `korean-law-mcp` when `USE_MCP=true`.
3. Gathered evidence is normalized into cited regulation sources `[R#]` and document sources `[D#]`.
4. A final synthesis prompt produces the fixed review format:
   - `준수 판단`
   - `핵심 근거`
   - `판단 근거`
   - `추가 확인 필요사항`

This keeps the existing answer structure while allowing selective external legal lookup.

## Implemented Components

- `src/config.py`
  - Loads `.env`
  - Builds LLM, embeddings, multi-reranker, and MCP runtime settings
- `src/vectorstore.py`
  - Handles local ChromaDB ingestion and similarity search
- `src/reranker.py`
  - Calls either the vLLM `/v1/score` reranker endpoint or the Infinity `/rerank` endpoint
- `src/agent_tools.py`
  - Defines the LangChain local search tool
  - Defines the LangChain `korean-law-mcp` wrapper tool
  - Collects evidence across tool calls
- `src/mcp_client.py`
  - Connects to `korean-law-mcp` over stdio or HTTP/SSE
  - Calls `search_law`, `get_law_text`, `search_precedents`, `search_interpretations`, `search_all`
- `src/rag_pipeline.py`
  - Runs the LangChain agent for tool selection
  - Synthesizes the final compliance review from gathered evidence
- `app.py`
  - Streamlit UI for ingestion, model selection, querying, and evidence review
- `scripts/ingest_regulations.py`
  - Pre-embeds regulations into the local `regulations` collection

## Environment Variables

Copy `.env.template` to `.env`, then adjust values:

```powershell
Copy-Item .env.template .env
```

Template:

```dotenv
LLM_BASE_URL=
MODEL_URL_gpt-oss-120b=http://192.168.1.149:58888
MODEL_URL_exaone-4.0-32b=http://192.168.1.149:58889
MODEL_URL_qwen3.5-35b-a3b=http://192.168.1.149:58890
MODEL_URL_qwen3.5-9b=http://192.168.1.149:58891
LLM_MODEL_NAME=gpt-oss-120b
LLM_DISPLAY_NAME=gpt-oss-120b
AVAILABLE_MODELS=gpt-oss-120b,exaone-4.0-32b,qwen3.5-35b-a3b,qwen3.5-9b
LLM_API_KEY=empty

EMBEDDING_BASE_URL=http://192.168.1.166:58001
EMBEDDING_MODEL_NAME=/model
EMBEDDING_DISPLAY_NAME=BAAI/bge-m3
EMBEDDING_API_KEY=empty

RERANKER_BASE_URL=http://192.168.1.166:58002
RERANKER_MODEL_NAME=/model
RERANKER_DISPLAY_NAME=BAAI/bge-reranker-v2-m3
RERANKER_API_KEY=empty
RERANKER_ENGINE=vllm

RERANKER_BASE_URL2=http://127.0.0.1:58002
RERANKER_MODEL_NAME2=/model
RERANKER_DISPLAY_NAME2=BAAI/bge-reranker-v2-m3
RERANKER_API_KEY2=empty
RERANKER_ENGINE2=infinity
DEFAULT_RERANKER_KEY=default

CHROMA_PERSIST_DIR=.chroma
CHROMA_COLLECTION_NAME=procurement_regulations
REGULATIONS_COLLECTION_NAME=regulations
REGULATIONS_DATA_DIR=data/regulations

CHUNK_SIZE=1200
CHUNK_OVERLAP=200
REGULATIONS_CHUNK_SIZE=500
REGULATIONS_CHUNK_OVERLAP=50
RETRIEVAL_TOP_K=20
RERANK_TOP_K=5
RERANK_RELATIVE_THRESHOLD=0.15

REQUEST_TIMEOUT=600
LLM_TEMPERATURE=0
LLM_MAX_TOKENS=2048
LLM_INCLUDE_REASONING=false

USE_MCP=false
KOREAN_LAW_MCP_TRANSPORT=stdio
KOREAN_LAW_MCP_COMMAND=npx
KOREAN_LAW_MCP_ARGS=-y --package korean-law-mcp korean-law-mcp
KOREAN_LAW_MCP_URL=
LAW_OC=
```

Notes:

- If `MODEL_URL_<selected-model>` is missing, the app falls back to `LLM_BASE_URL`.
- `USE_MCP=false` keeps the system fully local.
- `USE_MCP=true` enables the optional MCP tool.
- `USE_MCP` is the default mode on startup, but the Streamlit sidebar can override it at runtime.
- `RERANKER_ENGINE` supports `vllm`, `infinity`, or `auto`.
- Each domain template batch uses its fixed domain template string as the rerank query instead of the full user question.
- Chunks with rerank score below `max_score_in_batch * RERANK_RELATIVE_THRESHOLD` are excluded before they reach the LLM.
- Additional reranker profiles can be added with numeric suffixes such as `RERANKER_BASE_URL2`.
- The Streamlit sidebar lets you switch rerankers at runtime without editing code.
- For stdio mode, install `korean-law-mcp` separately and set `LAW_OC`.
- For HTTP mode, set `KOREAN_LAW_MCP_TRANSPORT=http` and `KOREAN_LAW_MCP_URL`.

Example stdio configuration:

```dotenv
USE_MCP=true
KOREAN_LAW_MCP_TRANSPORT=stdio
KOREAN_LAW_MCP_COMMAND=npx
KOREAN_LAW_MCP_ARGS=-y --package korean-law-mcp korean-law-mcp
LAW_OC=your-law-go-kr-open-api-key
```

Example hosted MCP configuration:

```dotenv
USE_MCP=true
KOREAN_LAW_MCP_TRANSPORT=http
KOREAN_LAW_MCP_URL=https://korean-law-mcp.fly.dev/mcp
```

## Run

Pre-embed regulations once before starting the app:

```powershell
.\.venv\Scripts\activate
python scripts/ingest_regulations.py
```

Then start Streamlit:

```powershell
.\.venv\Scripts\activate
streamlit run app.py
```

## vLLM Notes

This project now relies on model-side tool calling because the local Chroma search is exposed as a LangChain tool and the legal lookup path is exposed as an optional MCP-backed tool.

Make sure the selected vLLM model/server is configured for OpenAI-compatible tool calling before enabling the agent workflow.

## Validation

Offline validation completed:

- `python -m compileall app.py src scripts`
- `python scripts/test_retrieval_determinism.py --query "적용 법령 및 규정 검토"`
- `python scripts/audit_regulations_coverage.py`

Live validation is still required against:

- The selected vLLM chat endpoint with tool calling enabled
- The embedding endpoint
- The reranker endpoint
- `korean-law-mcp` when `USE_MCP=true`
