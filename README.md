# Code Search / Embedding Pipeline

RAG backend for code understanding: it retrieves the most relevant code chunks from a FAISS index and generates technical explanations with a local LLM stack.

This project is designed as a portfolio-ready foundation for:

- codebase Q&A,
- component explanation,
- retrieval-first GenAI workflows.

## Why this project

Reading unfamiliar repositories is slow. This backend reduces that friction by combining:

- embedding-based retrieval (`jina-embeddings-v5-text-small`),
- vector similarity search (FAISS),
- prompt-driven generation (`CYFRAGOVPL/Llama-PLLuM-8B-chat`).

## Core Features

- Batch embedding generation for code chunks
- Top-k retrieval from FAISS vector store
- Context-augmented generation (`/generate` endpoint)
- Prompt template system (`system --- user` format)
- FastAPI service with CORS support
- Parser test baseline in `tests/`

## Architecture

```text
Repository files
		-> Parser (chunking)
		-> Embedding model (Jina)
		-> FAISS index build/load
		-> Similarity search (top-k chunks)
		-> Prompt template + LLM (PLLuM)
		-> API response
```

### High-level flow

1. Parse source files into line-based chunks.
2. Encode chunks into embeddings in batches.
3. Persist vectors in FAISS.
4. For each user question, retrieve top-k relevant chunks.
5. Inject chunks into a prompt template.
6. Generate an explanation/answer with the chat model.

## Tech Stack

- **Backend:** FastAPI, Uvicorn
- **Retrieval:** LangChain + FAISS
- **Embeddings:** `jinaai/jina-embeddings-v5-text-small`
- **Generation:** `CYFRAGOVPL/Llama-PLLuM-8B-chat`
- **ML Runtime:** PyTorch, Transformers, bitsandbytes, accelerate
- **Testing:** Pytest

## Project Structure

```text
app/
	main.py                         # FastAPI entrypoint
	services/genAI/
		transformer.py                # LLM inference chain
		rag/
			parser.py                   # Repository parser + chunking
			embeddings.py               # Batch embedding pipeline
			pipeline.py                 # Embedding model wrapper
			utils.py                    # Similarity helpers
		context/
			utils.py                    # Prompt loader/parser
			prompts/                    # Prompt templates
db/
	save.py                         # Build and persist FAISS index
	load.py                         # Load FAISS and serve retrieval context
tests/
	test_parser.py                  # Basic parser test
```

## API

### `POST /generate`

Generates an answer from user input augmented with retrieved code context.

Request body:

```json
{
  "text": "Explain how this React component manages state"
}
```

Response:

- Generated text answer using retrieved code chunks (`k=3` currently).

## Quickstart

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Build FAISS index (offline step)

```bash
python db/save.py
```

### 3) Run API server

```bash
uvicorn app.main:app --reload
```

### 4) Test endpoint

```bash
curl -X POST http://127.0.0.1:8000/generate \
	-H "Content-Type: application/json" \
	-d "{\"text\":\"Explain this component architecture\"}"
```

## Configuration Notes

- The indexed repository path is currently hardcoded in `app/services/genAI/rag/embeddings.py`.
- FAISS path is loaded from `db/faiss_index` in `db/load.py`.
- Retrieval currently uses `k=3` in `app/main.py`.
- Prompt template is currently `explain_component` in `app/services/genAI/transformer.py`.

## Evaluation & Monitoring (Portfolio-ready)

Track and report these metrics in experiments:

- **Retrieval quality:** Recall@k, hit-rate by file/module
- **Context quality:** file diversity, confidence score, chunk relevance
- **Generation quality:** LLM-as-Judge rubric + human spot checks
- **System metrics:** latency (P50/P95), error rate, uptime, cost/request

Suggested benchmark set:

- 20-50 representative engineering questions,
- expected files/components per question,
- pass/fail or 1-5 scoring rubric.

## Prompt Strategy

Current prompt focus:

- hooks usage,
- rendering behavior,
- performance concerns.

Prompt files are stored in:

- `app/services/genAI/context/prompts/`

Recommended future prompt packs:

- bug localization,
- dependency tracing,
- refactor suggestions,
- architecture summary.

## Known Limitations

- Hardcoded repository path for ingestion.
- No metadata filtering (language/module/owner) in retrieval.
- No reranking layer after dense retrieval.
- Minimal automated tests (parser-focused baseline).
- No dedicated health endpoint yet.

## Roadmap

### Near-term

- Add environment-based configuration (`.env`) for paths and model IDs
- Add `/health` and `/ready` endpoints
- Add reranker support for better precision
- Add integration tests for `/generate`
- Add structured logging + request IDs

### Mid-term

- Add hybrid retrieval (dense + keyword/BM25)
- Add metadata-aware retrieval filters
- Add multi-repo ingestion pipeline
- Add caching for repeated queries

### Long-term

- Add streaming responses - DONE
- Add conversation memory/history controls
- Add evaluation dashboard with metric tracking over time
- Add CI/CD checks for retrieval and quality regressions

## Running Tests

```bash
pytest -q
```

## Portfolio Tips

To make this project stand out in interviews:

- include one architecture diagram image,
- include one short demo GIF,
- publish a small benchmark table before/after improvements,
- show 2-3 concrete trade-off decisions (quality vs latency vs cost).

## License

Add your preferred license (MIT/Apache-2.0) before public release.



## TODO: 

- kontekst rozmowy (historia)

