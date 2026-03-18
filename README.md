# Codebase Q&A Backend (RAG + FAISS)

Production-style backend for codebase question answering.  
The system indexes repository code, retrieves relevant fragments with FAISS, and generates answers with a local chat model.

## What Makes This Project Stand Out

- AST-based chunking for Python and syntax-aware chunking for JS/TS (tree-sitter with fallback).
- Cross-reference retrieval that expands context with linked definitions from other modules.
- FAISS retrieval with inner product (dot product) on L2-normalized embeddings, effectively matching cosine similarity behavior.
- Intent-based prompt routing (`explain_component`, `bug_check`, `find_dependency`, `summarize_module`).
- Streaming API responses and retrieval trace logging.

## Main Use Cases

- Explain how a component/module works.
- Diagnose likely code issues from repository context.
- Find dependencies and cross-module links.
- Summarize module responsibilities and logic flow.

## Architecture

```text
Repository files
	-> Parser (AST / syntax-aware chunking)
	-> Embedding generation (Jina)
	-> FAISS index build/save
	-> Query embedding + top-k similarity search
	-> Cross-reference expansion (symbols/imported modules)
	-> Prompt selection by intent
	-> LLM answer streaming
```

## Core Implementation Details

### 1. AST-Based Chunking

Implemented in `app/services/genAI/rag/parser.py`.

- Python files: chunks are extracted from AST nodes (`FunctionDef`, `AsyncFunctionDef`, `ClassDef`).
- JS/TS files: top-level syntax chunks are extracted via tree-sitter.
- Fallback: line-based chunking when AST/syntax extraction is not possible.

Each chunk stores metadata, including:

- `module_key`
- `defined_symbols`
- `referenced_symbols`
- `imports`
- `resolved_imports`
- `start_line` / `end_line`

### 2. Cross-Reference Retrieval

Implemented in `app/services/genAI/rag/FAISS/load.py`.

After base similarity retrieval, the system expands context by searching indexed chunks for:

- symbol definitions referenced in initial hits,
- modules targeted by resolved relative imports.

This improves answers for multi-file questions by attaching missing definitions from related files.

### 3. Similarity Strategy

- FAISS uses `MAX_INNER_PRODUCT` distance strategy.
- Embeddings and queries are normalized to unit vectors (`normalize_l2`).

With normalization, dot product corresponds to cosine similarity ranking.

## Tech Stack

- Backend: FastAPI, Uvicorn
- Retrieval: LangChain + FAISS
- Embeddings: `jinaai/jina-embeddings-v5-text-small`
- Generation: `CYFRAGOVPL/Llama-PLLuM-8B-chat`
- Runtime: PyTorch, Transformers, bitsandbytes, accelerate
- Tests: Pytest
