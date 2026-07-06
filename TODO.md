# Atlas V2 Architecture Audit & Refactor — TODO

## Plan summary (approved)
Refactor Atlas V2 into a production-ready, modular local RAG foundation.

## Steps
- [ ] Create TODO scaffolding + refactor checklist
- [x] Add logging module (logger.py) and initialize logging
- [x] Expand config.py to contain ALL configurable values
- [x] Add prompts folder with system/retrieval prompt templates
- [ ] Refactor chunker.py to output structured chunks (chunk_id, page_number, chunk_index, text)
- [ ] Refactor document_loader.py to return page-aware loaded documents (supports PDFs; extensible)
- [ ] Refactor vector_store.py into a Chroma-only module (collection init lazy, add/query/update/delete-by-doc)
- [ ] Add indexer.py for automatic hashing + incremental indexing (detect new/modified; skip unchanged; rebuild affected only)
- [ ] Refactor knowledge_search.py into retrieval-only (apply similarity threshold; return empty knowledge on low confidence)
- [ ] Refactor llm.py to only call Ollama
- [ ] Refactor atlas.py into entry-point only (startup orchestration + chat loop wiring)
- [ ] Remove duplicated logic (stop using knowledge_search’s PDF helpers)
- [ ] Add type hints + docstrings everywhere touched
- [ ] Smoke test: run atlas.py twice; verify incremental indexing
- [ ] Smoke test: intentionally low-relevance query returns exact "I don't know based on my knowledge base." 
- [ ] Final cleanup: remove unused imports, ensure no circular imports

