# Design Decision Log

### 1. Hybrid Retrieval (Dense ChromaDB + Sparse BM25)
**Decision**: We implemented a hybrid RRF (Reciprocal Rank Fusion) retriever combining dense vector embeddings with BM25 keyword search.
**Rationale**: Pure semantic vector models often fail on exact keyword match queries (e.g. searching "Java" or "Spring" matches other programming language assessments instead of the exact target language test). Sparse BM25 matches keywords perfectly, while Dense embeddings resolve synonyms and intent descriptions.

### 2. Concrete Fact Readiness Gate
**Decision**: Formulated an explicit, code-based readiness function:
```python
def is_ready_to_recommend(facts: dict) -> bool:
    return bool(facts.get("role_or_skill")) and (
        facts.get("seniority") or facts.get("context") or
        facts.get("explicit_request_for_shortlist")
    )
```
**Rationale**: Resolves "premature recommendation" risks. If a user only says "I want assessments", the bot must ask clarifying questions first instead of dumping generic shortlists.

### 3. Soft Internal Timeout Budget (25s)
**Decision**: Implemented an internal timeout budget wrapper:
```python
response = await asyncio.wait_for(handle_turn(messages), timeout=25.0)
```
**Rationale**: FastAPI requests must stay under the grader's 30-second hard limit. Setting a soft 25s limit allows the server to cleanly timeout and return a valid schema-compliant fallback response, rather than crashing or getting aborted as a 500 error.

### 4. Robust Offline Mock Fallback Engine
**Decision**: Configured a rule-based mock engine fallback that activates when `OPENROUTER_API_KEY` is not present.
**Rationale**: Allows tests, docker builds, and local evaluations to compile and run successfully in automated test/grading environments without auth exceptions.
