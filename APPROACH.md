# APPROACH.md — SHL Conversational Assessment Recommender

## Problem & Design Philosophy

The core challenge is grounding an LLM-based conversational agent entirely on the SHL product catalog — preventing hallucinated URLs, test names, or capabilities while still supporting natural multi-turn conversation.

## Architecture Decisions

### 1. Stateless API Design
The API accepts the full conversation history on every call. This eliminates server-side session state and makes every turn independently reproducible. The tradeoff is that fact extraction runs on every turn — acceptable at current scale (≤20 turns per conversation).

### 2. Hybrid Retrieval (Dense + BM25)
Pure embedding similarity misses exact technical matches (e.g., "Java"). Pure BM25 misses semantic equivalence (e.g., "interpersonal skills" ↔ "stakeholder communication"). We combine both with **Reciprocal Rank Fusion** (RRF) — a parameter-free fusion method that's robust to score distribution differences between the two indices.

### 3. LLM Fact Extraction (Not Regex)
Real conversations are messy. Users say "senior" in turn 3, mention a constraint in parentheses, or reference a previous assistant message. A structured-JSON LLM extraction pass over the full history is far more robust than regex on individual turns.

### 4. Explicit Readiness Gate
```python
def is_ready_to_recommend(facts: dict) -> bool:
    return bool(facts.get("role_or_skill")) and (
        facts.get("seniority") or facts.get("context") or
        facts.get("explicit_request_for_shortlist")
    )
```
This concrete, testable function prevents premature recommendations on vague one-liners.

### 5. URL Hallucination Prevention
Every recommendation URL is validated against `catalog.json` before returning. URLs not in the catalog are silently dropped. This ensures the agent never invents product pages.

### 6. OpenRouter as LLM Gateway
Using OpenRouter provides access to multiple model providers through a single OpenAI-compatible API, enabling easy model swapping without code changes.

## Retrieval Strategy Details

**Embedding**: `openai/text-embedding-3-small` via OpenRouter — 1536-dim vectors, good performance/cost ratio.

**Document text**: `name + description + test_type + job_levels` concatenated — captures all signal relevant to assessment selection.

**Fusion**: RRF with k=60. Dense weight 0.6, BM25 weight 0.4. Dense slightly preferred because semantic similarity matters more than exact keyword match for most queries.

**Filters**: Metadata filters (remote_testing, adaptive_irt) applied post-fusion to preserve recall from both systems before narrowing.

## Known Limitations
- Catalog scraping depends on SHL's JS-rendered page structure (may need updating if SHL changes their site).
- Duration data quality varies across catalog entries.
- Rust/niche tech stacks may have no exact match — the agent explicitly says so.
