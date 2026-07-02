"""
Hybrid retriever: dense (ChromaDB cosine similarity) + keyword (BM25).
Scores are fused with Reciprocal Rank Fusion (RRF).
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from app.config import get_settings
from app.models import CatalogItem

logger = logging.getLogger(__name__)
settings = get_settings()

# Catalog URL → item dict lookup (populated lazily)
_catalog_lookup: dict[str, dict] = {}


def get_catalog_lookup() -> dict[str, dict]:
    """Return URL→item mapping, loading catalog if needed."""
    global _catalog_lookup
    if not _catalog_lookup:
        from app.retrieval.index import get_catalog
        catalog = get_catalog()
        _catalog_lookup = {item["url"]: item for item in catalog}
    return _catalog_lookup



def _rrf_score(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion score."""
    return 1.0 / (k + rank)


def _get_embedding_sync(text: str) -> list[float]:
    """Synchronous embedding call via OpenRouter (OpenAI-compatible)."""
    if not settings.openrouter_api_key:
        return [0.0] * 1536

    from openai import OpenAI
    client = OpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
    )
    response = client.embeddings.create(
        model=settings.embedding_model,
        input=text,
    )
    return response.data[0].embedding


async def _get_embedding(text: str) -> list[float]:
    """Async wrapper for embedding call."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _get_embedding_sync, text)


async def search(
    query: str,
    k: int = 10,
    filters: dict | None = None,
) -> list[CatalogItem]:
    """
    Hybrid search: dense + BM25, fused with RRF.

    Args:
        query: natural-language query
        k: number of results to return
        filters: optional metadata filters (e.g. {"remote_testing": True})

    Returns:
        List of CatalogItem sorted by relevance (highest first).
    """
    from app.retrieval.index import get_chroma_collection, get_bm25_index, get_bm25_corpus

    collection = get_chroma_collection()
    bm25 = get_bm25_index()
    bm25_corpus = get_bm25_corpus()

    # Fallback: if no index loaded, return from catalog directly
    if collection is None and bm25 is None:
        logger.warning("No retrieval index loaded — returning empty results")
        return []

    scores: dict[str, float] = {}
    url_to_metadata: dict[str, dict] = {}

    # ── Dense retrieval (ChromaDB) ────────────────────────────────────
    if collection is not None:
        try:
            query_embedding = await _get_embedding(query)
            n_results = min(k * 3, collection.count())
            if n_results > 0:
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    include=["metadatas", "distances"],
                )
                metas = results["metadatas"][0]
                for rank, meta in enumerate(metas):
                    url = meta.get("url", "")
                    if url:
                        url_to_metadata[url] = meta
                        scores[url] = scores.get(url, 0.0) + (
                            settings.dense_weight * _rrf_score(rank + 1, settings.rrf_k)
                        )
        except Exception as exc:
            logger.warning("Dense retrieval failed: %s", exc)

    # ── BM25 retrieval ────────────────────────────────────────────────
    if bm25 is not None and bm25_corpus:
        try:
            tokenized_query = query.lower().split()
            bm25_scores = bm25.get_scores(tokenized_query)
            # Get top k*3 indices by score
            ranked_indices = sorted(
                range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
            )[: k * 3]
            for rank, idx in enumerate(ranked_indices):
                item = bm25_corpus[idx]
                url = item.get("url", "")
                if url:
                    if url not in url_to_metadata:
                        url_to_metadata[url] = item
                    scores[url] = scores.get(url, 0.0) + (
                        settings.bm25_weight * _rrf_score(rank + 1, settings.rrf_k)
                    )
        except Exception as exc:
            logger.warning("BM25 retrieval failed: %s", exc)

    # ── Apply metadata filters ────────────────────────────────────────
    if filters:
        filtered_scores = {}
        catalog_lookup = get_catalog_lookup()
        for url, score in scores.items():
            item = catalog_lookup.get(url, url_to_metadata.get(url, {}))
            if _passes_filter(item, filters):
                filtered_scores[url] = score
        scores = filtered_scores

    # ── Sort + build results ──────────────────────────────────────────
    sorted_urls = sorted(scores, key=lambda u: scores[u], reverse=True)[:k]
    catalog_lookup = get_catalog_lookup()

    results = []
    for url in sorted_urls:
        item = catalog_lookup.get(url) or url_to_metadata.get(url)
        if item:
            results.append(_dict_to_catalog_item(item))

    return results


def _passes_filter(item: dict, filters: dict) -> bool:
    """Check if a catalog item passes all specified metadata filters."""
    for key, value in filters.items():
        item_val = item.get(key)
        # Handle stringified booleans from ChromaDB metadata
        if isinstance(item_val, str):
            item_val = item_val.lower() == "true" if value in (True, False) else item_val
        if item_val != value:
            return False
    return True


async def search_by_names(names: list[str]) -> list[CatalogItem]:
    """
    Retrieve catalog items by fuzzy name matching.
    Used for comparison requests where user names specific assessments.
    """
    from rapidfuzz import process, fuzz
    catalog_lookup = get_catalog_lookup()
    name_to_url: dict[str, str] = {v["name"]: k for k, v in catalog_lookup.items()}
    all_names_list = list(name_to_url.keys())
    aliases = {
        "opq": "Occupational Personality Questionnaire OPQ32r",
        "opq32r": "Occupational Personality Questionnaire OPQ32r",
        "gsa": "Global Skills Assessment",
        "global skills": "Global Skills Assessment",
        "global skills assessment": "Global Skills Assessment",
        "verify g+": "SHL Verify Interactive G+",
        "verify interactive g+": "SHL Verify Interactive G+",
    }

    results = []
    seen_urls = set()
    for query_name in names:
        query_name_clean = query_name.strip()
        if not query_name_clean:
            continue

        # 1. Direct case-insensitive substring match (highly effective for short terms like "OPQ" or "GSA")
        matched_name = aliases.get(query_name_clean.lower())
        for canonical_name in all_names_list:
            if matched_name:
                break
            if query_name_clean.lower() in canonical_name.lower():
                matched_name = canonical_name
                break

        # 2. Fallback to fuzzy matching
        if not matched_name:
            match = process.extractOne(
                query_name_clean,
                all_names_list,
                scorer=fuzz.partial_ratio,
                score_cutoff=60,
            )
            if match:
                matched_name = match[0]

        if matched_name:
            url = name_to_url[matched_name]
            if url in seen_urls:
                continue
            item = catalog_lookup.get(url)
            if item:
                results.append(_dict_to_catalog_item(item))
                seen_urls.add(url)

    return results


def _dict_to_catalog_item(item: dict) -> CatalogItem:
    test_type = item.get("test_type", [])
    if isinstance(test_type, str):
        test_type = [t.strip() for t in test_type.split(",") if t.strip()]
    return CatalogItem(
        name=item.get("name", ""),
        url=item.get("url", ""),
        test_type=test_type,
        description=item.get("description", ""),
        remote_testing=item.get("remote_testing"),
        adaptive_irt=item.get("adaptive_irt"),
        duration=item.get("duration"),
        job_levels=_split_list_field(item.get("job_levels", [])),
        languages=_split_list_field(item.get("languages", [])),
    )


def _split_list_field(val) -> list[str]:
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        return [v.strip() for v in val.split(",") if v.strip()]
    return []
