#!/usr/bin/env python3
"""
Build the retrieval index from catalog.json.

Usage:
    uv run python scripts/build_index.py

Reads app/data/catalog.json, embeds each entry via OpenRouter,
builds ChromaDB (dense) + BM25 (keyword) indices.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)
settings = get_settings()

DATA_DIR = Path(__file__).resolve().parent.parent / "app" / "data"
CATALOG_PATH = DATA_DIR / "catalog.json"


_TEST_TYPE_FULL = {
    "A": "Ability & Aptitude cognitive reasoning test",
    "B": "Biodata & Situational Judgment test",
    "C": "Competencies assessment exercise",
    "D": "Development & 360 feedback report",
    "K": "Knowledge & Skills test",
    "P": "Personality & Behavior questionnaire",
    "S": "Simulation coding exercise",
}


def _expand_test_types(codes: list[str]) -> str:
    return ". ".join(_TEST_TYPE_FULL.get(c, c) for c in codes if c)


def _make_doc_text(item: dict) -> str:
    """Produce semantically rich text for embedding and BM25 indexing."""
    name = item.get("name", "")
    description = item.get("description", "") or ""
    test_types = item.get("test_type", []) or []
    job_levels = item.get("job_levels", []) or []
    duration = item.get("duration") or ""
    remote = item.get("remote_testing")

    type_text = _expand_test_types(test_types) if test_types else ""
    levels_text = ("Suitable for: " + ", ".join(job_levels)) if job_levels else ""
    duration_text = (f"Duration: {duration}") if duration and duration != "-" else ""
    remote_text = "Supports remote online testing." if remote is True else ""

    parts = [name, description, type_text, levels_text, duration_text, remote_text]
    return " ".join(p for p in parts if p).strip()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts using OpenRouter.
    Batches requests to avoid hitting rate limits.
    If no API key is present, returns dummy 1536-dimensional zero vectors.
    """
    if not settings.openrouter_api_key:
        logger.warning("No OPENROUTER_API_KEY set. Generating mock 1536-dimensional zero vectors.")
        return [[0.0] * 1536 for _ in texts]

    from openai import OpenAI
    client = OpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
    )

    embeddings = []
    batch_size = 50  # OpenRouter embedding batch limit

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        logger.info(
            "Embedding batch %d/%d (%d texts)…",
            i // batch_size + 1,
            (len(texts) + batch_size - 1) // batch_size,
            len(batch),
        )
        try:
            response = client.embeddings.create(
                model=settings.embedding_model,
                input=batch,
            )
            batch_embeddings = [d.embedding for d in sorted(response.data, key=lambda x: x.index)]
            embeddings.extend(batch_embeddings)
        except Exception as exc:
            logger.error("Embedding batch failed: %s", exc)
            raise
        # Rate limiting
        time.sleep(0.5)

    return embeddings


def main():
    # Load catalog
    if not CATALOG_PATH.exists():
        logger.error("catalog.json not found at %s — run scrape_catalog.py first", CATALOG_PATH)
        sys.exit(1)

    with CATALOG_PATH.open() as f:
        raw_catalog = json.load(f)
    
    from app.retrieval.index import normalize_catalog
    catalog = normalize_catalog(raw_catalog)
    logger.info("Loaded and normalized %d catalog items", len(catalog))

    if len(catalog) == 0:
        logger.error("Catalog is empty!")
        sys.exit(1)


    # Create document texts
    texts = [_make_doc_text(item) for item in catalog]

    # Embed
    logger.info("Embedding %d items with model '%s'…", len(catalog), settings.embedding_model)
    embeddings = embed_batch(texts)
    logger.info("Embeddings done: %d vectors", len(embeddings))

    # Build indices
    from app.retrieval.index import build_index
    build_index(catalog, embeddings)

    # Verify
    logger.info("Index build complete ✓")
    print(f"\n{'='*50}")
    print("INDEX BUILD STATS")
    print(f"  Items indexed: {len(catalog)}")
    print(f"  Embedding dim: {len(embeddings[0]) if embeddings else 'N/A'}")
    print(f"  ChromaDB path: {settings.chroma_db_path}")
    print(f"  BM25 path:     {settings.bm25_index_path}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
