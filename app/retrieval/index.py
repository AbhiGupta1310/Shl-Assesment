"""
Index management: build and load ChromaDB (dense) + BM25 (keyword) indices.
The index is built once via scripts/build_index.py and loaded at startup.
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Module-level singletons (loaded once at startup)
_chroma_collection = None
_bm25_index = None
_bm25_corpus: list[dict] = []   # parallel list of catalog dicts
_catalog: list[dict] = []


def normalize_catalog(raw_catalog: list[dict]) -> list[dict]:
    normalized = []
    test_type_mapping = {
        "Knowledge & Skills": ["K"],
        "Simulations": ["S"],
        "Personality & Behavior": ["P"],
        "Assessment Exercises": ["C"],
        "Biodata & Situational Judgement": ["B"],
        "Biodata & Situational Judgment": ["B"],
        "Competencies": ["C"],
        "Development & 360": ["D"],
        "Ability & Aptitude": ["A"]
    }
    for item in raw_catalog:
        name = item.get("name", "").strip()
        
        # 1. URL / Link
        url = item.get("url") or item.get("link") or ""
        url = url.strip()
        
        # 2. Test type / keys
        test_type = item.get("test_type")
        if not test_type:
            keys = item.get("keys") or []
            if isinstance(keys, str):
                keys = [keys]
            test_type = []
            for key in keys:
                for mapping_key, val in test_type_mapping.items():
                    if mapping_key.lower() in key.lower():
                        test_type.extend(val)
            if not test_type:
                test_type = ["K"]  # fallback default
        
        # Deduplicate list
        if isinstance(test_type, list):
            test_type = sorted(list(set(test_type)))
        else:
            test_type = [test_type]

        # 3. Remote testing
        remote_testing = item.get("remote_testing")
        if remote_testing is None:
            remote = str(item.get("remote", "")).lower()
            remote_testing = True if remote == "yes" else (False if remote == "no" else None)

        # 4. Adaptive/IRT
        adaptive_irt = item.get("adaptive_irt")
        if adaptive_irt is None:
            adaptive = str(item.get("adaptive", "")).lower()
            adaptive_irt = True if adaptive == "yes" else (False if adaptive == "no" else None)

        # 5. Duration
        duration = item.get("duration") or item.get("duration_raw") or "-"
        duration = str(duration).strip()
        if not duration:
            duration = "-"

        # 6. Description
        description = item.get("description") or f"SHL assessment for {name}."
        description = str(description).strip()

        # 7. Job levels
        job_levels = item.get("job_levels") or ["Individual Contributor"]

        # 8. Languages
        languages = item.get("languages") or ["English (USA)"]

        normalized.append({
            "name": name,
            "url": url,
            "test_type": test_type,
            "description": description,
            "remote_testing": remote_testing,
            "adaptive_irt": adaptive_irt,
            "duration": duration,
            "job_levels": job_levels,
            "languages": languages
        })
    return normalized


def load_index() -> None:
    """Load ChromaDB collection + BM25 index from disk. Called at startup."""
    global _chroma_collection, _bm25_index, _bm25_corpus, _catalog

    # Load catalog
    catalog_path = Path(settings.catalog_path)
    if not catalog_path.exists():
        logger.error("catalog.json missing — retrieval will be empty")
        return
    with catalog_path.open() as f:
        raw_catalog = json.load(f)
    _catalog = normalize_catalog(raw_catalog)
    logger.info("Loaded and normalized %d catalog items", len(_catalog))

    # Load BM25
    bm25_path = Path(settings.bm25_index_path)
    if bm25_path.exists():
        with bm25_path.open("rb") as f:
            saved = pickle.load(f)
        _bm25_index = saved["index"]
        _bm25_corpus = normalize_catalog(saved["corpus"])
        logger.info("BM25 index loaded (%d docs)", len(_bm25_corpus))
    else:
        logger.warning("BM25 index not found at %s", bm25_path)

    # Load ChromaDB
    try:
        import chromadb
        chroma_path = Path(settings.chroma_db_path)
        if chroma_path.exists():
            client = chromadb.PersistentClient(path=str(chroma_path))
            _chroma_collection = client.get_collection("shl_catalog")
            logger.info("ChromaDB collection loaded (%d vectors)", _chroma_collection.count())
        else:
            logger.warning("ChromaDB not found at %s", chroma_path)
    except Exception as exc:
        logger.warning("ChromaDB load failed: %s", exc)



def get_chroma_collection():
    global _chroma_collection
    if _chroma_collection is None:
        load_index()
    return _chroma_collection


def get_bm25_index():
    global _bm25_index
    if _bm25_index is None:
        load_index()
    return _bm25_index


def get_bm25_corpus() -> list[dict]:
    global _bm25_corpus
    if not _bm25_corpus:
        load_index()
    return _bm25_corpus


def get_catalog() -> list[dict]:
    global _catalog
    if not _catalog:
        load_index()
    return _catalog


def build_index(catalog: list[dict], embeddings: list[list[float]]) -> None:
    """
    Build and persist ChromaDB + BM25 indices.
    Called from scripts/build_index.py — NOT from API startup.
    """
    import chromadb
    from rank_bm25 import BM25Okapi

    # ── ChromaDB ────────────────────────────────────────────────────────
    chroma_path = Path(settings.chroma_db_path)
    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path))

    # Delete existing collection to allow rebuild
    try:
        client.delete_collection("shl_catalog")
    except Exception:
        pass

    collection = client.create_collection(
        "shl_catalog",
        metadata={"hnsw:space": "cosine"},
    )

    ids = [str(i) for i in range(len(catalog))]
    documents = [_make_doc_text(item) for item in catalog]
    metadatas = [_make_metadata(item) for item in catalog]

    # Batch upsert (ChromaDB limit ~5461 per call)
    batch_size = 500
    for start in range(0, len(catalog), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
    logger.info("ChromaDB: %d items indexed", len(catalog))

    # ── BM25 ────────────────────────────────────────────────────────────
    tokenized_corpus = [_make_doc_text(item).lower().split() for item in catalog]
    bm25 = BM25Okapi(tokenized_corpus)

    bm25_path = Path(settings.bm25_index_path)
    bm25_path.parent.mkdir(parents=True, exist_ok=True)
    with bm25_path.open("wb") as f:
        pickle.dump({"index": bm25, "corpus": catalog}, f)
    logger.info("BM25 index saved (%d docs)", len(catalog))


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
    """Expand short test-type codes to full human-readable names."""
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


def _make_metadata(item: dict) -> dict:
    """Flatten catalog item into ChromaDB-compatible metadata (strings/numbers only)."""
    return {
        "name": item.get("name", ""),
        "url": item.get("url", ""),
        "test_type": ",".join(item.get("test_type", [])),
        "remote_testing": str(item.get("remote_testing", "")),
        "adaptive_irt": str(item.get("adaptive_irt", "")),
        "duration": item.get("duration", "") or "",
        "job_levels": ",".join(item.get("job_levels", [])),
        "languages": ",".join(item.get("languages", [])),
    }
