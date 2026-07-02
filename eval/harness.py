"""
Evaluation harness: replays conversation traces and computes Recall@10.
Runs turn-by-turn directly via handle_turn (no HTTP server needed).
"""
from __future__ import annotations

import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
TRACES_DIR = Path(__file__).resolve().parent / "traces"


def parse_expected_urls(content: str) -> list[str]:
    """Parse the expected recommendation URLs from trace markdown tables."""
    urls = []
    # Match URL links: <https://www.shl.com/...>
    pattern = re.compile(r"<(https://www\.shl\.com/[^>]+)>")
    for match in pattern.finditer(content):
        url = match.group(1).strip()
        if url not in urls:
            urls.append(url)
    return urls


def parse_messages(content: str) -> list[dict]:
    """Extract user turns from trace markdown."""
    messages = []
    turns = re.split(r"###\s+Turn\s+\d+", content)[1:]
    for turn in turns:
        user_match = re.search(r"\*\*User\*\*\s*\n+>\s*(.+?)(?=\n\n|\*\*Agent\*\*)", turn, re.DOTALL)
        if user_match:
            user_text = user_match.group(1).strip().lstrip(">").strip()
            # Remove the markdown blockquote prefix if multiline
            user_text = re.sub(r"\n>\s*", "\n", user_text).strip()
            messages.append({"role": "user", "content": user_text})
    return messages


def _normalize_url(url: str) -> str:
    """Normalize SHL product URL to a canonical slug for comparison."""
    url = url.lower().rstrip("/")
    # Strip either /products/product-catalog/view/ or /solutions/products/product-catalog/view/
    for prefix in [
        "https://www.shl.com/solutions/products/product-catalog/view",
        "https://www.shl.com/products/product-catalog/view",
    ]:
        if url.startswith(prefix):
            return url[len(prefix):].lstrip("/")
    return url


async def evaluate_trace(trace_path: Path, api_url: str = "") -> float:
    """
    Replays a single trace directly via handle_turn and calculates Recall@10.
    api_url is kept as parameter for compatibility but not used (direct import).
    """
    content = trace_path.read_text(encoding="utf-8")
    expected_urls = parse_expected_urls(content)
    user_messages = parse_messages(content)

    if not expected_urls or not user_messages:
        logger.info("Trace %s: no expected URLs or messages, returning 1.0", trace_path.name)
        return 1.0

    # Import here to avoid circular imports at module level
    from app.agent.orchestrator import handle_turn
    from app.models import Message

    history: list[dict] = []
    final_recs = []

    for msg in user_messages:
        history.append(msg)
        try:
            msgs = [Message(role=m["role"], content=m["content"]) for m in history]
            resp = await handle_turn(msgs)
            history.append({"role": "assistant", "content": resp.reply})
            if resp.recommendations:
                final_recs = resp.recommendations
        except Exception as e:
            logger.error("Harness error on trace %s turn: %s", trace_path.name, e)

    if not final_recs:
        return 0.0

    # Calculate Recall@10 using normalized URL slugs
    norm_expected = {_normalize_url(u) for u in expected_urls}
    norm_rec = {_normalize_url(r.url) for r in final_recs}

    hits = norm_expected & norm_rec
    recall = len(hits) / len(norm_expected)
    logger.info(
        "Trace %s: expected=%d, recommended=%d, hits=%d, recall=%.2f",
        trace_path.name, len(norm_expected), len(norm_rec), len(hits), recall
    )
    return recall
