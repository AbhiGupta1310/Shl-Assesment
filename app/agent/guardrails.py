"""
Guardrails: scope checking, injection detection, and recommendation validation.
"""
from __future__ import annotations

import logging
import re
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Scope keywords ──────────────────────────────────────────────────────────
# Words that indicate a conversation is about hiring, assessments, or jobs.
# Broadly inclusive — we want to pass-through almost anything HR/work-related.
_SCOPE_KEYWORDS = re.compile(
    r"\b(assess|test|hire|hiring|skill|candidate|role|job|interview|"
    r"recruit|talent|seniority|personality|aptitude|cognitive|OPQ|SHL|"
    r"engineer|developer|manager|analyst|leadership|competenc|program|code|"
    r"experience|senior|junior|mid|intern|tech|languages|cv|resume|staff|"
    r"differ|solution|position|shortlist|recommend|evaluation|selection|"
    r"director|executive|CXO|graduate|entry.level|level|benchmark|"
    r"spring|java|python|rust|sql|aws|docker|web|cloud|finance|banking|"
    r"marketing|sales|customer|support|data|team|pool|candidate|project|"
    r"worker|employee|colleague|report|compare|short|duration|remote|adaptive|"
    r"behavior|behaviour|trait|ability|aptitude|verbal|numerical|abstract|"
    r"spatial|reasoning|situational|judgement|judgment|biodata|360|"
    r"development|competency|career|talent|workforce|onboarding|screening)\w*\b",
    re.IGNORECASE,
)

# ── Injection patterns ───────────────────────────────────────────────────────
_INJECTION_PATTERNS = re.compile(
    r"(ignore (previous|all|above|prior)|forget (your|all|previous)|"
    r"you are now|act as [a-z]+|new (instruction|persona|role)|"
    r"system prompt|jailbreak|override (all|previous|your)|disregard|pretend you|"
    r"reveal system|reveal instruction|output .jailbreak.)",
    re.IGNORECASE,
)

# ── Hard off-topic patterns (explicit non-HR topics) ────────────────────────
_OFF_TOPIC_PATTERNS = re.compile(
    r"\b(weather|forecast|temperature|recipe|cook|movie|film|sport|football|"
    r"basketball|cricket|stock market|crypto|bitcoin|ethereum|politics|election|"
    r"celebrity|instagram|tiktok|netflix|gaming|game|restaurant|hotel|travel|"
    r"flight|vacation|holiday|health|medical|doctor|medicine|drug|legal advice|"
    r"lawyer|law firm|investment advice|financial advice|tax advice|"
    r"best programming language to learn|what programming language|learn in 20\d\d)\b",
    re.IGNORECASE,
)

_HR_ADVICE_PATTERNS = re.compile(
    r"\b(should i fire|can i fire|terminate|lay off|layoff|discipline|"
    r"performance improvement plan|pip|employment law|legal risk|sue|lawsuit|"
    r"compensation advice|salary advice)\b",
    re.IGNORECASE,
)


def is_in_scope(messages: list) -> bool:
    """
    Check if the conversation history is within scope (HR assessment/hiring).
    Strategy: refuse ONLY messages that clearly match hard off-topic patterns
    AND do not have any HR scope signals at all.

    This is intentionally permissive — false-positive refusals hurt UX more
    than false-negatives (the LLM will handle off-topic gracefully anyway).
    """
    if not messages:
        return True

    last_user_msg = next((m.content for m in reversed(messages) if m.role == "user"), "")
    if not last_user_msg:
        return True

    words = last_user_msg.split()

    if _HR_ADVICE_PATTERNS.search(last_user_msg):
        return False

    # Very short turns are always in-scope (acknowledgements, clarifications)
    if len(words) <= 6:
        return True

    # Clear non-assessment questions should be refused even if they contain words
    # like "programming" that overlap with technical assessment vocabulary.
    if _OFF_TOPIC_PATTERNS.search(last_user_msg):
        return False

    # If there are any HR/assessment scope signals, it's in-scope
    if _SCOPE_KEYWORDS.search(last_user_msg):
        return True

    # Default: allow through (let LLM handle ambiguous cases)
    return True


def is_injection_attempt(messages: list) -> bool:
    """
    Detect if the user is attempting a prompt injection/jailbreak
    by examining the latest user message.
    """
    if not messages:
        return False

    last_user_msg = next((m.content for m in reversed(messages) if m.role == "user"), "")
    return bool(_INJECTION_PATTERNS.search(last_user_msg))


def validate_recommendations(candidate_list: list[dict], catalog: list[dict]) -> list[dict]:
    """
    Programmatically validates every (name, url) pair in the candidate list
    against the catalog.json.
    - If URL is not in catalog: try normalizing the URL path first, then drop if still not found.
    - If URL matches but name is slightly different, resolve it to the canonical catalog name.
    """
    validated = []
    catalog_by_url = {item["url"]: item for item in catalog}

    # Build a slug-to-item lookup for fuzzy matching on URL path
    def _get_slug(url: str) -> str:
        """Extract the last meaningful path segment from URL."""
        url = url.rstrip("/")
        return url.split("/")[-1].lower()

    slug_to_item = {}
    for item in catalog:
        slug = _get_slug(item["url"])
        if slug:
            slug_to_item[slug] = item

    for idx, item in enumerate(candidate_list, 1):
        url = item.get("url", "").strip()
        name = item.get("name", "").strip()

        # Direct lookup first
        if url in catalog_by_url:
            canonical_item = catalog_by_url[url]
        else:
            # Try slug-based fallback (handles /products/ vs /solutions/products/ mismatch)
            slug = _get_slug(url)
            canonical_item = slug_to_item.get(slug)
            if not canonical_item:
                logger.warning(
                    "Dropping recommendation: URL '%s' (name: '%s') not found in catalog",
                    url, name
                )
                continue

        validated_item = {
            "rank": idx,
            "name": canonical_item["name"],
            "url": canonical_item["url"],
            "test_type": canonical_item.get("test_type", []),
            "duration": canonical_item.get("duration"),
            "languages": canonical_item.get("languages", [])
        }
        validated.append(validated_item)

    return validated
