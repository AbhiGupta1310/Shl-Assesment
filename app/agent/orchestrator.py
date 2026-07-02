"""
Turn-level orchestrator: decides what to do each conversational turn.
Implements Phase 6 & 7 logic: turn caps, guardrail checks, branch routing,
closed-set recommendation validation, and schema-valid ChatResponse outputs.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Literal

from openai import AsyncOpenAI

from app.config import get_settings
from app.models import ChatResponse, Message, Recommendation
from app.agent.guardrails import is_injection_attempt, is_in_scope, validate_recommendations
from app.agent.prompts import (
    CLARIFY_PROMPT,
    RECOMMEND_PROMPT,
    COMPARE_PROMPT,
    REFUSAL_PROMPT,
    OFF_TOPIC_REPLY,
    INJECTION_REPLY,
)

logger = logging.getLogger(__name__)
settings = get_settings()


def _get_client() -> AsyncOpenAI | None:
    if not settings.use_llm or not settings.openrouter_api_key:
        return None
    return AsyncOpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
    )


def _format_conversation(messages: list[Message]) -> str:
    return "\n".join(f"{m.role.upper()}: {m.content}" for m in messages)


def _get_last_user_message(messages: list[Message]) -> str:
    for m in reversed(messages):
        if m.role == "user":
            return m.content
    return ""


def _extract_recommendations_json(reply_text: str) -> list[dict]:
    """Extract recommendations_json array from LLM response."""
    pattern = r"```recommendations_json\s*(\[.*?\])\s*```"
    match = re.search(pattern, reply_text, re.DOTALL)
    if not match:
        try:
            cleaned = reply_text.strip()
            if cleaned.startswith("[") and cleaned.endswith("]"):
                return json.loads(cleaned)
        except Exception:
            pass
        return []
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return []


def _clean_reply(reply_text: str) -> str:
    """Remove recommendations_json block from response text."""
    return re.sub(r"```recommendations_json.*?```", "", reply_text, flags=re.DOTALL).strip()


def _is_closing(text: str) -> bool:
    """Detect if the message is a closing turn."""
    closing_words = {"thanks", "thank you", "that works", "perfect", "great", "no further", "bye", "goodbye", "resolved"}
    normalized = text.lower().strip().rstrip(".,!")
    return any(phrase in normalized for phrase in closing_words) and len(text.split()) <= 12


# ── Offline fallback mock implementation when API key is missing ──────────────
_TECH_TERMS = {
    "java": ["java", "core java", "java 8", "java frameworks", "java web services"],
    "javascript": ["javascript"],
    "python": ["python"],
    "sql": ["sql", "database"],
    "aws": ["amazon web services", "aws"],
    "docker": ["docker"],
    "kubernetes": ["kubernetes"],
    "react": ["react"],
    "angular": ["angular"],
    "spring": ["spring"],
    "net": [".net", "asp.net", "c#"],
    "c#": ["c#", ".net"],
    "c++": ["c++"],
    "linux": ["linux"],
    "excel": ["excel"],
}

_ROLE_TERMS = {
    "developer": ["software", "programming", "coding", "development", "automata"],
    "engineer": ["engineering", "technical", "software", "development"],
    "manager": ["management", "manager", "leadership", "scenarios"],
    "analyst": ["analyst", "analysis", "data", "excel", "sql"],
    "sales": ["sales"],
    "customer service": ["customer service", "customer"],
    "support": ["support", "customer service"],
    "graduate": ["graduate", "entry"],
}

_PERSONALITY_HINTS = ("personality", "behavior", "behaviour", "trait", "opq", "leadership")
_ABILITY_HINTS = ("cognitive", "reasoning", "aptitude", "ability", "numerical", "deductive", "inductive")
_SIMULATION_HINTS = ("simulation", "simulate", "coding", "hands-on", "practical")


def _catalog_text(item: dict) -> str:
    parts = [
        item.get("name", ""),
        item.get("description", ""),
        " ".join(item.get("test_type", []) or []),
        " ".join(item.get("job_levels", []) or []),
        " ".join(item.get("languages", []) or []),
    ]
    return " ".join(parts).lower()


def _contains_any(text: str, needles: list[str] | tuple[str, ...]) -> bool:
    return any(n.lower() in text for n in needles)


def _score_catalog_item(item: dict, conversation_text: str) -> int:
    text = _catalog_text(item)
    name = item.get("name", "").lower()
    test_types = set(item.get("test_type") or [])
    score = 0

    for term, aliases in _TECH_TERMS.items():
        if term in conversation_text:
            for alias in aliases:
                if alias in text:
                    score += 45 if alias in name else 25

    if "java" in conversation_text and "javascript" not in conversation_text and "javascript" in name:
        score -= 80

    for term, aliases in _ROLE_TERMS.items():
        if term in conversation_text:
            for alias in aliases:
                if alias in text:
                    score += 12

    if _contains_any(conversation_text, _PERSONALITY_HINTS):
        if "P" in test_types:
            score += 40
        if "opq" in name or "personality" in name:
            score += 30

    if _contains_any(conversation_text, _ABILITY_HINTS):
        if "A" in test_types:
            score += 40
        if _contains_any(name, ("verify", "reasoning", "g+")):
            score += 20

    if _contains_any(conversation_text, _SIMULATION_HINTS):
        if "S" in test_types:
            score += 25
        if "automata" in name:
            score += 25

    if "senior" in conversation_text and _contains_any(" ".join(item.get("job_levels", [])).lower(), ("professional", "manager", "director", "executive")):
        score += 8
    if _contains_any(conversation_text, ("entry", "junior", "graduate")) and _contains_any(" ".join(item.get("job_levels", [])).lower(), ("entry", "graduate")):
        score += 8
    if "short" in conversation_text and re.search(r"\b([1-9]|1\d|20) minutes\b", str(item.get("duration", "")).lower()):
        score += 6

    if "report" in name and "report" not in conversation_text:
        score -= 35
    if not item.get("url"):
        score -= 100
    return score


def _offline_candidates(catalog: list[dict], conversation_text: str, limit: int = 10) -> list[dict]:
    scored = [(_score_catalog_item(item, conversation_text), item) for item in catalog]
    ranked = [item for score, item in sorted(scored, key=lambda row: row[0], reverse=True) if score > 0]

    if _contains_any(conversation_text, _PERSONALITY_HINTS):
        opq = next((item for item in catalog if item.get("name") == "Occupational Personality Questionnaire OPQ32r"), None)
        if opq and opq not in ranked:
            ranked.insert(0, opq)

    if _contains_any(conversation_text, _ABILITY_HINTS):
        verify = next((item for item in catalog if item.get("name") == "SHL Verify Interactive G+"), None)
        if verify and verify not in ranked:
            ranked.insert(0, verify)

    if not ranked:
        ranked = [
            item for item in catalog
            if item.get("name") in {
                "Global Skills Assessment",
                "Occupational Personality Questionnaire OPQ32r",
                "SHL Verify Interactive G+",
            }
        ]

    deduped = []
    seen_urls = set()
    for item in ranked:
        url = item.get("url")
        if url and url not in seen_urls:
            deduped.append(item)
            seen_urls.add(url)
        if len(deduped) >= limit:
            break
    return deduped


def _merge_catalog_candidates(primary: list, supplemental: list[dict], limit: int = 10) -> list:
    """Merge CatalogItem objects and raw catalog dicts, preserving URL uniqueness."""
    from app.retrieval.retriever import _dict_to_catalog_item

    merged = []
    seen_urls = set()
    for item in supplemental + list(primary):
        catalog_item = _dict_to_catalog_item(item) if isinstance(item, dict) else item
        if catalog_item.url and catalog_item.url not in seen_urls:
            merged.append(catalog_item)
            seen_urls.add(catalog_item.url)
        if len(merged) >= limit:
            break
    return merged


def _offline_extract_comparison_targets(last_user: str) -> list[str]:
    known_aliases = {
        "opq": "Occupational Personality Questionnaire OPQ32r",
        "opq32r": "Occupational Personality Questionnaire OPQ32r",
        "gsa": "Global Skills Assessment",
        "global skills assessment": "Global Skills Assessment",
        "verify g+": "SHL Verify Interactive G+",
        "verify interactive g+": "SHL Verify Interactive G+",
    }
    text = last_user.lower()
    targets = [canonical for alias, canonical in known_aliases.items() if alias in text]
    if targets:
        return list(dict.fromkeys(targets))
    pieces = re.split(r"\b(?:and|vs|versus|between|compare|difference)\b", last_user, flags=re.IGNORECASE)
    return [p.strip(" ?.,") for p in pieces if len(p.strip(" ?.,").split()) <= 6 and p.strip(" ?.,")]


async def _mock_handle_turn(messages: list[Message], last_user: str, user_turn_count: int) -> ChatResponse:
    """
    Simulates the pipeline when no OpenRouter API key is configured.
    Ensures tests and replays pass successfully without auth errors.
    """
    from app.retrieval.index import get_catalog
    catalog = get_catalog()

    conversation_text = " ".join(m.content for m in messages).lower()
    last_user_lower = last_user.lower()
    has_role_or_skill = any(term in conversation_text for term in (*_TECH_TERMS.keys(), *_ROLE_TERMS.keys()))
    has_qualifier = any(term in conversation_text for term in (
        "senior", "junior", "entry", "graduate", "mid", "lead", "manager", "5 years",
        "personality", "cognitive", "reasoning", "short", "remote", "recommend", "shortlist",
    ))

    if "compare" in last_user_lower or "difference between" in last_user_lower or "vs" in last_user_lower:
        intent = "compare"
    elif has_role_or_skill and (has_qualifier or user_turn_count >= 3):
        intent = "ready_to_recommend"
    elif has_role_or_skill:
        intent = "clarify_needed"
    else:
        intent = "clarify_needed"

    if user_turn_count >= 4 and intent == "clarify_needed":
        logger.info("Overriding intent to ready_to_recommend due to turn limit cap")
        intent = "ready_to_recommend"

    # 1. Compare branch
    if intent == "compare":
        from app.retrieval.retriever import search_by_names
        targets = _offline_extract_comparison_targets(last_user)
        items = await search_by_names(targets) if targets else []
        if not items:
            return ChatResponse(
                reply="I could not confidently match those assessment names in the SHL catalog. Please share the exact assessment names or URLs you want to compare.",
                recommendations=[],
                end_of_conversation=False,
            )
        lines = ["Here is a catalog-grounded comparison:"]
        for item in items:
            type_text = ",".join(item.test_type)
            duration = item.duration or "-"
            desc = item.description or "No catalog description available."
            lines.append(f"- {item.name}: type {type_text}; duration {duration}. {desc}")
        reply = "\n".join(lines)
        return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)

    # 2. Recommend/Refine branch
    if intent == "ready_to_recommend":
        matched = _offline_candidates(catalog, conversation_text, limit=10)
        recs = []
        for item in matched:
            recs.append(Recommendation(
                name=item["name"],
                url=item["url"],
                test_type=",".join(item["test_type"])
            ))

        # Filter out sales-specific items unless sales was mentioned
        if "sales" not in conversation_text:
            recs = [r for r in recs if "sales" not in r.name.lower()]

        reply = "Here is a catalog-grounded SHL assessment shortlist based on the role details so far:"
        return ChatResponse(reply=reply, recommendations=recs, end_of_conversation=False)

    # 3. Clarify branch
    reply = "Could you clarify the seniority level or any specific tech stack constraints for this role?"
    return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)


# ── Core Orchestrator handle_turn ─────────────────────────────────────────────
async def handle_turn(messages: list[Message]) -> ChatResponse:
    """
    Main orchestrator for turns. Executes the compiled LangGraph agent state machine.
    Accepts full messages list, checks for offline fallback first, and returns
    a structured ChatResponse.
    """
    # Truncate messages list if it exceeds 8 turns (16 messages total) to stay within constraints
    if len(messages) > 16:
        logger.info("Truncating message history from %d to 16 messages", len(messages))
        messages = messages[-16:]

    last_user = _get_last_user_message(messages)
    # Safely handle whitespace-only messages
    if not last_user or not last_user.strip():
        return ChatResponse(
            reply="It looks like you sent an empty message. Could you tell me what role you're hiring for?",
            recommendations=[],
            end_of_conversation=False
        )

    user_turns = [m for m in messages if m.role == "user"]
    user_turn_count = len(user_turns)

    # ── API Key Fallback check ────────────────────────────────────────────
    client = _get_client()
    if not client:
        logger.info("Using deterministic catalog engine")
        return await _mock_handle_turn(messages, last_user, user_turn_count)

    # ── LangGraph Agent Execution ─────────────────────────────────────────
    from app.agent.agent_graph import get_agent_graph

    agent = get_agent_graph()
    initial_state = {
        "messages": messages,
        "facts": {},
        "intent": "clarify_needed",
        "reply": "",
        "recommendations": [],
        "end_of_conversation": False
    }

    try:
        final_state = await agent.ainvoke(initial_state)
        return ChatResponse(
            reply=final_state["reply"],
            recommendations=final_state["recommendations"],
            end_of_conversation=final_state["end_of_conversation"]
        )
    except Exception as exc:
        logger.exception("Global exception handler caught during LangGraph execution: %s", exc)
        return ChatResponse(
            reply="I ran into an unexpected issue processing that request. Could we clarify the assessment skills you are looking for?",
            recommendations=[],
            end_of_conversation=False
        )


from tenacity import retry, stop_after_attempt, wait_random_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(min=1, max=10),
    reraise=True
)
async def _call_llm_with_retry(client, model, prompt):
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1500,
    )
    return response.choices[0].message.content.strip()


async def _call_llm(prompt: str) -> str:
    """Helper to perform standard chat completions via OpenRouter with retries."""
    client = _get_client()
    if not client:
        return "Offline stub reply"
    try:
        return await _call_llm_with_retry(client, settings.llm_model, prompt)
    except Exception as exc:
        logger.error("LLM call failed after retries: %s", exc)
        raise
