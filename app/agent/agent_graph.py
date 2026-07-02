from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from app.config import get_settings
from app.models import ChatResponse, Message, Recommendation
from app.agent.prompts import (
    CLARIFY_PROMPT,
    RECOMMEND_PROMPT,
    COMPARE_PROMPT,
    OFF_TOPIC_REPLY,
    INJECTION_REPLY,
)
from app.agent.guardrails import (
    is_injection_attempt,
    is_in_scope,
    validate_recommendations,
)

logger = logging.getLogger(__name__)
settings = get_settings()

class AgentState(TypedDict):
    messages: list[Message]
    facts: dict[str, Any]
    intent: Literal["clarify_needed", "ready_to_recommend", "refine", "compare", "off_topic", "injection", "closing"]
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool


# ── Structured Output Schema ──────────────────────────────────────────────────
class AnalysisSchema(BaseModel):
    role_or_skill: Optional[str] = Field(None, description="The job role or primary skill candidate is being evaluated for, e.g., 'Java developer'.")
    seniority: Optional[str] = Field(None, description="The required seniority level of the role, e.g., 'senior', 'junior', 'graduate', 'entry-level'.")
    context: Optional[str] = Field(None, description="The purpose of assessment. Must be exactly 'selection' (hiring/comparing candidates) or 'development' (feedback/growth for existing staff). Must be null/None if not explicitly mentioned by the user yet.")
    tools_tech: list[str] = Field(default_factory=list, description="List of technical tools, libraries, frameworks, or databases requested, e.g. ['Spring', 'AWS'].")
    behavioral_needs: list[str] = Field(default_factory=list, description="List of behavioral, cognitive, or personality traits requested, e.g. ['personality', 'numerical reasoning'].")
    remote_testing: Optional[bool] = Field(None, description="Whether remote testing or online monitoring is explicitly requested (True/False).")
    adaptive_irt: Optional[bool] = Field(None, description="Whether adaptive or IRT tests are explicitly requested (True/False).")
    languages: list[str] = Field(default_factory=list, description="List of languages mentioned or requested for the test, e.g. ['English'].")
    explicit_request_for_shortlist: bool = Field(False, description="True if the user explicitly demands a shortlist, recommendations, or a list of assessments.")
    comparison_targets: list[str] = Field(default_factory=list, description="Assessment names or types requested to compare, e.g. ['OPQ32r', 'GSA'].")
    intent: Literal["clarify_needed", "ready_to_recommend", "refine", "compare"] = Field(
        ...,
        description=(
            "Classified intent based on the user's latest message and conversation. "
            "Use 'clarify_needed' if there is not enough information to recommend (missing role, seniority, or context). "
            "Use 'ready_to_recommend' ONLY if role + seniority + context (selection vs development) are ALL known. "
            "Use 'refine' if user is modifying a previous recommendation. "
            "Use 'compare' if comparing tests."
        )
    )


# ── Helper functions ──────────────────────────────────────────────────────────
def _format_conversation(messages: list[Message]) -> str:
    return "\n".join(f"{m.role.upper()}: {m.content}" for m in messages)


def _get_last_user_message(messages: list[Message]) -> str:
    for m in reversed(messages):
        if m.role == "user":
            return m.content
    return ""


def _is_closing(text: str) -> bool:
    closing_words = {"thanks", "thank you", "that works", "perfect", "great", "no further", "bye", "goodbye", "resolved"}
    normalized = text.lower().strip().rstrip(".,!")
    return any(phrase in normalized for phrase in closing_words) and len(text.split()) <= 12


def _is_ready_to_recommend(facts: dict) -> bool:
    has_role = bool(facts.get("role_or_skill"))
    has_qualifier = bool(
        facts.get("seniority")
        or facts.get("explicit_request_for_shortlist")
        or facts.get("tools_tech")
        or facts.get("behavioral_needs")
        or facts.get("remote_testing") is not None
        or facts.get("adaptive_irt") is not None
    )
    has_context = bool(facts.get("context"))  # selection vs development
    return has_role and has_qualifier and has_context


# ── Node: analyze_node ────────────────────────────────────────────────────────
async def analyze_node(state: AgentState) -> dict:
    messages = state["messages"]
    last_user = _get_last_user_message(messages)

    # 1. Fast Regex Guardrails
    if is_injection_attempt(messages):
        logger.warning("Injection attempt detected via regex")
        return {
            "intent": "injection",
            "reply": INJECTION_REPLY,
            "end_of_conversation": False,
            "recommendations": []
        }

    if not is_in_scope(messages):
        logger.warning("Off-topic request detected via regex")
        return {
            "intent": "off_topic",
            "reply": OFF_TOPIC_REPLY,
            "end_of_conversation": False,
            "recommendations": []
        }

    if _is_closing(last_user):
        logger.info("Closing turn detected")
        return {
            "intent": "closing",
            "reply": "You're welcome! Let me know if you need to set up any other assessment profiles.",
            "end_of_conversation": True,
            "recommendations": []
        }

    # 2. Unified LLM Structured Call
    ANALYSIS_SYSTEM = (
        "You are an expert SHL assessment consultant. "
        "Your job is to analyze a conversation about hiring needs and extract structured facts. "
        "SHL assessments include personality questionnaires (OPQ), cognitive ability tests (Verify), "
        "knowledge tests, simulations, and development reports. "
        "Identify whether the need is for SELECTION (hiring decision) or DEVELOPMENT (coaching/growth/feedback). "
        "Do NOT guess or assume the context; leave context as null/None if the user hasn't explicitly mentioned selection vs development. "
        "Do NOT classify intent as ready_to_recommend unless role, seniority, AND context (selection/development) are all known."
    )
    client = ChatOpenAI(
        model=settings.llm_model,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base=settings.openrouter_base_url,
        temperature=0.0
    )
    structured_llm = client.with_structured_output(AnalysisSchema)

    conv_text = _format_conversation(messages)
    try:
        analysis: AnalysisSchema = await structured_llm.ainvoke(
            [
                {"role": "system", "content": ANALYSIS_SYSTEM},
                {"role": "user", "content": conv_text}
            ]
        )
    except Exception as exc:
        logger.error(f"Unified LLM analysis failed: {exc}")
        # Fail-safe fallback to clarify
        return {
            "intent": "clarify_needed",
            "facts": {},
            "reply": "Could you clarify the seniority level or any specific tech stack constraints for this role?",
            "end_of_conversation": False,
            "recommendations": []
        }

    facts = {
        "role_or_skill": analysis.role_or_skill,
        "seniority": analysis.seniority,
        "context": analysis.context,
        "tools_tech": analysis.tools_tech,
        "behavioral_needs": analysis.behavioral_needs,
        "constraints": {
            "remote_testing": analysis.remote_testing,
            "adaptive_irt": analysis.adaptive_irt,
            "languages": analysis.languages
        },
        "explicit_request_for_shortlist": analysis.explicit_request_for_shortlist,
        "comparison_targets": analysis.comparison_targets
    }

    intent = analysis.intent

    # Enforce turn cap and guard against premature recommendations
    user_turn_count = len([m for m in messages if m.role == "user"])

    # Always ask at least one clarifying question on turn 1 unless user explicitly asked for shortlist
    if user_turn_count <= 1 and intent == "ready_to_recommend" and not facts.get("explicit_request_for_shortlist"):
        logger.info("Turn 1: forcing clarify_needed to gather more context before recommending")
        intent = "clarify_needed"

    # For leadership/senior/executive: require context (selection vs development) before recommending
    if intent == "ready_to_recommend":
        has_role = bool(facts.get("role_or_skill"))
        has_details = bool(
            facts.get("seniority")
            or facts.get("tools_tech")
            or facts.get("behavioral_needs")
            or facts.get("explicit_request_for_shortlist")
            or (facts.get("constraints") and facts["constraints"].get("remote_testing") is not None)
            or (facts.get("constraints") and facts["constraints"].get("adaptive_irt") is not None)
        )
        has_context = bool(facts.get("context"))
        if not has_role or not has_details:
            logger.info("Overriding intent to clarify_needed because details are too vague")
            intent = "clarify_needed"
        elif not has_context and user_turn_count <= 2:
            logger.info("Overriding intent to clarify_needed: need to know selection vs development context")
            intent = "clarify_needed"

    if user_turn_count >= 4 and intent == "clarify_needed":
        logger.info(f"Turn cap reached ({user_turn_count} user turns). Forcing recommendation intent.")
        intent = "ready_to_recommend"

    return {
        "facts": facts,
        "intent": intent
    }


# ── Node: clarify_node ────────────────────────────────────────────────────────
async def clarify_node(state: AgentState) -> dict:
    client = ChatOpenAI(
        model=settings.llm_model,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base=settings.openrouter_base_url,
        temperature=0.2
    )

    facts = state.get("facts", {})
    messages = state["messages"]
    conversation = _format_conversation(messages)

    prompt = CLARIFY_PROMPT.format(
        facts_json=json.dumps(facts, indent=2),
        conversation=conversation
    )

    response = await client.ainvoke([{"role": "user", "content": prompt}])
    reply = response.content.strip()

    return {
        "reply": reply,
        "recommendations": [],
        "end_of_conversation": False
    }


# ── Node: compare_node ────────────────────────────────────────────────────────
async def compare_node(state: AgentState) -> dict:
    from app.retrieval.retriever import search_by_names
    client = ChatOpenAI(
        model=settings.llm_model,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base=settings.openrouter_base_url,
        temperature=0.2
    )

    facts = state.get("facts", {})
    last_user = _get_last_user_message(state["messages"])
    targets = facts.get("comparison_targets", [])

    if not targets:
        # Simple extraction fallback
        targets = re.findall(r"between\s+([\w\s+-]+)\s+and\s+([\w\s+-]+)", last_user, re.IGNORECASE)
        if targets:
            targets = list(targets[0])
        else:
            targets = [last_user]

    items = await search_by_names(targets)
    if not items:
        return {
            "reply": "I could not confidently match those assessment names in the SHL catalog. Please share the exact assessment names or URLs you want to compare.",
            "recommendations": [],
            "end_of_conversation": False
        }

    items_json = json.dumps([dict(it) for it in items], indent=2)
    prompt = COMPARE_PROMPT.format(items_json=items_json, user_question=last_user)

    response = await client.ainvoke([{"role": "user", "content": prompt}])
    reply = response.content.strip()

    return {
        "reply": reply,
        "recommendations": [],
        "end_of_conversation": False
    }


# ── Node: recommend_node ──────────────────────────────────────────────────────
async def recommend_node(state: AgentState) -> dict:
    from app.retrieval.retriever import search, get_catalog_lookup
    from app.retrieval.index import get_catalog
    from app.agent.orchestrator import _offline_candidates, _merge_catalog_candidates, _extract_recommendations_json, _clean_reply

    client = ChatOpenAI(
        model=settings.llm_model,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base=settings.openrouter_base_url,
        temperature=0.2
    )

    facts = state.get("facts", {})
    messages = state["messages"]

    # ── Build semantically rich retrieval query ───────────────────────────────
    seniority = facts.get("seniority") or ""
    context = facts.get("context") or ""  # e.g. "selection" or "development"
    role = facts.get("role_or_skill") or ""
    behavioral = facts.get("behavioral_needs") or []
    tech = facts.get("tools_tech") or []

    query_parts = []
    if seniority:
        query_parts.append(seniority)
    if role:
        query_parts.append(role)
    if tech:
        query_parts.extend(tech)
    if behavioral:
        query_parts.extend(behavioral)
    if context:
        query_parts.append(context)
    query = " ".join(query_parts) if query_parts else "assessment"

    # For very generic "leadership" queries explicitly add personality/behavior to steer embeddings
    if any(w in role.lower() for w in ("leadership", "leader", "executive", "director", "cxo", "ceo", "cfo", "cto")):
        query = f"personality behavior assessment {query}"

    # ── Metadata filters ──────────────────────────────────────────────────────
    filters = {}
    constraints = facts.get("constraints") or {}
    if constraints.get("remote_testing") is not None:
        filters["remote_testing"] = constraints["remote_testing"]
    if constraints.get("adaptive_irt") is not None:
        filters["adaptive_irt"] = constraints["adaptive_irt"]

    # ── Retrieval ─────────────────────────────────────────────────────────────
    candidates = await search(query, k=settings.top_k, filters=filters)
    conversation_text = " ".join(m.content for m in messages).lower()
    catalog = get_catalog()
    supplemental = _offline_candidates(catalog, conversation_text, limit=settings.top_k)
    candidates = _merge_catalog_candidates(candidates, supplemental, limit=settings.top_k)

    # ── Post-retrieval job-level filter for senior/executive roles ────────────
    senior_keywords = ("senior", "director", "executive", "cxo", "ceo", "cfo", "cto", "vp", "c-suite", "c-level")
    if any(kw in conversation_text for kw in senior_keywords):
        senior_levels = {"Director", "Executive", "Manager", "Mid-Professional", "Professional Individual Contributor"}
        senior_candidates = [c for c in candidates if set(c.job_levels) & senior_levels]
        # Only apply filter if it doesn't eliminate all candidates
        if len(senior_candidates) >= 3:
            candidates = senior_candidates
            logger.info("Applied senior/executive job-level filter: %d candidates remain", len(candidates))

    catalog_items_json = json.dumps([dict(c) for c in candidates], indent=2)
    prompt = RECOMMEND_PROMPT.format(
        catalog_items_json=catalog_items_json,
        facts_json=json.dumps(facts, indent=2)
    )

    response = await client.ainvoke([{"role": "user", "content": prompt}])
    raw_reply = response.content.strip()

    raw_recs = _extract_recommendations_json(raw_reply)
    validated_recs = validate_recommendations(raw_recs, catalog)

    catalog_lookup = get_catalog_lookup()
    final_recs = []
    for rec in validated_recs:
        url = rec["url"]
        cat_item = catalog_lookup.get(url, {})
        test_type = cat_item.get("test_type", rec.get("test_type", []))
        test_type_str = ",".join(test_type) if isinstance(test_type, list) else str(test_type)

        final_recs.append(Recommendation(
            name=rec["name"],
            url=url,
            test_type=test_type_str
        ))

    final_recs = final_recs[:10]
    end_of_conv = state["intent"] != "refine"

    return {
        "reply": _clean_reply(raw_reply),
        "recommendations": final_recs,
        "end_of_conversation": end_of_conv
    }


# ── Edge Routing function ─────────────────────────────────────────────────────
def route_after_analyze(state: AgentState) -> str:
    intent = state["intent"]
    if intent in ("injection", "off_topic", "closing"):
        return END
    elif intent == "clarify_needed":
        return "clarify"
    elif intent == "compare":
        return "compare"
    else:
        return "recommend"


# ── Build & Compile the Graph ─────────────────────────────────────────────────
def get_agent_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("analyze", analyze_node)
    workflow.add_node("clarify", clarify_node)
    workflow.add_node("compare", compare_node)
    workflow.add_node("recommend", recommend_node)

    workflow.set_entry_point("analyze")

    workflow.add_conditional_edges(
        "analyze",
        route_after_analyze,
        {
            END: END,
            "clarify": "clarify",
            "compare": "compare",
            "recommend": "recommend"
        }
    )

    workflow.add_edge("clarify", END)
    workflow.add_edge("compare", END)
    workflow.add_edge("recommend", END)

    return workflow.compile()
