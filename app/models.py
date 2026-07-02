"""
Pydantic schemas mirroring the required API spec exactly.
"""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=1)

class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str

class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool

# Internal helper model for retrieval layer (not returned directly by endpoint)
class CatalogItem(BaseModel):
    name: str
    url: str
    test_type: list[str] = Field(default_factory=list)
    description: str = ""
    remote_testing: bool | None = None
    adaptive_irt: bool | None = None
    duration: str | None = None
    job_levels: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
