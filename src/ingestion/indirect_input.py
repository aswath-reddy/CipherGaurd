"""
Indirect Input Adapter for CipherGuard
Handles untrusted external content (RAG retrieval chunks, web crawls, tool outputs, emails).
"""

from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class IndirectSourceType(str, Enum):
    RETRIEVAL_CHUNK = "retrieval_chunk"
    WEB_PAGE = "web_page"
    EMAIL_BODY = "email_body"
    TOOL_OUTPUT = "tool_output"
    API_RESPONSE = "api_response"
    FILE_CONTENT = "file_content"


class IndirectInput(BaseModel):
    """
    Represents external content ingested by an LLM agent as passive data.
    Surface: 'indirect'
    """
    content: str = Field(..., description="External data or tool return to be inspected.")
    source_type: IndirectSourceType = Field(
        default=IndirectSourceType.RETRIEVAL_CHUNK,
        description="Type of external source."
    )
    source_id: Optional[str] = Field(default=None, description="Identifier of external document/tool.")
    parent_tool: Optional[str] = Field(default=None, description="Tool name that returned this content.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata associated with retrieval.")

    @property
    def surface(self) -> str:
        return "indirect"

    def get_eval_text(self) -> str:
        return self.content.strip()
