"""
Direct Input Adapter for CipherGuard
Handles user-turn prompt ingestion and caller metadata.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class DirectInput(BaseModel):
    """
    Represents an input arriving directly from a human user in a conversational turn.
    Surface: 'direct'
    """
    text: str = Field(..., description="The user prompt text to be inspected.")
    session_id: Optional[str] = Field(default=None, description="Optional conversation session ID.")
    user_id: Optional[str] = Field(default=None, description="Optional user identifier.")
    user_role: str = Field(default="standard", description="Caller role (e.g. 'standard', 'admin', 'internal').")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context metadata.")

    @property
    def surface(self) -> str:
        return "direct"

    def get_eval_text(self) -> str:
        return self.text.strip()
