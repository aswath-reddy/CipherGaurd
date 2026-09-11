"""
Pydantic Policy Schema for Dynamic Policy Matrix.
Matches Protocol §6.3.
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class RoutingAction(str, Enum):
    BLOCK = "BLOCK"
    REVIEW = "REVIEW"
    ALLOW = "ALLOW"


class PolicyRule(BaseModel):
    category: str = Field(..., description="Target risk category (e.g. 'Jailbreak', 'Prompt Injection').")
    surface: str = Field(default="*", description="Applicable surface ('direct', 'indirect', or '*' for all).")
    threshold: float = Field(..., ge=0.0, le=1.0, description="Risk threshold in [0.0, 1.0].")
    action: RoutingAction = Field(..., description="Routing action to trigger if score >= threshold.")

    @field_validator("surface")
    @classmethod
    def validate_surface(cls, v: str) -> str:
        v_clean = v.strip().lower()
        if v_clean not in ("direct", "indirect", "*"):
            raise ValueError("surface must be 'direct', 'indirect', or '*'")
        return v_clean


class PolicyConfig(BaseModel):
    version: str = Field(default="1.0", description="Policy version tag.")
    policies: List[PolicyRule] = Field(..., description="List of policy evaluation rules.")
    default_action: RoutingAction = Field(default=RoutingAction.ALLOW, description="Default action if no threshold is breached.")
