"""
FastAPI Gateway Application for CipherGuard.
Exposes /route, /policy, /policy/reload, /logs, and /explain endpoints.
Matches Protocol §4, §5.
"""

from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.router import CipherGuardRouter
from src.policy import RoutingAction

app = FastAPI(
    title="CipherGuard Safety Router API",
    description="A Policy-Aware Safety Router with Removal-Based Token Attribution for LLM Applications",
    version="1.0.0"
)

# Global router instance
router_instance = CipherGuardRouter(
    policy_path="policy/policy.json",
    db_path="cipherguard_audit.db"
)


class RouteRequest(BaseModel):
    text: str = Field(..., description="Prompt or external retrieved document to inspect.")
    surface: str = Field(default="direct", description="Input surface: 'direct' or 'indirect'.")
    beam_width: Optional[int] = Field(default=5, description="Beam search width for token attribution.")
    removal_cap: Optional[int] = Field(default=6, description="Max tokens allowed to be removed in explanation.")
    enable_attribution: bool = Field(default=True, description="Whether to run attribution if blocked.")


class PolicyReloadResponse(BaseModel):
    success: bool
    message: str
    active_version: str


@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "CipherGuard Safety Router",
        "policy_version": router_instance.policy_loader.config.version
    }


@app.post("/route")
def route_input(req: RouteRequest):
    """
    Evaluates input text through CipherGuard:
    1. Surface-specific risk classification.
    2. Dynamic policy resolution.
    3. Contrastive removal-based token attribution (if blocked).
    4. Regulatory audit logging.
    """
    try:
        res = router_instance.route(
            input_data=req.text,
            surface=req.surface,
            beam_width=req.beam_width,
            removal_cap=req.removal_cap,
            enable_attribution=req.enable_attribution
        )
        return res.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/policy")
def get_policy():
    """
    Returns the currently active dynamic policy matrix configuration.
    """
    return router_instance.policy_loader.config.model_dump()


@app.post("/policy/reload", response_model=PolicyReloadResponse)
def reload_policy():
    """
    Hot-reloads the policy matrix from disk without restarting the service.
    Fails safe if invalid JSON is provided.
    """
    success, msg = router_instance.policy_loader.reload()
    return PolicyReloadResponse(
        success=success,
        message=msg,
        active_version=router_instance.policy_loader.config.version
    )


@app.get("/logs")
def get_audit_logs(limit: int = 50):
    """
    Returns recent audit logs from SQLite store.
    """
    return router_instance.audit_logger.get_recent_logs(limit=limit)


@app.get("/explain/{audit_id}")
def get_explanation(audit_id: int):
    """
    Retrieves full contrastive explanation for an audit entry.
    """
    logs = router_instance.audit_logger.get_recent_logs(limit=200)
    for entry in logs:
        if entry["id"] == audit_id:
            return {
                "id": entry["id"],
                "input_text": entry["input_text"],
                "surface": entry["surface"],
                "action": entry["action"],
                "rationale": entry["rationale"],
                "is_flipped": bool(entry["is_flipped"]),
                "removed_tokens": entry["removed_tokens"],
                "cardinality": entry["cardinality"],
                "delta_score": entry["delta_score"],
                "sanitized_text": entry["sanitized_text"]
            }
    raise HTTPException(status_code=404, detail=f"Audit record {audit_id} not found.")
