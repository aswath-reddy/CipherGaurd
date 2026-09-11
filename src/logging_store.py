"""
Audit Logging Store: Records every routing decision, risk score, and attribution result.
Provides auditable trail for regulatory compliance.
Matches Protocol §2 and §5.
"""

import json
import sqlite3
import time
import os
from typing import Dict, Any, List, Optional


class AuditLogger:
    """
    SQLite-backed audit log for CipherGuard decisions and explanations.
    """

    def __init__(self, db_path: str = "cipherguard_audit.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    surface TEXT,
                    input_text TEXT,
                    risk_scores_json TEXT,
                    action TEXT,
                    rationale TEXT,
                    is_flipped INTEGER,
                    removed_tokens_json TEXT,
                    cardinality INTEGER,
                    delta_score REAL,
                    sanitized_text TEXT,
                    latency_ms REAL
                )
            """)
            conn.commit()

    def log_decision(
        self,
        surface: str,
        input_text: str,
        risk_scores: Dict[str, float],
        action: str,
        rationale: str,
        explanation: Optional[Dict[str, Any]] = None,
        latency_ms: float = 0.0
    ) -> int:
        """
        Inserts a single decision record. Returns generated record ID.
        """
        is_flipped = 0
        removed_tokens_json = "[]"
        cardinality = 0
        delta_score = 0.0
        sanitized_text = input_text

        if explanation:
            is_flipped = 1 if explanation.get("is_flipped") else 0
            removed_tokens_json = json.dumps(explanation.get("removed_tokens", []))
            cardinality = explanation.get("cardinality", 0)
            delta_score = explanation.get("delta_score", 0.0)
            sanitized_text = explanation.get("sanitized_text", input_text)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO audit_logs (
                    timestamp, surface, input_text, risk_scores_json,
                    action, rationale, is_flipped, removed_tokens_json,
                    cardinality, delta_score, sanitized_text, latency_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                time.time(),
                surface,
                input_text,
                json.dumps(risk_scores),
                action,
                rationale,
                is_flipped,
                removed_tokens_json,
                cardinality,
                delta_score,
                sanitized_text,
                latency_ms
            ))
            conn.commit()
            return cursor.lastrowid

    def get_recent_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM audit_logs ORDER BY id DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            results = []
            for r in rows:
                item = dict(r)
                item["risk_scores"] = json.loads(item["risk_scores_json"])
                item["removed_tokens"] = json.loads(item["removed_tokens_json"])
                results.append(item)
            return results
