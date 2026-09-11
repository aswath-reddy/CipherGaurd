"""
Policy Loader with Hot-Reload and Fail-Safe Schema Enforcement.
Maintains in-memory compiled policy and rejects malformed edits without downtime.
"""

import json
import os
import time
from typing import Optional, Tuple
from .schema import PolicyConfig


class PolicyLoader:
    """
    Manages loading, validating, and hot-reloading policy.json.
    Guarantees fail-safe behavior: if an invalid JSON or schema error is introduced,
    the loader rejects the reload and maintains the last-known-good policy.
    """

    def __init__(self, policy_path: str):
        self.policy_path = policy_path
        self._current_config: Optional[PolicyConfig] = None
        self._last_mtime: float = 0.0
        self._last_loaded_at: float = 0.0
        self.reload()

    @property
    def config(self) -> PolicyConfig:
        if self._current_config is None:
            raise RuntimeError("No valid policy loaded.")
        return self._current_config

    def reload(self) -> Tuple[bool, str]:
        """
        Reloads and validates policy from disk.
        Returns: (success: bool, message: str)
        """
        if not os.path.exists(self.policy_path):
            return False, f"Policy file not found: {self.policy_path}"

        try:
            with open(self.policy_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            # Validate against Pydantic schema
            new_config = PolicyConfig.model_validate(raw_data)
            self._current_config = new_config
            self._last_mtime = os.path.getmtime(self.policy_path)
            self._last_loaded_at = time.time()
            return True, f"Policy successfully loaded (v{new_config.version}, {len(new_config.policies)} rules)."
        except Exception as e:
            # Reject edit, fail safe (keep previous good config)
            err_msg = f"Failed to reload policy: {str(e)}. Retaining existing active policy."
            return False, err_msg

    def check_and_reload(self) -> bool:
        """
        Checks if the policy file has been modified on disk, and reloads if so.
        """
        if not os.path.exists(self.policy_path):
            return False
        mtime = os.path.getmtime(self.policy_path)
        if mtime > self._last_mtime:
            success, _ = self.reload()
            return success
        return False
