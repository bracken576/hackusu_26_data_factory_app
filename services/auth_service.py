"""User identity resolution and role enforcement for Gradio + Databricks Apps."""
import os
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)
_ROLES_PATH = Path(__file__).parent.parent / "governance" / "roles.yaml"


def get_user_from_request(request) -> dict | None:
    """
    Resolve the current user from a gr.Request object (Databricks Apps / IAP).
    Falls back to DEV_USER_EMAIL env var for local dev.
    Returns: {"email": str, "role": str} or None
    """
    try:
        email = None
        if request and hasattr(request, "headers"):
            email = request.headers.get("X-Forwarded-Email")
        if not email:
            email = os.getenv("DEV_USER_EMAIL", "dev@local")
        if not email:
            return None
        return {"email": email, "role": resolve_role(email)}
    except Exception as exc:
        logger.warning("auth_service.get_user_from_request failed: %s", exc)
        return None


def resolve_role(email: str) -> str:
    """Look up role from governance/roles.yaml — never hardcode role assignments."""
    try:
        with open(_ROLES_PATH) as f:
            config = yaml.safe_load(f)
        user_map = config.get("users", {})
        return user_map.get(email, config.get("default_role", "viewer"))
    except Exception:
        return "viewer"


def require_role(user: dict | None, min_role: str) -> bool:
    """
    Return True if user's role is >= min_role.
    Hierarchy: admin > analyst > viewer
    """
    hierarchy = {"viewer": 0, "analyst": 1, "admin": 2}
    if user is None:
        return False
    user_level = hierarchy.get(user.get("role", "viewer"), 0)
    required_level = hierarchy.get(min_role, 0)
    return user_level >= required_level
