"""
Domain → Event mapping utilities.

Maps 64 pre-classified domains to one of the core events:
- escalation
- churn
- refund
- compliance
- general (fallback)

Uses configurable keyword-based heuristics by default. You can supply an
explicit mapping dict via Config or extend this module to read a mapping file.
"""

import logging
import re
from typing import Optional

from .config import Config

logger = logging.getLogger(__name__)


class DomainEventMapper:
    """Maps domain labels to event types using heuristics or config overrides."""

    def __init__(self, override_map: Optional[dict[str, str]] = None):
        # Normalize keys to lowercase for robust matching
        self.override_map = {k.lower(): v.lower() for k, v in (override_map or {}).items()}

        # Heuristic keyword buckets; extend as needed
        self.heu = {
            "escalation": [
                r"escala", r"supervisor", r"manager", r"complaint", r"angry", r"frustrat",
            ],
            "churn": [
                r"cancel", r"switch", r"competitor", r"retention", r"promotion", r"pricing",
                r"loyalty", r"winback", r"renewal", r"downgrade",
            ],
            "refund": [
                r"refund", r"chargeback", r"billing", r"payment", r"invoice", r"return",
            ],
            "compliance": [
                r"compliance", r"privacy", r"legal", r"gdpr", r"hipaa", r"policy", r"kyc",
                r"verification", r"consent",
            ],
        }

    def map(self, domain: str) -> str:
        """Return event type for the given domain string."""
        if not domain:
            return "general"

        d = str(domain).strip().lower()

        # Exact override mapping wins
        if d in self.override_map:
            return self._sanitize(self.override_map[d])

        # Heuristic regex buckets
        for event, patterns in self.heu.items():
            for pat in patterns:
                if re.search(pat, d):
                    return event

        # Default fallback
        return "general"

    def _sanitize(self, label: str) -> str:
        lab = label.lower().strip()
        if lab in {"escalation", "churn", "refund", "compliance"}:
            return lab
        return "general"
