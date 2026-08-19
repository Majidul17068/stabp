"""
STABP core library — the Set-Theoretic Agent Boundary Protocol.

This module is the reusable implementation of the formalism in the paper:
  * Label / Field                     — the labeling  ell_A: A -> {SHARED, PRIVATE, OUT}
  * BoundaryMatrix.shareable(...)     — the shareable set  S(A->B)   (Eq. 1)
  * Middleware.enforce(...)           — filter(m, A->B) = m ∩ S(A->B) (Eq. 2)
  * redact_freetext(...)              — the free-text residual tier   (Eq. 3)

Non-interference (Proposition 1) holds by construction on the typed channel:
enforce() only ever emits fields whose label is SHARED, so no PRIVATE/OUT item
can reach a peer.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum


def load_env(path: str = ".env") -> None:
    """Minimal .env loader (no dependency). Reads KEY=VALUE lines from `path`
    and sets them in os.environ if not already set. Silently does nothing if
    the file is missing."""
    try:
        with open(path) as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key, val = key.strip(), val.strip().strip('"').strip("'")
                if key and val and key not in os.environ:
                    os.environ[key] = val
    except FileNotFoundError:
        pass


# --------------------------------------------------------------------------
# labeling:  ell_A : A -> {SHARED, PRIVATE, OUT}
# --------------------------------------------------------------------------
class Label(str, Enum):
    SHARED = "SHARED"    # A ∩ B  — the only legal channel
    PRIVATE = "PRIVATE"  # A \ B  — never leaves the agent
    OUT = "OUT"          # complement — out of scope


@dataclass(frozen=True)
class Field:
    name: str
    value: str
    label: Label


@dataclass
class TypedMessage:
    """An inter-agent message: a set of labeled fields (the typed channel) and
    optional free text (the leaky channel STABP falls back to scanning)."""
    sender: str
    fields: list[Field] = field(default_factory=list)
    text: str = ""


@dataclass
class EnforcementLog:
    """Audit record for one enforced delivery."""
    sender: str
    recipient: str
    cross_tenant: bool
    fields_delivered: list[str] = field(default_factory=list)
    fields_dropped: list[str] = field(default_factory=list)
    redactions: int = 0


# --------------------------------------------------------------------------
# boundary matrix: which agents exist, in which tenant
# --------------------------------------------------------------------------
class BoundaryMatrix:
    def __init__(self, tenants: dict[str, str]):
        """tenants: {agent_id -> tenant_id}."""
        self.tenants = dict(tenants)

    def same_tenant(self, a: str, b: str) -> bool:
        return self.tenants.get(a) == self.tenants.get(b)

    def shareable(self, msg: TypedMessage, recipient: str) -> list[Field]:
        """S(A->B): fields labeled SHARED that may cross to `recipient`.

        Same tenant: SHARED fields are shareable. Cross tenant: nothing is
        shareable by default (strict tenant isolation) — the single-tenant
        experiments never hit this branch, but it keeps the protocol total."""
        if not self.same_tenant(msg.sender, recipient):
            return []
        return [f for f in msg.fields if f.label == Label.SHARED]


# --------------------------------------------------------------------------
# free-text residual tier (Eq. 3) — the role spyv's checkers fill
# --------------------------------------------------------------------------
_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9-]{8,}"),       # api key
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),      # US SSN
    re.compile(r"\b(?:\d[ -]?){13,19}\b"),      # card number
]


def redact_freetext(text: str, known_secrets: list[str]) -> tuple[str, int]:
    """Redact known private values + regex-detectable secrets. Returns
    (redacted_text, n_redactions). Recall of this tier is the rho in Eq. 3."""
    out, n = text, 0
    for secret in known_secrets:
        if secret and secret in out:
            out = out.replace(secret, "[REDACTED]")
            n += 1
    for pat in _SECRET_PATTERNS:
        out, k = pat.subn("[REDACTED]", out)
        n += k
    return out, n


# --------------------------------------------------------------------------
# the middleware: a reference monitor between agents
# --------------------------------------------------------------------------
class Middleware:
    def __init__(self, matrix: BoundaryMatrix, redact_free_text: bool = True):
        self.matrix = matrix
        self.redact_free_text = redact_free_text
        # every agent's PRIVATE values, for scanning free text
        self._private_values: dict[str, list[str]] = {}

    def register_private(self, agent_id: str, values: list[str]) -> None:
        self._private_values[agent_id] = list(values)

    def enforce(self, msg: TypedMessage, recipient: str) -> tuple[TypedMessage, EnforcementLog]:
        """filter(m, sender->recipient) = m ∩ S(sender->recipient), plus
        free-text redaction. Returns the delivered message + an audit log."""
        cross = not self.matrix.same_tenant(msg.sender, recipient)
        allowed = self.matrix.shareable(msg, recipient)
        allowed_names = {f.name for f in allowed}
        dropped = [f.name for f in msg.fields if f.name not in allowed_names]

        # free-text tier: redact ALL agents' private values + regex hits
        text = msg.text
        redactions = 0
        if self.redact_free_text and text:
            all_secrets = [v for vals in self._private_values.values() for v in vals]
            text, redactions = redact_freetext(text, all_secrets)

        delivered = TypedMessage(sender=msg.sender, fields=allowed, text=text)
        log = EnforcementLog(
            sender=msg.sender, recipient=recipient, cross_tenant=cross,
            fields_delivered=list(allowed_names), fields_dropped=dropped,
            redactions=redactions,
        )
        return delivered, log


def rendered(msg: TypedMessage) -> str:
    """What a recipient actually sees: free text + surviving typed fields."""
    parts = [msg.text] + [f"{f.name}={f.value}" for f in msg.fields]
    return " ".join(p for p in parts if p)


__all__ = [
    "Label", "Field", "TypedMessage", "EnforcementLog",
    "BoundaryMatrix", "Middleware", "redact_freetext", "rendered", "load_env",
]
