"""What a proposal may not carry into memory: credentials, and personal data.

Run on the proposal's text before anything is stored, sealed or not. A hit
refuses the proposal with the kind of what was found, never the match itself,
so the agent can rewrite it without the refusal repeating what it refused.
"""

import re
from typing import Literal

from gateway.secret_shapes import CREDENTIAL_SHAPES

ScreenFinding = Literal["credential", "email_address", "card_number", "private_key"]

_PRIVATE_KEY = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
# Thirteen to nineteen digits, with spaces or hyphens between groups, checked by Luhn below.
_CARD = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")
# Assignments whose value is a secret, whatever its shape: ``password=...``, ``"api_key": "..."``.
_SECRET_ASSIGNMENT = re.compile(
    r"\b(?:password|passwd|secret|api[_-]?key|access[_-]?token|client[_-]?secret)\b\s*[\"']?\s*[:=]\s*[\"']?[^\s\"']{6,}",
    re.IGNORECASE,
)


def screen(text: str) -> ScreenFinding | None:
    """The first kind of thing ``text`` may not carry into memory, or None when it carries none."""
    if _PRIVATE_KEY.search(text):
        return "private_key"
    if any(shape.search(text) for shape in CREDENTIAL_SHAPES) or _SECRET_ASSIGNMENT.search(text):
        return "credential"
    if _EMAIL.search(text):
        return "email_address"
    if any(_luhn(re.sub(r"[ -]", "", match.group())) for match in _CARD.finditer(text)):
        return "card_number"
    return None


def _luhn(digits: str) -> bool:
    total = 0
    for index, char in enumerate(reversed(digits)):
        digit = int(char)
        if index % 2:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0
