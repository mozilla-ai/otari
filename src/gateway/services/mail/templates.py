"""Renders an outgoing message from the static templates in ``gateway/templates/email/``.

Filled with a single regular-expression pass rather than a templating engine,
the same approach ``gateway.root_page`` takes for the tutorial page: the whole
of what needs rendering here is a shared layout plus one small body per message,
which is not enough to justify a dependency (see ``mail.transports`` for the
same choice about the send itself).

Three properties the one-pass substitution buys, each of which a sequential
``str.replace`` per value does not have:

* **A value cannot be re-substituted.** ``re.sub`` never rescans what it
  inserted, so an organization named ``{{ROLE}}`` (or, before the placeholders
  were delimited at all, one named "Role Corp") is emitted verbatim instead of
  being partly overwritten by a later pass.
* **A missing value is caught here, not in an inbox.** An unknown placeholder
  raises :class:`MailTemplateError` rather than shipping a literal
  ``{{ACCEPT_LINK}}`` to a recipient, so a template and its caller drifting
  apart fails in that message's test.
* **Escaping is applied per variant, once.** The HTML body escapes every value,
  because an organization name and a person's full name are free text; the
  plain-text body does not, since there is no markup there to break out of.
"""

import re
from functools import lru_cache
from html import escape
from importlib import resources
from typing import Final

from gateway.services.mail.message import MailMessage
from gateway.services.mail.transports import sanitize_header_value

_PLACEHOLDER: Final = re.compile(r"\{\{([A-Z0-9_]+)\}\}")

# Substituted into the layout before the value pass, so a body template is the
# only thing a message has to write. Not available to a caller as a value name:
# BODY is consumed here, and SUBJECT is supplied from the rendered subject.
_LAYOUT_SLOT: Final = "{{BODY}}"
_RESERVED_VALUES: Final = frozenset({"BODY", "SUBJECT"})


class MailTemplateError(Exception):
    """A template and the values it was rendered with do not agree."""


# A template name is a fixed identifier this codebase ships, never caller data.
# Enforced rather than documented because ``name`` is interpolated into a
# resource path and the result is cached forever: a name from outside would be
# both a traversal read and an unbounded cache.
_TEMPLATE_NAME = re.compile(r"\A_?[a-z][a-z0-9_]*\.(html|txt)\Z")


@lru_cache(maxsize=None)
def _load(name: str) -> str:
    """Read one template file out of the installed package.

    Cached because the set of templates is fixed at build time and a message is
    rendered per request. ``resources.files`` rather than a path relative to
    this module, so it also resolves inside a wheel or a zipimport.
    """
    if not _TEMPLATE_NAME.match(name):
        raise MailTemplateError(f"Not a valid email template name: {name!r}")
    try:
        return resources.files("gateway").joinpath(f"templates/email/{name}").read_text(encoding="utf-8")
    # KeyError is how a missing entry surfaces from a zipimported package
    # (zipfile.Path.read_text), where OSError is what a real filesystem raises;
    # a wheel installed as a zip would otherwise fail differently from a
    # checkout, which is the environment least likely to be tested.
    except (OSError, KeyError) as exc:
        raise MailTemplateError(f"Missing email template: {name}") from exc


def _fill(document: str, values: dict[str, str], *, template: str) -> str:
    def substitute(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in values:
            raise MailTemplateError(f"Template {template!r} uses {{{{{name}}}}}, which was not supplied")
        return values[name]

    return _PLACEHOLDER.sub(substitute, document)


def render_email(template: str, *, subject: str, values: dict[str, str]) -> MailMessage:
    """Render ``<template>.html`` and ``<template>.txt`` into one message.

    ``subject`` may carry placeholders too and is filled from the same values,
    unescaped (it is a header, not markup) and stripped of CR/LF, which is what
    keeps a free-text organization name in a subject line from being either a
    ``HeaderParseError`` at send time or, on a mail stack that tolerates it, a
    header-injection vector.
    """
    reserved = _RESERVED_VALUES & values.keys()
    if reserved:
        # A caller-supplied SUBJECT would be silently overwritten below, and a
        # caller-supplied BODY would never be substituted at all; both are a
        # mistake worth naming rather than a value worth honoring.
        raise MailTemplateError(f"Reserved template value(s): {sorted(reserved)}")

    rendered_subject = sanitize_header_value(_fill(subject, values, template=f"{template} subject"))

    # The body's own trailing newline is dropped, so a layout that puts a blank
    # line before its footer produces one blank line rather than two.
    html_document = _load("_layout.html").replace(_LAYOUT_SLOT, _load(f"{template}.html").rstrip("\n"))
    html_values = {name: escape(value) for name, value in values.items()}
    html_values["SUBJECT"] = escape(rendered_subject)

    text_document = _load("_layout.txt").replace(_LAYOUT_SLOT, _load(f"{template}.txt").rstrip("\n"))
    text_values = dict(values)
    text_values["SUBJECT"] = rendered_subject

    return MailMessage(
        subject=rendered_subject,
        html=_fill(html_document, html_values, template=f"{template}.html"),
        text=_fill(text_document, text_values, template=f"{template}.txt"),
    )


__all__ = ["MailTemplateError", "render_email"]
