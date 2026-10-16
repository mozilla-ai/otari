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
# BODY is consumed here, and the rest are filled from render_email's arguments.
_LAYOUT_SLOT: Final = "{{BODY}}"
_RESERVED_VALUES: Final = frozenset({"BODY", "SUBJECT", "PREHEADER", "MARK", "FONT_FACES"})

# Files the dashboard serves from ``web/public``, by their path under its root.
# A test holds them to that directory, since a moved file would break every
# message without failing anything.
MARK_PATH: Final = "pwa/icon-192.png"
FONT_FILES: Final = (
    ("Mozilla Text", "fonts/MozillaText-Variable.woff2", "200 700"),
    ("Mozilla Headline", "fonts/MozillaHeadline-Variable.woff2", "200 700"),
    ("Fira Code", "fonts/FiraCode-VariableFont_wght.woff2", "300 700"),
)


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


def _preheader(text: str) -> str:
    """The line an inbox shows after the subject, hidden in the message itself.

    Padded so a client does not fill the rest of the preview with the first
    words of the card.
    """
    if not text:
        return ""
    padding = "&#847;&zwnj;&nbsp;" * 40
    return (
        '<div style="display:none; max-height:0; max-width:0; overflow:hidden; opacity:0; mso-hide:all;">'
        f"{escape(text)}{padding}</div>"
    )


def _mark(asset_root: str | None) -> str:
    if asset_root is None:
        return ""
    return (
        f'<img src="{escape(asset_root)}/{MARK_PATH}" width="24" height="24" alt="" '
        'style="display:inline-block; vertical-align:middle; border:0; outline:none; width:24px; height:24px;">'
        "&nbsp;&nbsp;"
    )


def _font_faces(asset_root: str | None) -> str:
    if asset_root is None:
        return ""
    return "".join(
        f"@font-face {{ font-family:'{family}'; src:url('{escape(asset_root)}/{path}') format('woff2'); "
        f"font-weight:{weights}; font-style:normal; }}"
        for family, path, weights in FONT_FILES
    )


def render_email(
    template: str,
    *,
    subject: str,
    values: dict[str, str],
    preheader: str = "",
    asset_root: str | None = None,
) -> MailMessage:
    """Render ``<template>.html`` and ``<template>.txt`` into one message.

    ``subject`` may carry placeholders too and is filled from the same values,
    unescaped (it is a header, not markup) and stripped of CR/LF, which is what
    keeps a free-text organization name in a subject line from being either a
    ``HeaderParseError`` at send time or, on a mail stack that tolerates it, a
    header-injection vector. ``preheader`` is filled from the same values and HTML-escaped.

    ``asset_root`` is where this deployment's dashboard serves its static
    files (``Mailer.asset_root``). The header's mark and the brand fonts load
    from it; without it the message goes out with neither rather than with a
    broken link.
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
    html_values["PREHEADER"] = _preheader(_fill(preheader, values, template=f"{template} preheader"))
    html_values["MARK"] = _mark(asset_root)
    html_values["FONT_FACES"] = _font_faces(asset_root)

    text_document = _load("_layout.txt").replace(_LAYOUT_SLOT, _load(f"{template}.txt").rstrip("\n"))
    text_values = dict(values)
    text_values["SUBJECT"] = rendered_subject

    return MailMessage(
        subject=rendered_subject,
        html=_fill(html_document, html_values, template=f"{template}.html"),
        text=_fill(text_document, text_values, template=f"{template}.txt"),
    )


__all__ = ["MailTemplateError", "render_email"]
