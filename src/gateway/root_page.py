"""Static assets for the gateway root tutorial page.

The page is served at the deployment root and links to the quickstart docs.
The markup lives in ``templates/root_tutorial.html`` and the favicon in
``templates/favicon.svg``; this module loads them once at import time and
injects the canonical quickstart URL.
"""

from importlib import resources

QUICKSTART_URL = "https://github.com/mozilla-ai/otari/blob/main/docs/quickstart.md"


def _load_root_tutorial_html() -> str:
    template = resources.files("gateway").joinpath("templates/root_tutorial.html").read_text(encoding="utf-8")
    return template.replace("QUICKSTART_URL", QUICKSTART_URL)


def _load_favicon_svg() -> str:
    return resources.files("gateway").joinpath("templates/favicon.svg").read_text(encoding="utf-8")


ROOT_TUTORIAL_HTML = _load_root_tutorial_html()
FAVICON_SVG = _load_favicon_svg()
