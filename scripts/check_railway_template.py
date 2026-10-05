#!/usr/bin/env python3
"""Check deploy/railway/template.json and listing.md against the live Railway template.

A Railway template is a platform object edited by hand in the Railway editor,
and the API has no mutation that writes its services or variables, so the repo
snapshot can only be kept honest by reading the live template back. Railway's
public GraphQL API answers ``template(code: ...)`` with no token: its
``serializedConfig`` holds every service's image, deploy settings, domains and
variables, and ``readme`` holds the README pasted from listing.md.

Compared per service (matched by name): the image (``docker.io/`` ignored),
start command, healthcheck path, the public domain's port, and each variable's
default, optional flag and description. Defaults are compared exactly, since a
stray space there changes the value; descriptions ignore surrounding
whitespace. Buckets are compared by name, which is all the serialized config
carries about one. The README ignores trailing whitespace.

Standard library only, like oss_edition_smoke.py, so CI runs it with no
environment. Exits 0 when the two agree, 1 on drift, 2 when the fetch fails.

Usage:
    python scripts/check_railway_template.py
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = REPO_ROOT / "deploy" / "railway" / "template.json"
LISTING_PATH = REPO_ROOT / "deploy" / "railway" / "listing.md"
GRAPHQL_URL = "https://backboard.railway.com/graphql/v2"
QUERY = "query($code: String!) { template(code: $code) { readme serializedConfig } }"


class FetchError(Exception):
    """The live template could not be read."""


def fetch_template(code: str) -> dict[str, Any]:
    """Return the live template's ``readme`` and ``serializedConfig``."""
    request = urllib.request.Request(
        GRAPHQL_URL,
        data=json.dumps({"query": QUERY, "variables": {"code": code}}).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "otari-railway-template-check"},
    )
    body: dict[str, Any]
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.load(response)
    except urllib.error.HTTPError as exc:
        # GraphQL errors (an unknown code among them) arrive with a non-2xx status.
        try:
            body = json.loads(exc.read())
        except ValueError:
            raise FetchError(f"Railway answered HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        raise FetchError(f"could not reach Railway: {exc}") from exc

    if body.get("errors"):
        raise FetchError("; ".join(error.get("message", str(error)) for error in body["errors"]))
    template: dict[str, Any] | None = (body.get("data") or {}).get("template")
    if not template:
        raise FetchError(f"Railway returned no template for code {code!r}")
    return template


def _one_side_only(label: str, expected: set[str], live: set[str]) -> list[str]:
    return [
        f"{label} {key} is in template.json but missing from the live template" for key in sorted(expected - live)
    ] + [f"{label} {key} is on the live template but not in template.json" for key in sorted(live - expected)]


def normalize_image(image: str | None) -> str | None:
    if image is None:
        return None
    return image.removeprefix("docker.io/")


def _domain_ports(live_service: dict[str, Any]) -> list[int]:
    domains = (live_service.get("networking") or {}).get("serviceDomains") or {}
    return sorted(domain.get("port") for domain in domains.values())


def _expected_ports(service: dict[str, Any]) -> list[int]:
    networking = service.get("networking") or {}
    return [networking["targetPort"]] if networking.get("publicDomain") else []


def _bucket_names(buckets: dict[str, Any] | None) -> set[str]:
    return {bucket["name"] for bucket in (buckets or {}).values()}


def _diff_variables(name: str, expected: dict[str, Any], live: dict[str, Any]) -> list[str]:
    problems = _one_side_only(f"{name}: variable", set(expected), set(live))
    for var in sorted(expected.keys() & live.keys()):
        want, got = expected[var], live[var]
        if want.get("defaultValue") != got.get("defaultValue"):
            problems.append(
                f"{name}: variable {var} default: template.json has {want.get('defaultValue')!r}, "
                f"live has {got.get('defaultValue')!r}"
            )
        if bool(want.get("isOptional")) != bool(got.get("isOptional")):
            problems.append(
                f"{name}: variable {var} optional: template.json has {bool(want.get('isOptional'))}, "
                f"live has {bool(got.get('isOptional'))}"
            )
        if (want.get("description") or "").strip() != (got.get("description") or "").strip():
            problems.append(
                f"{name}: variable {var} description: template.json has {want.get('description')!r}, "
                f"live has {got.get('description')!r}"
            )
    return problems


def diff_config(snapshot: dict[str, Any], live_config: dict[str, Any]) -> list[str]:
    """List every way the live ``serializedConfig`` disagrees with the snapshot."""
    expected = {service["name"]: service for service in snapshot["services"].values()}
    live = {service["name"]: service for service in live_config.get("services", {}).values()}

    problems = _one_side_only("service", set(expected), set(live))
    problems += _one_side_only(
        "bucket", _bucket_names(snapshot.get("buckets")), _bucket_names(live_config.get("buckets"))
    )

    for name in sorted(expected.keys() & live.keys()):
        want, got = expected[name], live[name]

        want_image = normalize_image((want.get("source") or {}).get("image"))
        got_image = normalize_image((got.get("source") or {}).get("image"))
        if want_image != got_image:
            problems.append(f"{name}: image: template.json has {want_image!r}, live has {got_image!r}")

        for key in ("startCommand", "healthcheckPath"):
            want_value = (want.get("deploy") or {}).get(key)
            got_value = (got.get("deploy") or {}).get(key)
            if want_value != got_value:
                problems.append(f"{name}: {key}: template.json has {want_value!r}, live has {got_value!r}")

        want_ports, got_ports = _expected_ports(want), _domain_ports(got)
        if want_ports != got_ports:
            problems.append(f"{name}: public domain port: template.json has {want_ports}, live has {got_ports}")

        problems += _diff_variables(name, want.get("variables", {}), got.get("variables") or {})
    return problems


def diff_readme(listing: str, live_readme: str) -> list[str]:
    """Return a unified diff from listing.md to the live README, empty when they agree."""
    want = listing.rstrip().splitlines()
    got = (live_readme or "").rstrip().splitlines()
    return list(difflib.unified_diff(want, got, "deploy/railway/listing.md", "live template readme", lineterm=""))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--template", type=Path, default=TEMPLATE_PATH, help="the template.json snapshot")
    parser.add_argument("--listing", type=Path, default=LISTING_PATH, help="the template's Railway README")
    args = parser.parse_args(argv)

    snapshot = json.loads(args.template.read_text())
    try:
        live = fetch_template(snapshot["code"])
    except FetchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    problems = diff_config(snapshot, live.get("serializedConfig") or {})
    readme_diff = diff_readme(args.listing.read_text(), live.get("readme") or "")

    if not problems and not readme_diff:
        print(f"The live Railway template {snapshot['code']!r} matches template.json and listing.md.")
        return 0

    print(f"The live Railway template {snapshot['code']!r} has drifted from the repo.\n")
    for problem in problems:
        print(f"- {problem}")
    if readme_diff:
        if problems:
            print()
        print("The live README differs from listing.md:")
        print("\n".join(readme_diff))
    print(
        "\nEdit the live template or the repo files until they agree; "
        "see 'Maintaining the template' in deploy/railway/README.md."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
