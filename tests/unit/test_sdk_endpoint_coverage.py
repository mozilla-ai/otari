"""Endpoint-coverage drift gate for the SDK manifest.

Checks ``scripts/sdk_codegen/sdk-endpoints.txt`` against
``docs/public/openapi.json`` from the same commit. Adding a gateway endpoint
fails here until it is classified as ``[covered]`` or ``[excluded]``.

This lives in the gateway rather than in the four SDK repos on purpose. The SDK
copies of the manifest are pushed by the codegen workflow, and their own checks
cannot see a spec change until a regen lands, so a gateway-side gate is what
catches drift at the moment it is introduced. Both files are read from disk, so
the result depends only on the commit under test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "sdk_codegen"))

import endpoint_manifest as em  # noqa: E402

SPEC_PATH = REPO_ROOT / "docs" / "public" / "openapi.json"


@pytest.fixture(scope="module")
def manifest() -> tuple[set[str], set[str]]:
    return em.load_manifest()


@pytest.fixture(scope="module")
def endpoints() -> set[str]:
    return em.spec_endpoints(em.load_spec(SPEC_PATH))


def test_manifest_is_well_formed(manifest: tuple[set[str], set[str]]) -> None:
    covered, excluded = manifest
    assert covered, "manifest [covered] section is empty"
    overlap = sorted(covered & excluded)
    assert not overlap, f"endpoints listed in both sections: {overlap}"


def test_spec_endpoints_are_accounted_for(
    manifest: tuple[set[str], set[str]], endpoints: set[str]
) -> None:
    covered, excluded = manifest
    missing = em.unaccounted(endpoints, covered, excluded)
    assert not missing, (
        f"docs/public/openapi.json exposes {len(missing)} endpoint(s) absent from "
        f"scripts/sdk_codegen/sdk-endpoints.txt: {missing}. Add each under [covered] "
        "if an SDK wrapper surfaces it, or under [excluded] with a reason if not. "
        "The codegen workflow pushes this manifest to the SDK repos, so classifying "
        "it here is what keeps all four in step."
    )


def test_manifest_has_no_stale_entries(
    manifest: tuple[set[str], set[str]], endpoints: set[str]
) -> None:
    """A manifest entry the spec no longer exposes is a leftover, not a deferral.

    Unlike the old per-SDK check, the manifest and the spec are in the same
    commit here, so a mismatch is always an editing slip and always fixable in
    the PR that caused it. That makes failing appropriate where warning was not.
    """
    covered, excluded = manifest
    leftovers = em.stale(endpoints, covered, excluded)
    assert not leftovers, (
        f"scripts/sdk_codegen/sdk-endpoints.txt lists {len(leftovers)} endpoint(s) "
        f"the spec no longer exposes: {leftovers}. Remove them, or restore the routes."
    )
