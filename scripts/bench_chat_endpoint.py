#!/usr/bin/env python3
"""Minimal latency benchmark for POST /v1/chat/completions.

Boots the gateway in-process against a throwaway SQLite DB and drives it
with an in-process ASGI transport (`httpx.AsyncClient(transport=ASGITransport(...))`),
so requests share the *same* asyncio event loop as the app -- this is what
lets the ``dns-stall`` scenario reproduce the event-loop-blocking bug the
"async MCP/guardrail URL safety check" fix addresses: if any single request
handler blocks the loop with a synchronous syscall, every other concurrent
request measured here pays for it, in-process, without needing a second
machine or a packet-capture setup.

The upstream provider call (``any_llm.acompletion``) is patched to return a
fixed response instantly, so these numbers measure *gateway* overhead only
(auth, budget reservation, guardrails, MCP-server URL validation, provider
resolution, usage logging), not real provider latency.

Scenarios (see --scenario):

* baseline    N concurrent plain chat-completion requests. Reports
              p50/p95/p99/mean latency -- the number to compare before/after
              a gateway change.
* dns-stall   Same N concurrent plain requests, but one extra request in the
              same batch carries an `mcp_servers` entry pointing at a
              hostname that will not resolve (forces a real, slow DNS
              failure). Compares the plain requests' p95 with and without
              the stalling request in the batch. On the pre-fix code (DNS
              resolution running synchronously inside a Pydantic validator
              during request-body parsing) this inflates every concurrent
              request's latency by roughly the DNS timeout; after the fix
              (async resolution in the request pipeline) it should not.
* resolve-count
              Single request; counts how many times `resolve_provider_selector`
              is invoked while handling one standalone request. Verifies the
              "resolve provider once, reuse for dispatch" fix directly
              (expect 1, not 2) rather than inferring it from latency noise.

Usage:
    uv run python scripts/bench_chat_endpoint.py --scenario baseline -n 200 -c 20
    uv run python scripts/bench_chat_endpoint.py --scenario dns-stall -n 50 -c 25
    uv run python scripts/bench_chat_endpoint.py --scenario resolve-count
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import statistics
import sys
import tempfile
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import httpx
from any_llm.types.completion import ChatCompletion, ChatCompletionMessage, Choice, CompletionUsage
from fastapi.testclient import TestClient

from gateway.core.config import API_KEY_HEADER, GatewayConfig
from gateway.main import create_app

MODEL = "anthropic:claude-3-5-sonnet-20241022"
UNRESOLVABLE_HOST = "this-host-does-not-exist.invalid"


def _fake_completion() -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-bench",
        object="chat.completion",
        created=int(time.time()),
        model="claude-3-5-sonnet-20241022",
        choices=[
            Choice(index=0, message=ChatCompletionMessage(role="assistant", content="ok"), finish_reason="stop")
        ],
        usage=CompletionUsage(prompt_tokens=5, completion_tokens=1, total_tokens=6),
    )


async def _instant_acompletion(**_kwargs: Any) -> ChatCompletion:
    return _fake_completion()


@asynccontextmanager
async def _bench_app(
    *, master_key: str = "bench-master-key"
) -> AsyncIterator[tuple[httpx.AsyncClient, dict[str, str]]]:
    """Boot the gateway against a throwaway SQLite DB, bootstrap an API key,
    and yield an in-process ASGI-backed async client plus its auth header.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        database_url = f"sqlite:///{Path(tmpdir) / 'bench.db'}"
        config = GatewayConfig(database_url=database_url, master_key=master_key, require_pricing=False)
        app = create_app(config)

        # TestClient's context manager drives FastAPI startup (table creation
        # + bootstrap key) synchronously; cheaper than hand-rolling lifespan
        # handling for a one-off script.
        master_headers = {API_KEY_HEADER: f"Bearer {master_key}"}
        with TestClient(app) as sync_client:
            key_resp = sync_client.post("/v1/keys", json={"key_name": "bench"}, headers=master_headers)
            key_resp.raise_for_status()
            api_key = key_resp.json()["key"]

            # Configure pricing so the per-request "no pricing configured"
            # warning (real logging I/O on every request) doesn't add noise
            # to the latency measurement -- an operator would set this too.
            pricing_resp = sync_client.post(
                "/v1/pricing",
                json={"model_key": MODEL, "input_price_per_million": 3.0, "output_price_per_million": 15.0},
                headers=master_headers,
            )
            pricing_resp.raise_for_status()

        headers = {API_KEY_HEADER: f"Bearer {api_key}"}
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://bench.local", timeout=30.0) as client:
            yield client, headers


def _chat_body(*, mcp_servers: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
    }
    if mcp_servers is not None:
        body["mcp_servers"] = mcp_servers
    return body


def _percentiles(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "mean": statistics.mean(ordered),
        "p50": statistics.median(ordered),
        "p95": ordered[int(len(ordered) * 0.95) - 1] if len(ordered) >= 20 else max(ordered),
        "p99": ordered[int(len(ordered) * 0.99) - 1] if len(ordered) >= 100 else max(ordered),
        "max": max(ordered),
    }


def _print_stats(label: str, samples: list[float]) -> None:
    stats = _percentiles(samples)
    print(
        f"{label}: n={len(samples)} "
        f"mean={stats['mean'] * 1000:.1f}ms p50={stats['p50'] * 1000:.1f}ms "
        f"p95={stats['p95'] * 1000:.1f}ms p99={stats['p99'] * 1000:.1f}ms max={stats['max'] * 1000:.1f}ms"
    )


async def _timed_post(client: httpx.AsyncClient, headers: dict[str, str], body: dict[str, Any]) -> float:
    start = time.perf_counter()
    resp = await client.post("/v1/chat/completions", json=body, headers=headers)
    elapsed = time.perf_counter() - start
    if resp.status_code != 200:
        print(f"  !! non-200 response ({resp.status_code}): {resp.text[:200]}", file=sys.stderr)
    return elapsed


async def run_baseline(n: int, concurrency: int) -> None:
    with patch("gateway.api.routes.chat.acompletion", new=_instant_acompletion):
        async with _bench_app() as (client, headers):
            sem = asyncio.Semaphore(concurrency)

            async def _one() -> float:
                async with sem:
                    return await _timed_post(client, headers, _chat_body())

            samples = await asyncio.gather(*(_one() for _ in range(n)))
    _print_stats("baseline (plain requests)", list(samples))


async def run_dns_stall(n: int, concurrency: int) -> None:
    with patch("gateway.api.routes.chat.acompletion", new=_instant_acompletion):
        async with _bench_app() as (client, headers):
            sem = asyncio.Semaphore(concurrency)

            async def _plain() -> float:
                async with sem:
                    return await _timed_post(client, headers, _chat_body())

            async def _stalling() -> None:
                async with sem:
                    body = _chat_body(mcp_servers=[{"name": "x", "url": f"https://{UNRESOLVABLE_HOST}/mcp"}])
                    await client.post("/v1/chat/completions", json=body, headers=headers)

            print(f"Running {n} plain requests (concurrency={concurrency}), no stalling request...")
            baseline_samples = await asyncio.gather(*(_plain() for _ in range(n)))
            _print_stats("dns-stall: without stalling request", list(baseline_samples))

            print(f"Running {n} plain requests + 1 unresolvable-host mcp_servers request in the same batch...")
            start = time.perf_counter()
            results = await asyncio.gather(*([_plain() for _ in range(n)] + [_stalling()]), return_exceptions=True)
            total = time.perf_counter() - start
            stalled_samples = [r for r in results if isinstance(r, float)]
            _print_stats("dns-stall: with stalling request present", stalled_samples)
            print(f"  (batch wall time: {total * 1000:.1f}ms)")

            delta_p95 = _percentiles(stalled_samples)["p95"] - _percentiles(list(baseline_samples))["p95"]
            print(
                f"\np95 delta caused by the stalling request: {delta_p95 * 1000:+.1f}ms "
                "(large positive delta => the DNS lookup is blocking the event loop; "
                "near-zero => the async fix is working)"
            )


async def run_resolve_count() -> None:
    from gateway.services import provider_kwargs as provider_kwargs_module

    calls = 0
    real_resolve = provider_kwargs_module.resolve_provider_selector

    def _counting_resolve(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return real_resolve(*args, **kwargs)

    with (
        patch("gateway.api.routes.chat.acompletion", new=_instant_acompletion),
        patch("gateway.api.routes._pipeline.resolve_provider_selector", new=_counting_resolve),
    ):
        async with _bench_app() as (client, headers):
            resp = await client.post("/v1/chat/completions", json=_chat_body(), headers=headers)
            resp.raise_for_status()

    print(f"resolve_provider_selector call count for one standalone request: {calls}")
    print("(expect 1 -- the pricing-gate resolution reused for dispatch, not resolved twice)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", choices=["baseline", "dns-stall", "resolve-count"], default="baseline")
    parser.add_argument("-n", "--requests", type=int, default=200, help="number of plain requests")
    parser.add_argument("-c", "--concurrency", type=int, default=20, help="max concurrent in-flight requests")
    args = parser.parse_args()

    # Quiet the gateway's own request-path logging (structured logs on every
    # request add real I/O overhead that would otherwise show up as noise in
    # the very latency numbers this script measures).
    logging.getLogger("gateway").setLevel(logging.ERROR)

    if args.scenario == "baseline":
        asyncio.run(run_baseline(args.requests, args.concurrency))
    elif args.scenario == "dns-stall":
        asyncio.run(run_dns_stall(args.requests, args.concurrency))
    else:
        asyncio.run(run_resolve_count())


if __name__ == "__main__":
    main()
