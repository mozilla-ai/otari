# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.27"]
# ///
"""Open-loop load generator for the load test.

Sends chat completions at a fixed rate whatever the gateway's latency is (open
loop, so a slow gateway builds a queue rather than slowing the test down), with
a configurable request mix: prompt sizes, streaming share and end users behind one
service key. Writes one JSON result per run.

    uv run loadgen.py --phase 300:120 --phase 1000:120 --label steady

Phases run back to back; each is RPM:SECONDS.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

# An assumed mix until a real sample replaces it: mostly short prompts with a
# long tail. Each entry is (weight, approximate prompt tokens).
DEFAULT_PROMPT_MIX = [(60, 200), (30, 1_000), (9, 4_000), (1, 16_000)]
WORDS = "the quick brown fox jumps over the lazy dog while summarizing a long article".split()


@dataclass
class Result:
    start: float
    status: int
    latency_ms: float
    stream: bool
    ttft_ms: float | None = None
    served_by: str | None = None
    stream_error_event: bool = False
    saw_done: bool = False
    error: str | None = None
    detail: str | None = None


@dataclass
class Summary:
    label: str
    phases: list[str]
    target: str
    model: str
    started_at: str
    sent: int = 0
    status_counts: dict[str, int] = field(default_factory=dict)
    served_by: dict[str, int] = field(default_factory=dict)
    latency_ms: dict[str, float] = field(default_factory=dict)
    ttft_ms: dict[str, float] = field(default_factory=dict)
    gateway_overhead_ms: dict[str, float] = field(default_factory=dict)
    streams: int = 0
    stream_error_events: int = 0
    streams_without_done: int = 0
    client_errors: dict[str, int] = field(default_factory=dict)
    refusals_429: dict[str, int] = field(default_factory=dict)
    late_starts: int = 0
    per_phase: list[dict] = field(default_factory=list)


def percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)

    def pick(q: float) -> float:
        return round(ordered[min(len(ordered) - 1, int(q * len(ordered)))], 1)

    return {
        "n": len(ordered),
        "p50": pick(0.50),
        "p90": pick(0.90),
        "p99": pick(0.99),
        "max": round(ordered[-1], 1),
        "mean": round(statistics.fmean(ordered), 1),
    }


def make_prompt(tokens: int) -> str:
    # About four characters a token, which is what the fake provider counts.
    words = []
    length = 0
    while length < tokens * 4:
        word = random.choice(WORDS)
        words.append(word)
        length += len(word) + 1
    return " ".join(words)


def pick_prompt_tokens(mix: list[tuple[int, int]]) -> int:
    weights, sizes = zip(*mix, strict=True)
    return random.choices(sizes, weights=weights)[0]


async def one_request(
    client: httpx.AsyncClient, args: argparse.Namespace, headers: dict[str, str], prompts: dict[int, str]
) -> Result:
    stream = random.random() < args.stream_share
    tokens = pick_prompt_tokens(args.prompt_mix)
    body: dict = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompts[tokens]}],
        "max_tokens": args.max_tokens,
        "stream": stream,
    }
    if args.users > 0:
        body["user"] = f"enduser-{random.randrange(args.users):05d}"
    started = time.perf_counter()
    wall = time.time()
    try:
        if not stream:
            response = await client.post(args.path, json=body, headers=headers)
            latency = (time.perf_counter() - started) * 1000
            result = Result(wall, response.status_code, latency, False)
            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"] or ""
                result.served_by = content.split("served-by:")[-1][:2] if "served-by:" in content else "?"
            else:
                result.detail = response.text[:300]
            return result

        async with client.stream("POST", args.path, json=body, headers=headers) as response:
            result = Result(wall, response.status_code, 0.0, True)
            if response.status_code != 200:
                result.detail = (await response.aread()).decode(errors="replace")[:300]
                result.latency_ms = (time.perf_counter() - started) * 1000
                return result
            async for line in response.aiter_lines():
                if not line:
                    continue
                if line.startswith("event: error"):
                    result.stream_error_event = True
                    continue
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    result.saw_done = True
                    continue
                payload = json.loads(data)
                if "error" in payload:
                    result.stream_error_event = True
                    continue
                for choice in payload.get("choices") or []:
                    content = (choice.get("delta") or {}).get("content")
                    if content and result.ttft_ms is None:
                        result.ttft_ms = (time.perf_counter() - started) * 1000
                    if content and "served-by:" in content:
                        result.served_by = content.split("served-by:")[-1][:2]
            result.latency_ms = (time.perf_counter() - started) * 1000
            return result
    except httpx.HTTPError as exc:
        return Result(wall, 0, (time.perf_counter() - started) * 1000, stream, error=type(exc).__name__)


async def run_phase(
    rpm: int,
    seconds: int,
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    headers: dict[str, str],
    prompts: dict[int, str],
    results: list[Result],
) -> tuple[int, int]:
    interval = 60.0 / rpm
    total = int(rpm * seconds / 60)
    tasks: list[asyncio.Task] = []
    late = 0
    begin = time.perf_counter()
    print(f"[{args.label}] phase {rpm} RPM for {seconds}s ({total} requests)", flush=True)
    for index in range(total):
        due = begin + index * interval
        delay = due - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        elif delay < -0.25:
            late += 1
        task = asyncio.create_task(one_request(client, args, headers, prompts))
        task.add_done_callback(lambda t: results.append(t.result()))
        tasks.append(task)
        if index and index % max(1, rpm) == 0:
            done = [r for r in results if r.start >= time.time() - 60]
            ok = sum(1 for r in done if r.status == 200)
            print(f"[{args.label}]   {index}/{total} sent, last 60s: {ok}/{len(done)} ok", flush=True)
    await asyncio.gather(*tasks)
    return total, late


def summarize(label: str, results: list[Result], args: argparse.Namespace, sent: int, late: int) -> Summary:
    summary = Summary(
        label=label,
        phases=args.phase,
        target=args.base_url + args.path,
        model=args.model,
        started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(results[0].start if results else time.time())),
        sent=sent,
        late_starts=late,
    )
    for result in results:
        key = str(result.status) if result.status else f"conn:{result.error}"
        summary.status_counts[key] = summary.status_counts.get(key, 0) + 1
        if result.served_by:
            summary.served_by[result.served_by] = summary.served_by.get(result.served_by, 0) + 1
        if result.error:
            summary.client_errors[result.error] = summary.client_errors.get(result.error, 0) + 1
        if result.status == 429:
            reason = (result.detail or "")[:160]
            summary.refusals_429[reason] = summary.refusals_429.get(reason, 0) + 1
        if result.stream and result.status == 200:
            summary.streams += 1
            summary.stream_error_events += result.stream_error_event
            summary.streams_without_done += not result.saw_done and not result.stream_error_event
    ok = [r for r in results if r.status == 200]
    summary.latency_ms = percentiles([r.latency_ms for r in ok if not r.stream])
    summary.ttft_ms = percentiles([r.ttft_ms for r in ok if r.stream and r.ttft_ms is not None])
    if args.provider_latency_ms is not None:
        # What the gateway adds on a non-streamed call: the client's latency minus
        # the fake provider's fixed sleep. Includes the network hops and the
        # load balancer, which the baseline run measures on their own.
        summary.gateway_overhead_ms = percentiles(
            [r.latency_ms - args.provider_latency_ms for r in ok if not r.stream and r.served_by != "m3"]
        )
    return summary


async def main_async(args: argparse.Namespace) -> Summary:
    headers = {"Content-Type": "application/json"}
    if args.key_file:
        state = json.loads(Path(args.key_file).read_text())
        headers["Authorization"] = f"Bearer {state['key']}"
    elif args.key:
        headers["Authorization"] = f"Bearer {args.key}"
    prompts = {tokens: make_prompt(tokens) for _, tokens in args.prompt_mix}
    limits = httpx.Limits(max_connections=args.max_connections, max_keepalive_connections=args.max_connections)
    results: list[Result] = []
    sent = late = 0
    async with httpx.AsyncClient(base_url=args.base_url, timeout=args.timeout, limits=limits) as client:
        for phase in args.phase:
            rpm, seconds = (int(part) for part in phase.split(":"))
            before = len(results)
            phase_sent, phase_late = await run_phase(rpm, seconds, client, args, headers, prompts, results)
            sent += phase_sent
            late += phase_late
            phase_summary = summarize(f"{args.label}@{rpm}", results[before:], args, phase_sent, phase_late)
            print(json.dumps(_brief(phase_summary)), flush=True)
    summary = summarize(args.label, results, args, sent, late)
    summary.per_phase = [
        _brief(summarize(f"{args.label}@{phase}", chunk, args, len(chunk), 0))
        for phase, chunk in _split(args.phase, results)
    ]
    return summary


def _split(phases: list[str], results: list[Result]) -> list[tuple[str, list[Result]]]:
    ordered = sorted(results, key=lambda r: r.start)
    chunks = []
    cursor = 0
    for phase in phases:
        rpm, seconds = (int(part) for part in phase.split(":"))
        count = int(rpm * seconds / 60)
        chunks.append((phase, ordered[cursor : cursor + count]))
        cursor += count
    return chunks


def _brief(summary: Summary) -> dict:
    return {
        "label": summary.label,
        "sent": summary.sent,
        "status": summary.status_counts,
        "served_by": summary.served_by,
        "latency_ms": summary.latency_ms,
        "ttft_ms": summary.ttft_ms,
        "overhead_ms": summary.gateway_overhead_ms,
        "stream_error_events": summary.stream_error_events,
        "streams_without_done": summary.streams_without_done,
    }


def parse_mix(text: str) -> list[tuple[int, int]]:
    """``60:200,30:1000`` → weight 60 for 200-token prompts, and so on."""
    return [tuple(int(part) for part in item.split(":")) for item in text.split(",")]  # type: ignore[misc]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://lb")
    parser.add_argument("--path", default="/api/v1/chat/completions")
    parser.add_argument("--key-file", help="state file written by setup_tenant.py")
    parser.add_argument("--key", help="API key, when there is no state file")
    parser.add_argument("--model", default="summarize")
    parser.add_argument("--phase", action="append", default=[], help="RPM:SECONDS, repeatable")
    parser.add_argument("--stream-share", type=float, default=0.5)
    parser.add_argument("--users", type=int, default=200, help="end users behind the key; 0 sends no user")
    parser.add_argument("--prompt-mix", type=parse_mix, default=DEFAULT_PROMPT_MIX)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--provider-latency-ms", type=float, default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-connections", type=int, default=1000)
    parser.add_argument("--label", default="run")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    if not args.phase:
        args.phase = ["300:60"]

    summary = asyncio.run(main_async(args))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.label}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(asdict(summary), indent=2))
    print(json.dumps(asdict(summary), indent=2))
    print(f"wrote {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
