"""What automatic recall adds to a request's time: the same request with memory declined and with it, alternated.

    overhead.py MODE [N]     MODE as in bench.py (similarity, broad, rerank:0.9 ...); N rounds, 15 by default
"""

import statistics
import sys
import time

from bench import KEYS, active, http, mode_switches, switches

QUESTION = sys.argv[3] if len(sys.argv) > 3 else "How do I roll back a staging deploy?"


def timed(declined: bool) -> float:
    headers = {"Otari-Memory": "off"} if declined else {}
    body = {"model": "ollama:qwen2.5:3b", "max_tokens": 1, "messages": [{"role": "user", "content": QUESTION}]}
    started = time.monotonic()
    status, _, answer = http("POST", "/chat/completions", KEYS["ALICE_KEY"], body, headers)
    if status != 200:
        raise SystemExit(f"{status} {answer}")
    return (time.monotonic() - started) * 1000


mode, rounds = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 15
switches(auto_recall=True, **mode_switches(mode))
timed(True), timed(False)  # warm both paths
without, with_memory = [], []
for _ in range(rounds):
    without.append(timed(True))
    with_memory.append(timed(False))
print(f"{mode} with {active()} entries: without memory p50 {statistics.median(without):.0f} ms, "
      f"with recall p50 {statistics.median(with_memory):.0f} ms, "
      f"added p50 {statistics.median(w - o for w, o in zip(with_memory, without)):.0f} ms")
