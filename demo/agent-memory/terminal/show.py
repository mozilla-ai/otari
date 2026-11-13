"""Formats what the demo's calls answer. Reads JSON on stdin: show.py tool | chat | experiment."""

import json
import sys

RED, BOLD, DIM, OFF = "\033[31m", "\033[1m", "\033[2m", "\033[0m"


def tool(reply: dict) -> None:
    result = reply["result"]
    if result.get("isError"):
        print(f"  {RED}refused:{OFF} {result['content'][0]['text']}")
        return
    data = result.get("structuredContent") or {}
    entries = data.get("results", [data])
    if not entries:
        print("  (nothing found)")
    for entry in entries:
        if "text" in entry:
            label = f"{entry['kind']}, {entry['scope']}, v{entry['version']}, confidence {entry['confidence']}"
            print(f"  [{label}] {entry['text']}")
        else:
            print(f"  {json.dumps(entry)}")


def chat(body: dict) -> None:
    if "choices" not in body:
        print(f"  {RED}{body}{OFF}")
        return
    print(f"{BOLD}{body['choices'][0]['message']['content'].strip()}{OFF}")
    usage = body.get("usage") or {}
    print(f"{DIM}  tokens: {usage.get('prompt_tokens')} in, {usage.get('completion_tokens')} out{OFF}")


def experiment(report: dict) -> None:
    if not report["started_at"]:
        print("  no experiment ran: prepare.sh --quick skips it")
        return
    for arm in ("control", "memory"):
        summary = report[arm]
        print(f"  {arm:8} {summary['sessions']:3} sessions   {summary['tokens_per_session']:7.0f} tokens per session")
    change = report["tokens"]
    if change:
        verdict = "conclusive" if change["conclusive"] else "not conclusive"
        print(
            f"  change with memory: {change['relative']:+.1%} "
            f"(95% interval {change['low']:+.0f} to {change['high']:+.0f} tokens; {verdict})"
        )
    else:
        print(f"  change: needs {report['min_sessions']} sessions in each arm")
    print(f"  recall found something in {report['recalls_with_results']} of {report['recalls']} turns")


if __name__ == "__main__":
    {"tool": tool, "chat": chat, "experiment": experiment}[sys.argv[1]](json.load(sys.stdin))
