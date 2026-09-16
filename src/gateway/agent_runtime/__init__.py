"""Agent Gates: policy evaluation for coding-agent sessions.

Otari never reads a caller's repository directly. The caller (an agent hook,
eventually the native ``otari hook`` dispatcher) collects its own policy body
and Git evidence locally and submits both in one request; this package
parses and evaluates them. Every function here is pure: no filesystem,
network, subprocess, or clock access. That is what lets the same evaluator
run identically from a route handler and from a test, with nothing to fake
but the evidence passed in.

Evaluation results are ``client_reported``: Otari did not observe the
repository itself, only what the caller claims about it. See
docs/agent-gates.md for the request/response contract, and
docs/agent-gates.md for where this is going.
"""
