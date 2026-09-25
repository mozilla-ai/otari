"""Agent Guardrails: policy evaluation for coding-agent sessions.

Otari never reads a caller's repository directly. The caller (`otari hook`)
collects its own policy body and Git evidence locally; this package parses and
evaluates them. Every function here is pure: no filesystem, network,
subprocess, or clock access. That is what lets the same evaluator run
identically in process from `otari hook`, from the Hook Server route
(`gateway.api.routes.hooks`) and from a test, with nothing to fake but the
evidence passed in.

It lives in the otari-agent distribution rather than in the gateway so that
`otari hook` can be installed without the server. Evaluation results are
``client_reported``: Otari did not observe the repository itself, only what the
caller claims about it. See docs/agent-guardrails.md for the contract.
"""
