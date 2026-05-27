"""Compatibility aliases for legacy `api.routes.*` imports in tests."""

import sys

from gateway.api.routes import (
    admin,
    budgets,
    chat,
    embeddings,
    health,
    keys,
    messages,
    models,
    pricing,
    projects,
    route_traces,
    routing,
    routing_policies,
    users,
)

admin = admin
chat = chat
messages = messages
embeddings = embeddings
users = users
keys = keys
budgets = budgets
pricing = pricing
projects = projects
route_traces = route_traces
routing = routing
routing_policies = routing_policies
models = models
health = health

sys.modules[f"{__name__}.chat"] = chat
sys.modules[f"{__name__}.admin"] = admin
sys.modules[f"{__name__}.messages"] = messages
sys.modules[f"{__name__}.embeddings"] = embeddings
sys.modules[f"{__name__}.users"] = users
sys.modules[f"{__name__}.keys"] = keys
sys.modules[f"{__name__}.budgets"] = budgets
sys.modules[f"{__name__}.pricing"] = pricing
sys.modules[f"{__name__}.projects"] = projects
sys.modules[f"{__name__}.route_traces"] = route_traces
sys.modules[f"{__name__}.routing"] = routing
sys.modules[f"{__name__}.routing_policies"] = routing_policies
sys.modules[f"{__name__}.models"] = models
sys.modules[f"{__name__}.health"] = health

__all__ = [
    "admin",
    "budgets",
    "chat",
    "embeddings",
    "health",
    "keys",
    "messages",
    "models",
    "pricing",
    "projects",
    "route_traces",
    "routing",
    "routing_policies",
    "users",
]
