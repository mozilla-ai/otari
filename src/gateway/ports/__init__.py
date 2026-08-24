"""Ports: the domain-named interfaces Otari's core depends on.

Each port is a seam between the core and whichever build supplies its adapter,
and each ships with a working adapter in ``gateway.adapters`` so Otari stands
alone with no overlay present. Only a capability with a real second
implementation earns one; plain, single-implementation services stay concrete
(``ARCHITECTURE.md``, "Cardinal rules for contributors", rule 7).

Ports are asynchronous ``Protocol`` classes: an adapter satisfies one by shape,
without importing or subclassing it, which is what lets an overlay implement a
port it only depends on structurally. The composition-root container
(``gateway.container``) keys its registry on the protocol class object itself,
so a caller names the port and never a concrete adapter.
"""
