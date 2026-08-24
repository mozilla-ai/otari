"""Adapters: Otari's own implementation of every port in ``gateway.ports``.

The cardinal property of the seam is that each port ships with a working
adapter here, a real lightweight one or an honest Null Object, so Otari stands
alone with no overlay present (``ARCHITECTURE.md``, rule 3). An overlay binds
richer adapters of its own through the bootstrap hook and never edits this
package.

Only the composition root (``gateway.container``) names anything in here. Every
other layer refers to the port and asks the container for whatever is bound,
which is what ``scripts/check_architecture.py`` enforces.
"""
