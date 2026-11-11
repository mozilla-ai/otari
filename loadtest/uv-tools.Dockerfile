# The container the load test drives uv from.
#
# Built rather than pulled, because `ghcr.io/astral-sh/uv` publishes a Python
# flavour and a version, never both: `python3.14-bookworm-slim` carries whatever
# uv it was built with, which lags, and `0.13.7` carries the binary alone. uv
# refuses every command when its version falls outside the range
# `pyproject.toml` declares, so the version has to be the pinned one.
#
# `COPY --from` does not expand a variable in an image reference, so the image is
# pinned through a named stage, as the repository's own Dockerfile does.
# `tests/unit/test_uv_version_pin.py` holds the default below to that range.
ARG UV_VERSION=0.13.7

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM python:3.14-slim

COPY --from=uv /uv /uvx /bin/
