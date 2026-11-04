"""Run the README's Docker quickstart exactly as written, against a given image.

The OSS-edition smoke and the liveness check each set the secrets they need, so
neither notices when the one command a newcomer copies from the README stops
starting. This runs that command, changing nothing but the image it names, and
checks the first run the README promises: the gateway answers, prints a
bootstrap key once, and accepts that key.

Usage: uv run --frozen --no-dev python scripts/readme_quickstart_check.py --image <tag>

Standard library only, like the OSS-edition smoke, so it runs under ``--no-dev``.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

README = Path(__file__).resolve().parent.parent / "README.md"
PUBLISHED_IMAGE = "mzdotai/otari:latest"
BASE_URL = "http://127.0.0.1:8000"
HEALTH_TIMEOUT_SECONDS = 120
DOCKER_TIMEOUT_SECONDS = 60

_FENCE = re.compile(r"^```bash\n(.*?)^```", re.DOTALL | re.MULTILINE)
_BOOTSTRAP_KEY = re.compile(r"Save this key now:\s*\n(tk-\S+)\s*$", re.MULTILINE)
# The first run prints a generated master key and an API key, one per line.
_PRINTED_SECRET = re.compile(r"\b(?:otari-mk-|tk-)\S+")


class QuickstartFailure(Exception):
    """The README quickstart did not give the first run it promises."""


def quickstart_command(readme: str) -> str:
    """Return the first ``bash`` block under ``## Quickstart``, which must be a ``docker run``."""
    start = readme.find("\n## Quickstart\n")
    if start == -1:
        raise QuickstartFailure("README.md has no '## Quickstart' section")
    end = readme.find("\n## ", start + 1)
    section = readme[start : end if end != -1 else len(readme)]
    match = _FENCE.search(section)
    if match is None:
        raise QuickstartFailure("The README quickstart has no bash block")
    command = match.group(1).strip()
    if not command.startswith("docker run "):
        raise QuickstartFailure(f"The README quickstart's first bash block is not a 'docker run': {command!r}")
    return command


def for_image(command: str, image: str) -> str:
    """Swap the published image for ``image``, the one change made to the command."""
    if command.count(PUBLISHED_IMAGE) != 1:
        raise QuickstartFailure(f"The README quickstart must name {PUBLISHED_IMAGE} exactly once")
    return command.replace(PUBLISHED_IMAGE, image)


def bootstrap_key(log: str) -> str:
    """Return the API key the first run printed, on its own line after the banner."""
    match = _BOOTSTRAP_KEY.search(log)
    if match is None:
        raise QuickstartFailure("The gateway printed no bootstrap API key")
    return match.group(1)


def _status(path: str, api_key: str | None = None) -> int:
    request = urllib.request.Request(f"{BASE_URL}{path}")
    if api_key is not None:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310 (a fixed loopback URL)
            return int(response.status)
    except urllib.error.HTTPError as exc:
        return exc.code


def _await_health(process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + HEALTH_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise QuickstartFailure(f"The quickstart command exited with code {process.returncode} before serving")
        try:
            if _status("/api/v1/health") == 200:
                return
        except OSError:
            pass
        time.sleep(0.5)
    raise QuickstartFailure(f"The gateway did not answer /api/v1/health within {HEALTH_TIMEOUT_SECONDS}s")


def _await_key(log_path: Path) -> str:
    # The key is printed during startup, but stdout reaches the file through
    # the docker client, so give a moment for it to land after health answers.
    deadline = time.monotonic() + 10
    while True:
        try:
            return bootstrap_key(log_path.read_text(encoding="utf-8", errors="replace"))
        except QuickstartFailure:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.5)


def redact(log: str) -> str:
    """Hide the keys the first run printed, so a failure's output carries none."""
    return _PRINTED_SECRET.sub("[redacted]", log)


def _containers_of(image: str) -> set[str]:
    listed = subprocess.run(
        ["docker", "ps", "--all", "--quiet", "--no-trunc", "--filter", f"ancestor={image}"],
        capture_output=True,
        check=False,
        text=True,
        timeout=DOCKER_TIMEOUT_SECONDS,
    )
    return set(listed.stdout.split())


def _remove(containers: set[str]) -> None:
    if containers:
        subprocess.run(
            ["docker", "rm", "--force", *sorted(containers)],
            capture_output=True,
            check=False,
            timeout=DOCKER_TIMEOUT_SECONDS,
        )


def check(image: str, log_path: Path) -> None:
    """Run the quickstart against ``image`` and walk the first run it describes."""
    command = for_image(quickstart_command(README.read_text(encoding="utf-8")), image)
    print(f"Running the README quickstart against {image}:\n{command}", flush=True)
    # The command names no container, so the one it starts is the one that was not there before it.
    already_there = _containers_of(image)
    with log_path.open("wb") as log_file:
        process = subprocess.Popen(["bash", "-c", command], stdout=log_file, stderr=subprocess.STDOUT)  # noqa: S603, S607
        try:
            _await_health(process)
            key = _await_key(log_path)
            if (status := _status("/api/v1/models")) != 401:
                raise QuickstartFailure(f"/api/v1/models answered {status} without a key, expected 401")
            if (status := _status("/api/v1/models", key)) != 200:
                raise QuickstartFailure(f"/api/v1/models answered {status} with the bootstrap key, expected 200")
        finally:
            _remove(_containers_of(image) - already_there)
            try:
                process.wait(timeout=DOCKER_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image", required=True, help="The image to run in place of the published one.")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory() as directory:
        log_path = Path(directory) / "quickstart.log"
        try:
            check(args.image, log_path)
        except QuickstartFailure as exc:
            log = redact(log_path.read_text(encoding="utf-8", errors="replace")) if log_path.exists() else ""
            print(f"README quickstart check failed: {exc}\n=== quickstart output ===\n{log}", file=sys.stderr)
            return 1
    print("The README quickstart starts and serves its first run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
