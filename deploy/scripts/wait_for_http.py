"""Wait for an HTTP endpoint to answer with a success status.

A tiny stdlib-only helper the container entrypoints use to make startup robust
(the Compose healthchecks already order services; this is belt-and-suspenders
for the Prefect worker). Usage:

    python wait_for_http.py <url> [timeout_seconds]
"""

from __future__ import annotations

import sys
import time
import urllib.error
import urllib.request

_DEFAULT_TIMEOUT_SECONDS = 120.0
_POLL_SECONDS = 2.0


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: wait_for_http.py <url> [timeout_seconds]", file=sys.stderr)
        return 2
    url = sys.argv[1]
    timeout = float(sys.argv[2]) if len(sys.argv) > 2 else _DEFAULT_TIMEOUT_SECONDS

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status < 400:
                    print(f"{url} is up")
                    return 0
        except (urllib.error.URLError, OSError, ValueError):
            pass
        time.sleep(_POLL_SECONDS)

    print(f"timed out waiting for {url}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
