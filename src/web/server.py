"""``graph-hypoth-web``: serve the local web app for the complete pipeline.

Binds to the loopback address by default. The pipeline's own progress lines still print on this
terminal (``--quiet`` silences them); the browser shows the same events from the run directory.
"""

from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from collections.abc import Sequence
from pathlib import Path

DEFAULT_RUNS_ROOT = Path("runtime_artifacts/web")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Serve the local web app: start runs from a research profile, confirm hypotheses in "
            "the browser, and browse each run's claim graph and report pages."
        )
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: loopback only)")
    parser.add_argument("--port", type=int, default=8765, help="TCP port (default: 8765)")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
        help="directory holding one subdirectory per run started here "
        "(default: runtime_artifacts/web)",
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="do not open the page in the default browser"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress the pipeline's progress lines on this terminal "
        "(the browser still shows them)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import uvicorn

        from src.web.app import create_app
    except ImportError as exc:
        print(
            f"error: the web app needs the `web` extra ({exc}).\n"
            'Install it with: python -m pip install -e ".[web]"',
            file=sys.stderr,
        )
        return 1
    from src.web.runs import RunManager, is_loopback_host

    manager = RunManager(args.runs_root, quiet=args.quiet)
    app = create_app(manager)
    url = f"http://{args.host}:{args.port}/"
    print(f"GraphHypoth web app at {url} (runs are saved under {manager.runs_root})", flush=True)
    if not is_loopback_host(args.host):
        print(
            "WARNING: bound beyond loopback — the API has no authentication. "
            "Profiles/configs/imports are sandboxed to runs_root plus cwd "
            "examples/, config/, and runtime_artifacts/, but anyone who can "
            "reach this host can still list runs, start jobs from those roots, "
            "and read imported artifacts. Prefer --host 127.0.0.1.",
            file=sys.stderr,
            flush=True,
        )
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
