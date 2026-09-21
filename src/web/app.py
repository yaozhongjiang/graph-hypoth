"""The FastAPI application behind ``graph-hypoth-web``.

Every route is a thin adapter over ``RunManager`` and ``build_graph_view``; the browser page under
``static/`` does the rendering. Errors map to the usual codes: 400 for bad input, 404 for an unknown
run or file, 409 when the action does not fit the run's state (busy, not waiting).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import DEFAULT_CONFIG_PATH
from src.web.graph_view import build_graph_view
from src.web.runs import RunBusyError, RunManager, RunStateError

STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_PROFILE_PATH = Path("examples/research_profile.yaml")


class LaunchRequest(BaseModel):
    profile_path: str = Field(min_length=1)
    config_path: str = str(DEFAULT_CONFIG_PATH)
    elaborate: bool = True
    connected_render: bool = True
    confirm_in_browser: bool = False


class ImportRequest(BaseModel):
    run_dir: str = Field(min_length=1)


class ConfirmRequest(BaseModel):
    candidate_ids: list[str] | None = None
    choice: str | None = None  # "all" | "none"; ``candidate_ids`` wins when both are given


def _yaml_choices() -> list[str]:
    """Profiles and configs a user is likely to pick from, relative to the working directory."""
    found: list[str] = []
    patterns = (
        "examples/*.yaml",
        "config/*.yaml",
        "runtime_artifacts/*.yaml",
        "runtime_artifacts/*/*.yaml",
    )
    for pattern in patterns:
        found.extend(path.as_posix() for path in sorted(Path().glob(pattern)) if path.is_file())
    return found


def _config_choices(manager: RunManager, candidates: list[str]) -> list[dict[str, Any]]:
    """The loadable configs among ``candidates`` with their readiness (no placeholder model ids)."""
    choices: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate.startswith("examples/"):
            continue
        try:
            summary = manager.config_summary(Path(candidate))
        except (OSError, ValueError):
            continue  # a profile or another YAML that is not a system config
        choices.append(
            {
                "path": candidate,
                "ready": not summary["placeholders"],
                "placeholders": summary["placeholders"],
                "providers": summary["providers"],
                "models": sorted({entry["model_id"] for entry in summary["roles"].values()}),
            }
        )
    return choices


def _default_config(choices: list[dict[str, Any]], cli: dict[str, bool]) -> str:
    """A config that can actually run: the user's own copy first, then the CLI the machine has."""
    ready = [choice["path"] for choice in choices if choice["ready"]]
    for path in ready:
        if path.startswith("runtime_artifacts/"):
            return path
    preferred = []
    if cli.get("claude"):
        preferred.append("config/claude-code.yaml")
    if cli.get("codex"):
        preferred.append("config/codex.yaml")
    for path in preferred:
        if path in ready:
            return path
    if ready:
        return ready[0]
    return Path(DEFAULT_CONFIG_PATH).as_posix()


def create_app(manager: RunManager) -> FastAPI:
    app = FastAPI(
        title="GraphHypoth web app", docs_url="/api/docs", openapi_url="/api/openapi.json"
    )
    app.state.manager = manager

    def require(run_id: str) -> Any:
        record = manager.get(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"unknown run {run_id}")
        return record

    @app.get("/", include_in_schema=False)
    def index() -> HTMLResponse:
        # Version the asset URLs by their modification time so a browser never keeps a stale script
        # or stylesheet after the package is updated.
        version = int(max(path.stat().st_mtime for path in STATIC_DIR.iterdir() if path.is_file()))
        page = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        for asset in ("style.css", "app.js", "logo.svg"):
            page = page.replace(f"/static/{asset}", f"/static/{asset}?v={version}")
        return HTMLResponse(page)

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        active = manager.active_run()
        choices = _yaml_choices()
        cli = {"claude": shutil.which("claude") is not None, "codex": shutil.which("codex") is not None}
        config_choices = _config_choices(manager, choices)
        return {
            "runs_root": str(manager.runs_root),
            "cwd": str(Path.cwd()),
            "python": sys.version.split()[0],
            "active_run_id": active.run_id if active else None,
            "defaults": {
                "profile_path": DEFAULT_PROFILE_PATH.as_posix(),
                "config_path": _default_config(config_choices, cli),
            },
            "choices": choices,
            "config_choices": config_choices,
            "cli": cli,
        }

    @app.get("/api/runs")
    def list_runs() -> dict[str, Any]:
        return {"runs": [record.to_json() for record in manager.list_runs()]}

    @app.post("/api/runs", status_code=201)
    def launch(body: LaunchRequest) -> dict[str, Any]:
        try:
            record = manager.launch(
                profile_path=Path(body.profile_path),
                config_path=Path(body.config_path),
                elaborate=body.elaborate,
                connected_render=body.connected_render,
                confirm_in_browser=body.confirm_in_browser,
            )
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except RunBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return record.to_json()

    @app.post("/api/runs/import", status_code=201)
    def import_run(body: ImportRequest) -> dict[str, Any]:
        try:
            record = manager.import_run(Path(body.run_dir))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        return record.to_json()

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        return require(run_id).to_json()

    @app.get("/api/runs/{run_id}/events")
    def events(run_id: str, after: int = Query(0, ge=0)) -> dict[str, Any]:
        require(run_id)
        return manager.events(run_id, after=after)

    @app.get("/api/runs/{run_id}/confirmation")
    def confirmation(run_id: str) -> dict[str, Any]:
        record = require(run_id)
        pending = manager.pending_confirmation(run_id)
        if pending is None:
            raise HTTPException(
                status_code=409,
                detail=f"run {run_id} is {record.status}, not waiting for confirmation",
            )
        return pending

    @app.post("/api/runs/{run_id}/confirmation")
    def confirm(run_id: str, body: ConfirmRequest) -> dict[str, Any]:
        require(run_id)
        if body.candidate_ids is not None:
            candidate_ids = list(body.candidate_ids)
        elif body.choice == "all":
            pending = manager.pending_confirmation(run_id) or {}
            candidate_ids = [row["candidate_id"] for row in pending.get("candidates", [])]
        elif body.choice == "none":
            candidate_ids = []
        else:
            raise HTTPException(
                status_code=400, detail="send candidate_ids, or choice = 'all' | 'none'"
            )
        try:
            selected = manager.confirm(run_id, candidate_ids)
        except RunStateError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"run_id": run_id, "selected": selected}

    @app.get("/api/runs/{run_id}/graph")
    def graph(run_id: str) -> dict[str, Any]:
        record = require(run_id)
        logical_run_id = None if record.source == "imported" else record.run_id
        try:
            return build_graph_view(Path(record.run_dir), run_id=logical_run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/runs/{run_id}/artifacts")
    def artifacts(run_id: str) -> dict[str, Any]:
        require(run_id)
        return manager.artifacts(run_id)

    @app.get("/runs/{run_id}/files/{path:path}", include_in_schema=False)
    def run_file(run_id: str, path: str) -> FileResponse:
        require(run_id)
        try:
            target = manager.resolve_file(run_id, path)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"no such file: {exc}") from exc
        media_type = "text/markdown; charset=utf-8" if target.suffix == ".md" else None
        return FileResponse(target, media_type=media_type)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
