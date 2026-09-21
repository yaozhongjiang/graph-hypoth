"""The FastAPI routes are thin adapters over the run manager; these tests drive the same flows over
HTTP with the hermetic fake pipeline. Skipped when the ``web`` extra (or httpx) is not installed."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from src.web.app import create_app
from src.web.runs import RunManager
from tests.unit.web_fakes import FakePipeline, write_minimal_artifacts

PROFILE = "examples/research_profile.yaml"
CONFIG = "config/claude-code.yaml"  # model ids set; the shipped default is a template
TEMPLATE = "config/evidence-evaluation.yaml"


def _client(tmp_path: Path, pipeline: FakePipeline) -> TestClient:
    manager = RunManager(tmp_path / "web", pipeline=pipeline, quiet=True)
    return TestClient(create_app(manager))


def _wait(client: TestClient, run_id: str, status: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = client.get(f"/api/runs/{run_id}").json()
        if record["status"] == status:
            return record
        time.sleep(0.02)
    raise AssertionError(f"run {run_id} never reached {status}")


def test_status_page_and_empty_run_list(tmp_path):
    client = _client(tmp_path, FakePipeline())

    status = client.get("/api/status").json()

    assert status["active_run_id"] is None
    assert status["defaults"]["profile_path"] == PROFILE
    assert PROFILE in status["choices"] and CONFIG in status["choices"]
    by_path = {choice["path"]: choice for choice in status["config_choices"]}
    assert PROFILE not in by_path  # profiles are offered but are not configs
    assert by_path[TEMPLATE]["ready"] is False and "builder" in by_path[TEMPLATE]["placeholders"]
    assert by_path[CONFIG]["ready"] is True and by_path[CONFIG]["providers"] == ["claude-cli"]
    assert by_path[status["defaults"]["config_path"]]["ready"] is True
    template = client.post("/api/runs", json={"profile_path": PROFILE, "config_path": TEMPLATE})
    assert template.status_code == 400 and "placeholders" in template.json()["detail"]
    assert client.get("/api/runs").json() == {"runs": []}
    page = client.get("/")
    assert page.status_code == 200 and "GraphHypoth" in page.text
    assert client.get("/static/app.js").status_code == 200


def test_launch_follow_and_browse_a_run(tmp_path):
    pipeline = FakePipeline()
    client = _client(tmp_path, pipeline)

    created = client.post("/api/runs", json={"profile_path": PROFILE, "config_path": CONFIG})

    assert created.status_code == 201, created.text
    run_id = created.json()["run_id"]
    record = _wait(client, run_id, "completed")
    assert record["summary"]["version"] == 3
    assert client.get("/api/status").json()["active_run_id"] is None

    events = client.get(f"/api/runs/{run_id}/events").json()
    assert events["events"][0]["event"] == "run_started" and events["next"] > 0
    later = client.get(f"/api/runs/{run_id}/events", params={"after": events["next"]}).json()
    assert later == {"events": [], "next": events["next"]}

    artifacts = client.get(f"/api/runs/{run_id}/artifacts").json()
    assert artifacts["pages"]["graph"] == "graph.json"
    assert artifacts["pages"]["trace_index"] == "trace/index.html"
    memo = client.get(f"/runs/{run_id}/files/audit_memo.md")
    assert memo.status_code == 200 and memo.text.startswith("# audit memo")
    assert memo.headers["content-type"].startswith("text/markdown")
    assert client.get(f"/runs/{run_id}/files/trace/index.html").status_code == 200
    assert client.get(f"/runs/{run_id}/files/missing.html").status_code == 404
    escaped = client.get(f"/runs/{run_id}/files/..%2F..%2Fpyproject.toml")
    assert escaped.status_code in {403, 404} and "[project]" not in escaped.text

    graph = client.get(f"/api/runs/{run_id}/graph").json()
    assert graph["nodes"] == [] and graph["edges"] == [] and graph["seed"] == "seed"
    assert client.get(f"/api/runs/{run_id}/confirmation").status_code == 409
    assert client.get("/api/runs/nope").status_code == 404


def test_confirmation_flow_over_http(tmp_path):
    pipeline = FakePipeline(surfaced=("h1", "h2"))
    client = _client(tmp_path, pipeline)
    run_id = client.post(
        "/api/runs",
        json={"profile_path": PROFILE, "config_path": CONFIG, "confirm_in_browser": True},
    ).json()["run_id"]
    _wait(client, run_id, "waiting_confirmation")

    pending = client.get(f"/api/runs/{run_id}/confirmation").json()
    assert [c["candidate_id"] for c in pending["candidates"]] == ["h1", "h2"]
    busy = client.post("/api/runs", json={"profile_path": PROFILE, "config_path": CONFIG})
    assert busy.status_code == 409 and run_id in busy.json()["detail"]
    bad = client.post(f"/api/runs/{run_id}/confirmation", json={"candidate_ids": ["nope"]})
    assert bad.status_code == 400 and "nope" in bad.json()["detail"]
    assert client.post(f"/api/runs/{run_id}/confirmation", json={}).status_code == 400

    answer = client.post(f"/api/runs/{run_id}/confirmation", json={"choice": "all"})

    assert answer.status_code == 200 and answer.json()["selected"] == ["h1", "h2"]
    _wait(client, run_id, "completed")
    assert pipeline.confirmed == ["h1", "h2"]
    late = client.post(f"/api/runs/{run_id}/confirmation", json={"choice": "none"})
    assert late.status_code == 409


def test_launch_validation_and_import(tmp_path):
    client = _client(tmp_path, FakePipeline())

    missing = client.post(
        "/api/runs",
        json={"profile_path": "examples/nope-missing.yaml", "config_path": CONFIG},
    )
    assert missing.status_code == 400 and "does not exist" in missing.json()["detail"]
    assert client.post("/api/runs", json={"config_path": CONFIG}).status_code == 422

    # Imports must stay under an allowed root (runs_root / examples / config / runtime_artifacts).
    external = tmp_path / "web" / "cli-run"
    write_minimal_artifacts(external)
    imported = client.post("/api/runs/import", json={"run_dir": str(external)})
    assert imported.status_code == 201 and imported.json()["status"] == "imported"
    run_id = imported.json()["run_id"]
    assert client.get(f"/api/runs/{run_id}/graph").json()["seed"] == "seed"
    assert client.get(f"/api/runs/{run_id}/events").json() == {"events": [], "next": 0}
    nowhere = client.post("/api/runs/import", json={"run_dir": str(tmp_path / "web" / "nowhere")})
    assert nowhere.status_code == 400

    outside = tmp_path / "secrets"
    write_minimal_artifacts(outside)
    (outside / "credentials.env").write_text("SECRET=hunter2\n", encoding="utf-8")
    denied = client.post("/api/runs/import", json={"run_dir": str(outside)})
    assert denied.status_code == 403
    assert "allowed roots" in denied.json()["detail"]


def test_launch_rejects_profile_outside_allowed_roots(tmp_path):
    client = _client(tmp_path, FakePipeline())
    evil = tmp_path / "evil-profile.yaml"
    evil.write_text("seed_claim: x\n", encoding="utf-8")
    denied = client.post(
        "/api/runs",
        json={"profile_path": str(evil), "config_path": CONFIG},
    )
    assert denied.status_code == 403
    assert "allowed roots" in denied.json()["detail"]
