"""The web app's run manager drives the CLI's own run function on a worker thread and swaps the
stdin confirmation prompt for a browser-answered gate.

Hermetic: the pipeline is ``FakePipeline``. It writes the artifact files a real run exports, reports
progress through the real ``ProgressReporter`` the manager installs, and calls the injected confirm
hook exactly the way ``run_synthesist`` does.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from src.web.runs import RunBusyError, RunManager, RunStateError, candidate_row
from tests.unit.web_fakes import FakePipeline, scored_candidate, write_minimal_artifacts

PROFILE = Path("examples/research_profile.yaml")
CONFIG = Path("config/claude-code.yaml")  # model ids set; the shipped default is a template
TEMPLATE = Path("config/evidence-evaluation.yaml")


def wait_until(predicate, *, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition not met in time")


def _manager(tmp_path: Path, pipeline: FakePipeline) -> RunManager:
    return RunManager(tmp_path / "web", pipeline=pipeline, quiet=True)


def test_launch_runs_the_pipeline_in_the_background_and_records_the_result(tmp_path):
    pipeline = FakePipeline()
    manager = _manager(tmp_path, pipeline)

    record = manager.launch(profile_path=PROFILE, config_path=CONFIG)

    assert record.status in {"queued", "running"}
    assert record.seed.startswith("When does structured pruning")
    from src.research_profile import load_research_profile

    assert record.seed_kind == load_research_profile(PROFILE).seed_kind
    assert record.confirm_policy["mode"] == "top_k" and record.confirm_policy["k"] == 2
    assert record.models["builder"]["provider"] == "claude-cli"
    assert record.models["builder"]["model_id"] and "<" not in record.models["builder"]["model_id"]
    wait_until(lambda: manager.get(record.run_id).status == "completed")
    done = manager.get(record.run_id)
    assert done.summary["version"] == 3
    assert done.summary["surfaced"] == 1
    assert done.summary["committed_edges"] == 1
    assert done.summary["hypotheses"][0]["candidate_id"] == "h1"
    assert done.output == ["done: 2 hypotheses surfaced"]
    assert done.started_at and done.finished_at and done.error is None

    kwargs = pipeline.calls[0]
    run_dir = Path(record.run_dir)
    assert kwargs["run_id"] == kwargs["thread_id"] == record.run_id
    assert kwargs["export_dir"] == run_dir
    assert kwargs["trace_dir"] == run_dir / "trace"
    assert kwargs["log_store"].path == run_dir / "events.sqlite"
    assert kwargs["elaborate"] is True and kwargs["connected_render"] is True
    assert "confirm_fn" not in kwargs  # the profile's top_k policy applies, as on the CLI

    persisted = json.loads((run_dir / "web_run.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "completed" and persisted["is_active"] is False
    events = [
        json.loads(line)
        for line in (run_dir / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert events[0]["event"] == "run_started"
    assert events[-1]["event"] == "run_finished" and events[-1]["status"] == "completed"
    assert any(event["stage"] == "Retrieval" for event in events)
    page = manager.events(record.run_id)
    assert page["next"] == len(events)
    assert [e["event"] for e in page["events"]] == [e["event"] for e in events]
    assert manager.events(record.run_id, after=page["next"]) == {
        "events": [], "next": page["next"],
    }


def test_browser_confirmation_parks_the_worker_until_the_selection_arrives(tmp_path):
    pipeline = FakePipeline(surfaced=("h1", "h2", "h3"))
    manager = _manager(tmp_path, pipeline)

    record = manager.launch(
        profile_path=PROFILE, config_path=CONFIG, confirm_in_browser=True
    )

    wait_until(lambda: manager.get(record.run_id).status == "waiting_confirmation")
    pending = manager.pending_confirmation(record.run_id)
    rows = pending["candidates"]
    assert [row["candidate_id"] for row in rows] == ["h1", "h2", "h3"]
    assert [row["rank"] for row in rows] == [1, 2, 3]
    assert rows[0]["new_edges"][0]["sources"] == ["h1 cause"]  # ids mapped to labels
    assert pipeline.confirmed is None  # the worker is parked
    time.sleep(0.2)
    assert manager.get(record.run_id).status == "waiting_confirmation"

    selected = manager.confirm(record.run_id, ["h3", "h1"])

    assert selected == ["h1", "h3"]  # surfaced order, like the CLI hook
    wait_until(lambda: manager.get(record.run_id).status == "completed")
    assert pipeline.confirmed == ["h1", "h3"]
    done = manager.get(record.run_id)
    assert done.confirmation["selected"] == ["h1", "h3"]
    assert done.confirmation["resolved_at"]
    events = manager.events(record.run_id)["events"]
    assert any(
        e["status"] == "waiting" and e["stage"] == "Confirming hypotheses" for e in events
    )


def test_an_interactive_profile_policy_uses_the_browser_gate_by_itself(tmp_path):
    profile = tmp_path / "web" / "profile.yaml"
    profile.parent.mkdir(parents=True, exist_ok=True)
    profile.write_text("claim: x causes y\nconfirm:\n  mode: interactive\n", encoding="utf-8")
    pipeline = FakePipeline(surfaced=("h1",))
    manager = _manager(tmp_path, pipeline)

    record = manager.launch(profile_path=profile, config_path=CONFIG)

    assert record.confirm_policy["mode"] == "interactive"
    wait_until(lambda: manager.get(record.run_id).status == "waiting_confirmation")
    manager.confirm(record.run_id, [])
    wait_until(lambda: manager.get(record.run_id).status == "completed")
    assert pipeline.confirmed == []


def test_confirm_rejects_unknown_ids_and_the_wrong_state(tmp_path):
    pipeline = FakePipeline(surfaced=("h1", "h2"))
    manager = _manager(tmp_path, pipeline)
    record = manager.launch(
        profile_path=PROFILE, config_path=CONFIG, confirm_in_browser=True
    )
    wait_until(lambda: manager.get(record.run_id).status == "waiting_confirmation")

    with pytest.raises(ValueError, match="unknown candidate ids: zzz"):
        manager.confirm(record.run_id, ["zzz"])
    assert manager.get(record.run_id).status == "waiting_confirmation"  # still parked

    manager.confirm(record.run_id, ["h2"])
    wait_until(lambda: manager.get(record.run_id).status == "completed")
    assert pipeline.confirmed == ["h2"]
    with pytest.raises(RunStateError):
        manager.confirm(record.run_id, ["h1"])
    assert manager.pending_confirmation(record.run_id) is None


def test_a_second_launch_is_refused_while_a_run_is_active(tmp_path):
    release = threading.Event()
    manager = _manager(tmp_path, FakePipeline(gate=release))

    first = manager.launch(profile_path=PROFILE, config_path=CONFIG)

    wait_until(lambda: manager.get(first.run_id).status == "running")
    with pytest.raises(RunBusyError, match=first.run_id):
        manager.launch(profile_path=PROFILE, config_path=CONFIG)
    assert manager.active_run().run_id == first.run_id
    release.set()
    wait_until(lambda: manager.get(first.run_id).status == "completed")
    assert manager.active_run() is None

    second = manager.launch(profile_path=PROFILE, config_path=CONFIG)

    wait_until(lambda: manager.get(second.run_id).status == "completed")
    assert [r.run_id for r in manager.list_runs()] == [second.run_id, first.run_id]


def test_a_failing_pipeline_marks_the_run_failed_with_the_error(tmp_path):
    manager = _manager(tmp_path, FakePipeline(fail=True))

    record = manager.launch(profile_path=PROFILE, config_path=CONFIG)

    wait_until(lambda: manager.get(record.run_id).status == "failed")
    done = manager.get(record.run_id)
    assert done.error == "RuntimeError: boom" and done.finished_at
    assert "RuntimeError: boom" in (Path(record.run_dir) / "error.txt").read_text(encoding="utf-8")
    last = manager.events(record.run_id)["events"][-1]
    assert last["event"] == "run_finished" and last["status"] == "failed"
    assert manager.active_run() is None


def test_restart_marks_records_that_were_in_progress_as_interrupted(tmp_path):
    root = tmp_path / "web"
    (root / "r1").mkdir(parents=True)
    (root / "r1" / "web_run.json").write_text(
        json.dumps({
            "run_id": "r1", "created_at": "2026-09-16T10:00:00+00:00",
            "status": "waiting_confirmation", "run_dir": str(root / "r1"),
        }),
        encoding="utf-8",
    )
    (root / "broken").mkdir()
    (root / "broken" / "web_run.json").write_text("{not json", encoding="utf-8")

    manager = RunManager(root, pipeline=FakePipeline(), quiet=True)

    record = manager.get("r1")
    assert record.status == "interrupted" and not record.is_active
    assert "server stopped" in record.error
    assert manager.active_run() is None
    stored = json.loads((root / "r1" / "web_run.json").read_text(encoding="utf-8"))
    assert stored["status"] == "interrupted"
    assert [r.run_id for r in manager.list_runs()] == ["r1"]


def test_import_registers_an_existing_run_directory_and_survives_restart(tmp_path):
    manager = _manager(tmp_path, FakePipeline())
    external = tmp_path / "web" / "cli-run"
    write_minimal_artifacts(external)
    with pytest.raises(FileNotFoundError):
        manager.import_run(tmp_path / "web" / "nowhere")
    with pytest.raises(PermissionError, match="allowed roots"):
        manager.import_run(tmp_path / "secrets-outside")

    record = manager.import_run(external)

    assert record.status == "imported" and record.source == "imported"
    assert record.seed == "seed" and not record.is_active
    assert manager.import_run(external) is record  # idempotent
    again = RunManager(tmp_path / "web", pipeline=FakePipeline(), quiet=True)
    assert [r.run_id for r in again.list_runs()] == [record.run_id]
    assert again.artifacts(record.run_id)["pages"]["graph"] == "graph.json"
    assert not (external / "web_run.json").exists()  # nothing is written into a foreign dir


def test_launch_rejects_missing_files_and_unloadable_profiles(tmp_path):
    manager = _manager(tmp_path, FakePipeline())
    root = tmp_path / "web"

    with pytest.raises(FileNotFoundError, match="profile file does not exist"):
        manager.launch(profile_path=root / "missing.yaml", config_path=CONFIG)
    with pytest.raises(FileNotFoundError, match="config file does not exist"):
        manager.launch(profile_path=PROFILE, config_path=root / "missing.yaml")
    with pytest.raises(PermissionError, match="allowed roots"):
        manager.launch(profile_path=tmp_path / "outside.yaml", config_path=CONFIG)
    bad = root / "bad.yaml"
    bad.write_text("claim: x\nnot_a_field: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="profile could not be loaded"):
        manager.launch(profile_path=bad, config_path=CONFIG)
    assert manager.list_runs() == []  # nothing was recorded


def test_launch_refuses_a_template_config_before_starting_a_worker(tmp_path):
    manager = _manager(tmp_path, FakePipeline())

    with pytest.raises(ValueError, match="placeholders for builder"):
        manager.launch(profile_path=PROFILE, config_path=TEMPLATE)

    assert manager.list_runs() == []  # nothing was recorded and no thread started
    template = manager.config_summary(TEMPLATE)
    assert "builder" in template["placeholders"] and template["providers"] == ["claude-cli"]
    ready = manager.config_summary(CONFIG)
    assert ready["placeholders"] == [] and ready["roles"]["builder"]["provider"] == "claude-cli"
    assert manager.config_summary(CONFIG) is ready  # cached until the file changes
    with pytest.raises(ValueError, match="config could not be loaded"):
        manager.config_summary(PROFILE)  # a profile is not a system config


def test_candidate_row_projects_the_candidate_for_the_browser():
    row = candidate_row(2, scored_candidate("h7", field_novelty=0.8, saturation=0.1))

    assert row["rank"] == 2 and row["candidate_id"] == "h7"
    assert row["field_novelty"] == 0.8 and row["saturation"] == 0.1
    assert row["rank_score"] == pytest.approx(0.45) and row["hyp_score"] == 0.5
    assert row["cross_concept"] is True and row["common_sense"] is False
    assert [node["label"] for node in row["new_nodes"]] == ["h7 cause", "h7 effect"]
    assert row["new_edges"] == [{
        "sources": ["h7 cause"], "targets": ["h7 effect"], "relation_type": "increases",
        "direction": "causal", "mechanism": "via a shared pathway",
    }]
    assert row["mechanism_chain"][0]["from"] == "h7 cause"
    assert row["idea_scaffold"] == {"problem": "latency vs accuracy", "method": "ablation"}
    assert row["assumptions"] == ["the effect is measurable"]
    assert row["source_quotes"][0]["quote_span"] == "a verbatim span"
    assert row["lineage"] is None


def test_resolve_file_refuses_paths_outside_the_run_directory(tmp_path):
    manager = _manager(tmp_path, FakePipeline())
    external = tmp_path / "web" / "cli-run"
    write_minimal_artifacts(external)
    record = manager.import_run(external)

    assert manager.resolve_file(record.run_id, "graph.json") == external / "graph.json"
    with pytest.raises(PermissionError):
        manager.resolve_file(record.run_id, "../bad.yaml")
    with pytest.raises(FileNotFoundError):
        manager.resolve_file(record.run_id, "missing.html")
