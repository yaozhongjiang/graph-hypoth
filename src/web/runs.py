"""Background runs for the local web app: launch, watch, confirm, list, and serve artifacts.

One ``RunManager`` owns every run the server started. A run is the same ``run_synthesist`` call the
CLI makes, executed on a worker thread with the CLI's own ``ProgressReporter`` appending
``progress.jsonl`` inside the run directory; the browser tails that file. The only pipeline seam the
web layer uses is the confirmation hook: instead of ``input()`` on stdin, the hook parks the worker on
an event until the browser posts a selection. The deterministic gates, receipts, and exports are the
CLI's, untouched. Records persist as ``web_run.json`` next to the artifacts, so the run list survives
a server restart; a run that was in progress when the server stopped is shown as interrupted.
"""

from __future__ import annotations

import dataclasses
import json
import os
import threading
import traceback
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

RECORD_FILE = "web_run.json"
IMPORTS_FILE = "imports.json"
ACTIVE_STATUSES = frozenset({"queued", "running", "waiting_confirmation"})
MAX_OUTPUT_LINES = 400
MAX_EVENTS_PER_PAGE = 500
# Paths the unauthenticated local API may read for launch/import. Anything outside
# these roots is rejected so a mis-bound host cannot turn the API into arbitrary LFI.
_ALLOWED_PATH_ROOT_NAMES: tuple[str, ...] = ("examples", "config", "runtime_artifacts")
# The roles a complete run may build a model backend for (see ``run_synthesist``).
WORKFLOW_ROLES: tuple[str, ...] = (
    "builder",
    "skeptical_verifier",
    "evidence_reviewer",
    "research_synthesist",
    "critic_panel",
    "experiment_designer",
    "experiment_validator",
    "elaboration_writer",
    "reader_translator",
    "translation_verifier",
)


def _is_placeholder(model_id: Any) -> bool:
    """An unedited template value such as ``<MODEL_ID>`` (the pipeline's own preflight rule)."""
    text = str(model_id or "").strip()
    return text.startswith("<") and text.endswith(">")


def describe_config(config_path: Path) -> dict[str, Any]:
    """Provider and model id per workflow role, plus the roles whose id is still a placeholder.

    Mirrors the pipeline's preflight so a template such as the shipped
    ``config/evidence-evaluation.yaml`` is refused before a worker starts, instead of failing a
    second later with a traceback. ``ValueError`` when the file is not a loadable config.
    """
    from src.config import load_config

    try:
        config = load_config(config_path)
    except Exception as exc:  # any loader failure is the caller's 400, not a crash
        raise ValueError(f"config could not be loaded: {exc}") from exc
    roles: dict[str, dict[str, str]] = {}
    placeholders: list[str] = []
    for role in WORKFLOW_ROLES:
        try:
            model = config.agents.for_role(role).model
        except ValueError:
            continue
        if model is None or not getattr(model, "model_id", None):
            continue
        roles[role] = {"provider": str(model.provider), "model_id": str(model.model_id)}
        if _is_placeholder(model.model_id):
            placeholders.append(role)
    return {
        "roles": roles,
        "placeholders": placeholders,
        "providers": sorted({entry["provider"] for entry in roles.values()}),
    }


class RunBusyError(RuntimeError):
    """Another run is active; the app runs one pipeline at a time."""


class RunStateError(RuntimeError):
    """The requested action does not fit the run's current status."""


def _now() -> str:
    # Microseconds keep the run list ordered even when two records land in the same second.
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _new_run_id() -> str:
    return f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid4().hex[:6]}"


def path_is_under(path: Path, root: Path) -> bool:
    """True when ``path`` is ``root`` or a descendant (after resolve)."""
    resolved = path.expanduser().resolve()
    base = root.expanduser().resolve()
    return resolved == base or base in resolved.parents


def is_loopback_host(host: str) -> bool:
    """True for bind addresses that stay on the local machine."""
    lowered = (host or "").strip().lower()
    if lowered in {"127.0.0.1", "::1", "localhost"}:
        return True
    try:
        import ipaddress

        return ipaddress.ip_address(lowered).is_loopback
    except ValueError:
        return False


@dataclass
class RunRecord:
    """Everything the browser needs to list and follow a run; persisted as JSON."""

    run_id: str
    created_at: str
    status: str
    run_dir: str
    source: str = "web"  # "web" (started here) | "imported" (an existing directory)
    profile_path: str = ""
    config_path: str = ""
    options: dict[str, Any] = field(default_factory=dict)
    seed: str = ""
    seed_kind: str | None = None
    confirm_policy: dict[str, Any] = field(default_factory=dict)
    models: dict[str, Any] = field(default_factory=dict)  # role -> {provider, model_id}
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    summary: dict[str, Any] | None = None
    confirmation: dict[str, Any] | None = None
    output: list[str] = field(default_factory=list)

    @property
    def is_active(self) -> bool:
        return self.status in ACTIVE_STATUSES

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["is_active"] = self.is_active
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> RunRecord:
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known})


# --- candidate projection for the confirmation screen -------------------------------------------
def _lineage_of(candidate: Any) -> dict[str, Any] | None:
    """A derived candidate's parents and strategy (mirrors the trace report), else ``None``."""
    provenance = getattr(candidate, "provenance", ()) or ()
    first = provenance[0] if provenance else None
    if not isinstance(first, dict) or "wasDerivedFrom" not in first:
        return None
    return {"wasDerivedFrom": first.get("wasDerivedFrom"), "strategy": first.get("strategy")}


def candidate_row(rank: int, scored: Any) -> dict[str, Any]:
    """One surfaced ``ScoredCandidate`` as the confirmation screen shows it.

    Carries the same ranking fields the CLI prints plus the content a researcher needs to decide:
    the coined concepts with definitions, the proposed edges (ids mapped to labels), the mechanism
    chain, rationale, scaffold, assumptions, and grounding quotes. Tolerant of partial objects.
    """
    candidate = scored.candidate
    labels = {
        str(getattr(node, "node_id", "")): str(getattr(node, "label", ""))
        for node in getattr(candidate, "new_nodes", ()) or ()
    }

    def name(node_id: Any) -> str:
        return labels.get(str(node_id)) or str(node_id)

    return {
        "rank": rank,
        "candidate_id": str(candidate.candidate_id),
        "field_novelty": float(getattr(candidate, "ranking_novelty", 0.0) or 0.0),
        "saturation": float(getattr(candidate, "saturation", 0.0) or 0.0),
        "hyp_score": float(getattr(scored, "hyp_score", 0.0) or 0.0),
        "rank_score": float(getattr(scored, "rank_score", 0.0) or 0.0),
        "cross_concept": bool(getattr(candidate, "cross_concept", False)),
        "common_sense": bool(getattr(candidate, "common_sense", False)),
        "rationale": str(getattr(candidate, "rationale", "") or ""),
        "new_nodes": [
            {
                "label": str(getattr(node, "label", "")),
                "type": str(getattr(node, "type", "")),
                "definition": str(getattr(node, "definition", "")),
            }
            for node in getattr(candidate, "new_nodes", ()) or ()
        ],
        "new_edges": [
            {
                "sources": [name(s) for s in getattr(edge, "source_node_ids", ()) or ()],
                "targets": [name(t) for t in getattr(edge, "target_node_ids", ()) or ()],
                "relation_type": str(getattr(edge, "relation_type", "") or ""),
                "direction": str(getattr(edge, "direction", "") or ""),
                "mechanism": str(getattr(edge, "mechanism", "") or ""),
            }
            for edge in getattr(candidate, "new_edges", ()) or ()
        ],
        "mechanism_chain": [dict(step) for step in getattr(candidate, "mechanism_chain", ()) or ()],
        "assumptions": [str(a) for a in getattr(candidate, "assumptions", ()) or ()],
        "idea_scaffold": {
            str(key): str(value)
            for key, value in (getattr(candidate, "idea_scaffold", {}) or {}).items()
            if value
        },
        "source_quotes": [dict(quote) for quote in getattr(candidate, "source_quotes", ()) or ()],
        "lineage": _lineage_of(candidate),
    }


class ConfirmationGate:
    """Parks the worker thread until the browser answers.

    ``confirm`` has the pipeline's ``ConfirmFn`` shape and runs on the worker thread; ``normalize``
    and ``release`` run on request threads. The manager updates the record between the two so the
    browser never sees the worker finish before the selection is on disk.
    """

    def __init__(self, on_wait: Callable[[list[dict[str, Any]]], None]) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._on_wait = on_wait
        self.candidates: list[dict[str, Any]] = []
        self.selection: list[str] | None = None

    def confirm(self, surfaced: Sequence[Any]) -> list[str]:
        from src.progress import report_progress

        if not surfaced:
            return []
        rows = [candidate_row(rank, scored) for rank, scored in enumerate(surfaced, start=1)]
        with self._lock:
            self.candidates = rows
            self.selection = None
            self._event.clear()
        self._on_wait(rows)
        report_progress(
            "Confirming hypotheses", "waiting for a selection in the browser", status="waiting"
        )
        self._event.wait()
        report_progress("Confirming hypotheses", "applying the browser selection")
        with self._lock:
            wanted = set(self.selection or [])
        return [row["candidate_id"] for row in rows if row["candidate_id"] in wanted]

    def normalize(self, candidate_ids: Sequence[str]) -> list[str]:
        """The selection in surfaced order; rejects ids that are not pending."""
        with self._lock:
            known = [row["candidate_id"] for row in self.candidates]
        unknown = [cid for cid in candidate_ids if cid not in known]
        if unknown:
            raise ValueError(f"unknown candidate ids: {', '.join(unknown)}")
        wanted = set(candidate_ids)
        return [cid for cid in known if cid in wanted]

    def release(self, selection: Sequence[str]) -> None:
        with self._lock:
            self.selection = list(selection)
            self._event.set()


def _no_stdin(prompt: str) -> str:
    raise RuntimeError(
        "the web app does not read stdin; confirm in the browser or use a non-interactive "
        f"confirm policy (prompt was: {prompt!r})"
    )


def _summarize(result: Any) -> dict[str, Any]:
    surfaced = list(getattr(result, "surfaced", ()) or ())
    edge_table = list(getattr(result, "edge_table", ()) or ())
    return {
        "version": getattr(result, "version", None),
        "surfaced": len(surfaced),
        "committed_edges": sum(1 for row in edge_table if getattr(row, "status", "")),
        "open_risks": len(list(getattr(result, "open_risks", ()) or ())),
        "hypotheses": [
            {
                "rank": getattr(row, "rank", index),
                "candidate_id": str(getattr(row, "candidate_id", "")),
                "rank_score": getattr(row, "rank_score", None),
                "experiment_plan": bool(getattr(row, "experiment_plan", None)),
                "hypothesis_edge_ids": list(getattr(row, "hypothesis_edge_ids", ()) or ()),
            }
            for index, row in enumerate(surfaced, start=1)
        ],
    }


def _default_pipeline() -> Callable[..., Any]:
    from src.synthesist_run import run_synthesist

    return run_synthesist


class RunManager:
    """Owns the runs the server started, their worker threads, and their confirmation gates.

    ``pipeline`` defaults to ``run_synthesist`` and is injectable for tests. ``quiet`` silences the
    pipeline's progress lines on the server's stderr (the browser still sees them via the file).
    """

    def __init__(
        self,
        runs_root: Path,
        *,
        pipeline: Callable[..., Any] | None = None,
        quiet: bool = False,
    ) -> None:
        self.runs_root = Path(runs_root).expanduser().resolve()
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self._pipeline = pipeline
        self.quiet = quiet
        self._lock = threading.RLock()
        self._records: dict[str, RunRecord] = {}
        self._gates: dict[str, ConfirmationGate] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._config_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._load_existing()

    def allowed_path_roots(self) -> list[Path]:
        """Directories the API may read for profiles, configs, and imported run dirs."""
        roots = [self.runs_root]
        cwd = Path.cwd().resolve()
        for name in _ALLOWED_PATH_ROOT_NAMES:
            candidate = (cwd / name).resolve()
            if candidate not in roots:
                roots.append(candidate)
        return roots

    def ensure_allowed_path(self, path: Path, *, kind: str) -> Path:
        """Resolve ``path`` and refuse anything outside ``allowed_path_roots``."""
        resolved = Path(path).expanduser().resolve()
        if not any(path_is_under(resolved, root) for root in self.allowed_path_roots()):
            allowed = ", ".join(str(root) for root in self.allowed_path_roots())
            raise PermissionError(
                f"{kind} path escapes the allowed roots ({allowed}): {resolved}"
            )
        return resolved

    # --- persistence -----------------------------------------------------------------------
    def _load_existing(self) -> None:
        for record_path in sorted(self.runs_root.glob(f"*/{RECORD_FILE}")):
            record = self._read_record(record_path)
            if record is None:
                continue
            if record.is_active:  # no thread owns it any more: the server stopped mid-run
                record.status = "interrupted"
                record.error = record.error or "the server stopped while this run was in progress"
                record.finished_at = record.finished_at or _now()
                self._persist(record)
            self._records[record.run_id] = record
        imports_path = self.runs_root / IMPORTS_FILE
        if imports_path.is_file():
            try:
                entries = json.loads(imports_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                entries = []
            for entry in entries if isinstance(entries, list) else []:
                try:
                    record = RunRecord.from_json(entry)
                except (TypeError, ValueError):
                    continue
                if (Path(record.run_dir) / "graph.json").is_file():
                    self._records[record.run_id] = record

    @staticmethod
    def _read_record(path: Path) -> RunRecord | None:
        try:
            return RunRecord.from_json(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            return None

    def _persist(self, record: RunRecord) -> None:
        if record.source == "imported":
            self._save_imports()
            return
        run_dir = Path(record.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        target = run_dir / RECORD_FILE
        temp = run_dir / f"{RECORD_FILE}.tmp"
        temp.write_text(json.dumps(record.to_json(), indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temp, target)  # readers never see a half-written record

    def _save_imports(self) -> None:
        entries = [
            record.to_json()
            for record in self._records.values()
            if record.source == "imported"
        ]
        temp = self.runs_root / f"{IMPORTS_FILE}.tmp"
        temp.write_text(json.dumps(entries, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temp, self.runs_root / IMPORTS_FILE)

    def _update(self, run_id: str, **fields: Any) -> RunRecord:
        with self._lock:
            record = self._records[run_id]
            for key, value in fields.items():
                setattr(record, key, value)
            self._persist(record)
            return record

    def _append_output(self, run_id: str, text: str) -> None:
        with self._lock:
            record = self._records[run_id]
            record.output.extend(str(text).splitlines() or [""])
            del record.output[:-MAX_OUTPUT_LINES]
            self._persist(record)

    # --- queries ---------------------------------------------------------------------------
    def list_runs(self) -> list[RunRecord]:
        with self._lock:
            return sorted(self._records.values(), key=lambda r: r.created_at, reverse=True)

    def get(self, run_id: str) -> RunRecord | None:
        with self._lock:
            return self._records.get(run_id)

    def _require(self, run_id: str) -> RunRecord:
        record = self.get(run_id)
        if record is None:
            raise KeyError(run_id)
        return record

    def active_run(self) -> RunRecord | None:
        with self._lock:
            for record in self._records.values():
                if record.is_active:
                    return record
        return None

    # --- launching -------------------------------------------------------------------------
    @staticmethod
    def describe_profile(profile_path: Path) -> dict[str, Any]:
        """The seed text, seed kind, and confirm policy of a profile; ``ValueError`` if unloadable."""
        from src.research_profile import load_research_profile

        try:
            profile = load_research_profile(profile_path)
        except Exception as exc:  # any loader failure is the caller's 400, not a crash
            raise ValueError(f"profile could not be loaded: {exc}") from exc
        seed = (
            profile.research_question
            or profile.claim
            or profile.seed_input
            or profile.interest
            or ""
        )
        return {
            "seed": str(seed),
            "seed_kind": profile.seed_kind,
            "confirm_policy": profile.confirm.model_dump(exclude_none=True),
        }

    def config_summary(self, config_path: Path) -> dict[str, Any]:
        """``describe_config`` with a per-file cache keyed on modification time."""
        path = Path(config_path).expanduser().resolve()
        stamp = path.stat().st_mtime
        with self._lock:
            cached = self._config_cache.get(str(path))
            if cached is not None and cached[0] == stamp:
                return cached[1]
        summary = describe_config(path)
        with self._lock:
            self._config_cache[str(path)] = (stamp, summary)
        return summary

    def launch(
        self,
        *,
        profile_path: Path,
        config_path: Path,
        elaborate: bool = True,
        connected_render: bool = True,
        confirm_in_browser: bool = False,
    ) -> RunRecord:
        """Start the complete pipeline on a worker thread and return its record immediately.

        Raises ``FileNotFoundError`` for a missing file, ``ValueError`` for an unloadable profile
        or config (including a template whose model ids are still placeholders), and
        ``RunBusyError`` while another run is active.
        """
        profile_path = Path(profile_path).expanduser().resolve()
        config_path = Path(config_path).expanduser().resolve()
        profile_path = self.ensure_allowed_path(profile_path, kind="profile")
        config_path = self.ensure_allowed_path(config_path, kind="config")
        if not profile_path.is_file():
            raise FileNotFoundError(f"profile file does not exist: {profile_path}")
        if not config_path.is_file():
            raise FileNotFoundError(f"config file does not exist: {config_path}")
        described = self.describe_profile(profile_path)
        config = self.config_summary(config_path)
        if config["placeholders"]:
            raise ValueError(
                f"{config_path.name} is a template: its model ids are still placeholders for "
                f"{', '.join(config['placeholders'])}. Choose config/claude-code.yaml or "
                "config/codex.yaml, whose model ids are set for a saved CLI login, or copy a "
                "config under runtime_artifacts/ and set real model ids as the README's "
                "settings 2 and 3 describe."
            )
        with self._lock:
            active = self.active_run()
            if active is not None:
                raise RunBusyError(
                    f"run {active.run_id} is still {active.status.replace('_', ' ')}; "
                    "the app runs one pipeline at a time"
                )
            run_id = _new_run_id()
            record = RunRecord(
                run_id=run_id,
                created_at=_now(),
                status="queued",
                run_dir=str(self.runs_root / run_id),
                profile_path=str(profile_path),
                config_path=str(config_path),
                options={
                    "elaborate": bool(elaborate),
                    "connected_render": bool(connected_render),
                    "confirm_in_browser": bool(confirm_in_browser),
                },
                models=config["roles"],
                **described,
            )
            self._records[run_id] = record
            self._persist(record)
            gate = ConfirmationGate(on_wait=lambda rows: self._enter_waiting(run_id, rows))
            self._gates[run_id] = gate
            thread = threading.Thread(
                target=self._execute, args=(run_id, gate), name=f"graph-hypoth-web-{run_id}",
                daemon=True,
            )
            self._threads[run_id] = thread
            thread.start()
        return record

    def _enter_waiting(self, run_id: str, rows: list[dict[str, Any]]) -> None:
        self._update(
            run_id,
            status="waiting_confirmation",
            confirmation={
                "candidates": rows,
                "requested_at": _now(),
                "selected": None,
                "resolved_at": None,
            },
        )

    def _execute(self, run_id: str, gate: ConfirmationGate) -> None:
        from src.log_store import SQLiteLogStore
        from src.progress import ProgressReporter

        record = self._update(run_id, status="running", started_at=_now())
        run_dir = Path(record.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        kwargs: dict[str, Any] = {
            "profile_path": record.profile_path,
            "config_path": record.config_path,
            "log_store": SQLiteLogStore(run_dir / "events.sqlite"),
            "run_id": run_id,
            "thread_id": run_id,
            "export_dir": run_dir,
            "trace_dir": run_dir / "trace",
            "elaborate": bool(record.options.get("elaborate", True)),
            "connected_render": bool(record.options.get("connected_render", True)),
            "input_fn": _no_stdin,
            "output_fn": lambda text: self._append_output(run_id, text),
        }
        # The browser gate replaces the stdin prompt whenever the profile would have prompted, and
        # it overrides an automatic policy when the researcher asked to decide in the browser.
        if record.options.get("confirm_in_browser") or (
            record.confirm_policy.get("mode") == "interactive"
        ):
            kwargs["confirm_fn"] = gate.confirm
        pipeline = self._pipeline or _default_pipeline()
        try:
            with ProgressReporter(run_id, quiet=self.quiet, jsonl_path=run_dir / "progress.jsonl"):
                result = pipeline(**kwargs)
        except (Exception, SystemExit) as exc:  # noqa: BLE001 -- the browser must see any failure
            try:
                (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            except OSError:
                pass
            self._update(
                run_id,
                status="failed",
                finished_at=_now(),
                error=f"{type(exc).__name__}: {exc}",
            )
            return
        self._update(run_id, status="completed", finished_at=_now(), summary=_summarize(result))

    # --- confirmation ----------------------------------------------------------------------
    def pending_confirmation(self, run_id: str) -> dict[str, Any] | None:
        record = self._require(run_id)
        if record.status != "waiting_confirmation" or record.confirmation is None:
            return None
        return record.confirmation

    def confirm(self, run_id: str, candidate_ids: Sequence[str]) -> list[str]:
        """Apply the browser's selection; returns the confirmed ids in surfaced order.

        ``RunStateError`` when the run is not waiting, ``ValueError`` for unknown ids.
        """
        with self._lock:
            record = self._require(run_id)
            gate = self._gates.get(run_id)
            if record.status != "waiting_confirmation" or gate is None:
                raise RunStateError(f"run {run_id} is not waiting for confirmation")
            selection = gate.normalize(candidate_ids)
            confirmation = dict(record.confirmation or {})
            confirmation.update(selected=selection, resolved_at=_now())
            self._update(run_id, status="running", confirmation=confirmation)
            gate.release(selection)
        return selection

    # --- imports ---------------------------------------------------------------------------
    def import_run(self, run_dir: Path) -> RunRecord:
        """Register an existing run directory (a CLI run, say) so its graph and pages can be browsed.

        The directory must lie under an allowed root (``runs_root``, or cwd
        ``examples`` / ``config`` / ``runtime_artifacts``) so the unauthenticated
        files API cannot be pointed at arbitrary filesystem trees.
        """
        run_dir = self.ensure_allowed_path(Path(run_dir), kind="import")
        graph_path = run_dir / "graph.json"
        if not graph_path.is_file():
            raise FileNotFoundError(f"no graph.json under {run_dir}")
        with self._lock:
            for record in self._records.values():
                if Path(record.run_dir) == run_dir:
                    return record
            try:
                scope = json.loads(graph_path.read_text(encoding="utf-8")).get("scope_context") or {}
            except (OSError, ValueError, AttributeError):
                scope = {}
            finished = datetime.fromtimestamp(graph_path.stat().st_mtime, UTC).isoformat(
                timespec="seconds"
            )
            record = RunRecord(
                run_id=f"imported-{run_dir.name}-{uuid4().hex[:6]}",
                created_at=_now(),
                status="imported",
                run_dir=str(run_dir),
                source="imported",
                seed=str(scope.get("seed_claim") or ""),
                finished_at=finished,
            )
            self._records[record.run_id] = record
            self._persist(record)
        return record

    # --- artifacts -------------------------------------------------------------------------
    def events(self, run_id: str, after: int = 0) -> dict[str, Any]:
        """Progress rows ``after`` the given line index (a cursor the browser hands back).

        Only complete lines count, so a row the reporter is still writing is delivered next time.
        """
        record = self._require(run_id)
        path = Path(record.run_dir) / "progress.jsonl"
        rows: list[dict[str, Any]] = []
        cursor = after
        if not path.is_file():
            return {"events": rows, "next": cursor}
        with path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if not line.endswith("\n"):
                    break
                if index < after:
                    continue
                cursor = index + 1
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
                if len(rows) >= MAX_EVENTS_PER_PAGE:
                    break
        return {"events": rows, "next": cursor}

    def artifacts(self, run_id: str) -> dict[str, Any]:
        """The files under the run directory plus the named pages the browser links to."""
        record = self._require(run_id)
        base = Path(record.run_dir)
        files: list[dict[str, Any]] = []
        if base.is_dir():
            for path in sorted(base.rglob("*")):
                if not path.is_file() or path.name.endswith(".tmp"):
                    continue
                relative = path.relative_to(base).as_posix()
                if len(relative.split("/")) > 3:
                    continue
                stat = path.stat()
                files.append({"path": relative, "size": stat.st_size, "modified": stat.st_mtime})
        present = {entry["path"] for entry in files}
        pages = {
            "trace_index": "trace/index.html" if "trace/index.html" in present else None,
            "connected": sorted(
                p for p in present if p.endswith("-connected.html") and "/" not in p
            ),
            "audit_memo": "audit_memo.md" if "audit_memo.md" in present else None,
            "edge_table": "edge_table.csv" if "edge_table.csv" in present else None,
            "graph": "graph.json" if "graph.json" in present else None,
            "error": "error.txt" if "error.txt" in present else None,
            "progress": "progress.jsonl" if "progress.jsonl" in present else None,
        }
        return {"run_dir": str(base), "files": files, "pages": pages}

    def resolve_file(self, run_id: str, relative_path: str) -> Path:
        """The on-disk file for a run-relative path; refuses anything outside the run directory."""
        record = self._require(run_id)
        base = Path(record.run_dir).resolve()
        target = (base / relative_path).resolve()
        if target != base and base not in target.parents:
            raise PermissionError(f"{relative_path} escapes the run directory")
        if not target.is_file():
            raise FileNotFoundError(relative_path)
        return target
