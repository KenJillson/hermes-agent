"""Behavior tests for the owned build-graph implementer subprocess."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import sys
import time

import pytest

from hermes_cli import build_graph_implementer as impl


IDENTITY = {
    "provider": "custom",
    "model": "fixture-local-model",
    "base_url": "http://127.0.0.1:9/v1",
    "api_mode": "chat_completions",
}

FIXTURE = r'''import json, os, pathlib, signal, subprocess, sys, time
request_path = pathlib.Path(sys.argv[1])
request = json.loads(request_path.read_text())
control = json.loads(request["goal"])
pathlib.Path(control["observed"]).write_text(json.dumps({
    "cwd": os.getcwd(), "pid": os.getpid(), "pgid": os.getpgrp(),
    "argv": sys.argv, "request_mode": request_path.stat().st_mode & 0o777,
    "kanban": sorted(k for k in os.environ if k.startswith("HERMES_KANBAN_")),
    "marker": os.environ.get("HERMES_DELEGATED_CHILD_CONTEXT"),
    "terminal_cwd": os.environ.get("TERMINAL_CWD"), "request": request,
}))
mode = control["mode"]
if mode in ("hang", "interrupt"):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    grand = subprocess.Popen([sys.executable, "-c",
        "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"])
    pathlib.Path(control["grandchild"]).write_text(str(grand.pid))
    time.sleep(60)
if mode == "malformed":
    print("not-json"); raise SystemExit(0)
if mode == "oversized":
    print("x" * 70000); raise SystemExit(0)
identity = dict(request["expected_identity"])
if mode == "wrong_identity": identity["model"] = "wrong"
status = "failed" if mode == "failed" else "completed"
print(json.dumps({"protocol": 1, "status": status, "identity": identity,
                  "summary": "fixture complete"}))
raise SystemExit(1 if status == "failed" else 0)
'''


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


@pytest.fixture
def fixture_child(tmp_path: Path) -> Path:
    path = tmp_path / "fixture_child.py"
    path.write_text(FIXTURE)
    return path


def _run(monkeypatch, tmp_path: Path, fixture_child: Path, mode: str,
         *, timeout: float = 3.0, interrupt_check=None):
    monkeypatch.setattr(impl, "_identity", lambda: dict(IDENTITY))
    workspace = tmp_path / ("workspace-" + mode)
    workspace.mkdir()
    observed = tmp_path / ("observed-" + mode + ".json")
    grandchild = tmp_path / ("grandchild-" + mode + ".txt")
    goal = json.dumps({
        "mode": mode, "observed": str(observed), "grandchild": str(grandchild),
    })
    result = impl.run_implementer(
        goal=goal, workspace=str(workspace), max_iterations=7,
        timeout_seconds=timeout, interrupt_check=interrupt_check,
        child_argv=[sys.executable, str(fixture_child), "{request_file}"],
        evidence_dir=str(tmp_path / ("evidence-" + mode)),
        poll_seconds=0.02, term_grace_seconds=0.05,
        env={
            **os.environ,
            "HERMES_KANBAN_TASK": "remove",
            "HERMES_KANBAN_GOAL_MODE": "remove",
            "HERMES_KANBAN_MAX_CARD_SPEND": "remove",
        },
    )
    return result, observed, grandchild, workspace


def test_real_child_contract_and_environment(monkeypatch, tmp_path, fixture_child):
    result, observed_path, _, workspace = _run(
        monkeypatch, tmp_path, fixture_child, "completed")
    assert result["status"] == "completed"
    assert result["identity"] == IDENTITY
    evidence = Path(result["evidence_path"])
    assert stat.S_IMODE(evidence.stat().st_mode) == 0o600
    observed = json.loads(observed_path.read_text())
    assert observed["cwd"] == str(workspace.resolve())
    assert observed["terminal_cwd"] == str(workspace.resolve())
    assert observed["pid"] == observed["pgid"]
    assert observed["marker"] == "1"
    assert observed["kanban"] == []
    assert observed["request_mode"] == 0o600
    assert observed["request"]["goal"] not in " ".join(observed["argv"])
    assert observed["request"]["max_iterations"] == 7


@pytest.mark.parametrize("mode, expected", [
    ("failed", "failed"),
    ("malformed", "protocol_error"),
    ("oversized", "protocol_error"),
    ("wrong_identity", "protocol_error"),
])
def test_closed_child_outcomes(monkeypatch, tmp_path, fixture_child, mode, expected):
    result, *_ = _run(monkeypatch, tmp_path, fixture_child, mode)
    assert result["status"] == expected


@pytest.mark.parametrize("mode, expected", [
    ("hang", "timeout"),
    ("interrupt", "interrupted"),
])
def test_process_group_and_descendant_are_reaped(
    monkeypatch, tmp_path, fixture_child, mode, expected,
):
    at = time.monotonic() + 0.15
    result, _, grandchild_path, _ = _run(
        monkeypatch, tmp_path, fixture_child, mode,
        timeout=0.2 if mode == "hang" else 3.0,
        interrupt_check=(lambda: time.monotonic() >= at) if mode == "interrupt" else None,
    )
    assert result["status"] == expected
    pid = int(grandchild_path.read_text())
    for _ in range(100):
        if not _alive(pid):
            break
        time.sleep(0.02)
    assert not _alive(pid)


def test_provider_locality_uses_real_config_and_resolver(monkeypatch, tmp_path):
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(
        "model:\n"
        "  default: fixture-model\n"
        "  provider: custom\n"
        "  base_url: http://127.0.0.1:9/v1\n"
        "  api_mode: chat_completions\n"
    )
    identity = impl._identity()
    assert identity["provider"] == "custom"
    assert identity["model"] == "fixture-model"
    assert identity["base_url"] == "http://127.0.0.1:9/v1"

    (home / "config.yaml").write_text(
        "model:\n"
        "  default: fixture-model\n"
        "  provider: custom\n"
        "  base_url: https://example.com/v1\n"
        "  api_mode: chat_completions\n"
    )
    with pytest.raises(impl.ProtocolError, match="provider_not_local"):
        impl._identity()


def test_locality_failure_prevents_spawn(monkeypatch, tmp_path):
    monkeypatch.setattr(
        impl, "_identity", lambda: (_ for _ in ()).throw(impl.ProtocolError("no")))
    called = False
    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("spawned")
    monkeypatch.setattr(impl.subprocess, "Popen", forbidden)
    result = impl.run_implementer(
        goal="x", workspace=str(tmp_path), max_iterations=1,
        timeout_seconds=1, evidence_dir=str(tmp_path / "evidence"),
    )
    assert result["status"] == "protocol_error"
    assert called is False


def test_request_validation_is_strict(tmp_path):
    base = {
        "protocol": 1, "goal": "x", "workspace": str(tmp_path),
        "max_iterations": 1, "deadline_monotonic": time.monotonic() + 1,
        "expected_identity": dict(IDENTITY),
    }
    bad = [
        {**base, "extra": 1}, {**base, "protocol": 2},
        {**base, "max_iterations": 0},
        {**base, "deadline_monotonic": float("nan")},
    ]
    for request in bad:
        with pytest.raises(impl.ProtocolError):
            impl._validate_request(request)
