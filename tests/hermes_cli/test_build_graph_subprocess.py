"""Behavior tests for build_graph's A1 runner seam while interim C is active."""

from __future__ import annotations

import sqlite3

import pytest

from hermes_cli import build_graph as graph
from hermes_cli.build_graph_state import new_workflow_state


def _state():
    return new_workflow_state("t_a1", "component", diff="")


def _deps(tmp_path, runner, derive, *, body="Implement the fixture"):
    return graph.Deps(
        agent=None,
        implement_runner=runner,
        derive=derive,
        workspace=str(tmp_path),
        body=body,
        changed_files=[],
    )


def test_completed_runner_rederives_work_product(monkeypatch, tmp_path):
    seen = {}
    def runner(**kwargs):
        seen.update(kwargs)
        return {
            "status": "completed",
            "identity": {"model": "fixture-local-model"},
            "evidence_path": "/private/evidence",
        }
    def derive(workspace):
        assert workspace == str(tmp_path)
        return {
            "ok": True,
            "diff": "diff --git a/x b/x\n+fixture\n",
            "diff_files": 1,
            "diff_added_lines": 1,
            "changed_files": ["x"],
        }
    node = graph.make_implement(_deps(tmp_path, runner, derive))
    update = node(_state())
    assert seen["workspace"] == str(tmp_path)
    assert seen["max_iterations"] == graph.IMPLEMENT_MAX_ITERATIONS
    assert seen["timeout_seconds"] == graph.IMPLEMENT_TIMEOUT_SECONDS
    assert callable(seen["interrupt_check"])
    assert update["diff_files"] == 1
    assert update["diff_added_lines"] == 1
    assert update["implementer_model"] == "fixture-local-model"
    assert update["rung_attempts"]["implement"] == 1
    assert "evidence_path" not in update


@pytest.mark.parametrize("status", [
    "failed", "timeout", "interrupted", "protocol_error", "exec_error",
])
def test_closed_runner_failures_park_and_consume_attempt(tmp_path, status):
    node = graph.make_implement(_deps(
        tmp_path,
        lambda **kwargs: {"status": status, "identity": None, "evidence_path": None},
        lambda workspace: pytest.fail("derive called"),
    ))
    update = node(_state())
    assert update == {
        "terminal_reason": "implement_" + status,
        "rung_attempts": {"implement": 1},
    }


def test_unknown_or_nonmapping_result_fails_closed(tmp_path):
    for raw, reason in (
        ("completed", "implement_protocol_error"),
        ({"status": "invented"}, "implement_unknown_status"),
    ):
        node = graph.make_implement(_deps(
            tmp_path, lambda **kwargs: raw,
            lambda workspace: pytest.fail("derive called"),
        ))
        update = node(_state())
        assert update["terminal_reason"] == reason
        assert update["rung_attempts"]["implement"] == 1


def test_empty_spec_and_attempt_cap_do_not_spawn(tmp_path):
    def forbidden(**kwargs):
        pytest.fail("runner called")
    empty = graph.make_implement(_deps(
        tmp_path, forbidden, forbidden, body=" \n\t"))(_state())
    assert empty == {"terminal_reason": "implement_no_spec"}

    state = _state()
    state["rung_attempts"]["implement"] = graph.IMPLEMENT_ATTEMPT_CAP
    capped = graph.make_implement(_deps(tmp_path, forbidden, forbidden))(state)
    assert capped == {"terminal_reason": "implement_attempt_cap"}


def test_completed_noop_parks_before_gate(tmp_path):
    node = graph.make_implement(_deps(
        tmp_path,
        lambda **kwargs: {
            "status": "completed",
            "identity": {"model": "fixture-local-model"},
            "evidence_path": None,
        },
        lambda workspace: {"ok": False, "reason": "graph_diff_empty"},
    ))
    update = node(_state())
    assert update["terminal_reason"] == "graph_diff_empty"
    assert update["rung_attempts"]["implement"] == 1
    assert "diff" not in update


def test_delegate_task_regression_does_not_affect_a1(monkeypatch, tmp_path):
    import tools.delegate_tool
    monkeypatch.setattr(
        tools.delegate_tool, "delegate_task",
        lambda **kwargs: pytest.fail("delegate_task called"),
    )
    node = graph.make_implement(_deps(
        tmp_path,
        lambda **kwargs: {"status": "failed", "identity": None, "evidence_path": None},
        lambda workspace: pytest.fail("derive called"),
    ))
    assert node(_state())["terminal_reason"] == "implement_failed"


def test_interim_c_full_run_never_spawns_runner(tmp_path):
    calls = {"runner": 0, "gate": 0, "model": 0}
    def forbidden_runner(**kwargs):
        calls["runner"] += 1
        pytest.fail("runner became reachable")
    def gate(*args, **kwargs):
        calls["gate"] += 1
        return {"verdict": "no_checks", "results": []}
    def model(*args, **kwargs):
        calls["model"] += 1
        pytest.fail("model called")
    deps = graph.Deps(
        gate=gate,
        arbiter=lambda *args, **kwargs: ("no_arbiter", "offline"),
        model=model,
        parse_ac=lambda body: (None, None, None, None),
        implement_runner=forbidden_runner,
        derive=lambda workspace: pytest.fail("derive called"),
        over_cap=lambda *args, **kwargs: False,
        workspace=str(tmp_path),
        body="Implement fixture",
    )
    conn = sqlite3.connect(":memory:")
    try:
        state = graph.run(conn, "t_a1", str(tmp_path), deps=deps, diff="")
    finally:
        conn.close()
    assert state["terminal_reason"] == "human_review"
    assert calls == {"runner": 0, "gate": 0, "model": 0}
