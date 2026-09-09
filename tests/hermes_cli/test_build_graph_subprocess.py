"""Behavior tests for build_graph's enabled A1 entry and owned runner seam."""

from __future__ import annotations

import sqlite3
import subprocess
from functools import partial

import pytest

from hermes_cli import build_graph as graph
from hermes_cli.build_graph_state import new_workflow_state


@pytest.fixture
def worktree(tmp_path):
    """A real disposable anchor plus card worktree; no network or user Git config."""
    anchor = tmp_path / "anchor"
    anchor.mkdir()
    def git(*args):
        return subprocess.run(
            ["git", "-C", str(anchor), *args], check=True,
            capture_output=True, text=True,
        )
    git("init", "-b", "main")
    git("config", "user.name", "A1 Fixture")
    git("config", "user.email", "a1@example.invalid")
    (anchor / "fixture.txt").write_text("before\n")
    (anchor / ".gitignore").write_text(".graph-checkpoints/\nac-execution-*.json\n")
    git("add", "fixture.txt", ".gitignore")
    git("-c", "commit.gpgsign=false", "commit", "-m", "fixture base")
    path = tmp_path / "card"
    git("worktree", "add", "-b", "fixture-card", str(path))
    return path


def _full_deps(worktree, runner):
    from hermes_cli import ac_check_runner, kanban_db, spend_accounting
    calls = {"runner": 0, "gate": 0, "model": 0}
    def owned(**kwargs):
        calls["runner"] += 1
        return runner(**kwargs)
    def gate(*args, **kwargs):
        calls["gate"] += 1
        return ac_check_runner.gate(*args, **kwargs)
    def model(*args, **kwargs):
        calls["model"] += 1
        pytest.fail("paid adapter called")
    return graph.Deps(
        gate=gate, parse_ac=kanban_db.parse_ac_text,
        arbiter=partial(spend_accounting.arbiter_decision, ledger_lines=[]),
        over_cap=partial(spend_accounting.over_spend_cap, ledger_lines=[]),
        model=model, implement_runner=owned, workspace=str(worktree),
    ), calls


def _write_fixture(**kwargs):
    from pathlib import Path
    (Path(kwargs["workspace"]) / "fixture.txt").write_text("after\n")
    return {"status": "completed", "identity": {"model": "fixture-local-model"}}


@pytest.mark.parametrize("diff,reason,expected", [
    ("", None, "implement"), (" \n\t", None, "implement"),
    ("+x\n", None, "cheap_gate"), ("", "prior", "human"),
    ("+x\n", "prior", "human"),
])
def test_enabled_entry_and_compiled_destination(worktree, diff, reason, expected):
    deps, _ = _full_deps(worktree, _write_fixture)
    state = _state()
    state.update(diff=diff, terminal_reason=reason)
    assert graph.route_entry(state) == expected
    app, _ = graph.build(deps)
    branches = app.builder.branches["__start__"]
    assert len(branches) == 1
    assert next(iter(branches.values())).ends[expected] == expected


@pytest.mark.parametrize("diff", ["", " \n\t"])
def test_full_run_implements_derives_then_real_no_checks_gate(worktree, diff):
    deps, calls = _full_deps(worktree, _write_fixture)
    conn = sqlite3.connect(":memory:")
    try:
        state = graph.run(conn, "t_a1", str(worktree), deps=deps, diff=diff,
                          body="Change fixture.txt to after. No acceptance block.")
    finally:
        conn.close()
    assert state["terminal_reason"] == "human_review"
    assert state["gate_summary"]["verdict"] == "no_checks"
    assert "+after" in state["diff"] and "-before" in state["diff"]
    assert state["diff_files"] == 1 and state["diff_added_lines"] == 1
    assert deps.changed_files == ["fixture.txt"]
    assert state["rung_attempts"]["implement"] == 1
    assert state["implementer_model"] == "fixture-local-model"
    assert calls == {"runner": 1, "gate": 1, "model": 0}


def test_nonempty_diff_bypasses_implementer(worktree):
    deps, calls = _full_deps(worktree, lambda **kw: pytest.fail("runner called"))
    conn = sqlite3.connect(":memory:")
    try:
        state = graph.run(conn, "t_a1", str(worktree), deps=deps, diff="+existing\n",
                          body="Existing work product without acceptance block")
    finally:
        conn.close()
    assert state["terminal_reason"] == "human_review"
    assert calls == {"runner": 0, "gate": 1, "model": 0}


def test_prior_terminal_reason_skips_child_and_gate(worktree):
    deps, calls = _full_deps(worktree, lambda **kw: pytest.fail("runner called"))
    app, _ = graph.build(deps)
    state = _state()
    state["terminal_reason"] = "prior"
    result = app.invoke(state, {"configurable": {"thread_id": "prior"}})
    assert result["terminal_reason"] == "prior"
    assert calls == {"runner": 0, "gate": 0, "model": 0}


@pytest.mark.parametrize("raw,reason", [
    ({"status": "failed"}, "implement_failed"),
    ({"status": "timeout"}, "implement_timeout"),
    ({"status": "interrupted"}, "implement_interrupted"),
    ({"status": "protocol_error"}, "implement_protocol_error"),
    ({"status": "exec_error"}, "implement_exec_error"),
    ({"status": "invented"}, "implement_unknown_status"),
    ("completed", "implement_protocol_error"),
])
def test_full_run_failure_never_reaches_gate(worktree, raw, reason):
    deps, calls = _full_deps(worktree, lambda **kw: raw)
    conn = sqlite3.connect(":memory:")
    try:
        state = graph.run(conn, "t_a1", str(worktree), deps=deps,
                          body="Change fixture.txt to after")
    finally:
        conn.close()
    assert state["terminal_reason"] == reason
    assert state["rung_attempts"]["implement"] == 1
    assert calls == {"runner": 1, "gate": 0, "model": 0}


def test_full_run_noop_parks_before_gate(worktree):
    from hermes_cli.build_graph_diff import EMPTY_DIFF_REASON
    deps, calls = _full_deps(worktree, lambda **kw: {
        "status": "completed", "identity": {"model": "fixture-local-model"}})
    conn = sqlite3.connect(":memory:")
    try:
        state = graph.run(conn, "t_a1", str(worktree), deps=deps,
                          body="Change fixture.txt to after")
    finally:
        conn.close()
    assert state["terminal_reason"] == EMPTY_DIFF_REASON
    assert calls == {"runner": 1, "gate": 0, "model": 0}


def test_resume_preserves_consumed_attempt(worktree):
    deps, calls = _full_deps(worktree, lambda **kw: {"status": "failed"})
    conn = sqlite3.connect(":memory:")
    try:
        first = graph.run(conn, "t_a1", str(worktree), deps=deps,
                          body="Change fixture.txt to after", checkpoint="workspace",
                          resume="auto")
        second = graph.run(conn, "t_a1", str(worktree), deps=deps,
                           body="Change fixture.txt to after", checkpoint="workspace",
                           resume="auto")
    finally:
        conn.close()
    assert first["terminal_reason"] == "implement_failed"
    assert second["terminal_reason"] == "implement_attempt_cap"
    assert second["rung_attempts"]["implement"] == 1
    assert calls == {"runner": 1, "gate": 0, "model": 0}


def test_zero_cap_refuses_paid_review_after_implementation(worktree):
    from hermes_cli import kanban_db
    deps, calls = _full_deps(worktree, _write_fixture)
    # Keep the real arbiter and cap predicate; isolate the AC subprocess here.
    def passing_gate(*args, **kwargs):
        calls["gate"] += 1
        return {"verdict": "all_pass", "results": []}
    deps.gate = passing_gate
    conn = kanban_db.connect(worktree.parent / "board.db")
    try:
        task_id = kanban_db.create_task(conn, title="A1 zero-cap fixture")
        assert kanban_db.set_max_card_spend(conn, task_id, 0.0)
        state = graph.run(conn, task_id, str(worktree), deps=deps,
                          body="Change fixture.txt to after")
    finally:
        conn.close()
    assert state["terminal_reason"] == "over_spend_cap"
    assert calls == {"runner": 1, "gate": 1, "model": 0}


@pytest.mark.parametrize("with_ac", [False, True])
def test_dispatcher_clean_worktree_reaches_owned_implementation(monkeypatch, worktree, with_ac):
    """Exercise the imported dispatcher entry, including the empty-diff exemption."""
    from types import SimpleNamespace
    import cli
    from hermes_cli import kanban_db, build_graph_diff

    body = "Change fixture.txt to after."
    if with_ac:
        body += "\n\n## AC\n- [ ] Fixture is changed.\n"
    assert bool(kanban_db.parse_ac_text(body)[3]) is with_ac
    assert build_graph_diff.derive(str(worktree))["reason"] == build_graph_diff.EMPTY_DIFF_REASON
    db_path = worktree.parent / "dispatch.db"
    conn = kanban_db.connect(db_path)
    try:
        task_id = kanban_db.create_task(conn, title="A1 dispatcher fixture", body=body,
                                       workspace_kind="worktree", workspace_path=str(worktree))
        assert kanban_db.set_graph_executor(conn, task_id, "build_graph")
        assert kanban_db.set_max_card_spend(conn, task_id, 0.0)
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(worktree))
    deps, calls = _full_deps(worktree, _write_fixture)
    if with_ac:
        # AC transport is a separate integration; this case proves the
        # dispatcher exemption, then the real graph's per-node spend refusal.
        def passing_gate(*args, **kwargs):
            calls["gate"] += 1
            assert kwargs["parse_fn"](body)[3]
            return {"verdict": "all_pass", "results": []}
        deps.gate = passing_gate
    run = graph.run
    observed = {}
    def injected_run(*args, **kwargs):
        assert kwargs["diff"] == ""
        result = run(*args, deps=deps, **kwargs)
        observed.update(result)
        return result
    monkeypatch.setattr(graph, "run", injected_run)
    parked = []
    def block(conn, card_id, *, reason, kind):
        parked.append((card_id, reason, kind))
        return True
    monkeypatch.setattr(kanban_db, "block_task", block)
    assert cli._run_build_graph_q(SimpleNamespace(agent=None)) == 0
    expected = "over_spend_cap" if with_ac else "human_review"
    assert parked == [(task_id, "graph_terminal:" + expected, "needs_input")]
    assert observed["terminal_reason"] == expected
    assert "+after" in observed["diff"]
    assert calls == {"runner": 1, "gate": 1, "model": 0}


def test_full_run_empty_spec_never_spawns(worktree):
    deps, calls = _full_deps(worktree, lambda **kw: pytest.fail("runner called"))
    conn = sqlite3.connect(":memory:")
    try:
        state = graph.run(conn, "t_a1", str(worktree), deps=deps, body=" \n\t")
    finally:
        conn.close()
    assert state["terminal_reason"] == "implement_no_spec"
    assert not state["rung_attempts"].get("implement")
    assert calls == {"runner": 0, "gate": 0, "model": 0}


def test_missing_runner_import_parks(monkeypatch, worktree):
    import sys
    deps, calls = _full_deps(worktree, lambda **kw: pytest.fail("runner called"))
    deps.implement_runner = None
    monkeypatch.setitem(sys.modules, "hermes_cli.build_graph_implementer", None)
    conn = sqlite3.connect(":memory:")
    try:
        state = graph.run(conn, "t_a1", str(worktree), deps=deps,
                          body="Change fixture.txt to after")
    finally:
        conn.close()
    assert state["terminal_reason"] == "implement_runner_unavailable"
    assert calls == {"runner": 0, "gate": 0, "model": 0}


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
