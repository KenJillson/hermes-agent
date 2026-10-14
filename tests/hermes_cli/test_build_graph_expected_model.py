"""Offline expected-model propagation and refusal qualification."""
import json
import ast
import logging
import os
import sys
import types
from pathlib import Path

import pytest

from hermes_cli import build_graph as graph
from hermes_cli import build_graph_constraints as constraints
from hermes_cli.build_graph_state import SCHEMA_VERSION, new_workflow_state, validate
from tests.hermes_cli.expected_model_fixtures import contract_kwargs

SONNET = {"schema": 1, "model": "sonnet", "lane": "lane_a", "invoke": "sonnet"}


def test_registry_requires_explicit_classification_when_checkpoint_is_absent():
    raw = json.dumps({"version": 1, "entries": []}).encode()
    with pytest.raises(constraints.ConstraintRefused,
                       match="Explicit protected workflow classification required"):
        constraints.parse_registry(raw, board_database="/isolated/board.db",
                                   task_id="t_missing", component="main")


def test_registry_surrogate_text_is_a_typed_refusal():
    entry = contract_kwargs("t_bad")["review_contract"]
    entry["task_id"] = "t_\ud800"
    raw = json.dumps({"version": 1, "entries": [entry]}).encode()
    with pytest.raises(constraints.ConstraintRefused,
                       match="Invalid bounded task_id"):
        constraints.parse_registry(raw, board_database="/isolated/test-board.db",
                                   task_id="t_bad", component="main")


def test_state_schema_binds_contract_and_rejects_old_or_changed_contract():
    bound = contract_kwargs("t_bound")
    state = new_workflow_state("t_bound", "main", **bound)
    bound["review_contract"]["contract_version"] = 99
    assert state["review_contract"]["contract_version"] == 1
    validate(state)
    assert SCHEMA_VERSION == 3
    old = dict(state); old["schema_version"] = 2
    with pytest.raises(ValueError, match="schema_version"):
        validate(old)
    changed = dict(state); changed["review_contract"] = dict(state["review_contract"])
    changed["review_contract"]["contract_version"] = 2
    with pytest.raises(ValueError, match="SHA-256"):
        validate(changed)


def _review_state():
    bound = contract_kwargs("t_review", classification="constrained", expected=SONNET)
    state = new_workflow_state("t_review", "main", diff="+change\n", **bound)
    return state


def test_review_node_propagates_constraint_and_latches_mismatch_without_count(monkeypatch):
    seen = {}
    def model(**kwargs):
        seen.update(kwargs)
        return {"rc": 10, "klass": "constraint", "result": {
            "ok": False, "invoked": False,
            "error": {"stage": "constraint", "code": "expected_model_mismatch"}}}
    deps = graph.Deps(model=model, workspace="/isolated", body="spec")
    monkeypatch.setattr(graph, "guard", lambda update, state: update)
    monkeypatch.setattr(graph, "spend_gate", lambda *_: None)
    monkeypatch.setattr(graph.revidence, "render", lambda *_: "evidence")
    monkeypatch.setattr(graph.revidence, "render_denials", lambda *_: "")
    monkeypatch.setattr(graph, "review_diff_current", lambda *_: True)
    update = graph.make_cloud_review(
        deps, activity="code_review", node_name="cloud_review")(_review_state())
    assert seen["workflow_classification"] == "constrained"
    assert seen["expected_model"] == SONNET
    assert seen["review_contract_sha256"]
    assert update["terminal_reason"] == "expected_model_mismatch:code_review"
    assert update["expected_model_refusal"] == "expected_model_mismatch"
    assert "cloud_review_calls" not in update


def test_matching_review_counts_one_invocation(monkeypatch):
    def model(**kwargs):
        return {"rc": 0, "klass": "ok", "result": {
            "ok": True, "invoked": True, "model": "sonnet",
            "verdict": {"passed": True}, "text": "ok"}}
    deps = graph.Deps(model=model, workspace="/isolated", body="spec")
    monkeypatch.setattr(graph, "guard", lambda update, state: update)
    monkeypatch.setattr(graph, "spend_gate", lambda *_: None)
    monkeypatch.setattr(graph.revidence, "render", lambda *_: "evidence")
    monkeypatch.setattr(graph.revidence, "render_denials", lambda *_: "")
    monkeypatch.setattr(graph, "review_diff_current", lambda *_: True)
    monkeypatch.setattr(graph, "verdict_passed", lambda *_: True)
    update = graph.make_cloud_review(
        deps, activity="code_review", node_name="cloud_review")(_review_state())
    assert update["cloud_review_calls"] == 1
    assert update["reviewer_model"] == "sonnet"
    assert update["last_verdict_passed"] is True


def test_forbidden_activity_refuses_before_spend_gate_or_model(monkeypatch):
    bound = contract_kwargs("t_forbidden", classification="constrained", expected=SONNET)
    bound["review_contract"]["activity_constraints"]["code_review"] = {"mode": "forbidden"}
    raw = json.dumps(bound["review_contract"], sort_keys=True,
                     separators=(",", ":")).encode()
    import hashlib
    bound["review_contract_sha256"] = hashlib.sha256(raw).hexdigest()
    state = new_workflow_state("t_forbidden", "main", diff="+x\n", **bound)
    monkeypatch.setattr(graph, "spend_gate",
                        lambda *_: pytest.fail("spend gate reached"))
    deps = graph.Deps(model=lambda **_: pytest.fail("model reached"))
    update = graph.make_cloud_review(
        deps, activity="code_review", node_name="cloud_review")(state)
    assert update["terminal_reason"] == "expected_model_forbidden:code_review"


def test_wrapper_rejects_noncanonical_launcher_before_artifact_preparation():
    result = graph.call_model(activity="code_review", prompt="review",
        workspace="/path/that/must/not/be/opened", card_id="t_wrapper",
        workflow_classification="constrained", expected_model=SONNET,
        review_contract_sha256="a" * 64, exe="/tmp/fixture-launcher")
    assert result["rc"] == 10
    assert result["klass"] == "constraint"
    assert result["result"]["invoked"] is False


@pytest.mark.parametrize("kwargs", [
    {"workflow_classification": "constrained", "expected_model": SONNET,
     "review_contract_sha256": "not-a-digest"},
    {"workflow_classification": None, "expected_model": SONNET,
     "review_contract_sha256": "a" * 64},
    {"workflow_classification": "unconstrained", "expected_model": SONNET,
     "review_contract_sha256": "a" * 64},
])
def test_wrapper_rejects_malformed_constraint_before_artifact_preparation(kwargs):
    result = graph.call_model(activity="code_review", prompt="review",
        workspace="/path/that/must/not/be/opened", card_id="t_wrapper", **kwargs)
    assert result["rc"] == 10
    assert result["klass"] == "constraint"
    assert result["result"]["invoked"] is False
    assert result["result"]["error"]["code"] == "expected_model_invalid"


def test_node_update_cannot_replace_protected_contract(monkeypatch):
    monkeypatch.setattr(graph.sz, "state_halt_reason", lambda update: None)
    update = graph.guard({"review_contract": {"workflow_classification":
        "unconstrained"}}, _review_state())
    assert update == {"terminal_reason": "expected_model_invalid:contract_mutation",
                      "expected_model_refusal": "expected_model_invalid"}


def _actual_caller():
    source = Path(__file__).resolve().parents[2] / "cli.py"
    tree = ast.parse(source.read_bytes())
    node = next(item for item in tree.body if isinstance(item, ast.FunctionDef)
                and item.name == "_run_build_graph_q")
    namespace = {"logger": logging.getLogger("expected-model-caller-test")}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"),
         namespace)
    return namespace["_run_build_graph_q"]


def _caller_modules(loader, capture):
    task = types.SimpleNamespace(graph_executor="build_graph", body="spec",
        workspace_kind="scratch", current_run_id=7)
    class Cursor:
        def fetchall(self): return [(0, "main", "/isolated/board.db")]
    class DB:
        def close(self): pass
        def execute(self, sql): return Cursor()
    kb = types.ModuleType("hermes_cli.kanban_db")
    kb.GRAPH_EXECUTORS = {"build_graph"}; kb.connect = DB
    kb.get_task = lambda conn, task_id: task
    kb.parse_ac_text = lambda body: (None, None, None, None)
    kb.block_task = lambda conn, task_id, *, reason, kind: capture.update(
        parked=(task_id, reason, kind)) or True
    constraints_module = types.ModuleType("hermes_cli.build_graph_constraints")
    constraints_module.load_review_contract = loader
    implementer = types.ModuleType("hermes_cli.build_graph_implementer")
    implementer.A2ReconciliationRequired = type("A2ReconciliationRequired", (Exception,), {})
    implementer.make_a2_runner = lambda *args: capture.update(a2=True) or \
        types.SimpleNamespace(derive=lambda workspace: {"ok": False, "reason": "unused"})
    graph_module = types.ModuleType("hermes_cli.build_graph")
    graph_module.Deps = lambda **kwargs: types.SimpleNamespace(**kwargs)
    graph_module.run = lambda *args, **kwargs: capture.update(run=kwargs) or {
        "terminal_reason": "human_review"}
    package = types.ModuleType("hermes_cli")
    package.kanban_db = kb; package.build_graph_constraints = constraints_module
    package.build_graph_implementer = implementer; package.build_graph = graph_module
    return {"hermes_cli": package, kb.__name__: kb,
            constraints_module.__name__: constraints_module,
            implementer.__name__: implementer, graph_module.__name__: graph_module}


def test_actual_assigned_caller_propagates_protected_contract(monkeypatch):
    capture = {}
    bound = contract_kwargs("t_caller")
    modules = _caller_modules(lambda **kwargs: (
        bound["review_contract"], bound["review_contract_sha256"]), capture)
    monkeypatch.setattr(os, "environ", {"HERMES_KANBAN_TASK": "t_caller",
        "HERMES_KANBAN_WORKSPACE": "/isolated/workspace"})
    with monkeypatch.context() as context:
        for name, module in modules.items(): context.setitem(sys.modules, name, module)
        assert _actual_caller()(types.SimpleNamespace(agent=None)) == 0
    assert capture["run"]["review_contract"] == bound["review_contract"]
    assert capture["run"]["review_contract_sha256"] == bound["review_contract_sha256"]
    assert capture["parked"][1] == "graph_terminal:human_review"


def test_actual_assigned_caller_refuses_missing_classification_before_a2(monkeypatch):
    capture = {}
    def missing(**kwargs):
        raise constraints.ConstraintRefused(
            "Explicit protected workflow classification required")
    modules = _caller_modules(missing, capture)
    monkeypatch.setattr(os, "environ", {"HERMES_KANBAN_TASK": "t_caller",
        "HERMES_KANBAN_WORKSPACE": "/isolated/workspace"})
    with monkeypatch.context() as context:
        for name, module in modules.items(): context.setitem(sys.modules, name, module)
        assert _actual_caller()(types.SimpleNamespace(agent=None)) == 0
    assert "a2" not in capture and "run" not in capture
    assert capture["parked"][1] == "graph_constraint_refused:ConstraintRefused"


def test_workspace_checkpoint_preserves_latched_refusal_and_rejects_contract_drift(
        tmp_path, monkeypatch):
    from hermes_cli import build_graph_checkpoint as checkpoint
    # Use the real file-backed saver/serde path with disposable state. The
    # separately required Postgres adapter is qualified only when a server is available.
    monkeypatch.setenv("HERMES_CONFIG_ROOT",
        str(Path(__file__).resolve().parents[3] / "config"))
    bound = contract_kwargs("t_resume")
    pair = checkpoint.make_workspace_checkpointer(str(tmp_path), schema_version=3)
    deps = graph.Deps(gate=lambda *a, **k: {"verdict": "no_checks"},
        arbiter=lambda *a, **k: ("no_arbiter", "fixture"),
        parse_ac=lambda *a, **k: (None, None, None, None), workspace=str(tmp_path))
    app, _ = graph.build(deps, checkpointer=pair)
    state = new_workflow_state("t_resume", "main", **bound)
    state.update(terminal_reason="expected_model_mismatch:code_review",
                 expected_model_refusal="expected_model_mismatch")
    config = {"configurable": {"thread_id": "t_resume:main"}}
    persisted = app.invoke(state, config)
    assert persisted["review_contract"] == bound["review_contract"]

    resumed = graph._run_with_checkpointer(None, "t_resume", str(tmp_path),
        deps=deps, checkpoint="workspace", resume=True, _workspace_pair=pair,
        **bound)
    assert resumed["expected_model_refusal"] == "expected_model_mismatch"
    assert resumed["terminal_reason"] == "expected_model_mismatch:code_review"

    changed = contract_kwargs("t_resume")
    changed["review_contract"]["contract_version"] = 2
    raw = json.dumps(changed["review_contract"], sort_keys=True,
                     separators=(",", ":")).encode()
    import hashlib
    changed["review_contract_sha256"] = hashlib.sha256(raw).hexdigest()
    refused = graph._run_with_checkpointer(None, "t_resume", str(tmp_path),
        deps=deps, checkpoint="workspace", resume=True, _workspace_pair=pair,
        **changed)
    assert refused["terminal_reason"] == "resume_constraint_mismatch"
