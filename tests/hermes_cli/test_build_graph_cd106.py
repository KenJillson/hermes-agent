"""CD-106: a review park records its findings as one inert, bounded, sanitized event.

Measured basis: Test14e t_ccb5e9ab parked at `a2_stage_review_required:plan_review`
with one objection; the harvest kept only its length (graph-trace) and a checkpoint
pointer, so the finding itself was unrecoverable from any committed record.

Payload and record tests use a stub sanitizer and a fake board. The real-sanitizer
and real-board classes need the box (lib/model_call sanitizer, maintenance gate);
the box rehearsal requires that they RUN, not skip.
"""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from hermes_cli import build_graph_findings as f


class Stub:
    """A sanitizer stand-in: 'HALT' trips a rule, 'SECRET' is redacted."""
    def __init__(self, rule="private_key", halt_error=False, redact_error=False):
        self.rule, self.halt_error, self.redact_error = rule, halt_error, redact_error
    def halt_reason(self, text):
        if self.halt_error:
            raise RuntimeError("sanitizer missing")
        return self.rule if "HALT" in text else None
    def redact(self, text):
        if self.redact_error:
            raise RuntimeError("redactor broke")
        return text.replace("SECRET", "[REDACTED]")


def state(*texts):
    return {"objections_current": [{"text": t, "raw": t, "review_id": None} for t in texts]}


class Payload(unittest.TestCase):
    def test_only_review_parks_record(self):
        for terminal in (None, "", "human_review", "a2_stage_ready:plan_review",
                         "a2_stage_failed:plan_review:stage_result:exception",
                         "cloud_plan_review_cap", "graph_terminal:a2_stage_review_required:plan_review"):
            with self.subTest(terminal=terminal):
                self.assertIsNone(f.payload(terminal, state("x"), sanitize=Stub()))
        for terminal in ("a2_stage_review_required:plan_review",
                         "a2_stage_review_required:local_review", "cloud_plan_review_rejected"):
            with self.subTest(terminal=terminal):
                self.assertEqual(f.payload(terminal, state("x"), sanitize=Stub())["terminal"], terminal)

    def test_exact_shape(self):
        value = f.payload("a2_stage_review_required:plan_review", state("Plan omits AC4"), sanitize=Stub())
        self.assertEqual(value, {"terminal": "a2_stage_review_required:plan_review", "count": 1,
                                 "truncated": False, "findings": ["Plan omits AC4"]})
        json.dumps(value)  # the event payload is JSON

    def test_malformed_or_empty_objections(self):
        bad = {"objections_current": ["bare", None, {"text": ""}, {"text": "  "}, {"text": 3},
                                      {"raw": "no text"}, {"text": "kept"}]}
        self.assertEqual(f.payload("cloud_plan_review_rejected", bad, sanitize=Stub())["findings"], ["kept"])
        for value in (None, {}, {"objections_current": "x"}, "state"):
            with self.subTest(value=value):
                out = f.payload("cloud_plan_review_rejected", value, sanitize=Stub())
                self.assertEqual((out["count"], out["findings"], out["truncated"]), (0, [], False))

    def test_bounds(self):
        many = f.payload("cloud_plan_review_rejected", state(*["f%d" % i for i in range(11)]), sanitize=Stub())
        self.assertEqual((many["count"], len(many["findings"]), many["truncated"]), (11, f.MAX_FINDINGS, True))
        long = f.payload("cloud_plan_review_rejected", state("y" * 1000), sanitize=Stub())
        self.assertEqual(len(long["findings"][0]), f.MAX_CHARS)
        self.assertTrue(long["findings"][0].endswith("…") and long["truncated"])
        exact = f.payload("cloud_plan_review_rejected", state("z" * f.MAX_CHARS), sanitize=Stub())
        self.assertEqual((exact["findings"][0], exact["truncated"]), ("z" * f.MAX_CHARS, False))

    def test_halt_is_withheld_by_rule_name_only(self):
        out = f.payload("cloud_plan_review_rejected", state("HALT key material", "ok"), sanitize=Stub())
        self.assertEqual(out["findings"], ["[withheld: private_key]", "ok"])
        self.assertNotIn("key material", json.dumps(out))
        odd = f.payload("cloud_plan_review_rejected", state("HALT x"), sanitize=Stub(rule="Not A Rule!"))
        self.assertEqual(odd["findings"], ["[withheld: halt]"])

    def test_redaction_and_sanitizer_failures_withhold(self):
        self.assertEqual(f.payload("cloud_plan_review_rejected", state("token SECRET"), sanitize=Stub())["findings"],
                         ["token [REDACTED]"])
        for stub in (Stub(halt_error=True), Stub(redact_error=True)):
            with self.subTest(stub=vars(stub)):
                out = f.payload("cloud_plan_review_rejected", state("token SECRET"), sanitize=stub)
                self.assertEqual(out["findings"], ["[withheld: sanitizer_unavailable]"])
                self.assertNotIn("SECRET", json.dumps(out))
        broken = f.payload("cloud_plan_review_rejected", state("a", "b"), sanitize=object())
        self.assertEqual(broken["findings"], ["[withheld: sanitizer_unavailable]"] * 2)


class FakeBoard:
    def __init__(self, fail=None):
        self.fail, self.events, self.closed, self.txn = fail, [], 0, 0
    def connect(self):
        if self.fail == "connect":
            raise OSError("db SECRET path")
        return self
    def close(self):
        self.closed += 1
    def write_txn(self, conn):
        board = self
        class Txn:
            def __enter__(self): board.txn += 1
            def __exit__(self, *exc): return False
        return Txn()
    def _append_event(self, conn, task_id, kind, payload, *, run_id=None):
        if self.fail == "append":
            raise RuntimeError("insert failed SECRET")
        self.events.append((task_id, kind, payload, run_id))


class Record(unittest.TestCase):
    VALUE = {"terminal": "cloud_plan_review_rejected", "count": 1, "truncated": False, "findings": ["x"]}

    def test_records_one_event_on_the_parked_run(self):
        board = FakeBoard()
        self.assertTrue(f.record(board, "t_1", 42, self.VALUE))
        self.assertEqual(board.events, [("t_1", "graph_review_findings", self.VALUE, 42)])
        self.assertEqual((board.txn, board.closed), (1, 1))

    def test_none_writes_nothing(self):
        board = FakeBoard()
        self.assertFalse(f.record(board, "t_1", 42, None))
        self.assertEqual((board.events, board.txn), ([], 0))

    def test_failures_never_raise_and_log_type_only(self):
        for fail in ("connect", "append"):
            with self.subTest(fail=fail):
                board, log = FakeBoard(fail), mock.Mock()
                self.assertFalse(f.record(board, "t_1", 42, self.VALUE, log))
                args = log.error.call_args[0]
                self.assertIn(args[2], ("OSError", "RuntimeError"))
                self.assertNotIn("SECRET", repr(args))
                self.assertEqual(board.closed, 0 if fail == "connect" else 1)


# ---- the real cli._run_build_graph_q, executed against fake board/graph modules ----
# Same technique as test_build_graph_expected_model._actual_caller (that module needs
# pytest, absent on the box): the function is compiled from cli.py and RUN.

import ast
import logging
import sys
import types


def actual_caller():
    source = Path(__file__).resolve().parents[2] / "cli.py"
    node = next(item for item in ast.parse(source.read_bytes()).body
                if isinstance(item, ast.FunctionDef) and item.name == "_run_build_graph_q")
    namespace = {"logger": logging.getLogger("cd106-caller-test")}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["_run_build_graph_q"]


class Caller(unittest.TestCase):
    TERMINAL = "a2_stage_review_required:plan_review"

    def modules(self, state, parked=True, findings=f, append_error=None):
        from tests.hermes_cli.expected_model_fixtures import contract_kwargs
        bound = contract_kwargs("t_cd106")
        self.capture = capture = {"events": []}
        task = types.SimpleNamespace(graph_executor="build_graph", body="spec",
                                     workspace_kind="scratch", current_run_id=7)
        class Cursor:
            def fetchall(self): return [(0, "main", "/isolated/board.db")]
        class DB:
            def close(self): pass
            def execute(self, sql): return Cursor()
        class Txn:
            def __enter__(self): return None
            def __exit__(self, *exc): return False
        kb = types.ModuleType("hermes_cli.kanban_db")
        kb.GRAPH_EXECUTORS = {"build_graph"}; kb.connect = DB
        kb.get_task = lambda conn, task_id: task
        kb.parse_ac_text = lambda body: (None, None, None, None)
        kb.block_task = lambda conn, task_id, *, reason, kind: capture.update(
            parked=(task_id, reason, kind)) or parked
        kb.write_txn = lambda conn: Txn()
        def append(conn, task_id, kind, payload, *, run_id=None):
            if append_error:
                raise append_error
            capture["events"].append((task_id, kind, payload, run_id))
        kb._append_event = append
        constraints = types.ModuleType("hermes_cli.build_graph_constraints")
        constraints.load_review_contract = lambda **kw: (bound["review_contract"], bound["review_contract_sha256"])
        constraints.evaluation_target = lambda *a, **kw: None
        implementer = types.ModuleType("hermes_cli.build_graph_implementer")
        implementer.A2ReconciliationRequired = type("A2ReconciliationRequired", (Exception,), {})
        implementer.make_a2_runner = lambda *args: types.SimpleNamespace(
            derive=lambda workspace: {"ok": False, "reason": "unused"})
        graph = types.ModuleType("hermes_cli.build_graph")
        graph.Deps = lambda **kwargs: types.SimpleNamespace(**kwargs)
        graph.run = lambda *args, **kwargs: dict(state)
        package = types.ModuleType("hermes_cli")
        for name, module in (("kanban_db", kb), ("build_graph_constraints", constraints),
                             ("build_graph_implementer", implementer), ("build_graph", graph),
                             ("build_graph_findings", findings)):
            setattr(package, name, module)
        return {"hermes_cli": package, kb.__name__: kb, constraints.__name__: constraints,
                implementer.__name__: implementer, graph.__name__: graph,
                "hermes_cli.build_graph_findings": findings}

    def run_caller(self, modules):
        caller = actual_caller()
        env = {"HERMES_KANBAN_TASK": "t_cd106", "HERMES_KANBAN_WORKSPACE": "/isolated/workspace"}
        with mock.patch.dict(sys.modules, modules), mock.patch.object(os, "environ", env):
            return caller(types.SimpleNamespace(agent=None))

    def review_state(self):
        return dict(state("Plan never sets QUALIFICATION_VALUE"), terminal_reason=self.TERMINAL)

    def test_review_park_records_one_event_on_the_parked_run(self):
        self.assertEqual(self.run_caller(self.modules(self.review_state())), 0)
        self.assertEqual(self.capture["parked"], ("t_cd106", "graph_terminal:" + self.TERMINAL, "needs_input"))
        (task_id, kind, value, run_id), = self.capture["events"]
        self.assertEqual((task_id, kind, run_id), ("t_cd106", "graph_review_findings", 7))
        self.assertEqual((value["terminal"], value["count"]), (self.TERMINAL, 1))
        self.assertEqual(len(value["findings"]), 1)

    def test_refused_park_records_nothing(self):
        self.assertEqual(self.run_caller(self.modules(self.review_state(), parked=False)), 1)
        self.assertEqual(self.capture["events"], [])

    def test_other_parks_record_nothing(self):
        self.assertEqual(self.run_caller(self.modules({"terminal_reason": "human_review"})), 0)
        self.assertEqual(self.capture["parked"][1], "graph_terminal:human_review")
        self.assertEqual(self.capture["events"], [])

    def test_a_crashing_payload_never_changes_the_park(self):
        broken = types.ModuleType("hermes_cli.build_graph_findings")
        def boom(*args, **kwargs): raise RuntimeError("CD-106 bug")
        broken.payload = boom; broken.record = boom
        self.assertEqual(self.run_caller(self.modules(self.review_state(), findings=broken)), 0)
        self.assertEqual(self.capture["parked"][1], "graph_terminal:" + self.TERMINAL)
        self.assertEqual(self.capture["events"], [])

    def test_a_failing_event_write_never_fails_the_worker(self):
        rc = self.run_caller(self.modules(self.review_state(), append_error=OSError("locked")))
        self.assertEqual(rc, 0)
        self.assertEqual(self.capture["parked"][1], "graph_terminal:" + self.TERMINAL)
        self.assertEqual(self.capture["events"], [])


def box_modules():
    try:
        from hermes_cli import build_graph_sanitize, kanban_db
        build_graph_sanitize.halt_reason("probe")
        return build_graph_sanitize, kanban_db
    except Exception:
        return None


BOX = box_modules()


@unittest.skipIf(BOX is None, "needs the box sanitizer and maintenance gate")
class OnBox(unittest.TestCase):
    def test_real_sanitizer_withholds_private_key(self):
        san, _ = BOX
        text = "-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAAA\n-----END OPENSSH PRIVATE KEY-----"
        rule = san.halt_reason(text)
        self.assertIsNotNone(rule, "fixture assumption: this text trips a HALT rule")
        out = f.payload("cloud_plan_review_rejected", state(text, "plain finding"))
        self.assertEqual(out["findings"], ["[withheld: %s]" % rule, "plain finding"])
        self.assertNotIn("b3BlbnNzaC1", json.dumps(out))

    def test_real_board_event_and_worker_context(self):
        _, kb = BOX
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "board.db"
            conn = kb.connect(db)
            try:
                task_id = kb.create_task(conn, title="CD-106 fixture", body="Spec body.")
            finally:
                conn.close()
            value = f.payload("a2_stage_review_required:plan_review",
                              state("UNIQUE-FINDING-7f3a: plan omits AC4"))
            with mock.patch.dict(os.environ, {"HERMES_KANBAN_DB": str(db)}):
                self.assertTrue(f.record(kb, task_id, None, value))
            conn = kb.connect(db)
            try:
                events = [e for e in kb.list_events(conn, task_id) if e.kind == f.EVENT]
                self.assertEqual([e.payload for e in events], [value])
                self.assertEqual(kb.list_comments(conn, task_id), [])
                self.assertNotIn("UNIQUE-FINDING-7f3a", kb.build_worker_context(conn, task_id))
            finally:
                conn.close()


if __name__ == "__main__":
    unittest.main()
