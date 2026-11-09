"""CD-103 (F-L2): an output-compliance refusal at a cloud fix rung goes straight to the next
rung, upward only, keeping the AC-1 dispute edge; scope/safety/system refusals still park.
Fixtures are shared with CD-102's tests. Provider-free.

Measured basis: the Test14 Codex measurement (run 8c615804) returned a refusal sentence as a
file's content, and the 2026-11-09 whole-path review showed rungs 1/2 are never shown the
specification, so a decline is a likely answer to an unsatisfiable check.
"""
import ast
import inspect
import unittest
from unittest.mock import Mock

from hermes_cli import build_graph as bg
from tests.hermes_cli.test_build_graph_cd102 import ORIGINAL, REFUSAL, RungFixture, fix_reply


class Escalation(RungFixture):
    def test_compliance_refusal_climbs_without_a_gate(self):
        _, update, merged = self.run_rung("rung1", {"klass": "ok", "result": {"model": "sonnet", "text": "no JSON"}})
        self.assertIsNone(update.get("terminal_reason"))
        self.assertEqual(update["rung_attempts"]["fix_refused_next"], "rung2")
        self.assertEqual(update["rung_attempts"]["fix_refusals"], ["rung1:fix_unparseable"])
        self.assertEqual(bg.route_after_fix(merged), "fix_rung2")

    def test_content_refusal_is_a_compliance_refusal(self):
        _, update, merged = self.run_rung("rung1", fix_reply([{"path": "m.py", "content": REFUSAL}]))
        self.assertEqual(update["rung_attempts"]["fix_refusals"], ["rung1:fix_content_invalid_python"])
        self.assertEqual(bg.route_after_fix(merged), "fix_rung2")

    def test_disputed_card_skips_rung2_on_escalation(self):
        _, update, merged = self.run_rung("rung1", fix_reply([]), objection_recurred=True, holdout=False)
        self.assertEqual(update["rung_attempts"]["fix_refused_next"], "rung3")
        self.assertEqual(bg.route_after_fix(merged), "fix_rung3")

    def test_scope_refusal_still_parks(self):
        _, update, merged = self.run_rung("rung1", fix_reply([{"path": "other.py", "content": "x = 1\n"}]))
        self.assertEqual(update["terminal_reason"], "fix_path_not_in_work_product")
        self.assertNotIn("fix_refused_next", update["rung_attempts"])
        self.assertEqual(bg.route_after_fix(merged), "human")

    def test_no_higher_rung_left_parks_with_the_reason(self):
        _, update, merged = self.run_rung("rung2", fix_reply([]), rung_attempts={"rung3": 1})
        self.assertEqual(update["terminal_reason"], "fix_empty_files")
        self.assertEqual(bg.route_after_fix(merged), "human")

    def test_marker_is_cleared_when_the_next_rung_runs(self):
        _, first, merged = self.run_rung("rung1", fix_reply([]))
        _, second, merged2 = self.run_rung("rung2", fix_reply([{"path": "m.py", "content": ORIGINAL + "y = 2\n"}]),
                                           rung_attempts=first["rung_attempts"])
        self.assertNotIn("fix_refused_next", second["rung_attempts"])
        self.assertEqual(second["rung_attempts"]["fix_refusals"], ["rung1:fix_empty_files"])
        self.assertEqual(bg.route_after_fix(merged2), "cheap_gate")


class CompiledGraph(unittest.TestCase):
    def test_fix_nodes_can_route_up_the_ladder(self):
        app, _ = bg.build(bg.Deps(over_cap=lambda *a: False))
        edges = {(e.source, e.target) for e in app.get_graph().edges}
        for source in ("fix_rung1", "fix_rung2", "fix_rung3"):
            for target in ("cheap_gate", "human", "fix_rung2", "fix_rung3"):
                with self.subTest(source=source, target=target):
                    self.assertIn((source, target), edges)


class WiderImpact(RungFixture):
    def test_rung2_refusal_reaches_rung3_which_consumes_the_marker(self):
        _, first, _ = self.run_rung("rung2", fix_reply([]), rung_attempts={"rung1": 1})
        self.assertEqual(first["rung_attempts"]["fix_refused_next"], "rung3")
        root = __import__("tests.hermes_cli.test_build_graph_cd102", fromlist=["x"]).workspace_with({"m.py": ORIGINAL})
        codex = Mock(return_value=fix_reply([{"path": "m.py", "content": ORIGINAL + "z = 3\n"}]))
        deps = bg.Deps(codex=codex, over_cap=lambda *a: False, review_default=lambda: "sonnet",
                       workspace=root, changed_files=["m.py"], body="spec",
                       derive=lambda ws: {"ok": True, "diff": "+z\n", "diff_files": 1,
                                          "diff_added_lines": 1, "changed_files": ["m.py"]})
        state = __import__("tests.hermes_cli.test_build_graph_cd102", fromlist=["x"]).state_for(rung_attempts=first["rung_attempts"])
        update = bg.make_fix_rung3(deps)(state)
        merged = dict(state); merged.update(update)
        self.assertIsNone(update.get("terminal_reason"))
        self.assertNotIn("fix_refused_next", update["rung_attempts"])
        self.assertEqual(update["rung_attempts"]["fix_refusals"], ["rung2:fix_empty_files"])
        self.assertEqual(bg.route_after_fix(merged), "cheap_gate")

    def test_successive_refusals_are_all_recorded_in_order(self):
        _, first, _ = self.run_rung("rung1", {"klass": "ok", "result": {"model": "sonnet", "text": "no JSON"}})
        _, second, merged = self.run_rung("rung2", fix_reply([]), rung_attempts=first["rung_attempts"])
        self.assertEqual(second["rung_attempts"]["fix_refusals"], ["rung1:fix_unparseable", "rung2:fix_empty_files"])
        self.assertEqual(bg.route_after_fix(merged), "fix_rung3")

    def test_rung3_refusal_parks_with_its_reason(self):
        _, update, merged = self.run_rung("rung2", fix_reply([]), rung_attempts={"rung1": 1, "rung3": 1})
        self.assertEqual(update["terminal_reason"], "fix_empty_files")
        self.assertEqual(update["rung_attempts"]["fix_refusals"], ["rung2:fix_empty_files"])
        self.assertEqual(bg.route_after_fix(merged), "human")

    def test_capped_rung2_is_skipped(self):
        _, update, merged = self.run_rung("rung1", fix_reply([]), rung_attempts={"rung2": bg.RUNG_ATTEMPT_CAP})
        self.assertEqual(bg.route_after_fix(merged), "fix_rung3")

    def test_a_refusing_rung_is_not_recorded_as_the_author(self):
        _, update, _ = self.run_rung("rung1", {"klass": "ok", "result": {"model": "sonnet", "text": "no"}},
                                     implementer_model="qwen-local")
        self.assertEqual(update["implementer_model"], "qwen-local")
        self.assertNotIn("author_independence", update)

    def test_transport_and_provenance_failures_still_park(self):
        _, update, merged = self.run_rung("rung1", {"klass": "transient", "result": {}})
        self.assertEqual(update["terminal_reason"], "model_call_transient:fix_sonnet")
        self.assertEqual(bg.route_after_fix(merged), "human")
        _, update, merged = self.run_rung("rung1", {"klass": "ok", "result": {"text": "{}"}})
        self.assertEqual(update["terminal_reason"], "fix_model_provenance_unavailable")
        self.assertEqual(bg.route_after_fix(merged), "human")

    def test_next_rung_still_passes_the_spend_gate(self):
        _, first, _ = self.run_rung("rung1", fix_reply([]))
        model = Mock()
        deps = bg.Deps(model=model, over_cap=lambda *a: True, review_default=lambda: "sonnet",
                       workspace=__import__("tests.hermes_cli.test_build_graph_cd102", fromlist=["x"]).workspace_with({"m.py": ORIGINAL}),
                       changed_files=["m.py"])
        state = __import__("tests.hermes_cli.test_build_graph_cd102", fromlist=["x"]).state_for(rung_attempts=first["rung_attempts"])
        update = bg.make_fix(deps, rung="rung2", activity="fix_opus")(state)
        self.assertTrue(update.get("terminal_reason"))
        model.assert_not_called()


class Vocabulary(unittest.TestCase):
    PARK = frozenset({"fix_unsafe_path", "fix_path_not_in_work_product", "fix_path_outside_workspace",
                      "fix_path_not_a_file", "fix_read_failed", "fix_write_failed", "fix_restore_failed",
                      "fix_no_work_product"})

    def test_every_apply_fix_reason_is_classified_exactly_once(self):
        tree = ast.parse(inspect.getsource(bg))
        found = set()
        for name in ("apply_fix", "_content_refusal"):
            fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            for node in ast.walk(fn):
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith("fix_"):
                    found.add(node.value)
        self.assertFalse(bg.FIX_COMPLIANCE_REFUSALS & self.PARK)
        self.assertEqual(found, set(bg.FIX_COMPLIANCE_REFUSALS) | self.PARK, sorted(found ^ (set(bg.FIX_COMPLIANCE_REFUSALS) | self.PARK)))

if __name__ == "__main__":
    unittest.main()
