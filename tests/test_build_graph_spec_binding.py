"""Exact specification provenance at the implementation runner boundary.

The fixture retains the actual pre-implementation state from the failed trial.
Only its filesystem scope is relocated and coherently rebound for each test.
The real acceptance verifier, receipt renderer and goal builder run; no provider,
graph dispatch, task database or production checkpoint is touched.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from hermes_cli import build_graph as graph
from hermes_cli import build_graph_plan_connection as connection


FIXTURE = Path(__file__).parent / "fixtures/build_graph_spec_binding_state.json"
FILES = ["hermes_cli/build_graph_checks.py",
         "tests/test_build_graph_checks_distinct_paths.py"]


class BoundaryRunner:
    def __init__(self, state):
        self.run_id = 226
        self.source = state["plan_acceptance"]["source_sha256"]
        self.files = FILES.copy()
        self.capability = copy.deepcopy(
            state["rung_attempts"]["a2_capability_evidence"]["plan_review"]["profile"])
        self.calls = []
        self.binding_calls = 0

    def binding_inputs(self):
        self.binding_calls += 1
        return self.files, self.source

    def capability_evidence(self, state):
        return self.capability

    def stage(self, **kwargs):
        raise AssertionError("Implementation must use the protected runner boundary")

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        # End the offline test immediately after the protected runner boundary.
        return {"status": "interrupted"}


class SpecificationBindingTests(unittest.TestCase):
    def setUp(self):
        fixture = json.loads(FIXTURE.read_text())
        self.original_body = fixture["body"]
        self.state = fixture["state"]
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = str(Path(self.temp.name).resolve())
        receipt = self.state["plan_acceptance"]["cloud_receipt"]
        self.envelope = (Path(self.workspace) / ".graph-checkpoints" /
                         "model-call-fixture" /
                         f"{self.state['card_id']}-{receipt['run_id']}-result.json")
        self.envelope.parent.mkdir(parents=True)
        self.envelope.write_text(fixture["cloud_envelope"])
        receipt["envelope_path"] = str(self.envelope)
        self.runner = BoundaryRunner(self.state)
        self.bind(self.original_body)

    def bind(self, body):
        """Relocate real receipts without replacing validation with a stub."""
        self.body = body
        for row in self.state["rung_attempts"]["a2_denial_evidence"].values():
            row["workspace"] = self.workspace
            row["spec_sha256"] = connection.digest(body)
        old = self.state["plan_acceptance"]
        self.state["plan_acceptance"] = connection.binding(
            self.state, body, self.workspace, self.runner.source, FILES,
            status="cloud_accepted", guidance=old["guidance"],
            cloud=old["cloud_receipt"], capability=self.runner.capability)

    def invoke(self, body=None):
        deps = graph.Deps(implement_runner=self.runner, workspace=self.workspace,
                          body=self.body if body is None else body,
                          cloud_plan_review=True)
        return graph.make_implement(deps)(self.state)

    def assert_rejected_before_attempt(self, body=None):
        before = copy.deepcopy(self.state)
        result = self.invoke(body)
        self.assertEqual(result, {"terminal_reason": "plan_acceptance_stale_or_missing"})
        self.assertEqual(self.runner.calls, [])
        self.assertNotIn("rung_attempts", result)
        self.assertEqual(self.state, before)

    def assert_runner_receives_exact_body(self, body):
        self.bind(body)
        before = copy.deepcopy(self.state)
        result = self.invoke()
        self.assertEqual(result["terminal_reason"], "implement_interrupted")
        self.assertEqual(len(self.runner.calls), 1)
        self.assertEqual(result["rung_attempts"]["implement"], 1)
        self.assertEqual(self.state, before)
        call = self.runner.calls[0]
        self.assertIn("SPECIFICATION:\n" + body + "\n\nACCEPTED QWEN PLAN:\n", call["goal"])
        self.assertIn(self.state["plan"], call["goal"])
        self.assertIn(self.state["plan_acceptance"]["guidance"][0]["action"], call["goal"])
        self.assertEqual(call["expected_source_sha256"], self.runner.source)
        self.assertEqual(call["expected_policy_sha256"], self.runner.capability["policy_sha256"])

    def test_actual_trailing_newline_reaches_runner_with_original_spec(self):
        self.assertTrue(self.original_body.endswith("\n"))
        self.assert_runner_receives_exact_body(self.original_body)

    def test_leading_and_trailing_whitespace_preserves_provenance(self):
        self.assert_runner_receives_exact_body("\n \t" + self.original_body + "\t\n")

    def test_body_without_edge_whitespace_still_reaches_runner(self):
        self.assert_runner_receives_exact_body(self.original_body.strip())

    def test_empty_and_whitespace_only_body_stops_before_binding_and_attempt(self):
        for body in ("", " \t\n"):
            with self.subTest(body=body):
                before = copy.deepcopy(self.state)
                self.assertEqual(self.invoke(body), {"terminal_reason": "implement_no_spec"})
                self.assertEqual(self.runner.binding_calls, 0)
                self.assertEqual(self.runner.calls, [])
                self.assertEqual(self.state, before)

    def test_changed_spec_remains_rejected(self):
        self.assert_rejected_before_attempt(self.body + "Changed requirement.")

    def test_changed_plan_remains_rejected(self):
        self.state["plan"] += "\nChanged approach."
        self.assert_rejected_before_attempt()

    def test_changed_source_remains_rejected(self):
        self.runner.source = "0" * 64
        self.assert_rejected_before_attempt()

    def test_changed_declared_files_remain_rejected(self):
        self.runner.files.append("unexpected.py")
        self.assert_rejected_before_attempt()

    def test_changed_guidance_remains_rejected_even_with_updated_guidance_hash(self):
        record = self.state["plan_acceptance"]
        record["guidance"][0]["action"] += " Tampered."
        record["guidance_sha256"] = connection.sha(record["guidance"])
        self.state["rung_attempts"]["cloud_plan_review_evidence"]["guidance_sha256"] = record["guidance_sha256"]
        self.assert_rejected_before_attempt()

    def test_changed_review_identity_remains_rejected(self):
        self.state["rung_attempts"]["cloud_plan_review_evidence"]["run_id"] = "00000000"
        self.assert_rejected_before_attempt()

    def test_changed_review_contract_remains_rejected(self):
        self.state["review_contract_sha256"] = "0" * 64
        self.assert_rejected_before_attempt()

    def test_changed_capability_remains_rejected(self):
        self.runner.capability["policy_sha256"] = "0" * 64
        self.assert_rejected_before_attempt()

    def test_changed_planning_receipt_identity_remains_rejected(self):
        self.state["rung_attempts"]["a2_runs"]["plan"] += 1000
        self.assert_rejected_before_attempt()

    def test_changed_retained_envelope_remains_rejected(self):
        with self.envelope.open("a") as stream:
            stream.write(" ")
        self.assert_rejected_before_attempt()

    def test_exhausted_attempt_cap_still_prevents_runner_call(self):
        self.state["rung_attempts"]["implement"] = graph.IMPLEMENT_ATTEMPT_CAP
        before = copy.deepcopy(self.state)
        self.assertEqual(self.invoke(), {"terminal_reason": "implement_attempt_cap"})
        self.assertEqual(self.runner.calls, [])
        self.assertEqual(self.state, before)


if __name__ == "__main__":
    unittest.main()
