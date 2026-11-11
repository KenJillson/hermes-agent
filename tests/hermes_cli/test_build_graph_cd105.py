"""CD-105 (F-T14g-1): a protected local fix round that fails on the local model's own
account (timeout, abnormal completion, tool accounting) hands the card to cloud rung 1
instead of parking it; every other failure still parks with its exact reason.
Provider-free: the protected runner is a fake returning the controller's failure shape.

Measured basis: Test14g fix1 (run 297) ran past the 600 s payload budget. With the
payload watchdog that becomes `a2_stage_failed:fix1:conversation:timeout`, which before
this change parked the card and never reached the ladder it exists to qualify.
"""
import unittest
from unittest.mock import Mock, patch

from hermes_cli import build_graph as bg
from hermes_cli import build_graph_stages as stages

SPEC = "Write m.py so that value == 2.\n"
PLAN = "1. edit m.py\n"
FILES = ["m.py"]
SOURCE = "source-after-implement"


def failure(stage, reason):
    return {"status": "failed",
            "failure": {"stage": stage, "reason": reason, "exit_code": 74}}


class Fixture(unittest.TestCase):
    def node_state(self, **attempts):
        acceptance = {"plan_sha256": bg.connection.digest(PLAN),
                      "spec_sha256": bg.connection.digest(SPEC),
                      "files_sha256": bg.connection.sha(FILES)}
        base = {stages.LOCAL_FIX_PENDING: True, "local_fix": 0,
                "a2_source_chain": {"implement": SOURCE,
                                    "plan_acceptance_sha256": bg.connection.sha(acceptance)}}
        base.update(attempts)
        return {"rung_attempts": base, "plan": PLAN, "plan_acceptance": acceptance}

    def run_fix(self, raw, instance="fix1", **attempts):
        runner = Mock(return_value=raw)
        runner.stage = Mock()
        runner.run_id = 297
        runner.binding_inputs = Mock(return_value=(FILES, SOURCE))
        runner.capability_evidence = Mock(return_value={"policy_sha256": "p" * 64})
        deps = bg.Deps(over_cap=lambda *a: False)
        deps.body = SPEC
        deps.implement_runner = runner
        deps.workspace = "/ws"
        deps.agent = None
        state = self.node_state(**attempts)
        if instance == "fix2":
            state["rung_attempts"]["local_fix"] = 1
            state["rung_attempts"]["a2_source_chain"]["fix1"] = SOURCE
            state["rung_attempts"]["a2_source_chain"].pop("implement")
            state["rung_attempts"]["a2_source_chain"]["implement"] = "older"
        with patch.object(stages, "record_stage", return_value={"a2_runs": {}}), \
                patch.object(bg, "build_local_fix_goal", return_value="goal"):
            update = bg.make_local_fix(deps, instance)(state)
        merged = dict(state, **update)
        return runner, update, merged


class Escalation(Fixture):
    def test_timeout_escalates_to_rung1(self):
        runner, update, merged = self.run_fix(failure("conversation", "timeout"))
        runner.assert_called_once()
        self.assertIsNone(update.get("terminal_reason"))
        attempts = update["rung_attempts"]
        self.assertIs(attempts[stages.LOCAL_FIX_PENDING], False)
        self.assertEqual(attempts["local_fix"], 1)
        self.assertEqual(attempts["local_fix_escalated"], "fix1")
        self.assertEqual(attempts["local_fix_failures"], ["a2_stage_failed:fix1:conversation:timeout"])
        self.assertEqual(bg.route_after_local_fix(merged), "fix_rung1")

    def test_every_model_owned_stage_escalates(self):
        for stage in sorted(bg.LOCAL_FIX_ESCALATE_STAGES):
            with self.subTest(stage=stage):
                _, update, merged = self.run_fix(failure(stage, "exception"))
                self.assertEqual(bg.route_after_local_fix(merged), "fix_rung1")
                self.assertIsNone(update.get("terminal_reason"))

    def test_fix2_failure_also_escalates(self):
        _, update, merged = self.run_fix(failure("completion_flags", "exception"), instance="fix2")
        self.assertEqual(update["rung_attempts"]["local_fix_escalated"], "fix2")
        self.assertEqual(bg.route_after_local_fix(merged), "fix_rung1")


class StillParks(Fixture):
    def test_infrastructure_and_output_failures_park_with_exact_reason(self):
        for stage, reason in (("request", "exception"), ("dependencies", "exception"),
                              ("inputs", "exception"), ("agent_init", "exception"),
                              ("agent_close", "exception"),
                              ("artifact_collect", "outside_declared_scope"),
                              ("artifact_transport", "timeout"), ("stage_result", "invalid_result")):
            with self.subTest(stage=stage):
                _, update, merged = self.run_fix(failure(stage, reason))
                self.assertEqual(update["terminal_reason"],
                                 "a2_stage_failed:fix1:%s:%s" % (stage, reason))
                self.assertNotIn("local_fix_escalated", update["rung_attempts"])
                self.assertEqual(bg.route_after_local_fix(merged), "human")

    def test_unshaped_failure_parks_as_before(self):
        _, update, merged = self.run_fix({"status": "failed"})
        self.assertEqual(update["terminal_reason"], "local_fix_failed")
        self.assertEqual(bg.route_after_local_fix(merged), "human")

    def test_spent_rung1_parks_with_the_failure_reason(self):
        _, update, merged = self.run_fix(failure("conversation", "timeout"), rung1=1)
        self.assertEqual(update["terminal_reason"], "a2_stage_failed:fix1:conversation:timeout")
        self.assertEqual(bg.route_after_local_fix(merged), "human")

    def test_interrupted_and_other_statuses_unchanged(self):
        for status in sorted(set(bg.IMPLEMENT_STATUSES) - {bg.IMPLEMENT_OK, "failed"}):
            with self.subTest(status=status):
                _, update, merged = self.run_fix({"status": status})
                self.assertEqual(update["terminal_reason"], "local_fix_%s" % status)
                self.assertEqual(bg.route_after_local_fix(merged), "human")


class Router(unittest.TestCase):
    def test_success_still_goes_to_the_gate(self):
        self.assertEqual(bg.route_after_local_fix({"rung_attempts": {"a2_handoff": "local_review2"}}),
                         "cheap_gate")

    def test_terminal_wins(self):
        self.assertEqual(bg.route_after_local_fix({"terminal_reason": "x",
                                                   "rung_attempts": {"local_fix_escalated": "fix1"}}),
                         "human")


class CompiledGraph(unittest.TestCase):
    def test_local_fix_nodes_reach_rung1(self):
        app, _ = bg.build(bg.Deps(over_cap=lambda *a: False))
        edges = {(e.source, e.target) for e in app.get_graph().edges}
        for source in ("fix1", "fix2"):
            for target in ("cheap_gate", "human", "fix_rung1"):
                self.assertIn((source, target), edges)


if __name__ == "__main__":
    unittest.main()
