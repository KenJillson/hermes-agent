"""CD-104: the graph's wait for codex-call is the named, raised bound. Provider-free."""
import json
import subprocess
import unittest
from unittest.mock import patch

from hermes_cli import build_graph as bg


class CodexOuterBound(unittest.TestCase):
    def test_call_codex_waits_the_named_bound(self):
        done = subprocess.CompletedProcess([], 0, stdout=json.dumps({"ok": True}), stderr="")
        with patch.object(bg.subprocess, "run", return_value=done) as run:
            out = bg.call_codex(activity="fix_codex", prompt="p", card_id="t_x", component="main")
        self.assertEqual(out["klass"], "ok")
        self.assertEqual(run.call_args.kwargs["timeout"], bg.CODEX_CALL_SECONDS)

    def test_bound_outlives_the_config_chain(self):
        # F-W1c (config): bridge 540 s, then collect 30 s and ledger append 30 s.
        self.assertEqual(bg.CODEX_CALL_SECONDS, 660)
        self.assertGreater(bg.CODEX_CALL_SECONDS, 540 + 30 + 30)

    def test_a_timeout_still_fails_closed_with_collection_required(self):
        with patch.object(bg.subprocess, "run", side_effect=subprocess.TimeoutExpired("codex-call", 1)):
            out = bg.call_codex(activity="fix_codex", prompt="p", card_id="t_x", component="main")
        self.assertEqual(out["klass"], "failed")
        self.assertTrue(out["result"]["collection_required"])


if __name__ == "__main__":
    unittest.main()
