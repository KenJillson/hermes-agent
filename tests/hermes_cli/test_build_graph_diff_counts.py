"""Real worktree counts survive graph node and command-line boundaries."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock

from hermes_cli import build_graph as bg
from hermes_cli import build_graph_diff as diff
from hermes_cli.build_graph_state import new_workflow_state, selection_signals


class AtModel(Exception):
    pass


class DiffCounts(unittest.TestCase):
    def test_every_graph_model_call_has_state_counts(self):
        for kind in ("review", "re_review", "fix", "classify"):
            with self.subTest(kind=kind):
                state = new_workflow_state("t_fixture", "test", diff="+fixture\n",
                                           diff_files=2, diff_added_lines=7)
                model = Mock(side_effect=AtModel)
                deps = bg.Deps(model=model, over_cap=lambda *a: False,
                               review_default=lambda: "sonnet")
                if kind in ("review", "re_review"):
                    node = bg.make_cloud_review(deps, activity=(
                        "code_review" if kind == "review" else "re_review"),
                        node_name=kind)
                elif kind == "fix":
                    node = bg.make_fix(deps, rung="rung1", activity="fix_sonnet",
                                       prompt_builder=lambda s: "fixture")
                else:
                    state.update(rung="rung1", rung_attempts={"rung1": 1},
                                 objections_prior=[{"description": "a"}],
                                 objections_current=[{"description": "b"}])
                    node = bg.make_classify_failure(deps)
                with self.assertRaises(AtModel):
                    node(state)
                signals = model.call_args.kwargs["signals"]
                self.assertEqual(signals["diff_files"], state["diff_files"])
                self.assertEqual(signals["diff_added_lines"], state["diff_added_lines"])
                self.assertNotIn("large_diff", signals)

    def test_reviews_keep_zero_counts_and_activity_booleans(self):
        state = new_workflow_state("t_fixture", "test", security_sensitive=True)
        for activity in ("plan_review", "code_review", "re_review"):
            signals = selection_signals(state, activity=activity)
            self.assertEqual(signals["diff_files"], 0)
            self.assertEqual(signals["diff_added_lines"], 0)
            self.assertEqual(signals.get("security_sensitive", False),
                             activity != "re_review")

    def test_real_diff_counts_cross_subprocess_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo, workspace = root / "repo", root / "workspace"
            repo.mkdir()

            def git(*args):
                subprocess.run(["git", "-c", "core.hooksPath=/dev/null",
                    "-c", "user.name=Fixture", "-c", "user.email=fixture@invalid",
                    "-C", str(repo), *args], check=True, capture_output=True)

            git("init", "--initial-branch=main")
            (repo / "sample.py").write_text("value = 1\n")
            git("add", "sample.py")
            git("commit", "-m", "fixture")
            git("worktree", "add", "-b", "fixture", str(workspace), "main")
            (workspace / "sample.py").write_text("value = 2\nextra = 3\n")
            derived = diff.derive(str(workspace))
            self.assertTrue(derived["ok"], derived)
            state = new_workflow_state("t_fixture", "test", diff=derived["diff"],
                diff_files=derived["diff_files"], diff_added_lines=derived["diff_added_lines"])
            exe = root / "capture"
            exe.write_text("#!" + sys.executable + "\nimport json,sys\n"
                           "print(json.dumps({'ok':True,'argv':sys.argv[1:]}))\n")
            exe.chmod(0o700)
            for signals in (selection_signals(state, activity="code_review"),
                            {"diff_files": 0, "diff_added_lines": 0}):
                result = bg.call_model(activity="code_review", prompt="fixture",
                    workspace=str(workspace), card_id="t_fixture", exe=str(exe),
                    signals=signals)
                self.assertEqual(result["klass"], "ok")
                argv = result["result"]["argv"]
                for name in ("diff_files", "diff_added_lines"):
                    self.assertEqual(argv[argv.index("--" + name.replace("_", "-")) + 1],
                                     str(signals[name]))
                self.assertNotIn("--signal", argv)
