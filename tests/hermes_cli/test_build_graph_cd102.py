"""CD-102: fix rungs are shown the files they must return whole; apply_fix refuses
content that cannot be a repair; escalation is CD-103 and is not tested here.

Provider-free. Measured basis: Test14 Codex measurement run 8c615804 returned a
refusal sentence as the complete content of a modified 18.9 KB file it had only
seen as two diff hunks.
"""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from hermes_cli import build_graph as bg
from hermes_cli.build_graph_state import new_workflow_state
from tests.hermes_cli.expected_model_fixtures import contract_kwargs

ORIGINAL = "".join("def f%d():\n    return %d\n\n" % (i, i) for i in range(200))  # > 2 KB
REFUSAL = ("Cannot produce a valid replacement: the required QUALIFICATION_TOKEN plaintext and "
           "the file's complete existing contents are absent from the supplied evidence.")


def workspace_with(files):
    root = tempfile.mkdtemp()
    for rel, text in files.items():
        path = Path(root, rel); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text)
    return root


def state_for(**extra):
    state = new_workflow_state("t_cd102", "main", diff="--- a/m.py\n+++ b/m.py\n@@ hunk only @@\n",
                               **contract_kwargs("t_cd102", "main"))
    state.update(extra)
    return state


def fix_reply(files):
    return {"klass": "ok", "result": {"model": "sonnet", "text": json.dumps({"files": files})}}


class RungFixture(unittest.TestCase):
    def run_rung(self, rung, reply, **state_extra):
        root = workspace_with({"m.py": ORIGINAL})
        model = Mock(return_value=reply)
        deps = bg.Deps(model=model, over_cap=lambda *a: False, review_default=lambda: "sonnet",
                       workspace=root, changed_files=["m.py"],
                       derive=lambda ws: {"ok": True, "diff": "+x\n", "diff_files": 1,
                                          "diff_added_lines": 1, "changed_files": ["m.py"]})
        state = state_for(**state_extra)
        activity = {"rung1": "fix_sonnet", "rung2": "fix_opus"}[rung]
        update = bg.make_fix(deps, rung=rung, activity=activity)(state)
        merged = dict(state); merged.update(update)
        return model, update, merged


class RenderCurrentFiles(unittest.TestCase):
    def test_complete_contents_sorted_and_absent_named(self):
        root = workspace_with({"pkg/m.py": ORIGINAL, "a.txt": "alpha\n"})
        text, why = bg.render_current_files(root, ["pkg/m.py", "a.txt", "gone.py", "a.txt"])
        self.assertIsNone(why)
        self.assertIn(ORIGINAL, text)
        self.assertLess(text.index("--- FILE a.txt"), text.index("--- FILE pkg/m.py"))
        self.assertIn("--- FILE gone.py: not present in the worktree ---", text)
        self.assertEqual(text.count("--- FILE a.txt ("), 1)

    def test_refusals_are_closed_reasons(self):
        root = workspace_with({"m.py": "x = 1\n"})
        Path(root, "bin.dat").write_bytes(b"\xff\xfe\x00")
        Path(root, "big.txt").write_text("x" * (bg.MAX_CURRENT_FILES_BYTES + 1))
        cases = {"fix_no_work_product": [], "fix_context_unsafe_path": ["../m.py"],
                 "fix_context_unreadable": ["bin.dat"], "fix_context_bounds": ["big.txt"]}
        for reason, paths in cases.items():
            with self.subTest(reason=reason):
                self.assertEqual(bg.render_current_files(root, paths), ("", reason))
        self.assertEqual(bg.render_current_files(root, ["/etc/passwd"])[1], "fix_context_unsafe_path")

    def test_mixed_or_wrong_collection_types_never_raise(self):
        root = workspace_with({"m.py": "x = 1\n"})
        for paths in (["m.py", 3], ["m.py", None], ["", "m.py"], "m.py", {"m.py": 1}, 7):
            with self.subTest(paths=paths):
                self.assertEqual(bg.render_current_files(root, paths), ("", "fix_context_unsafe_path"))
        self.assertEqual(bg.render_current_files(root, None), ("", "fix_no_work_product"))


class FixPrompt(unittest.TestCase):
    def test_section_sits_between_diff_and_checks_and_default_is_unchanged(self):
        state = state_for()
        self.assertEqual(bg.build_fix_prompt(state), bg.build_fix_prompt(state, current_files=""))
        prompt = bg.build_fix_prompt(state, current_files="CURRENT FILE CONTENTS marker\n")
        self.assertLess(prompt.index("DIFF:"), prompt.index("CURRENT FILE CONTENTS marker"))
        self.assertLess(prompt.index("CURRENT FILE CONTENTS marker"), prompt.index("RESPONSE CONTRACT"))


class ApplyFixContent(unittest.TestCase):
    def test_refusal_sentence_as_python_is_refused_and_nothing_written(self):
        root = workspace_with({"m.py": ORIGINAL})
        written, why = bg.apply_fix(root, json.dumps({"files": [{"path": "m.py", "content": REFUSAL}]}), ["m.py"])
        self.assertEqual((written, why), (0, "fix_content_invalid_python"))
        self.assertEqual(Path(root, "m.py").read_text(), ORIGINAL)

    def test_drastic_shrink_is_refused(self):
        root = workspace_with({"notes.txt": "n" * 4096})
        written, why = bg.apply_fix(root, json.dumps({"files": [{"path": "notes.txt", "content": "n" * 100}]}), ["notes.txt"])
        self.assertEqual((written, why), (0, "fix_content_shrunk"))

    def test_valid_complete_replacement_is_written(self):
        root = workspace_with({"m.py": ORIGINAL})
        new = ORIGINAL + 'QUALIFICATION_TOKEN = "candidate"\n'
        written, why = bg.apply_fix(root, json.dumps({"files": [{"path": "m.py", "content": new}]}), ["m.py"])
        self.assertEqual((written, why), (1, None))
        self.assertEqual(Path(root, "m.py").read_text(), new)

    def test_all_or_nothing_holds_when_a_later_file_is_refused(self):
        root = workspace_with({"a.py": "a = 1\n", "b.py": ORIGINAL})
        files = [{"path": "a.py", "content": "a = 2\n"}, {"path": "b.py", "content": REFUSAL}]
        self.assertEqual(bg.apply_fix(root, json.dumps({"files": files}), ["a.py", "b.py"]),
                         (0, "fix_content_invalid_python"))
        self.assertEqual(Path(root, "a.py").read_text(), "a = 1\n")


class Rung102(RungFixture):
    def test_rung_prompt_carries_the_complete_file(self):
        model, _, _ = self.run_rung("rung1", fix_reply([{"path": "m.py", "content": ORIGINAL + "x = 1\n"}]))
        self.assertIn(ORIGINAL, model.call_args.kwargs["prompt"])

    def test_content_refusal_parks_when_no_rung_is_left(self):
        # CD-103 escalates a content refusal while a higher rung remains; with none left it parks.
        _, update, merged = self.run_rung("rung1", fix_reply([{"path": "m.py", "content": REFUSAL}]),
                                          rung_attempts={"rung2": 1, "rung3": 1})
        self.assertEqual(update["terminal_reason"], "fix_content_invalid_python")
        self.assertEqual(bg.route_after_fix(merged), "human")

    def test_context_refusal_happens_before_any_spend(self):
        model = Mock()
        deps = bg.Deps(model=model, over_cap=lambda *a: False, review_default=lambda: "sonnet",
                       workspace=workspace_with({}), changed_files=[])
        update = bg.make_fix(deps, rung="rung1", activity="fix_sonnet")(state_for())
        self.assertEqual(update, {"terminal_reason": "fix_no_work_product"})
        model.assert_not_called()


class Rung3Prompt(unittest.TestCase):
    def test_dispute_prompt_carries_the_complete_file(self):
        root = workspace_with({"m.py": ORIGINAL})
        codex = Mock(return_value=fix_reply([{"path": "m.py", "content": ORIGINAL + "z = 3\n"}]))
        deps = bg.Deps(codex=codex, over_cap=lambda *a: False, review_default=lambda: "sonnet",
                       workspace=root, changed_files=["m.py"], body="SPEC BODY",
                       derive=lambda ws: {"ok": True, "diff": "+z\n", "diff_files": 1,
                                          "diff_added_lines": 1, "changed_files": ["m.py"]})
        state = state_for(plan="plan text", objections_prior=[], objections_current=[])
        bg.make_fix_rung3(deps)(state)
        self.assertIn(ORIGINAL, codex.call_args.kwargs["prompt"])

if __name__ == "__main__":
    unittest.main()
