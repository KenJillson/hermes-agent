"""CD-098 (D5.5): blocking Ruff threshold E9,F63,F7,F82 in the cheap gate.

The observation-only check (CD-063) is unchanged and keeps its index; the new
enforcing check is appended after it. Behavioural cases run the real Ruff when
one is installed (the box's /usr/local/bin/ruff, or RUFF_FOR_TESTS), through
the real ac_check_runner with isolation disabled.
"""
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from hermes_cli import ac_check_runner as runner
from hermes_cli import build_graph_checks as checks


def workspace(files):
    root = Path(tempfile.mkdtemp())
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


class SpecTests(unittest.TestCase):
    def test_blocking_spec_follows_unchanged_observation(self):
        root = workspace({'a.py': 'x = 1\n', 'notes.md': 'n\n'})
        specs = checks.synthetic_specs(str(root), ['a.py', 'notes.md'])
        self.assertEqual([s['index'] for s in specs], [900, 901, 902, 903, 904])
        observation, blocking = specs[3], specs[4]
        self.assertIs(observation.get('observation_only'), True)
        self.assertIn('--select E4,E7,E9,F', observation['check']['command'])
        self.assertIn('--output-format json', observation['check']['command'])
        self.assertNotIn('observation_only', blocking)
        command = blocking['check']['command']
        self.assertTrue(command.startswith(checks.RUFF + ' check --isolated --no-cache --no-fix '))
        self.assertIn('--select E9,F63,F7,F82', command)
        self.assertIn('-- ./a.py', command)
        self.assertEqual((blocking['check']['expect_kind'], blocking['check']['expect_value']), ('exit', '0'))
        self.assertEqual(blocking.get('synthetic'), True)

    def test_no_python_no_lint(self):
        root = workspace({'notes.md': 'n\n'})
        specs = checks.synthetic_specs(str(root), ['notes.md'])
        self.assertFalse(any('ruff' in s['check']['command'] for s in specs))

    def test_unsafe_paths_excluded(self):
        root = workspace({'a.py': 'x = 1\n'})
        spec = checks.lint_blocking_spec(str(root), ['a.py', '../escape.py', '/abs.py'], 950)
        self.assertIn('-- ./a.py', spec['check']['command'])
        self.assertNotIn('escape', spec['check']['command'])
        self.assertIn('2 unsafe path(s) excluded', spec['text'])
        self.assertIsNone(checks.lint_blocking_spec(str(root), ['../escape.py'], 950))

    def test_enforcing_count_includes_blocking_lint_only(self):
        root = workspace({'a.py': 'x = 1\n'})
        parse = lambda body: (1, 0, 'derived', [{'index': 0, 'kind': 'check'}])
        count, _, _, results = checks.wrap_parse_fn(parse, str(root), ['a.py'])('body')
        self.assertEqual(len(results), 1 + 5)
        self.assertEqual(count, 1 + 4)  # syntax, tirith x2, blocking lint; observation excluded


RUFF = os.environ.get('RUFF_FOR_TESTS') or ('/usr/local/bin/ruff' if os.path.exists('/usr/local/bin/ruff')
                                             else shutil.which('ruff'))


@unittest.skipUnless(RUFF, 'ruff not installed')
class RealRuffTests(unittest.TestCase):
    def verdict(self, text):
        root = workspace({'m.py': text})
        old = checks.RUFF
        checks.RUFF = RUFF
        try:
            spec = checks.lint_blocking_spec(str(root), ['m.py'], 904)
        finally:
            checks.RUFF = old
        verdicts, summary = runner.run_ac_checks([spec], str(root), isolation='none')
        return verdicts[0], summary

    def test_clean_passes(self):
        v, s = self.verdict('def f(x):\n    return x\n')
        self.assertEqual((v['status'], s['verdict']), ('passed', 'all_pass'))

    def test_hygiene_findings_do_not_block(self):
        # F401 (the only finding class in the census) and F841 stay observation-only.
        v, s = self.verdict('import os\n\ndef f():\n    unused = 1\n    return 2\n')
        self.assertEqual((v['status'], s['verdict']), ('passed', 'all_pass'))

    def test_undefined_name_blocks(self):
        v, s = self.verdict('def f():\n    return missing_name\n')
        self.assertEqual((v['status'], s['verdict']), ('failed', 'exception'))
        self.assertIn('F821', v.get('stdout_tail', '') + v.get('stdout', ''))

    def test_misplaced_return_blocks(self):
        v, s = self.verdict('return 1\n')
        self.assertEqual(v['status'], 'failed')

    def test_is_literal_blocks(self):
        v, s = self.verdict('def f(x):\n    return x is "a"\n')
        self.assertEqual(v['status'], 'failed')

    def test_syntax_error_blocks(self):
        v, s = self.verdict('def f(:\n    pass\n')
        self.assertEqual(v['status'], 'failed')

    def test_missing_ruff_fails_closed(self):
        root = workspace({'m.py': 'x = 1\n'})
        old = checks.RUFF
        checks.RUFF = '/nonexistent/ruff'
        try:
            spec = checks.lint_blocking_spec(str(root), ['m.py'], 904)
        finally:
            checks.RUFF = old
        verdicts, summary = runner.run_ac_checks([spec], str(root), isolation='none')
        self.assertNotEqual(verdicts[0]['status'], 'passed')
        self.assertEqual(summary['verdict'], 'exception')


if __name__ == '__main__':
    unittest.main()
