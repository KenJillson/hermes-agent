"""CD-095: the A2 implement failure keeps the worker's stage and reason.

Stdlib only. The graph helper is tested directly; the two call sites are pinned
by AST so a revert to the collapsed 'implement_failed' cannot pass silently.
"""
import ast
import importlib.util
from pathlib import Path
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[2]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


IMPL = _load('cd095_implementer', 'hermes_cli/build_graph_implementer.py')
FAILURE = {'stage': 'tool_accounting', 'reason': 'incomplete', 'exit_code': 85}


def _helper():
    """Extract a2_failure_reason and its regex from build_graph.py by AST, so
    the test needs none of the graph's runtime dependencies."""
    tree = ast.parse((ROOT / 'hermes_cli/build_graph.py').read_text())
    nodes = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name == 'a2_failure_reason')
             or (isinstance(n, ast.Assign) and any(getattr(t, 'id', None) == 'A2_FAILURE_TOKEN' for t in n.targets))]
    assert len(nodes) == 2, 'a2_failure_reason and A2_FAILURE_TOKEN must both exist exactly once'
    scope = {'re': __import__('re')}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'build_graph-extract', 'exec'), scope)
    return scope['a2_failure_reason']


class GraphReason(unittest.TestCase):
    def setUp(self):
        self.reason = _helper()

    def test_exact_failure_maps_to_stage_form(self):
        self.assertEqual(self.reason('implement', {'status': 'failed', 'failure': FAILURE}),
                         'a2_stage_failed:implement:tool_accounting:incomplete')
        self.assertEqual(self.reason('implement', {'failure': {'stage': 'stage_result', 'reason': 'json',
                                                               'exit_code': 84}}),
                         'a2_stage_failed:implement:stage_result:json')

    def test_inexact_failure_is_none(self):
        bad = [None, {}, {'failure': None}, {'failure': 'x'},
               {'failure': dict(FAILURE, extra=1)},
               {'failure': {'stage': 'tool_accounting', 'reason': 'incomplete'}},
               {'failure': dict(FAILURE, exit_code='85')}, {'failure': dict(FAILURE, exit_code=True)},
               {'failure': dict(FAILURE, stage='Tool')}, {'failure': dict(FAILURE, stage='a:b')},
               {'failure': dict(FAILURE, reason='x y')}, {'failure': dict(FAILURE, reason='')},
               {'failure': dict(FAILURE, reason='a' * 65)}, {'failure': dict(FAILURE, stage=1)},
               {'failure': dict(FAILURE, reason='incomplete\n')}]
        for raw in bad:
            with self.subTest(raw=raw):
                self.assertIsNone(self.reason('implement', raw))


class ImplementerFailed(unittest.TestCase):
    def test_exact_failure_is_kept(self):
        out = IMPL._failed({'phase': 'failed', 'operation': 'implement', 'failure': dict(FAILURE)})
        self.assertEqual(out['status'], 'failed')
        self.assertEqual(out['failure'], FAILURE)
        self.assertEqual(set(out), {'status', 'identity', 'evidence_path', 'failure'})

    def test_inexact_failure_is_plain_failed(self):
        for result in ({'phase': 'failed'}, {'phase': 'failed', 'failure': None},
                       {'phase': 'failed', 'failure': dict(FAILURE, extra=1)},
                       {'phase': 'failed', 'failure': dict(FAILURE, exit_code='85')}, None):
            with self.subTest(result=result):
                self.assertEqual(IMPL._failed(result), IMPL._closed('failed'))


class CallSites(unittest.TestCase):
    def test_runner_failed_branch_uses_failed(self):
        source = (ROOT / 'hermes_cli/build_graph_implementer.py').read_text()
        self.assertIn("if result.get('phase') == 'failed':\n                return _failed(result)", source)
        self.assertNotIn("if result.get('phase') == 'failed':\n                return _closed('failed')", source)

    def test_graph_maps_only_a2_failed(self):
        source = (ROOT / 'hermes_cli/build_graph.py').read_text()
        self.assertIn('if status == "failed" and callable(getattr(runner, \'stage\', None)):\n'
                      '                reason = a2_failure_reason("implement", raw)', source)
        # The non-A2 (A1 owned subprocess) path keeps its closed vocabulary.
        self.assertIn('return _park("implement_%s" % status)', source)


if __name__ == '__main__':
    unittest.main()
