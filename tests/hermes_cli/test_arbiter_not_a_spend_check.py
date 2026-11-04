"""CD-096: defect B2-09 / section 6.1 stays closed.

arbiter_decision is AC-verdict-shaped: it returns PASS on all_pass BEFORE the
dollar ceiling is read. That is correct only while it is used for what it is --
the post-gate routing decision -- and the ceiling is enforced inside every
metered node by spend_gate (CD-045, wiring asserted by self-cert anchor
CD-045b). These tests pin the first half: the arbiter has exactly the call
sites it has today, so it cannot quietly be reused as a pre-spend check, which
would be a check that structurally cannot fire on a passing card.

Stdlib only; sources are read as text and parsed, never imported, except
spend_accounting, which needs only the standard library.
"""
import ast
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
SOURCES = sorted(ROOT.glob('hermes_cli/*.py')) + [ROOT / 'cli.py']


def _scan():
    """(file, enclosing qualname, kind) for every arbiter reference."""
    found, calls = [], []
    for path in SOURCES:
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        rel = path.relative_to(ROOT).as_posix()
        stack = []

        class Visit(ast.NodeVisitor):
            def scoped(self, node):
                stack.append(node.name)
                self.generic_visit(node)
                stack.pop()
            visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = scoped

            def visit_Name(self, node):
                if node.id == 'arbiter_decision':
                    found.append((rel, '.'.join(stack), 'name'))

            def visit_Attribute(self, node):
                if node.attr == 'arbiter_decision':
                    found.append((rel, '.'.join(stack), 'attribute'))
                self.generic_visit(node)

            def visit_ImportFrom(self, node):
                if any(alias.name == 'arbiter_decision' for alias in node.names):
                    found.append((rel, '.'.join(stack), 'import'))

            def visit_Call(self, node):
                if isinstance(node.func, ast.Attribute) and node.func.attr == 'arbiter':
                    calls.append((rel, '.'.join(stack)))
                self.generic_visit(node)
        Visit().visit(tree)
    return found, calls


class ArbiterCallSites(unittest.TestCase):
    def test_references_are_exactly_todays(self):
        found, _ = _scan()
        self.assertEqual(sorted(found), sorted([
            ('hermes_cli/build_graph.py', 'Deps.resolve_real', 'import'),
            ('hermes_cli/build_graph.py', 'Deps.resolve_real', 'name'),
            ('hermes_cli/kanban_db.py', 'complete_task', 'attribute'),
        ]), 'arbiter_decision gained or lost a call site: it is a routing decision, '
            'not a spend check -- use spend_gate (CD-045) before a metered call')

    def test_graph_consults_the_arbiter_only_at_the_gate_router(self):
        _, calls = _scan()
        self.assertEqual(calls, [('hermes_cli/build_graph.py', 'route_after_gate')])

    def test_board_use_is_observe_only_dark_launch(self):
        tree = ast.parse((ROOT / 'hermes_cli/kanban_db.py').read_text(encoding='utf-8'))
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        sites = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Attribute) and n.attr == 'arbiter_decision']
        self.assertEqual(len(sites), 1)
        node, guarded = sites[0], False
        while node in parents:
            node = parents[node]
            if isinstance(node, ast.If) and 'HERMES_KANBAN_ARBITER_DARKLAUNCH' in ast.unparse(node.test):
                guarded = True
                break
        self.assertTrue(guarded, 'board arbiter use must stay inside the observe-only dark-launch flag')


class ArbiterContract(unittest.TestCase):
    """The behaviour the call-site pins rely on, so a change to it is deliberate."""

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('cd096_spend', ROOT / 'hermes_cli/spend_accounting.py')
        cls.sa = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.sa)

    def test_pass_never_reads_the_ledger(self):
        def unreadable(*a, **k):
            raise AssertionError('all_pass must not consult the spend ceiling')
        original = self.sa.over_spend_cap
        self.sa.over_spend_cap = unreadable
        try:
            self.assertEqual(self.sa.arbiter_decision(None, 't_x', {'verdict': 'all_pass'}),
                             (self.sa.PASS, 'all_pass'))
        finally:
            self.sa.over_spend_cap = original

    def test_exception_reads_the_ceiling_and_fails_closed(self):
        original = self.sa.over_spend_cap
        try:
            self.sa.over_spend_cap = lambda *a, **k: True
            self.assertEqual(self.sa.arbiter_decision(None, 't_x', {'verdict': 'exception'})[0], self.sa.OVER_CAP)
            self.sa.over_spend_cap = lambda *a, **k: False
            self.assertEqual(self.sa.arbiter_decision(None, 't_x', {'verdict': 'exception'})[0], self.sa.ESCALATE)

            def broken(*a, **k):
                raise OSError('ssh')
            self.sa.over_spend_cap = broken
            self.assertEqual(self.sa.arbiter_decision(None, 't_x', {'verdict': 'exception'}),
                             (self.sa.OVER_CAP, 'ac_failure;spend_unknown:OSError'))
        finally:
            self.sa.over_spend_cap = original


if __name__ == '__main__':
    unittest.main()
