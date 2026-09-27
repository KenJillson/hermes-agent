"""Real graph/node behavior, disposable worktree, no model/board writes."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from hermes_cli import build_graph as bg
from hermes_cli.build_graph_state import new_workflow_state

class GraphContract(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.workspace=Path(self.tmp.name);self.file=self.workspace/'code.py';self.file.write_text('answer = 0\n')
        self.calls=[];self.prompts=[];self.derived=[];self.records=[]
        self.state=new_workflow_state('t_graph','main',plan='bounded integer result',diff='diff --git a/code.py b/code.py\n+answer = 0',diff_files=1)
        self.state['implementer_model']='qwen'
        self.deps=bg.Deps(workspace=str(self.workspace),body='Return the verified integer. Acceptance: answer is 2.',task_id='t_graph',
            changed_files=['code.py'],gate=lambda *a,**k: {'verdict':'all_pass'},
            arbiter=lambda *a:('pass','fixture'),parse_ac=lambda *a:[],over_cap=lambda *a:False,
            model=self.model,codex=self.codex,derive=self.derive)
        self.addCleanup(patch.stopall)
        patch.object(bg.ev,'append_classify_line',side_effect=lambda r:self.records.append(r) or True).start()
        patch.object(bg.ev,'is_holdout',return_value=False).start()
        real=bg.tl.Tracer
        patch.object(bg.tl,'Tracer',side_effect=lambda **k:real(**k,sink=lambda r:None)).start()

    def derive(self,workspace):
        self.derived.append(self.file.read_text())
        return dict(ok=True,diff='diff --git a/code.py b/code.py\n+'+self.file.read_text(),diff_files=1,diff_added_lines=1,changed_files=['code.py'])

    def model(self,**kw):
        activity=kw['activity'];self.calls.append(activity)
        if activity in ('code_review','re_review'):
            result=dict(model='opus',verdict={'passed':False,'findings':['integer is wrong: '+activity]})
        elif activity=='classify':result=dict(model='qwen',text='{"same": true}')
        elif activity in ('fix_sonnet','fix_opus'):
            self.assertEqual(kw['signals'],{})
            result=dict(model='sonnet' if activity=='fix_sonnet' else 'opus',text=json.dumps({'files':[{'path':'code.py','content':'answer = 1\n'}]}))
        else:raise AssertionError(activity)
        return dict(klass='ok',result=result)

    def codex(self,**kw):
        self.calls.append(kw['activity']);self.prompts.append(kw['prompt'])
        return dict(klass='ok',result=dict(model='codex',cost_usd=-1,text=json.dumps({'files':[{'path':'code.py','content':'answer = 2\n'}]})))

    def disputed(self):
        self.state.update(rung='rung1',rung_attempts={'rung1':1},cloud_review_calls=2,
            objections_prior=[{'text':'prior objection'}],objections_current=[{'text':'current objection'}],
            objection_recurred=True,dispute_class='approach_disputed',implementer_model='sonnet',reviewer_model='opus')
        return self.state

    def test_both_review_rounds_receive_specification_without_inventing_plan(self):
        self.state['plan'] = ''  # The actual production caller's current input.
        before = copy.deepcopy(self.state)
        seen = []
        def reviewer(**kw):
            seen.append(kw)
            return dict(klass='ok', result=dict(model='opus', verdict={'passed': True}))
        self.deps.model = reviewer
        for activity in ('code_review', 're_review'):
            update = bg.make_cloud_review(self.deps, activity=activity, node_name=activity)(self.state)
            self.assertTrue(update['last_verdict_passed'])
            self.assertEqual(update['cloud_review_calls'], 1)
        self.assertEqual([v['activity'] for v in seen], ['code_review', 're_review'])
        for request in seen:
            self.assertIn('SPECIFICATION (task requirements):\n' + self.deps.body, request['prompt'])
            self.assertIn('PLAN:\n\n\nDIFF:\n' + self.state['diff'], request['prompt'])
            self.assertEqual(request['mode'], 'read_only')
        self.assertEqual(self.state, before)
        self.assertEqual(self.deps.body, 'Return the verified integer. Acceptance: answer is 2.')

    def test_review_requirements_do_not_bypass_review_or_spend_caps(self):
        def forbidden(**kw):
            self.fail('A capped review reached the provider')
        self.deps.model = forbidden
        node = bg.make_cloud_review(self.deps, activity='re_review', node_name='cloud_re_review')
        self.state['cloud_review_calls'] = bg.CLOUD_REVIEW_CAP
        self.assertEqual(node(self.state)['terminal_reason'], 'cloud_review_cap')
        self.state['cloud_review_calls'] = 0
        self.deps.over_cap = lambda *args: True
        self.assertEqual(node(self.state)['terminal_reason'], 'over_spend_cap')

    def test_rung3_uses_both_positions_applies_local_and_rederives(self):
        state=self.disputed();before=copy.deepcopy(state)
        update=bg.make_fix_rung3(self.deps)(state)
        self.assertEqual(self.calls,['fix_codex']);self.assertIn('prior objection',self.prompts[0]);self.assertIn('current objection',self.prompts[0])
        self.assertIn(state['diff'],self.prompts[0]);self.assertIn(self.deps.body,self.prompts[0])
        self.assertEqual(self.file.read_text(),'answer = 2\n');self.assertEqual(self.derived,['answer = 2\n'])
        self.assertEqual(update['rung_attempts'],{'rung1':1,'rung3':1});self.assertEqual(state,before)

    def test_disputed_missing_position_refuses_no_write(self):
        self.disputed()['objections_prior']=[]
        update=bg.make_fix_rung3(self.deps)(self.state)
        self.assertEqual(update['terminal_reason'],'fix_prompt_refused');self.assertEqual(self.calls,[])
        self.assertEqual(self.file.read_text(),'answer = 0\n')

    def test_halt_in_dispute_refuses_before_provider(self):
        self.disputed()['objections_current']=[{'text':'honcho_context fixture'}]
        update=bg.make_fix_rung3(self.deps)(self.state)
        self.assertEqual(update['terminal_reason'],'fix_prompt_refused');self.assertEqual(self.calls,[])

    def test_rung_attempt_cap_refuses(self):
        self.disputed()['rung_attempts']['rung3']=1
        self.assertEqual(bg.make_fix_rung3(self.deps)(self.state)['terminal_reason'],'fix_rung_cap');self.assertEqual(self.calls,[])

    def test_spend_cap_and_unknown_refuse(self):
        self.disputed();self.deps.over_cap=lambda *a:True
        self.assertEqual(bg.make_fix_rung3(self.deps)(self.state)['terminal_reason'],'over_spend_cap')
        def unknown(*a):raise OSError('fixture')
        self.deps.over_cap=unknown
        self.assertEqual(bg.make_fix_rung3(self.deps)(self.state)['terminal_reason'],'spend_unknown:OSError');self.assertEqual(self.calls,[])

    def test_unknown_provider_does_not_apply(self):
        self.disputed();self.deps.codex=lambda **k:dict(klass='failed',result={'collection_required':True})
        update=bg.make_fix_rung3(self.deps)(self.state)
        self.assertEqual(update['terminal_reason'],'model_call_failed:fix_codex');self.assertEqual(update['rung_attempts']['rung3'],1)
        self.assertEqual(self.file.read_text(),'answer = 0\n');self.assertEqual(self.derived,[])

    def test_no_recurrence_invented_before_first_fix(self):
        self.state.update(objections_prior=['first'],objections_current=['second'])
        update=bg.make_classify_failure(self.deps)(self.state)
        self.assertFalse(update['objection_recurred']);self.assertEqual(update['recurrence_tier'],'default');self.assertEqual(self.calls,[])

    def test_actual_compiled_graph_post_first_fix_recurrence_and_caps(self):
        app,_=bg.build(self.deps)
        result=app.invoke(self.state,{'configurable':{'thread_id':'fixture'},'recursion_limit':40})
        self.assertEqual(self.calls,['code_review','fix_sonnet','re_review','classify','fix_codex'])
        self.assertEqual(result['cloud_review_calls'],2);self.assertEqual(result['terminal_reason'],'cloud_review_cap')
        self.assertEqual(result['rung_attempts'],{'rung1':1,'rung3':1});self.assertEqual(self.file.read_text(),'answer = 2\n')
        self.assertTrue(result['objection_recurred']);self.assertEqual(len(self.records),1)
        self.assertIn('integer is wrong: code_review',self.prompts[0]);self.assertIn('integer is wrong: re_review',self.prompts[0])



"""Approved promotion boundaries and real graph repair-review behavior; no provider."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hermes_cli import build_graph as bg
from hermes_cli import build_graph_state as state_module


class PolicyGraphTests(unittest.TestCase):
    def setUp(self):
        bg.sz._load_sanitizer()
        from model_call import policy
        self.policy = policy
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.workspace = Path(self.tmp.name)
        self.file = self.workspace / 'code.py'
        self.file.write_text('answer = 0\n')
        self.state = state_module.new_workflow_state('t_policy', 'main', diff='diff --git a/code.py b/code.py\n+answer = 0', diff_files=1, diff_added_lines=1)
        self.state['implementer_model'] = 'fixture-local-author'
        self.calls = []
        self.first_gate_fails = False
        self.gates = 0
        self.deps = bg.Deps(workspace=str(self.workspace), body='Return a verified integer.', task_id='t_policy',
                            gate=self.gate, arbiter=lambda c,t,s: ('escalate' if s['verdict']=='exception' else 'pass','fixture'),
                            parse_ac=lambda *a: [], model=self.model, codex=self.codex,
                            changed_files=['code.py'], derive=self.derive, over_cap=lambda *a: False)
        self.deps.review_default = lambda: self.policy.resolve('re_review').model
        self.addCleanup(patch.stopall)
        patch.object(bg.ev, 'append_classify_line', return_value=True).start()
        patch.object(bg.ev, 'is_holdout', return_value=False).start()
        tracer = bg.tl.Tracer
        patch.object(bg.tl, 'Tracer', side_effect=lambda **k: tracer(**k, sink=lambda r: None)).start()

    def gate(self, *args, **kwargs):
        self.gates += 1
        return {'verdict': 'exception' if self.first_gate_fails and self.gates == 1 else 'all_pass'}

    def derive(self, workspace):
        return {'ok': True, 'diff': 'diff --git a/code.py b/code.py\n+' + self.file.read_text(), 'diff_files': 1, 'diff_added_lines': 1, 'changed_files': ['code.py']}

    def model(self, **kw):
        resolution = self.policy.resolve(kw['activity'], directive=kw.get('directive'), signals=kw.get('signals'))
        self.calls.append((kw['activity'], resolution.model, kw.get('signals', {})))
        if kw['activity'] in ('code_review', 're_review'):
            result = {'model': resolution.model, 'verdict': {'passed': False, 'findings': ['Requirement remains unmet']}}
        elif kw['activity'] == 'classify':
            result = {'model': resolution.model, 'text': '{"same": true}'}
        else:
            result = {'model': resolution.model, 'text': json.dumps({'files': [{'path': 'code.py', 'content': 'answer = %s\n' % sum(x[0].startswith('fix_') for x in self.calls)}]})}
        return {'klass': 'ok', 'result': result}

    def codex(self, **kw):
        self.calls.append((kw['activity'], self.policy.resolve(kw['activity']).model, kw.get('signals', {})))
        return {'klass': 'ok', 'result': {'model': self.policy.resolve(kw['activity']).model, 'text': json.dumps({'files': [{'path': 'code.py', 'content': 'answer = 3\n'}]})}}

    def test_documented_size_boundaries(self):
        for files, lines, fired in [(1,1,False),(8,400,False),(9,400,True),(8,401,True)]:
            with self.subTest(files=files, lines=lines):
                self.state.update(diff_files=files, diff_added_lines=lines)
                self.assertEqual(bool(state_module.selection_signals(self.state).get('large_diff')), fired)
        for components, fired in [(1,False),(3,False),(4,True)]:
            self.state['component_count'] = components
            self.assertEqual(bool(state_module.selection_signals(self.state).get('many_components')), fired)

    def test_first_review_does_not_inherit_planning_or_repair_signals(self):
        self.state.update(component_count=4, novel_architecture=True, author_independence=True, re_review_after_severe=True)
        bg.make_cloud_review(self.deps, activity='code_review', node_name='cloud_review')(self.state)
        self.assertEqual(self.calls[-1], ('code_review', self.policy.resolve('code_review').model, {}))

    def test_re_review_does_not_inherit_diff_or_planning_signals(self):
        self.state.update(diff_files=9, diff_added_lines=401, component_count=4, novel_architecture=True, security_sensitive=True, financial_sensitive=True)
        bg.make_cloud_review(self.deps, activity='re_review', node_name='cloud_re_review')(self.state)
        self.assertEqual(self.calls[-1], ('re_review', self.policy.resolve('re_review').model, {}))

    def test_repair_sets_and_clears_independence_from_actual_policy_identity(self):
        first = bg.make_fix(self.deps, rung='rung1', activity='fix_sonnet')(self.state)
        self.assertTrue(first['author_independence'])
        self.state.update(first)
        second = bg.make_fix(self.deps, rung='rung2', activity='fix_opus')(self.state)
        self.assertFalse(second['author_independence'])
        self.assertTrue(self.state['author_independence'])

    def test_directive_priority_is_preserved(self):
        self.state['directive'] = self.policy.resolve('fix_opus').model
        update = bg.make_fix(self.deps, rung='rung1', activity='fix_sonnet')(self.state)
        self.assertFalse(update['author_independence'])
        self.state.update(update, author_independence=True)
        bg.make_cloud_review(self.deps, activity='re_review', node_name='cloud_re_review')(self.state)
        self.assertEqual(self.calls[-1][1], self.state['directive'])

    def test_unknown_policy_refuses_before_spending_attempt_or_writing(self):
        def unavailable():
            raise OSError('fixture unavailable')
        self.deps.review_default = unavailable
        before = copy.deepcopy(self.state)
        update = bg.make_fix(self.deps, rung='rung1', activity='fix_sonnet')(self.state)
        self.assertEqual(update, {'terminal_reason': 'review_policy_unavailable'})
        self.assertEqual(self.calls, [])
        self.assertEqual(self.file.read_text(), 'answer = 0\n')
        self.assertEqual(self.state, before)

    def test_missing_response_identity_consumes_attempt_without_applying(self):
        self.deps.model = lambda **kw: {'klass': 'ok', 'result': {'text': json.dumps({'files': [{'path':'code.py','content':'answer = 1\n'}]})}}
        update = bg.make_fix(self.deps, rung='rung1', activity='fix_sonnet')(self.state)
        self.assertEqual(update['terminal_reason'], 'fix_model_provenance_unavailable')
        self.assertEqual(update['rung_attempts'], {'rung1': 1})
        self.assertEqual(self.file.read_text(), 'answer = 0\n')

    def test_failed_application_does_not_claim_new_independent_author(self):
        self.deps.changed_files = []
        update = bg.make_fix(self.deps, rung='rung1', activity='fix_sonnet')(self.state)
        self.assertTrue(update['terminal_reason'])
        self.assertNotIn('author_independence', update)
        self.assertEqual(self.file.read_text(), 'answer = 0\n')

    def test_actual_graph_default_first_review_then_independent_dispute(self):
        app,_ = bg.build(self.deps)
        result = app.invoke(self.state, {'configurable': {'thread_id': 'policy'}, 'recursion_limit':40})
        self.assertEqual([x[0] for x in self.calls], ['code_review','fix_sonnet','re_review','classify','fix_codex'])
        self.assertEqual(self.calls[0][1], self.policy.resolve('code_review').model)
        self.assertNotEqual(self.calls[1][1], self.calls[2][1])
        self.assertEqual(self.calls[2][2], {'author_independence': True})
        self.assertTrue(result['objection_recurred'])
        self.assertEqual(result['rung_attempts'], {'rung1':1,'rung3':1})
        self.assertEqual(result['cloud_review_calls'], 2)
        self.assertEqual(result['terminal_reason'], 'cloud_review_cap')

    def test_mechanical_first_fix_has_independent_review_and_normal_climb(self):
        self.first_gate_fails = True
        app,_ = bg.build(self.deps)
        result = app.invoke(self.state, {'configurable': {'thread_id':'mechanical'}, 'recursion_limit':40})
        self.assertEqual([x[0] for x in self.calls], ['fix_sonnet','re_review','fix_opus','re_review','fix_codex'])
        self.assertEqual(self.calls[1][2], {'author_independence': True})
        self.assertEqual(self.calls[3][2], {})
        self.assertNotEqual(self.calls[2][1], self.calls[3][1])
        self.assertEqual(result['rung_attempts'], {'rung1':1,'rung2':1,'rung3':1})
        self.assertEqual(result['cloud_review_calls'],2)
        self.assertEqual(result['terminal_reason'],'cloud_review_cap')

    def test_explicit_default_directive_keeps_confounded_comparison_and_caps(self):
        self.state['directive'] = self.policy.resolve('re_review').model
        app,_ = bg.build(self.deps)
        result = app.invoke(self.state, {'configurable': {'thread_id':'directive'}, 'recursion_limit':40})
        self.assertEqual([x[0] for x in self.calls], ['code_review','fix_sonnet','re_review','classify','fix_opus'])
        self.assertEqual(self.calls[1][1], self.calls[2][1])
        self.assertTrue(result['recurrence_confounded'])
        self.assertFalse(result['objection_recurred'])
        self.assertEqual(result['rung_attempts'], {'rung1':1,'rung2':1})
        self.assertEqual(result['terminal_reason'],'cloud_review_cap')

    def test_canonical_policy_loader_reads_the_same_default(self):
        self.assertEqual(bg.normal_review_model(), self.policy.resolve('re_review').model)
        with patch.object(self.policy, '__file__', '/fixture/wrong-policy.py'):
            with self.assertRaises(RuntimeError): bg.normal_review_model()

if __name__ == '__main__':
    unittest.main()
