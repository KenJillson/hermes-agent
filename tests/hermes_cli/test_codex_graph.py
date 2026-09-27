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

if __name__=='__main__':unittest.main()
