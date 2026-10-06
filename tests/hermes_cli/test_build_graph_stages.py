import os
import unittest
from hermes_cli import build_graph as bg
from hermes_cli import build_graph_stages as stages
from hermes_cli.build_graph_state import new_workflow_state


class Runner:
    payload_workspace = '/workspace'
    def __init__(self): self.run_id = 1; self.calls = []; self.goals = []
    def stage(self, *, operation, goal, workspace):
        self.calls.append((operation, self.run_id))
        self.goals.append(goal)
        result = ({'operation':operation,'plan':'Create the declared file.'} if operation == 'plan'
                  else {'operation':operation,'passed':True,'findings':[]})
        return {'result':result}
    def __call__(self, **kwargs):
        self.calls.append(('implement', self.run_id))
        return {'status':'completed'}


class GraphStages(unittest.TestCase):
    def deps(self, runner, calls):
        def model(**kwargs):
            calls.append(kwargs)
            return {'klass':'ok','result':{'model':'reviewer','verdict':{'verdict':'pass','findings':[]}}}
        return bg.Deps(implement_runner=runner,
            derive=lambda w: {'ok':True,'diff':'diff --git a/a.py b/a.py\n+value = 1\n',
                              'diff_files':1,'diff_added_lines':1,'changed_files':['a.py']},
            body='Create a.py with value = 1.', workspace='/fixture',
            gate=lambda *a, **kw: {'all_pass':True}, arbiter=lambda *a: ('pass','fixture'),
            parse_ac=lambda b:(None,None,None,[]), model=model, over_cap=lambda *a:False)

    def test_compiled_graph_runs_one_protected_stage_per_fresh_run(self):
        runner = Runner(); calls=[]; deps=self.deps(runner,calls)
        app,_=bg.build(deps)
        cfg={'configurable':{'thread_id':'t_fixture'}}
        state=new_workflow_state('t_fixture','main')
        expected=['a2_stage_ready:plan_review','a2_stage_ready:implement',
                  'a2_stage_ready:local_review']
        for index in range(4):
            runner.run_id=index+1
            state['terminal_reason']=None
            state=app.invoke(state,cfg)
            if index<3:
                self.assertEqual(state['terminal_reason'],expected[index])
                self.assertEqual(calls,[])
        self.assertEqual(runner.calls,list(zip(stages.STAGES,range(1,5))))
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0]['signals']['diff_files'],1)
        self.assertEqual(calls[0]['signals']['diff_added_lines'],1)
        self.assertNotIn('large_diff', calls[0]['signals'])
        self.assertEqual(state['terminal_reason'],'assembled')
        # Resuming a completed stage sequence cannot buy another review.
        state['terminal_reason']=None
        app.invoke(state,cfg)
        self.assertEqual(len(calls),1)

    def test_same_run_cannot_advance(self):
        runner=Runner();deps=self.deps(runner,[])
        state=new_workflow_state('t_fixture','main')
        state.update(stages.make_stage(deps,'plan',bg.guard)(state))
        state['terminal_reason']=None
        result=stages.make_stage(deps,'plan_review',bg.guard)(state)
        self.assertEqual(result['terminal_reason'],'a2_stage_run_mismatch')
        self.assertEqual(runner.calls,[('plan',1)])

    def test_planning_keeps_spec_but_does_not_seed_from_old_plan_or_diff(self):
        runner=Runner();deps=self.deps(runner,[])
        state=new_workflow_state('t_fixture','main',plan='STALE_PLAN_COUNT_3')
        state.update(diff='UNMEASURED_DIFF',diff_files=99,diff_added_lines=999)
        out=stages.make_stage(deps,'plan',bg.guard)(state)
        goal=runner.goals[0]
        self.assertTrue(goal.endswith('SPECIFICATION:\n'+deps.body))
        self.assertNotIn('STALE_PLAN_COUNT_3',goal)
        self.assertNotIn('UNMEASURED_DIFF',goal)
        self.assertNotIn('Reviews: passed',goal)
        self.assertNotIn('diff_files',out)
        self.assertNotIn('diff_added_lines',out)

    def test_reviews_keep_full_spec_plan_and_actual_diff(self):
        for operation,receipts in [('plan_review',{'plan':1}),
                                  ('local_review',{'plan':1,'plan_review':2,'implement':3})]:
            runner=Runner();runner.run_id=4;deps=self.deps(runner,[])
            state=new_workflow_state('t_fixture','main',plan='Create a.py; validate value.')
            state.update(diff='diff --git a/a.py b/a.py\n+value = 1\n',
                         rung_attempts={'a2_runs':receipts})
            stages.make_stage(deps,operation,bg.guard)(state)
            goal=runner.goals[0]
            self.assertIn('SPECIFICATION:\n'+deps.body,goal)
            self.assertIn('PLAN:\n'+state['plan'],goal)
            if operation=='local_review':
                self.assertTrue(goal.endswith('ACTUAL DIFF:\n'+state['diff']))

    def test_failed_review_has_no_handoff(self):
        runner=Runner();runner.run_id=2;deps=self.deps(runner,[])
        runner.stage=lambda **kw: {'result':{'operation':'plan_review','passed':False,'findings':['Missing validation']}}
        state=new_workflow_state('t_fixture','main',plan='Create a.py')
        state['rung_attempts']={'a2_runs':{'plan':1}}
        out=stages.make_stage(deps,'plan_review',bg.guard)(state)
        self.assertEqual(out['terminal_reason'],'a2_stage_review_required:plan_review')
        self.assertNotIn('a2_stage_ready',out['terminal_reason'])

    def test_out_of_order_receipts_refused(self):
        state=new_workflow_state('t_fixture','main')
        for runs in ({'implement':3},{'plan':1,'plan_review':1},{'plan':True}):
            state['rung_attempts']={'a2_runs':runs}
            with self.assertRaises(ValueError):stages.next_stage(state)


class StageRunner(unittest.TestCase):
    def test_operation_and_run_are_bound_and_runner_cannot_relaunch(self):
        import types
        from unittest.mock import patch
        from hermes_cli import build_graph_implementer as adapter
        from pathlib import Path
        import sys
        source=Path(__file__).resolve().parents[3]/'worker'
        sys.path.insert(0,str(source))
        import launch_protocol
        calls=[]
        client=types.SimpleNamespace(launch_protocol=launch_protocol,DEADLINE_SECONDS=900,
            launch=lambda request: calls.append(request) or {'result':{'operation':'plan','plan':'x'}})
        task=types.SimpleNamespace(id='t_fixture',current_run_id=9,status='running',worker_pid=os.getpid(),
            body='## A2 Files\n```json\n["a.py"]\n```')
        env={'HERMES_KANBAN_TASK':task.id,'HERMES_KANBAN_RUN_ID':'9',
             'HERMES_KANBAN_BOARD':'ops','HERMES_KANBAN_WORKSPACE':'/fixture'}
        parent=types.SimpleNamespace(launch=lambda:None,committed=lambda:None)
        with patch.object(adapter,'_a2_client',return_value=client), \
             patch('hermes_cli.build_graph_parent_git.ParentGit',return_value=parent):
            runner=adapter.make_a2_runner(task,'/fixture',env)
            runner.stage(operation='plan',goal='Plan only',workspace='/fixture')
            self.assertEqual((calls[0]['operation'],calls[0]['run_id']),('plan',9))
            with self.assertRaises(adapter.A2ReconciliationRequired):
                runner.stage(operation='plan_review',goal='Review',workspace='/fixture')
            with self.assertRaises(adapter.A2ReconciliationRequired):
                runner(goal='Implement',workspace='/fixture',max_iterations=3,timeout_seconds=900)
        self.assertEqual(len(calls),1)
