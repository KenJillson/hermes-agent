"""Real completion serialization, graph nodes, JSON retention and cloud input."""
import copy
import json
import os
from pathlib import Path
import socket
import time
import unittest
import sys
from types import SimpleNamespace
from unittest.mock import patch

import hermes_cli
overlay=Path(__file__).parents[2]/'hermes_cli'
if overlay.is_dir():hermes_cli.__path__.insert(0,str(overlay))
worker=Path(__file__).parents[2]/'worker'
if not worker.is_dir():worker=Path(__file__).parents[3]/'scripts/michael-worker'
if worker.is_dir():sys.path.insert(0,str(worker))
from hermes_cli import build_graph as bg, build_graph_stages as stages
from hermes_cli import build_graph_review_evidence as ev, build_graph_implementer as implementer
from hermes_cli.build_graph_state import new_workflow_state
import launch_client as client
import launch_protocol as protocol
import launch_session as session
from test_denial_handoff import MemoryAttempts, Gate, summary
from test_launch_session import Authority, Runtime, BOOT


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.state=new_workflow_state('t_fixture','main')
        self.workspace='/tmp/isolated-denial-workspace';self.body='Change a.txt\n## A2 Files\n```json\n["a.txt"]\n```'
        self.sent=[];self.encoded=[]
        self.deps=bg.Deps(workspace=self.workspace,body=self.body,over_cap=lambda *a:False,
            model=self.model,derive=lambda w:dict(ok=True,diff='actual diff',diff_files=1,diff_added_lines=1,changed_files=['a.txt']))
        p=patch.object(bg,'guard',lambda u,s:u);p.start();self.addCleanup(p.stop)

    def model(self,**kwargs):
        self.sent.append(kwargs)
        return {'klass':'ok','result':{'model':'fixture-reviewer','verdict':'pass'}}

    def launch(self,request,interrupt_check=None):
        authority=Authority();authority.binding['run_id']=request['run_id']
        authority.bind=lambda b,t,r:copy.deepcopy(authority.binding)
        runtime=Runtime(authority);runtime.denial_evidence=lambda:summary(request['operation'])
        runtime.stage_result=lambda:({'operation':'plan','plan':'Change a.txt'} if request['operation']=='plan'
            else {'operation':request['operation'],'passed':True,'findings':[]})
        with patch.object(session.os,'geteuid',return_value=0):
            controller=session.LaunchSession(Gate(),MemoryAttempts(),lambda c:authority,lambda *a:runtime,BOOT)
        left,right=socket.socketpair()
        try:
            right.sendall(protocol.wire_frame(request,time.monotonic_ns()));controller.handle(left)
            raw=right.recv(client.MAX_RESPONSE);self.encoded.append(raw)
            value=client.response(raw)
            protocol.denial_evidence(value['denial_evidence'],request,value['token'])
            return value
        finally:left.close();right.close()

    def run_stages(self):
        real_client=SimpleNamespace(launch_protocol=protocol,DEADLINE_SECONDS=900,launch=self.launch)
        parent=SimpleNamespace(launch=lambda:None,committed=lambda:None)
        for run_id,operation in enumerate(stages.STAGES,10):
            task=SimpleNamespace(id='t_fixture',body=self.body,status='running',current_run_id=run_id,worker_pid=os.getpid())
            env=dict(HERMES_KANBAN_TASK=task.id,HERMES_KANBAN_BOARD='ops',HERMES_KANBAN_RUN_ID=str(run_id),HERMES_KANBAN_WORKSPACE=self.workspace)
            with patch.object(implementer,'_a2_client',return_value=real_client),patch('hermes_cli.build_graph_parent_git.ParentGit',return_value=parent):
                runner=implementer.make_a2_runner(task,self.workspace,env);self.deps.implement_runner=runner
                node=bg.make_implement(self.deps) if operation=='implement' else stages.make_stage(self.deps,operation,lambda u,s:u)
                self.state['terminal_reason']=None
                update=node(self.state)
                self.assertNotIn('unavailable',update.get('terminal_reason') or '')
                self.state.update(update)
                # The checkpoint boundary must retain JSON data, not live capabilities.
                self.state=json.loads(json.dumps(self.state))

    def review(self):
        # AC records have their own qualified path; isolate the changed denial seam.
        with patch.object(ev,'render',return_value='AC evidence'):
            return bg.make_cloud_review(self.deps,activity='code_review',node_name='review')(self.state)

    def test_session_to_real_runner_nodes_json_and_both_cloud_paths(self):
        self.run_stages()
        self.assertEqual(len(self.encoded),4)
        self.assertEqual(self.state['rung_attempts']['a2_runs'],dict(zip(stages.STAGES,range(10,14))))
        for activity in ('code_review','re_review'):
            with patch.object(ev,'render',return_value='AC evidence'):
                bg.make_cloud_review(self.deps,activity=activity,node_name='review')(self.state)
            prompt=self.sent[-1]['prompt']
            self.assertIn('PROTECTED-STAGE TOOL-DENIAL EVIDENCE',prompt)
            for raw in self.encoded:
                row=json.loads(raw)['denial_evidence']
                self.assertIn(row['token'],prompt);self.assertIn(row['journal_sha256'],prompt)
                self.assertIn(row['request_sha256'],prompt)

    def test_missing_stale_wrong_run_denied_incomplete_and_cross_stage_refuse_paid_call(self):
        self.run_stages();base=copy.deepcopy(self.state)
        mutations=[lambda s:s['rung_attempts'].pop('a2_denial_evidence'),
            lambda s:s['rung_attempts']['a2_denial_evidence'].pop('plan'),
            lambda s:s['rung_attempts']['a2_runs'].update(plan=99),
            lambda s:s.update(card_id='t_wrong'),lambda s:s.update(component='other')]
        for field,value in [('run_id',99),('task_id','t_wrong'),('denied',1),('unknown',1),('complete',False),('requested',True),('operation','plan'),('board','other')]:
            mutations.append(lambda s,f=field,v=value:s['rung_attempts']['a2_denial_evidence']['implement']['evidence'].update({f:v}))
        for mutate in mutations:
            self.state=copy.deepcopy(base);mutate(self.state)
            self.assertEqual(self.review()['terminal_reason'],'protected_denial_evidence_unavailable')
            self.assertEqual(self.sent,[])
        self.state=base;self.deps.body+=' changed'
        self.assertEqual(self.review()['terminal_reason'],'protected_denial_evidence_unavailable')

    def test_missing_completion_evidence_parks_stage_and_implement(self):
        self.run_stages();base=copy.deepcopy(self.state)
        for op in ('plan','implement'):
            self.state=copy.deepcopy(base)
            if op=='plan':self.state['rung_attempts']={}
            else:
                self.state['rung_attempts']={'a2_runs':{'plan':10,'plan_review':11}}
            runner=SimpleNamespace(stage=lambda **kw:{'phase':'committed','result':{'plan':'Change a.txt'}},run_id=99)
            if op=='implement':
                def runner(**kw):return {'status':'completed'}
                runner.stage=lambda **kw:None;runner.run_id=99
            self.deps.implement_runner=runner;self.state['terminal_reason']=None
            node=bg.make_implement(self.deps) if op=='implement' else stages.make_stage(self.deps,op,lambda u,s:u)
            self.assertEqual(node(self.state)['terminal_reason'],'a2_denial_evidence_unavailable')

if __name__=='__main__':unittest.main()
