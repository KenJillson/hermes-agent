"""Disposable Git + SQLite integration for parent exception; no root/auth proof.

On the Jetson validation host, every actual Git call additionally checks the
real all-board idle gate. The bound task/run and process are test fixtures.
"""
from contextlib import nullcontext
import importlib.util
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

from hermes_cli import build_graph_parent_git as parent
from hermes_cli import build_graph_diff as diff

class ParentGitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.home = self.root/'home'
        self.workspace = self.home/'kanban/boards/ops/workspaces/t_case'
        self.workspace.parent.mkdir(parents=True)
        self.anchor = self.root/'project'
        self.anchor.mkdir()
        self.real_run = subprocess.run
        def run(argv, *a, **kw):
            if argv[0] in ('git','/usr/bin/git') and os.environ.get('A2_TEST_LIVE_IDLE') == '1':
                spec = importlib.util.spec_from_file_location('_test_installed_admission', '/usr/local/lib/michael-maintenance/admission.py')
                module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
                module.Gate().no_active_runs()
            return self.real_run(argv,*a,**kw)
        self.run = run
        def git(*args):
            result = run(['/usr/bin/git','-C',str(self.anchor),*args],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
        git('init','-b','main')
        (self.anchor/'input.txt').write_text('input\n')
        git('add','input.txt')
        git('-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-m','fixture baseline')
        git('worktree','add','-b','wt/t_case',str(self.workspace),'HEAD')
        # Columns are the inspected production query contract; all databases
        # are disposable and never passed to the root production authority.
        schema='''CREATE TABLE task_runs (id INTEGER,task_id TEXT,status TEXT,claim_lock TEXT,claim_expires INTEGER,worker_pid INTEGER,started_at INTEGER,ended_at INTEGER,profile TEXT);
        CREATE TABLE tasks (id TEXT,status TEXT,current_run_id INTEGER,claim_lock TEXT,claim_expires INTEGER,worker_pid INTEGER,workspace_path TEXT,workspace_kind TEXT,assignee TEXT,branch_name TEXT,graph_executor TEXT);'''
        self.db = self.home/'kanban/boards/ops/kanban.db'
        for path in (self.home/'kanban.db',self.db):
            with sqlite3.connect(path) as db:db.executescript(schema)
        self.task=types.SimpleNamespace(id='t_case',current_run_id=7,branch_name='wt/t_case',claim_lock='fixture-claim',workspace_path=str(self.workspace))
        with sqlite3.connect(self.db) as db:
            db.execute('INSERT INTO task_runs VALUES (?,?,?,?,?,?,?,?,?)',(7,'t_case','running','fixture-claim',int(time.time())+3600,123,1,None,'coder'))
            db.execute('INSERT INTO tasks VALUES (?,?,?,?,?,?,?,?,?,?,?)',('t_case','running',7,'fixture-claim',int(time.time())+3600,123,str(self.workspace),'worktree','coder','wt/t_case','build_graph'))
        self.closed=False
        class Gate:
            def shared(inner,*args):
                if self.closed:raise RuntimeError('fixture admission closed/inhibit')
                return nullcontext()
        for target,value in [('ROOT',self.home),('process',lambda:(123,1,99)),('gate',lambda:Gate())]:
            p=patch.object(parent,target,value);p.start();self.addCleanup(p.stop)
        p=patch.object(parent.subprocess,'run',side_effect=run);p.start();self.addCleanup(p.stop)

    def context(self):return parent.ParentGit(self.task,str(self.workspace),'ops')
    def change(self,sql,args=()):
        with sqlite3.connect(self.db) as db:db.execute(sql,args)

    def test_real_diff_baseline_import_and_phase_gate(self):
        context=self.context()
        self.assertEqual(context.derive(str(self.workspace))['reason'],diff.EMPTY_DIFF_REASON)
        context.launch()
        (self.workspace/'output.txt').write_text('declared output\n')
        with self.assertRaisesRegex(parent.Refused,'pending'):context.derive(str(self.workspace))
        context.committed() # fixture stands in for validated protected receipt
        result=context.derive(str(self.workspace))
        self.assertTrue(result['ok']);self.assertIn('output.txt',result['changed_files'])
        self.assertIn('+declared output',result['diff'])
        with self.assertRaises(parent.Refused):context.launch()

    def test_exact_argv_refuses_maintenance_and_content_staging(self):
        context=self.context()
        commands=[['add','-A'],['commit','-m','x'],['fetch'],['push'],['worktree','add','x'],['rev-parse','--git-dir'],['diff','HEAD'],['merge-base','--all','HEAD'],['-c','x=y','diff']]
        with patch.object(parent.subprocess,'run',side_effect=AssertionError('forbidden spawn')):
            for args in commands:
                with self.subTest(args=args),self.assertRaises(parent.Refused):context(['git','-C',str(self.workspace),*args],60)
            with self.assertRaises(parent.Refused):context(['git','-C',str(self.anchor),'add','-A','-N'],60)

    def test_assigned_worker_claim_and_status_drift_prevent_git(self):
        cases=[("UPDATE tasks SET worker_pid=124",()),("UPDATE task_runs SET claim_lock='other'",()),("UPDATE tasks SET status='done'",()),("UPDATE task_runs SET claim_expires=0",()),("UPDATE tasks SET workspace_path=?",(str(self.anchor),)),("UPDATE tasks SET branch_name='main'",())]
        for sql,args in cases:
            with self.subTest(sql=sql):
                context=self.context()
                with sqlite3.connect(self.db) as db:before=list(db.iterdump())
                self.change(sql,args)
                with patch.object(parent.subprocess,'run',side_effect=AssertionError('no command after drift')):
                    with self.assertRaises(parent.Refused):context.derive(str(self.workspace))
                self.db.unlink()
                with sqlite3.connect(self.db) as db:db.executescript('\n'.join(before))

    def test_other_root_run_and_new_board_refused(self):
        context=self.context()
        with sqlite3.connect(self.home/'kanban.db') as db:
            db.execute('INSERT INTO task_runs VALUES (8,\'t_other\',\'running\',\'other\',9999999999,124,1,NULL,\'coder\')')
        with self.assertRaisesRegex(parent.Refused,'sole'):context.derive(str(self.workspace))
        with sqlite3.connect(self.home/'kanban.db') as db:db.execute('DELETE FROM task_runs')
        new=self.home/'kanban/boards/new';new.mkdir()
        with self.assertRaises((parent.Refused,FileNotFoundError)):context.derive(str(self.workspace))

    def test_inhibit_and_workspace_pointer_drift_refused(self):
        context=self.context();self.closed=True
        with self.assertRaisesRegex(parent.Refused,'uncertain'):context.derive(str(self.workspace))
        self.closed=False
        (self.workspace/'.git').write_text('gitdir: /nonexistent\n')
        with self.assertRaises((parent.Refused,FileNotFoundError)):context.derive(str(self.workspace))

    def test_diff_clean_filters_suppressed_without_config_mutation(self):
        config=self.anchor/'.git/config'
        with config.open('a') as stream:
            stream.write('\n[filter "test"]\n clean = "touch FILTER_EXECUTED; cat"\n required = true\n')
        before=config.read_bytes()
        (self.workspace/'.gitattributes').write_text('*.txt filter=test\n')
        (self.workspace/'output.txt').write_text('output\n')
        context=self.context();result=context.derive(str(self.workspace))
        self.assertTrue(result['ok']);self.assertFalse((self.workspace/'FILTER_EXECUTED').exists())
        self.assertEqual(config.read_bytes(),before)

    def test_repository_worktree_redirection_cannot_escape_task_scope(self):
        outside=self.root/'outside';outside.mkdir();(outside/'outside-secret.txt').write_text('outside\n')
        with (self.anchor/'.git/config').open('a') as stream:
            stream.write('\n[core]\n worktree = '+str(outside)+'\n')
        (self.workspace/'output.txt').write_text('inside\n')
        context=self.context();result=context.derive(str(self.workspace))
        self.assertTrue(result['ok']);self.assertIn('output.txt',result['changed_files'])
        self.assertNotIn('outside-secret',result['diff']);self.assertEqual((outside/'outside-secret.txt').read_text(),'outside\n')

    def test_external_git_config_refused_before_command(self):
        with (self.anchor/'.git/config').open('a') as stream:
            stream.write('\n[include]\n path = /tmp/unknown-config\n')
        with patch.object(parent.subprocess,'run',side_effect=AssertionError('no Git')):
            with self.assertRaisesRegex(parent.Refused,'includes'):self.context()

    def test_dispatcher_resolves_only_precreated_exact_worktree(self):
        self.change('UPDATE tasks SET worker_pid=NULL');self.change('UPDATE task_runs SET worker_pid=NULL')
        self.assertEqual(parent.precreated(self.task,board='ops'),(self.workspace,'wt/t_case'))
        context=parent.ParentGit(self.task,str(self.workspace),'ops',dispatcher=True)
        with self.assertRaises(parent.Refused):context(['git','-C',str(self.workspace),'add','-A','-N'],60)
        with self.assertRaises(parent.Refused):context.launch()
        self.task.workspace_path=str(self.anchor)
        with self.assertRaises(parent.Refused):parent.precreated(self.task,board='ops')

if __name__=='__main__':unittest.main(verbosity=2)
