"""Gate-to-review handoff behavior with real records and isolated Git worktrees."""
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

# Candidate overlay only for pre-placement qualification; removed for delivery.
import hermes_cli
_overlay=Path(__file__).parents[2]/'hermes_cli'
if _overlay.is_dir():hermes_cli.__path__.insert(0,str(_overlay))
from hermes_cli import build_graph as bg, build_graph_review_evidence as evidence
from hermes_cli import ac_check_runner as ac, build_graph_diff as diff
from hermes_cli.build_graph_state import new_workflow_state

class AtModel(Exception):pass

class ReviewEvidence(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.repo=self.root/'repo';self.repo.mkdir();self.ws=self.root/'work'
  def git(*args):return subprocess.check_output(['git','-c','core.hooksPath=/dev/null','-c','user.name=QA','-c','user.email=qa@invalid','-C',str(self.repo),*args],stderr=subprocess.DEVNULL,text=True)
  git('init','--initial-branch=main');(self.repo/'file.py').write_text('x=1\n');git('add','file.py');git('commit','-m','seed');git('worktree','add','-b','qa',str(self.ws));(self.ws/'file.py').write_text('x=2\n')
  derived=diff.derive(str(self.ws));assert derived['ok']
  self.state=new_workflow_state('t_qa','main',diff=derived['diff']);self.body='A test specification';self.sent=[]
  self.guard=patch.object(bg,'guard',lambda update,state:update);self.guard.start();self.addCleanup(self.guard.stop)
  self.deps=bg.Deps(workspace=str(self.ws),body=self.body,over_cap=lambda *a:False,model=self.model,derive=diff.derive)
 def model(self,**kwargs):self.sent.append(kwargs);raise AtModel
 def install_record(self):
  fixture=Path(__file__).with_name('review-evidence-record.json');rec=json.loads(fixture.read_text());rec['card_id']=self.state['card_id'];rec['workspace']=str(self.ws)
  path=evidence.record_path(str(self.ws),self.state['card_id'],self.state['component'],self.state['iteration']);Path(path).write_text(json.dumps(rec))
  summary=evidence.bind(self.state,str(self.ws),self.body,path,rec['summary']);self.state.update(gate_summary=summary,ac_record_path=path,iteration=self.state['iteration']+1);return Path(path)
 def review(self,activity='code_review'):return bg.make_cloud_review(self.deps,activity=activity,node_name='review')(self.state)
 def test_retained_five_checks_reach_both_review_paths(self):
  self.install_record()
  for activity in ('code_review','re_review'):
   with self.assertRaises(AtModel):self.review(activity)
   prompt=self.sent[-1]['prompt'];self.assertIn('ACCEPTANCE-CHECK EXECUTION EVIDENCE',prompt);self.assertIn('exact source mismatch',prompt);self.assertIn('"checks_passed": 5',prompt);self.assertIn('do not infer zero denials',prompt)
 def test_missing_binding_never_calls_model(self):
  self.assertEqual(self.review()['terminal_reason'],'review_evidence_unavailable');self.assertFalse(self.sent)
 def test_missing_changed_symlink_record_never_calls_model(self):
  for kind in ('missing','changed','symlink'):
   with self.subTest(kind=kind):
    self.state['iteration']=0;path=self.install_record();raw=path.read_bytes();path.unlink()
    if kind=='changed':path.write_bytes(raw+b' ')
    if kind=='symlink':target=self.root/'other';target.write_bytes(raw);path.symlink_to(target)
    self.assertEqual(self.review()['terminal_reason'],'review_evidence_unavailable');self.assertFalse(self.sent)
    if path.is_symlink():path.unlink()
 def test_stale_card_component_iteration_spec_or_diff_never_calls_model(self):
  self.install_record();old=copy.deepcopy(self.state)
  for key,val in [('card_id','t_other'),('component','other'),('iteration',99),('diff','different')]:
   with self.subTest(key=key):
    self.state=copy.deepcopy(old);self.state[key]=val;self.assertEqual(self.review()['terminal_reason'],'review_evidence_unavailable')
  self.state=old;self.deps.body='changed';self.assertEqual(self.review()['terminal_reason'],'review_evidence_unavailable');self.assertFalse(self.sent)
 def test_product_changes_after_checks_never_calls_model(self):
  self.install_record();(self.ws/'file.py').write_text('x=3\n');self.assertEqual(self.review()['terminal_reason'],'review_evidence_diff_changed');self.assertFalse(self.sent)
 def test_record_identity_counts_and_verdict_fail_closed(self):
  path=self.install_record();rec=json.loads(path.read_text());summary=rec['summary']
  for mutate in (lambda r:r.update(card_id='wrong'),lambda r:r['summary'].update(checks_passed=0),lambda r:r['verdicts'][0].update(status='failed'),lambda r:r['verdicts'][0].update(index=[])):
   bad=copy.deepcopy(rec);mutate(bad)
   with self.assertRaises(evidence.Refused):evidence.validate_record(bad,str(self.ws),'t_qa',summary)
 def test_prompt_bound_refuses_instead_of_truncating_checks(self):
  path=self.install_record();rec=json.loads(path.read_text());rec['verdicts'][0]['stdout_tail']='x'*70000;path.write_text(json.dumps(rec));self.state['iteration']=0;self.state['gate_summary']=evidence.bind(self.state,str(self.ws),self.body,str(path),rec['summary']);self.state['iteration']=1
  self.assertEqual(self.review()['terminal_reason'],'review_evidence_unavailable');self.assertFalse(self.sent)
 def test_real_check_execution_to_review(self):
  from hermes_cli.kanban_db import parse_ac_text
  self.deps.body='## AC\n- [ ] x is two\n```check expect=exit:0\npython3 -B -c "from file import x; assert x == 2"\n```\n';self.deps.gate=ac.gate;self.deps.parse_ac=parse_ac_text
  update=bg.make_cheap_gate(self.deps)(self.state);self.assertNotIn('terminal_reason',update);self.state.update(update)
  with self.assertRaises(AtModel):self.review()
  self.assertIn('from file import x; assert x == 2',self.sent[-1]['prompt']);self.assertIn('"exit_code": 0',self.sent[-1]['prompt'])
 def test_failed_record_persistence_parks_at_gate(self):
  self.deps.gate=lambda *a,**k:{'all_clean':True,'verdict':'all_pass'}
  update=bg.make_cheap_gate(self.deps)(self.state);self.assertEqual(update['terminal_reason'],'review_evidence_unavailable');self.assertFalse(self.sent)
if __name__=='__main__':unittest.main()
