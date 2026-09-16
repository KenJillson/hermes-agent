"""Real local model-process/file/diff tests; no provider, network or inference."""
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from hermes_cli import build_graph as bg
from hermes_cli import build_graph_diff as diff

class Artifacts(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(prefix='graph-artifacts-');self.addCleanup(self.tmp.cleanup)
  self.root=Path(self.tmp.name);self.workspace=self.root/'workspace';self.workspace.mkdir()
  self.exe=self.root/'model-stub'
  self.exe.write_text('#!'+sys.executable+'\n'+'''import json,sys
from pathlib import Path
a=sys.argv[1:];w=Path(a[a.index('--workspace')+1]);prompt=Path(a[a.index('--prompt-file')+1]);p=w/'t_fixture-abcdef12-result.json'
p.write_text(json.dumps({'fixture_provider_response':True,'argv':a,'prompt':prompt.read_text()}))
print(json.dumps({'ok':True,'envelope_path':str(p),'text':'fixture response'}))
''');self.exe.chmod(0o700)
 def call(self,**kwargs):
  return bg.call_model(activity='code_review',prompt='fixture prompt',workspace=str(self.workspace),card_id='t_fixture',exe=str(self.exe),**kwargs)
 def git(self,repo,*args):
  return subprocess.check_output(['git','-c','core.hooksPath=/dev/null','-c','core.fsmonitor=false','-c','user.name=Local test','-c','user.email=local@invalid','-C',str(repo),*args],stderr=subprocess.STDOUT).decode()
 def test_provider_envelope_excluded_without_hiding_project_result_files(self):
  repo=self.root/'repo';repo.mkdir();self.workspace.rmdir()
  self.git(repo,'init','--initial-branch=main');(repo/'module.py').write_text('value = 1\n');self.git(repo,'add','module.py');self.git(repo,'commit','-m','fixture');self.git(repo,'worktree','add','-b','case',str(self.workspace),'main')
  (self.workspace/'module.py').write_text('value = 2\n');(self.workspace/'project-result.json').write_text('{"project":true}\n')
  before=diff.derive(str(self.workspace));self.assertTrue(before['ok']);self.assertEqual(before['changed_files'],['module.py','project-result.json'])
  result=self.call(mode='read_only',cwd='/fixture-remote-cwd');self.assertEqual(result['klass'],'ok')
  artifact=Path(result['result']['envelope_path']);self.assertTrue(artifact.is_file());self.assertEqual(artifact.parent.parent,self.workspace.resolve()/'.graph-checkpoints')
  record=json.loads(artifact.read_text());self.assertEqual(record['prompt'],'fixture prompt')
  self.assertEqual(record['argv'][record['argv'].index('--mode')+1],'read_only');self.assertEqual(record['argv'][record['argv'].index('--cwd')+1],'/fixture-remote-cwd')
  after=diff.derive(str(self.workspace));self.assertEqual(after['changed_files'],before['changed_files']);self.assertEqual(after['diff'],before['diff']);self.assertEqual(after['diff_files'],before['diff_files']);self.assertEqual(after['diff_added_lines'],before['diff_added_lines'])
 def test_repeated_calls_private_and_checkpoint_preserved(self):
  metadata=self.workspace/'.graph-checkpoints';metadata.mkdir(mode=0o755);checkpoint=metadata/'checkpoints.json';checkpoint.write_bytes(b'checkpoint sentinel')
  roots=[]
  for _ in range(4):
   result=self.call();self.assertEqual(result['klass'],'ok');p=Path(result['result']['envelope_path']);roots.append(p.parent)
   self.assertEqual(stat.S_IMODE(p.parent.stat().st_mode),0o700);self.assertEqual(list(p.parent.glob('graph-prompt-*')),[])
  self.assertEqual(len(set(roots)),4);self.assertEqual(checkpoint.read_bytes(),b'checkpoint sentinel');self.assertEqual(stat.S_IMODE(metadata.stat().st_mode),0o755)
 def test_missing_workspace_refuses_before_subprocess(self):
  self.workspace.rmdir()
  with patch.object(bg.subprocess,'run') as run:
   result=self.call();run.assert_not_called()
  self.assertEqual(result['result']['error']['stage'],'prepare');self.assertFalse(self.workspace.exists())
 def test_symlink_metadata_refuses_before_subprocess(self):
  outside=self.root/'outside';outside.mkdir();(self.workspace/'.graph-checkpoints').symlink_to(outside,target_is_directory=True)
  with patch.object(bg.subprocess,'run') as run:
   result=self.call();run.assert_not_called()
  self.assertEqual(result['result']['error']['stage'],'prepare');self.assertEqual(list(outside.iterdir()),[])
 def test_file_metadata_refuses_without_overwrite(self):
  p=self.workspace/'.graph-checkpoints';p.write_bytes(b'project sentinel')
  with patch.object(bg.subprocess,'run') as run:
   result=self.call();run.assert_not_called()
  self.assertEqual(result['result']['error']['stage'],'prepare');self.assertEqual(p.read_bytes(),b'project sentinel')
 def test_shared_writable_metadata_refuses(self):
  p=self.workspace/'.graph-checkpoints';p.mkdir();p.chmod(0o777)
  with patch.object(bg.subprocess,'run') as run:
   result=self.call();run.assert_not_called()
  self.assertEqual(result['result']['error']['stage'],'prepare');self.assertEqual(list(p.iterdir()),[])
 def test_timeout_cleans_prompt_and_empty_call_directory(self):
  with patch.object(bg.subprocess,'run',side_effect=subprocess.TimeoutExpired('fixture',1)):
   result=self.call(timeout=1)
  self.assertEqual(result['result']['error']['stage'],'timeout');self.assertEqual(list((self.workspace/'.graph-checkpoints').iterdir()),[])
 def test_invalid_response_cleans_prompt_without_retry(self):
  with patch.object(bg.subprocess,'run',return_value=subprocess.CompletedProcess([],0,'not json','')) as run:
   result=self.call();self.assertEqual(run.call_count,1)
  self.assertFalse(result['result']['ok']);self.assertEqual(result['result']['error']['stage'],'parse');self.assertEqual(list((self.workspace/'.graph-checkpoints').iterdir()),[])

 def test_prompt_creation_failure_removes_empty_call_directory(self):
  with patch.object(bg.tempfile,'mkstemp',side_effect=OSError('fixture failure')),patch.object(bg.subprocess,'run') as run:
   result=self.call();run.assert_not_called()
  self.assertEqual(result['result']['error']['stage'],'prepare');self.assertEqual(list((self.workspace/'.graph-checkpoints').iterdir()),[])
