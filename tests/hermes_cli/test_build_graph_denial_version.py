"""Versioned denial summaries preserve the graph's closed evidence checks."""
import unittest
from hermes_cli import build_graph_review_evidence as ev
class VersionTests(unittest.TestCase):
 def test_versions_and_denial_refusal(self):
  v=dict(version=1,board='ops',task_id='t_fixture',run_id=1,operation='local_review',token='a'*32,request_sha256='b'*64,journal_sha256='c'*64,journal_version=3,complete=True,requested=2,observed=2,denied=0,unknown=0)
  for version in (1,2,3):self.assertEqual(ev.validate_denial(dict(v,journal_version=version),'t_fixture','local_review',1)['journal_version'],version)
  for k,bad in [('journal_version',4),('journal_version',True),('denied',1),('unknown',1),('observed',1),('complete',False),('run_id',2)]:
   with self.subTest(k=k,bad=bad),self.assertRaises(ev.Refused):ev.validate_denial(dict(v,**{k:bad}),'t_fixture','local_review',1)
