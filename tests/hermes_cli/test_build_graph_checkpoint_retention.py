"""Regression tests for the real WorkspaceSaver, using in-memory flush only."""
import copy,json,threading,unittest
from langgraph.graph import StateGraph,START,END
from typing import TypedDict
from hermes_cli import build_graph_checkpoint as ck

class MemorySaver(ck.WorkspaceSaver):
 def flush(self):
  self.last_envelope=self.envelope();self.flush_count+=1
class State(TypedDict):
 i:int
 payload:str
class Retention(unittest.TestCase):
 def setup_graph(self,cap=64):
  saver=MemorySaver('/__ac4_no_filesystem__',hydrate=False,max_checkpoints_per_thread=cap)
  graph=StateGraph(State)
  graph.add_node('step',lambda s:{'i':s['i']+1,'payload':str(s['i']+1)+'x'*1024})
  graph.add_edge(START,'step');graph.add_conditional_edges('step',lambda s:END if s['i']%16==0 else 'step',{END:END,'step':'step'})
  return saver,graph.compile(checkpointer=saver)
 def cfg(self,thread='a',ns=''):return {'configurable':{'thread_id':thread,'checkpoint_ns':ns},'recursion_limit':40}
 def test_sustained_retention_and_latest_resume(self):
  s,app=self.setup_graph();cfg=self.cfg();samples=[]
  for batch in range(8):
   result=app.invoke({'i':batch*16,'payload':'input'},cfg);self.assertEqual(result['i'],(batch+1)*16)
   samples.append((len(s.storage['a']['']),len(s.writes),len(s.blobs)))
  self.assertEqual(samples[-1][0],64);self.assertLessEqual(samples[-1][1],64);self.assertLessEqual(samples[-1][2],samples[-3][2])
  self.assertEqual(s.get_tuple(cfg).checkpoint['channel_values']['i'],128)
  self.assertEqual(len(list(s.list(cfg))),64)
 def test_all_retained_history_values_and_pending_writes(self):
  s,app=self.setup_graph(cap=8);cfg=self.cfg();app.invoke({'i':0,'payload':'input'},cfg)
  rows=list(s.list(cfg));self.assertEqual(len(rows),8)
  for row in rows:
   i=row.checkpoint['channel_values']['i'];self.assertEqual(row.checkpoint['channel_values']['payload'],str(i)+'x'*1024)
  self.assertGreater(sum(len(row.pending_writes) for row in rows),0)
 def test_pending_writes_before_checkpoint_are_preserved(self):
  s,app=self.setup_graph(cap=8);cfg=self.cfg();app.invoke({'i':0,'payload':'input'},cfg)
  latest=s.get_tuple(cfg);future=dict(latest.config['configurable'],checkpoint_id='z-future');f={'configurable':future}
  s.put_writes(f,[('i',17)],'pending-task');self.assertEqual(len(s.writes[('a','','z-future')]),1)
  cp=copy.deepcopy(latest.checkpoint);cp['id']='z-future';s.put(latest.config,cp,latest.metadata,{})
  self.assertEqual(s.get_tuple(f).pending_writes,[('pending-task','i',17)])
 def test_other_thread_and_namespace_preserved(self):
  s,app=self.setup_graph(cap=8);app.invoke({'i':0,'payload':'input'},self.cfg('b'));before=copy.deepcopy(s.storage['b'])
  app.invoke({'i':0,'payload':'input'},self.cfg());app.invoke({'i':16,'payload':'input'},self.cfg());self.assertEqual(before,s.storage['b'])
  self.assertEqual(s.get_tuple(self.cfg('b')).checkpoint['channel_values']['i'],16)
  latest=s.get_tuple(self.cfg());other=self.cfg('a','other');cp=copy.deepcopy(latest.checkpoint)
  s.put(other,cp,latest.metadata,cp['channel_versions']);self.assertEqual(s.get_tuple(other).checkpoint['channel_values'],latest.checkpoint['channel_values'])
  app.invoke({'i':32,'payload':'input'},self.cfg());self.assertEqual(s.get_tuple(other).checkpoint['channel_values'],latest.checkpoint['channel_values'])
 def test_pruning_disabled_retains_all_stores(self):
  s,app=self.setup_graph(cap=0);cfg=self.cfg()
  for i in range(5):app.invoke({'i':i*16,'payload':'input'},cfg)
  self.assertGreater(len(s.storage['a']['']),64);self.assertGreater(len(s.writes),64)
 def test_malformed_checkpoint_does_not_partially_prune(self):
  s,app=self.setup_graph(cap=0);app.invoke({'i':0,'payload':'input'},self.cfg());s.max_checkpoints_per_thread=2
  ns=s.storage['a'][''];last=max(ns);cp,meta,parent=ns[last];ns[last]=(s.serde.dumps_typed({'id':last}),meta,parent)
  before=(copy.deepcopy(s.storage),copy.deepcopy(s.writes),copy.deepcopy(s.blobs))
  with self.assertRaises(ck.CheckpointShapeError):s.envelope()
  self.assertEqual((s.storage,s.writes,s.blobs),before)
 def test_envelope_and_mutation_are_serialized(self):
  s,app=self.setup_graph();app.invoke({'i':0,'payload':'input'},self.cfg());last=s.get_tuple(self.cfg());paused=threading.Event();release=threading.Event();started=threading.Event();done=threading.Event();errors=[]
  class PauseDict(dict):
   def items(self):
    if threading.current_thread().name=='reader':paused.set();release.wait(3)
    return super().items()
  s.blobs=PauseDict(s.blobs)
  def read():
   try:s.envelope()
   except BaseException as e:errors.append(e)
  def write():
   started.set()
   try:s.put_writes(last.config,[('i',99)],'parallel-task')
   except BaseException as e:errors.append(e)
   finally:done.set()
  reader=threading.Thread(target=read,name='reader');writer=threading.Thread(target=write,name='writer');reader.start()
  try:
   self.assertTrue(paused.wait(2));writer.start();self.assertTrue(started.wait(2));self.assertFalse(done.wait(.1))
  finally:release.set();reader.join(3);writer.join(3) if writer.ident else None
  self.assertFalse(reader.is_alive());self.assertFalse(writer.is_alive());self.assertFalse(errors);self.assertTrue(done.is_set())
 def test_delete_thread_leaves_other_thread(self):
  s,app=self.setup_graph();app.invoke({'i':0,'payload':'input'},self.cfg());app.invoke({'i':0,'payload':'input'},self.cfg('b'));s.delete_thread('a');self.assertIsNone(s.get_tuple(self.cfg()));self.assertEqual(s.get_tuple(self.cfg('b')).checkpoint['channel_values']['i'],16)

import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from typing import TypedDict
from langgraph.graph import StateGraph,START,END
from hermes_cli import build_graph_checkpoint as ck
from hermes_cli import build_graph_sanitize as sz
class State(TypedDict):
 i:int
 payload:str
class Persistence(unittest.TestCase):
 def graph(self,saver):
  g=StateGraph(State);g.add_node('step',lambda s:{'i':s['i']+1,'payload':str(s['i']+1)+'x'*1024});g.add_edge(START,'step');g.add_conditional_edges('step',lambda s:END if s['i']%16==0 else 'step',{END:END,'step':'step'});return g.compile(checkpointer=saver)
 def saver(self,root,**kwargs):return ck.make_workspace_checkpointer(root,schema_version=2,**kwargs)[0]
 def test_atomic_file_hydration_and_continuation(self):
  with tempfile.TemporaryDirectory() as td:
   cfg={'configurable':{'thread_id':'persist'},'recursion_limit':40};s=self.saver(td);app=self.graph(s)
   for i in range(5):self.assertEqual(app.invoke({'i':i*16,'payload':'start'},cfg)['i'],(i+1)*16)
   disk=json.loads(Path(s.envelope_path).read_text());self.assertEqual(len(disk['storage']),64);self.assertLessEqual(len(disk['writes']),64)
   revived=self.saver(td);self.assertTrue(revived.has_thread('persist'));self.assertEqual(revived.get_tuple(cfg).checkpoint['channel_values']['i'],80)
   result=self.graph(revived).invoke({'i':80,'payload':'start'},cfg);self.assertEqual(result['i'],96)
   self.assertEqual(self.saver(td).get_tuple(cfg).checkpoint['channel_values']['i'],96)
   self.assertEqual(list(Path(td).glob('.checkpoints-*.tmp')),[])
 def test_atomic_replace_failure_preserves_prior_envelope(self):
  with tempfile.TemporaryDirectory() as td:
   s=self.saver(td);self.graph(s).invoke({'i':0,'payload':'start'},{'configurable':{'thread_id':'persist'}});prior=Path(s.envelope_path).read_bytes()
   with patch.object(ck.os,'replace',side_effect=OSError('injected replace failure')):
    with self.assertRaisesRegex(OSError,'injected'):s.flush()
   self.assertEqual(Path(s.envelope_path).read_bytes(),prior);self.assertEqual(list(Path(td).glob('.checkpoints-*.tmp')),[])
 def test_size_refusal_preserves_durable_state(self):
  with tempfile.TemporaryDirectory() as td:
   s=self.saver(td);self.graph(s).invoke({'i':0,'payload':'start'},{'configurable':{'thread_id':'persist'}});prior=Path(s.envelope_path).read_bytes();s.max_envelope_bytes=1
   with self.assertRaises(ck.CheckpointTooLarge):s.flush()
   self.assertEqual(Path(s.envelope_path).read_bytes(),prior)
