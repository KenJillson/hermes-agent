import ast,importlib.util,json,logging,os,socket,struct,sys,threading,time,types,unittest
from pathlib import Path
from unittest.mock import patch
PARENT=types.ModuleType('hermes_cli.build_graph_parent_git')
PARENT.ParentGit=unittest.mock.Mock()

ROOT=Path(__file__).resolve().parents[2]
WIRE=Path(os.environ.get('A2_TEST_WORKER_SOURCE',str(ROOT.parent/'scripts/michael-worker')))
KANBAN_SOURCE=ROOT/'hermes_cli/kanban_db.py'
CLI_SOURCE=Path(os.environ.get('A2_TEST_CLI_SOURCE',str(ROOT/'cli.py')))
sys.path.insert(0,str(WIRE))
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec);sys.modules[name]=obj;spec.loader.exec_module(obj);return obj
client=load('candidate_client',WIRE/'launch_client.py')
runner=load('candidate_implementer',ROOT/'hermes_cli/build_graph_implementer.py')
BODY='Write the declared file.\n\n## A2 Files\n```json\n["input.txt", "output.txt"]\n```\n'
ENV={'HERMES_KANBAN_TASK':'t_test','HERMES_KANBAN_BOARD':'ops','HERMES_KANBAN_RUN_ID':'7','HERMES_KANBAN_WORKSPACE':'/task'}
def task():return types.SimpleNamespace(id='t_test',current_run_id=7,status='running',worker_pid=os.getpid(),body=BODY,graph_executor='build_graph',workspace_kind='worktree')
def invoke(fn,**kwargs):return fn(goal='Do work in /workspace',workspace='/task',max_iterations=50,timeout_seconds=900,**kwargs)
class AdapterTests(unittest.TestCase):
 def setUp(self):
  for p in (patch.object(runner,'_a2_client',return_value=client),patch.dict(sys.modules,{'hermes_cli.build_graph_parent_git':PARENT})):
   p.start();self.addCleanup(p.stop)
 def test_success_exact_request_and_no_retry(self):
  fn=runner.make_a2_runner(task(),'/task',ENV)
  with patch.object(client,'launch',return_value={'version':1,'phase':'committed','token':'1'*32}) as launch:
   self.assertEqual(invoke(fn)['status'],'completed');request=launch.call_args.args[0]
   self.assertEqual(set(request),client.launch_protocol.KEYS);self.assertEqual(request['files'],['input.txt','output.txt']);self.assertEqual(request['run_id'],7)
   with self.assertRaises(runner.A2ReconciliationRequired):invoke(fn)
   self.assertEqual(launch.call_count,1)
 def test_failure_never_retries(self):
  for error in (TimeoutError(),ConnectionError(),client.Refused('disconnect')):
   with self.subTest(error=type(error).__name__),patch.object(client,'launch',side_effect=error) as launch:
    fn=runner.make_a2_runner(task(),'/task',ENV)
    for _ in range(2):
     with self.assertRaises(runner.A2ReconciliationRequired):invoke(fn)
    self.assertEqual(launch.call_count,1)
 def test_assignment_mismatch_no_channel(self):
  variants=[{'HERMES_KANBAN_TASK':'t_other'},{'HERMES_KANBAN_RUN_ID':'08'},{'HERMES_KANBAN_RUN_ID':'8'},{'HERMES_KANBAN_RUN_ID':'1.0'},{'HERMES_KANBAN_WORKSPACE':'/other'}]
  with patch.object(client,'launch') as launch:
   for diff in variants:
    with self.subTest(diff=diff),self.assertRaises(runner.ProtocolError):invoke(runner.make_a2_runner(task(),'/task',dict(ENV,**diff)))
   for field,value in [('worker_pid',os.getpid()+1),('status','done'),('current_run_id',8)]:
    t=task();setattr(t,field,value)
    with self.subTest(field=field),self.assertRaises(runner.ProtocolError):invoke(runner.make_a2_runner(t,'/task',ENV))
   launch.assert_not_called()
 def test_bad_declarations_no_channel(self):
  bad=['No declaration',BODY+BODY,'## A2 Files\n```json\n{}\n```', '## A2 Files\n```json\n["../secret"]\n```','## A2 Files\n```json\n["x","x"]\n```','## A2 Files\n```json\n["x","x/y"]\n```','## A2 Files\n```json\n[]\n```']
  with patch.object(client,'launch') as launch:
   for body in bad:
    t=task();t.body=body
    with self.subTest(body=body),self.assertRaises((runner.ProtocolError,client.launch_protocol.Refused)):invoke(runner.make_a2_runner(t,'/task',ENV))
   launch.assert_not_called()
 def test_interrupt_before_channel(self):
  with patch.object(client,'launch') as launch:
   self.assertEqual(invoke(runner.make_a2_runner(task(),'/task',ENV),interrupt_check=lambda:True)['status'],'interrupted');launch.assert_not_called()
 def test_lazy_binding_preserves_nonimplement_graph(self):
  with patch.object(runner,'_a2_client',side_effect=AssertionError('must not load')):
   fn=runner.make_a2_runner(types.SimpleNamespace(),'',{});self.assertEqual(fn.payload_workspace,'/workspace')

# The actual CLI function is compiled, not reproduced. Imports are replaced
# only at the graph/database boundary; the installed dispatch wiring executes.
def cli_function():
 tree=ast.parse(CLI_SOURCE.read_bytes());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_run_build_graph_q'];assert len(nodes)==1
 ns={'logger':logging.getLogger('dispatch-test')};exec(compile(ast.Module(body=nodes,type_ignores=[]),'actual-cli','exec'),ns);return ns['_run_build_graph_q']
class CliPathTests(unittest.TestCase):
 def setup_path(self,failure=False,implemented=True):
  kb=types.ModuleType('hermes_cli.kanban_db');kb.GRAPH_EXECUTORS={'build_graph'};kb.connect=lambda:types.SimpleNamespace(close=lambda:None);kb.get_task=lambda *a:task();kb.parse_ac_text=lambda b:(None,None,None,None);kb.block_task=unittest.mock.Mock(return_value=True)
  bg=types.ModuleType('hermes_cli.build_graph');bg.Deps=lambda **kw:types.SimpleNamespace(**kw)
  def graph(*args,**kw):
   self.assertEqual(kw['checkpoint'],'workspace');self.assertEqual(kw['resume'],'auto')
   if implemented:invoke(kw['deps'].implement_runner)
   return {'terminal_reason':'human_review'}
  bg.run=graph;pkg=types.ModuleType('hermes_cli');pkg.kanban_db=kb;pkg.build_graph=bg;pkg.build_graph_implementer=runner
  return kb,{'hermes_cli':pkg,'hermes_cli.kanban_db':kb,'hermes_cli.build_graph':bg,'hermes_cli.build_graph_implementer':runner,'hermes_cli.build_graph_parent_git':PARENT}
 def test_actual_cli_success_uses_a2_then_parks(self):
  kb,modules=self.setup_path()
  with patch.dict(sys.modules,modules),patch.dict(os.environ,ENV,clear=True),patch.object(runner,'_a2_client',return_value=client),patch.object(client,'launch',return_value={}) as launch:
   self.assertEqual(cli_function()(types.SimpleNamespace(agent=None)),0);self.assertEqual(launch.call_count,1);kb.block_task.assert_called_once()
 def test_actual_cli_uncertain_transport_never_parks(self):
  kb,modules=self.setup_path()
  with patch.dict(sys.modules,modules),patch.dict(os.environ,ENV,clear=True),patch.object(runner,'_a2_client',return_value=client),patch.object(client,'launch',side_effect=TimeoutError()) as launch:
   self.assertEqual(cli_function()(types.SimpleNamespace(agent=None)),1);launch.assert_called_once();kb.block_task.assert_not_called()
 def test_actual_cli_nonimplement_does_not_load_client(self):
  kb,modules=self.setup_path(implemented=False)
  with patch.dict(sys.modules,modules),patch.dict(os.environ,ENV,clear=True),patch.object(runner,'_a2_client',side_effect=AssertionError()):
   self.assertEqual(cli_function()(types.SimpleNamespace(agent=None)),0);kb.block_task.assert_called_once()
 def test_cli_baseline_binding_failure_never_parks(self):
  kb,modules=self.setup_path();kb.parse_ac_text=lambda b:(None,None,None,True)
  bgd=types.ModuleType('hermes_cli.build_graph_diff');modules['hermes_cli'].build_graph_diff=bgd;modules[bgd.__name__]=bgd
  with patch.dict(sys.modules,modules),patch.dict(os.environ,ENV,clear=True),patch.object(PARENT,'ParentGit',side_effect=RuntimeError('inhibit')):
   self.assertEqual(cli_function()(types.SimpleNamespace(agent=None)),1);kb.block_task.assert_not_called()
 def test_cli_baseline_and_post_diff_use_same_guard(self):
  kb,modules=self.setup_path();kb.parse_ac_text=lambda b:(None,None,None,True)
  bgd=types.ModuleType('hermes_cli.build_graph_diff');bgd.EMPTY_DIFF_REASON='empty';modules['hermes_cli'].build_graph_diff=bgd;modules[bgd.__name__]=bgd
  context=unittest.mock.Mock();context.derive.return_value={'ok':False,'reason':'empty'}
  def graph(*args,**kw):
   invoke(kw['deps'].implement_runner)
   kw['deps'].derive('/task')
   return {'terminal_reason':'human_review'}
  modules['hermes_cli.build_graph'].run=graph
  with patch.dict(sys.modules,modules),patch.dict(os.environ,ENV,clear=True),patch.object(PARENT,'ParentGit',return_value=context) as constructor,patch.object(runner,'_a2_client',return_value=client),patch.object(client,'launch',return_value={}):
   self.assertEqual(cli_function()(types.SimpleNamespace(agent=None)),0)
   constructor.assert_called_once();self.assertEqual(context.derive.call_count,2)
   self.assertEqual([c[0] for c in context.mock_calls],['derive','launch','committed','derive'])
 def test_dispatcher_actual_resolver_routes_graph_without_creation(self):
  source=KANBAN_SOURCE.read_text();tree=ast.parse(source);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_resolve_worktree_workspace')
  ns={'Task':object,'Optional':__import__('typing').Optional,'Path':Path,'GRAPH_EXECUTORS':('build_graph',)}
  exec(compile(ast.Module(body=[node],type_ignores=[]),'actual-kanban-resolver','exec'),ns)
  t=task();PARENT.precreated=unittest.mock.Mock(return_value=(Path('/task'),'wt/t_test'))
  with patch.dict(sys.modules,{'hermes_cli.build_graph_parent_git':PARENT}):
   self.assertEqual(ns['_resolve_worktree_workspace'](t,board='ops'),(Path('/task'),'wt/t_test'))
   PARENT.precreated.assert_called_once_with(t,board='ops')
   PARENT.precreated.side_effect=RuntimeError('not pre-created')
   with self.assertRaisesRegex(RuntimeError,'not pre-created'):ns['_resolve_worktree_workspace'](t,board='ops')
 def test_dispatcher_prior_uncertain_run_never_reclaims(self):
  import sqlite3
  tree=ast.parse(KANBAN_SOURCE.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='check_respawn_guard')
  ns={'sqlite3':sqlite3,'Optional':__import__('typing').Optional}
  exec(compile(ast.Module(body=[node],type_ignores=[]),'actual-respawn-guard','exec'),ns)
  with sqlite3.connect(':memory:') as db:
   db.row_factory=sqlite3.Row
   db.executescript('CREATE TABLE tasks(id TEXT,graph_executor TEXT,last_failure_error TEXT);CREATE TABLE task_runs(id INTEGER,task_id TEXT,ended_at INTEGER,outcome TEXT);')
   db.execute("INSERT INTO tasks VALUES ('t_test','build_graph',NULL)")
   for outcome in ('crashed','timed_out','reclaimed','stale','rate_limited','gave_up',None):
    db.execute('DELETE FROM task_runs');db.execute('INSERT INTO task_runs VALUES (1,?,?,?)',('t_test',1,outcome))
    self.assertEqual(ns['check_respawn_guard'](db,'t_test'),'a2_prior_run_requires_reconciliation')
   # A newer blocked outcome must not conceal older transport uncertainty.
   db.execute("INSERT INTO task_runs VALUES (2,'t_test',2,'blocked')")
   self.assertEqual(ns['check_respawn_guard'](db,'t_test'),'a2_prior_run_requires_reconciliation')
 def test_graph_prompt_uses_payload_workspace(self):
  tree=ast.parse((ROOT/'hermes_cli/build_graph.py').read_bytes());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='make_implement')
  seen=[]
  def run(**kwargs):seen.append(kwargs);return {'status':'completed'}
  run.payload_workspace='/workspace'
  ns={'Deps':object,'IMPLEMENT_ATTEMPT_CAP':1,'IMPLEMENT_MAX_ITERATIONS':50,'IMPLEMENT_TIMEOUT_SECONDS':900,'IMPLEMENT_STATUSES':('completed',),'IMPLEMENT_OK':'completed','bump_rung':lambda s,r:{'implement':1},'guard':lambda u,s:u,'build_implement_goal':lambda s,w,p:w}
  exec(compile(ast.Module(body=[node],type_ignores=[]),'actual-graph','exec'),ns)
  deps=types.SimpleNamespace(body=BODY,workspace='/host-workspace',implement_runner=run,agent=None,derive=lambda w:{'ok':True,'diff':'x','diff_files':1,'diff_added_lines':1,'changed_files':['output.txt']})
  ns['make_implement'](deps)({'rung_attempts':{}});self.assertEqual(seen[0]['goal'],'/workspace');self.assertEqual(seen[0]['workspace'],'/host-workspace')

class InterruptTests(unittest.TestCase):
 def test_real_socket_wait_interrupt_closes_once(self):
  # Actual socketpair/select/recv transport. Only credentials and fixed path
  # are fixture adapters; this test does not prove host-root authentication.
  left,right=socket.socketpair();sent=threading.Event();stop=threading.Event();observed=[]
  class Channel:
   def __enter__(self):return self
   def __exit__(self,*args):left.close()
   def connect(self,p):observed.append(p)
   def getsockopt(self,*args):return struct.pack('3i',0,0,0)
   def set_inheritable(self,v):left.set_inheritable(v)
   def settimeout(self,v):left.settimeout(v)
   def fileno(self):return left.fileno()
   def sendall(self,b):left.sendall(b);sent.set()
   def recv(self,n):return left.recv(n)
  def server():
   right.settimeout(3);right.recv(65536);stop.set();observed.append(right.recv(1));right.close()
  t=threading.Thread(target=server);t.start()
  request={'version':1,'operation':'implement','board':'ops','task_id':'t_test','run_id':7,'goal':'test','files':['x']}
  begin=time.monotonic()
  with patch.object(client.socket,'socket',return_value=Channel()),patch.object(client.socket,'SO_PEERCRED',17,create=True):
   with self.assertRaisesRegex(client.Refused,'interrupted'):client.launch(request,interrupt_check=stop.is_set)
  t.join(3);self.assertFalse(t.is_alive());self.assertLess(time.monotonic()-begin,2);self.assertEqual(observed,[client.SOCKET_PATH,b''])

if __name__=='__main__':unittest.main(verbosity=2)
