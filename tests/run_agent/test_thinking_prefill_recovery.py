"""Drive AIAgent's actual loop and request builder with a captured provider failure."""
import copy
from unittest.mock import MagicMock,patch
import httpx,openai
from openai.types.chat import ChatCompletion
from run_agent import AIAgent
ERROR='Assistant response prefill is incompatible with enable_thinking.'
def response(text=None,thinking=False,finish="stop"):
 message={'role':'assistant','content':text or ''}
 if thinking:message.update(content='<think>Answer briefly.</think>',reasoning_content='Answer briefly.')
 return ChatCompletion.model_validate({'id':'fixture','object':'chat.completion','created':1,'model':'fixture-model','choices':[{'index':0,'finish_reason':finish,'message':message}],'usage':{'prompt_tokens':8,'completion_tokens':8,'total_tokens':16}})
def error(message=ERROR,status=400):
 return openai.BadRequestError(message,response=httpx.Response(status,request=httpx.Request('POST','http://fixture.invalid/v1/chat/completions')),body={'error':{'message':message}})
def agent():
 with patch('run_agent.get_tool_definitions',return_value=[]),patch('run_agent.check_toolset_requirements',return_value={}),patch('run_agent.OpenAI'):
  a=AIAgent(api_key='fixture-only',base_url='http://fixture.invalid/v1',provider='custom',api_mode='chat_completions',model='fixture-model',quiet_mode=True,skip_context_files=True,skip_memory=True)
 a.client=MagicMock();a._cached_system_prompt='Be concise.';a._use_prompt_caching=False;a.compression_enabled=False;a.save_trajectories=False;a._api_max_retries=3
 return a
def run(a):
 with patch.object(a,'_persist_session'),patch.object(a,'_save_trajectory'),patch.object(a,'_cleanup_task_resources'):
  return a.run_conversation('Reply Ready.')
def test_exact_rejection_recovers_and_next_turn_keeps_defaults():
 a=agent();seen=[];original={'chat_template_kwargs':{'preserve_reasoning':True},'fixture_option':1};a.request_overrides={'extra_body':copy.deepcopy(original)}
 def create(**kw):
  seen.append(copy.deepcopy(kw))
  if len(seen)==1:return response(thinking=True)
  if len(seen)==2:raise error()
  return response('Ready.')
 a.client.chat.completions.create.side_effect=create
 result=run(a)
 assert result['final_response']=='Ready.',result
 assert len(seen)==3
 assert seen[1]['messages'][-1]['role']=='assistant'
 assert seen[2]['extra_body']['chat_template_kwargs']=={'preserve_reasoning':True,'enable_thinking':False}
 assert a.request_overrides=={'extra_body':original}
 assert run(a)['final_response']=='Ready.'
 assert len(seen)==4 and 'enable_thinking' not in seen[-1].get('extra_body',{}).get('chat_template_kwargs',{})
def test_repeated_rejection_does_not_loop():
 a=agent();a.client.chat.completions.create.side_effect=[response(thinking=True),error(),error()]
 result=run(a);assert result.get('failed') is True;assert a.client.chat.completions.create.call_count==3
 assert a.client.chat.completions.create.call_args.kwargs['extra_body']['chat_template_kwargs']['enable_thinking'] is False
def test_unrelated_400_is_not_recovered():
 a=agent();a.client.chat.completions.create.side_effect=[response(thinking=True),error('invalid tool schema')]
 result=run(a);assert result.get('failed') is True;assert a.client.chat.completions.create.call_count==2
 assert 'enable_thinking' not in a.client.chat.completions.create.call_args.kwargs.get('extra_body',{}).get('chat_template_kwargs',{})
def test_exact_error_without_marked_prefill_is_not_recovered():
 a=agent();a.client.chat.completions.create.side_effect=error();result=run(a)
 assert result.get('failed') is True;assert a.client.chat.completions.create.call_count==1
def test_exhausted_budget_does_not_add_attempt():
 a=agent();a._api_max_retries=1;a.client.chat.completions.create.side_effect=[response(thinking=True),error()];result=run(a)
 assert result.get('failed') is True;assert a.client.chat.completions.create.call_count==2
def test_explicitly_disabled_thinking_does_not_retry_same_override():
 a=agent();a.request_overrides={'extra_body':{'chat_template_kwargs':{'enable_thinking':False}}};a.client.chat.completions.create.side_effect=[response(thinking=True),error()];result=run(a)
 assert result.get('failed') is True;assert a.client.chat.completions.create.call_count==2

def test_length_exhaustion_keeps_existing_terminal_behavior():
 a=agent();a.client.chat.completions.create.return_value=response(thinking=True,finish='length');result=run(a)
 assert result['completed'] is False and 'Thinking Budget Exhausted' in result['final_response']
 assert a.client.chat.completions.create.call_count==1

def test_noncustom_provider_does_not_enable_recovery():
 a=agent();a.provider='openai';a.client.chat.completions.create.side_effect=[response(thinking=True),error()];result=run(a)
 assert result.get('failed') is True and a.client.chat.completions.create.call_count==2

def test_changed_provider_endpoint_or_model_does_not_receive_override():
 for field,value in [('provider','openai'),('base_url','http://another.invalid/v1'),('model','another-fixture-model')]:
  a=agent();original=a._build_api_kwargs;builds=[];seen=[]
  def build(*args,**kw):
   builds.append(1)
   if len(builds)==3:setattr(a,field,value)
   return original(*args,**kw)
  def create(**kw):
   seen.append(copy.deepcopy(kw))
   if len(seen)==1:return response(thinking=True)
   if len(seen)==2:raise error()
   return response('Ready.')
  a._build_api_kwargs=build;a.client.chat.completions.create.side_effect=create
  assert run(a)['final_response']=='Ready.'
  assert len(seen)==3 and 'enable_thinking' not in seen[-1].get('extra_body',{}).get('chat_template_kwargs',{}),field
