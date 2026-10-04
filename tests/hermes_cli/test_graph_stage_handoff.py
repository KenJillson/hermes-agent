import os
import pytest
from hermes_cli import kanban_db as kb


@pytest.fixture
def card(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    kb.init_db()
    with kb.connect_closing() as conn:
        task_id=kb.create_task(conn,title='Stage fixture',assignee='builder',initial_status='blocked')
        kb.set_graph_executor(conn,task_id,'build_graph')
        assert kb.unblock_task(conn,task_id)
        task=kb.claim_task(conn,task_id,claimer='fixture')
        assert task is not None
        kb._set_worker_pid(conn,task_id,os.getpid())
        yield conn,task_id,task.current_run_id


def test_handoff_waits_for_exit_and_creates_new_run(card):
    conn,task_id,run_id=card
    assert kb.finish_graph_stage(conn,task_id,run_id,'plan','plan_review',os.getpid())
    assert kb.get_task(conn,task_id).status=='blocked'
    assert kb.get_run(conn,run_id).ended_at is not None
    assert kb.recompute_ready(conn)==0
    assert kb.promote_graph_stages(conn,idle_check=lambda:None,pid_exists=lambda pid:True)==[]
    assert kb.promote_graph_stages(conn,idle_check=lambda:None,pid_exists=lambda pid:False)==[task_id]
    assert kb.promote_graph_stages(conn,idle_check=lambda:None,pid_exists=lambda pid:False)==[]
    next_run=kb.claim_task(conn,task_id,claimer='next-stage')
    assert next_run.current_run_id!=run_id
    assert next_run.status=='running'


def test_idle_gate_failure_cannot_promote(card):
    conn,task_id,run_id=card
    assert kb.finish_graph_stage(conn,task_id,run_id,'plan','plan_review',os.getpid())
    def active(): raise RuntimeError('active run')
    assert kb.promote_graph_stages(conn,idle_check=active,pid_exists=lambda pid:False)==[]
    assert kb.get_task(conn,task_id).status=='blocked'


def test_stale_worker_cannot_close_newer_run(card):
    conn,task_id,run_id=card
    assert not kb.finish_graph_stage(conn,task_id,run_id+1,'plan','plan_review',os.getpid())
    assert kb.get_run(conn,run_id).ended_at is None


def test_normal_block_never_continues(card):
    conn,task_id,run_id=card
    assert kb.block_task(conn,task_id,reason='review-required',kind='needs_input',expected_run_id=run_id)
    assert kb.promote_graph_stages(conn,idle_check=lambda:None,pid_exists=lambda pid:False)==[]
    assert kb.get_task(conn,task_id).status=='blocked'
