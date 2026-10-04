"""Checkpoint setup must not retain admission while an A2 stage launches."""
from contextlib import contextmanager
import fcntl
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from hermes_cli.build_graph_pg import workspace_postgres as factory


@pytest.fixture
def harness(tmp_path, monkeypatch):
    admission_path = tmp_path / 'admission.lock'
    runtime_path = tmp_path / 'runtime.lock'
    admission_path.touch()
    runtime_path.touch()
    held = []
    state = SimpleNamespace(open=True, connection_open=False, setup_guarded=False)

    @contextmanager
    def admitted():
        with admission_path.open('rb') as stream:
            fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
            if not state.open:
                raise RuntimeError('maintenance closed')
            yield

    def hold_runtime():
        if not held:
            stream = runtime_path.open('rb')
            fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
            held.append(stream)

    @contextmanager
    def exclusive(path):
        with path.open('rb') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield

    class Connection:
        @contextmanager
        def transaction(self):
            yield

        def execute(self, *args):
            return SimpleNamespace(fetchone=lambda: {
                'role': 'michael_checkpoint', 'database': 'michael_checkpoints'})

    connection = Connection()

    @contextmanager
    def connect():
        state.connection_open = True
        try:
            yield connection
        finally:
            state.connection_open = False

    def verify_schema(conn):
        with pytest.raises(BlockingIOError):
            with exclusive(admission_path):
                pass
        state.setup_guarded = True

    @contextmanager
    def marker_lock(identity, mutation_gate):
        mutation_gate()
        yield object()

    marker = dict(scope_id='a' * 32, board_database='board', task_id='task', workspace='workspace')
    scope_check = Mock()

    class Saver:
        serde = object()

        def __init__(self, conn, scope_id, *, schema_version, scope_guard):
            self.guard = scope_guard

    monkeypatch.setattr(factory, 'hold_runtime_for_process', hold_runtime)
    monkeypatch.setattr(factory, '_protected_module', lambda: SimpleNamespace(admit=admitted))
    monkeypatch.setattr(factory, 'fixed_connection', connect)
    monkeypatch.setattr(factory.registry, 'verify_schema', verify_schema)
    monkeypatch.setattr(factory.registry, 'identity_key', lambda identity: 1)
    monkeypatch.setattr(factory.registry, 'lookup', lambda *args: marker)
    monkeypatch.setattr(factory.registry, 'verify_association', scope_check)
    monkeypatch.setattr(factory.identity_module, 'identity_from_task', lambda *args: marker)
    monkeypatch.setattr(factory.identity_module, 'marker_lock', marker_lock)
    monkeypatch.setattr(factory.identity_module, 'read_marker', lambda *args: marker)
    monkeypatch.setattr(factory, 'ScopedPostgresSaver', Saver)
    state.exclusive = exclusive
    state.admission_path = admission_path
    state.runtime_path = runtime_path
    state.scope_check = scope_check
    try:
        yield state
    finally:
        for stream in held:
            stream.close()


def test_a2_registration_can_take_exclusive_admission_during_graph(harness):
    with factory.open_workspace_checkpointer(None, 'task', 'workspace') as (saver, serde):
        assert harness.setup_guarded and harness.connection_open
        # This is the flock operation which failed in Gate.worker_lifetime.
        with harness.exclusive(harness.admission_path):
            pass
        # Maintenance deployment must still be excluded for the process lifetime.
        with pytest.raises(BlockingIOError):
            with harness.exclusive(harness.runtime_path):
                pass
        before = harness.scope_check.call_count
        saver.guard(None)
        assert harness.scope_check.call_count == before + 1
        assert serde is saver.serde
    assert not harness.connection_open


def test_closed_maintenance_refuses_setup(harness):
    harness.open = False
    with pytest.raises(RuntimeError, match='maintenance closed'):
        with factory.open_workspace_checkpointer(None, 'task', 'workspace'):
            pytest.fail('graph must not start')
    assert not harness.connection_open


def test_setup_failure_releases_admission_and_connection(harness, monkeypatch):
    def fail(*args):
        raise ValueError('bad schema')
    monkeypatch.setattr(factory.registry, 'verify_schema', fail)
    with pytest.raises(ValueError, match='bad schema'):
        with factory.open_workspace_checkpointer(None, 'task', 'workspace'):
            pytest.fail('graph must not start')
    assert not harness.connection_open
    with harness.exclusive(harness.admission_path):
        pass


def test_graph_failure_closes_connection_without_releasing_runtime(harness):
    with pytest.raises(ValueError, match='graph failed'):
        with factory.open_workspace_checkpointer(None, 'task', 'workspace'):
            raise ValueError('graph failed')
    assert not harness.connection_open
    with harness.exclusive(harness.admission_path):
        pass
    with pytest.raises(BlockingIOError):
        with harness.exclusive(harness.runtime_path):
            pass
