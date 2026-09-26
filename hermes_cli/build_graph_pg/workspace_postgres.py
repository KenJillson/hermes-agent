"""Runtime workspace factory using the fixed peer route; never sets up schema."""
from contextlib import contextmanager
from psycopg import connect
from psycopg.rows import dict_row
from hermes_cli.maintenance_gate import _protected_module, hold_runtime_for_process
from hermes_cli.build_graph_state import SCHEMA_VERSION
from . import scope_registry as registry
from . import workspace_identity as identity_module
from .scoped_postgres_saver import ScopedPostgresSaver


def fixed_connection():
    # Verified dedicated substrate; never take destinations or credentials from
    # a card, graph config, payload JSON, environment variable or supplied DSN.
    return connect(host='/run/postgresql',dbname='michael_checkpoints',
                   user='michael_checkpoint',connect_timeout=8,autocommit=True,
                   row_factory=dict_row)


def registered_guard(conn, marker):
    identity={key:marker[key] for key in ('board_database','task_id','workspace')}
    row=registry.lookup(conn,identity)
    registry.verify_association(row,marker,SCHEMA_VERSION)


@contextmanager
def open_workspace_checkpointer(board_connection, task_id, workspace):
    hold_runtime_for_process()
    admission=_protected_module()
    with admission.admit():
        def admitted_write():
            with admission.admit():
                pass
        identity=identity_module.identity_from_task(board_connection,task_id,workspace)
        # Connection context closes on every refusal, yield exception and normal exit.
        with fixed_connection() as conn:
            peer=conn.execute('SELECT current_user AS role,current_database() AS database').fetchone()
            registry.require(peer=={'role':'michael_checkpoint','database':'michael_checkpoints'},
                             'Unexpected checkpoint peer identity')
            registry.verify_schema(conn)
            with identity_module.marker_lock(identity,mutation_gate=admitted_write) as directory:
                with conn.transaction():
                    conn.execute('SELECT pg_advisory_xact_lock(%s)',(registry.identity_key(identity),))
                    row=registry.lookup(conn,identity)
                    marker=identity_module.read_marker(directory,identity)
                    if marker is None:
                        marker=identity_module.create_marker(directory,identity,
                            mutation_gate=admitted_write,existing_database_scope=row is not None)
                        # If commit fails after publication, the marker remains as
                        # recovery evidence. A later invocation must refuse it.
                        registry.register_new(conn,identity,marker,SCHEMA_VERSION)
                    else:
                        registry.verify_association(row,marker,SCHEMA_VERSION)
            saver=ScopedPostgresSaver(conn,marker['scope_id'],schema_version=SCHEMA_VERSION,
                scope_guard=lambda connection:registered_guard(connection,marker))
            yield saver,saver.serde
