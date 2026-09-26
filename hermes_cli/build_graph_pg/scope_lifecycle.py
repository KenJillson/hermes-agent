"""Coordinated scratch cleanup with durable state and no unattended recovery."""
from contextlib import contextmanager
import os
from pathlib import Path
from hermes_cli.maintenance_gate import _protected_module,hold_runtime_for_process
from . import workspace_identity as identities
from . import scope_registry as registry
from .workspace_postgres import fixed_connection

TERMINAL = ('done','archived','failed','cancelled')


def scope_key(scope_id):
    return int.from_bytes(bytes.fromhex(scope_id[:16]),'big',signed=True)


def cleanup_scratch(board_connection, task_id, workspace, *, remove):
    """Caller must first apply kanban's managed-scratch and child-lifetime guards.

    remove is the caller's existing filesystem deletion primitive, not a shell
    command. Pending state is durable before deletion; any uncertainty preserves
    checkpoint data and requires an explicit maintenance recovery.
    """
    hold_runtime_for_process()
    admission=_protected_module()
    with admission.admit():
        def admitted_write():
            with admission.admit():pass
        identity=identities.identity_from_task(board_connection,task_id,workspace)
        assignment=board_connection.execute('SELECT workspace_kind FROM tasks WHERE id=?',(task_id,)).fetchone()
        registry.require(assignment is not None and assignment[0]=='scratch','Only scratch lifecycle may delete checkpoints')
        actual=identity['workspace'];before=os.stat(actual,follow_symlinks=False)
        with fixed_connection() as conn:
            registry.verify_schema(conn)
            with identities.marker_lock(identity,mutation_gate=admitted_write) as directory:
                with conn.transaction():
                    rows=conn.execute('''SELECT scope_id,board_database,task_id,workspace,state,
                        workflow_schema,legacy_md5 FROM michael_checkpoint_scopes
                        WHERE workspace=%s AND state<>'deleted' ORDER BY scope_id''',(actual,)).fetchall()
                    for row in rows:
                        registry.require(row['board_database']==identity['board_database'],
                                         'Shared workspace has a foreign-board scope; cleanup refused')
                        registry.require(row['state']=='active','Prior cleanup is unresolved; explicit recovery required')
                        assignment=board_connection.execute('SELECT status,workspace_path FROM tasks WHERE id=?',(row['task_id'],)).fetchone()
                        registry.require(assignment is not None and assignment[0] in TERMINAL
                            and os.path.realpath(assignment[1])==actual,'Registered task still needs its workspace')
                        owned={k:row[k] for k in ('board_database','task_id','workspace')}
                        marker=identities.read_marker(directory,owned)
                        registry.require(marker is not None,'Registered cleanup marker missing; explicit recovery required')
                        registry.verify_association(row,marker,row['workflow_schema'])
                        conn.execute('SELECT pg_advisory_xact_lock(%s)',(scope_key(row['scope_id']),))
                    for row in rows:
                        conn.execute("UPDATE michael_checkpoint_scopes SET state='cleanup_pending' WHERE scope_id=%s",(row['scope_id'],))
                current=os.stat(actual,follow_symlinks=False)
                registry.require((current.st_dev,current.st_ino)==(before.st_dev,before.st_ino),
                                 'Workspace identity changed before cleanup')
                admitted_write();remove(Path(actual))
                registry.require(not os.path.lexists(actual),'Scratch removal incomplete; explicit recovery required')
                # The removed directory descriptor still pins our flock until exit.
                # New writers cannot reopen this missing workspace; existing savers
                # see cleanup_pending and refuse through their registration guard.
                with conn.transaction():
                    for row in rows:
                        conn.execute('SELECT pg_advisory_xact_lock(%s)',(scope_key(row['scope_id']),))
                        found=registry.lookup(conn,{k:row[k] for k in ('board_database','task_id','workspace')})
                        registry.require(found is not None and found['scope_id']==row['scope_id']
                                         and found['state']=='cleanup_pending','Cleanup registration changed')
                        prefix=row['scope_id']+':'
                        for table in ('checkpoint_writes','checkpoint_blobs','checkpoints'):
                            conn.execute('DELETE FROM '+table+' WHERE left(thread_id,%s)=%s',(len(prefix),prefix))
                        conn.execute("UPDATE michael_checkpoint_scopes SET state='deleted' WHERE scope_id=%s",(row['scope_id'],))
        return {'workspace_removed':True,'scopes_deleted':len(rows)}
