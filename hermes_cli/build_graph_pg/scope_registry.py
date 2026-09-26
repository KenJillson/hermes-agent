"""Explicit Postgres scope registry; schema delivery is administrator-only."""
from __future__ import annotations
import hashlib
import re


class RegistrationRefused(ValueError):
    pass


# A new, explicit application schema contract; not a claim about installed tables.
SCHEMA_SQL = '''
CREATE TABLE michael_checkpoint_registry_version (
    singleton boolean PRIMARY KEY CHECK (singleton),
    version integer NOT NULL CHECK (version = 1)
);
INSERT INTO michael_checkpoint_registry_version(singleton,version) VALUES (true,1);
CREATE TABLE michael_checkpoint_scopes (
    scope_id text PRIMARY KEY CHECK (scope_id ~ '^[0-9a-f]{32}$'),
    board_database text NOT NULL,
    task_id text NOT NULL,
    workspace text NOT NULL,
    state text NOT NULL CHECK (state IN ('active','cleanup_pending','deleted')),
    workflow_schema integer NOT NULL CHECK (workflow_schema > 0),
    legacy_md5 text CHECK (legacy_md5 ~ '^[0-9a-f]{32}$'),
    UNIQUE (board_database,task_id)
);
'''


def require(ok, message):
    if not ok:
        raise RegistrationRefused(message)


def identity_key(identity):
    raw = (identity['board_database']+'\0'+identity['task_id']).encode()
    return int.from_bytes(hashlib.sha256(b'michael-checkpoint-registration\0'+raw).digest()[:8],
                          'big', signed=True)


def verify_schema(conn):
    rows = conn.execute('SELECT singleton,version FROM michael_checkpoint_registry_version').fetchall()
    require(len(rows)==1 and rows[0]['singleton'] is True and rows[0]['version']==1,
            'Checkpoint registration schema differs; administrator migration required')


def lookup(conn, identity):
    row = conn.execute('''SELECT scope_id,board_database,task_id,workspace,state,
                         workflow_schema,legacy_md5 FROM michael_checkpoint_scopes
                         WHERE board_database=%s AND task_id=%s''',
                       (identity['board_database'],identity['task_id'])).fetchone()
    if row is not None:
        require(all(row[key]==value for key,value in identity.items()),
                'Registered workspace moved; explicit migration required')
    return row


def verify_association(row, marker, schema_version):
    require(row is not None, 'Marker exists without database registration; explicit recovery required')
    require(row['scope_id']==marker['scope_id'] and all(row[k]==marker[k]
            for k in ('board_database','task_id','workspace')), 'Marker/database association differs')
    require(row['state']=='active', 'Checkpoint scope is not active; explicit recovery required')
    require(row['workflow_schema']==schema_version, 'Registered workflow schema differs')


def register_new(conn, identity, marker, schema_version):
    require(type(schema_version) is int and schema_version>0, 'Explicit workflow schema required')
    require(re.fullmatch('[0-9a-f]{32}',marker['scope_id']) is not None, 'Invalid new scope id')
    require(all(marker[k]==v for k,v in identity.items()), 'New marker identity differs')
    require(lookup(conn,identity) is None, 'Registration already exists; fresh scope refused')
    conn.execute('''INSERT INTO michael_checkpoint_scopes
                    (scope_id,board_database,task_id,workspace,state,workflow_schema,legacy_md5)
                    VALUES (%s,%s,%s,%s,'active',%s,NULL)''',
                 (marker['scope_id'],identity['board_database'],identity['task_id'],
                  identity['workspace'],schema_version))
    row=lookup(conn,identity)
    verify_association(row,marker,schema_version)
    return row
