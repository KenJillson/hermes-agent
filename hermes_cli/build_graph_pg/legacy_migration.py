"""Explicit administrator storage migration; never called by runtime resume.

Caller owns maintenance authorization, exact source-byte/mode checks and marker
publication. This function migrates only supplied bytes in one database transaction;
it never executes a graph, mutates its source file, or sets up a database schema."""
from __future__ import annotations
import base64
import copy
import hashlib
import json
from collections import Counter
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.base import get_serializable_checkpoint_metadata
from hermes_cli import build_graph_checkpoint as ck
from hermes_cli import build_graph_sanitize as sz
from .scoped_postgres_saver import ScopedPostgresSaver


class MigrationRefused(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise MigrationRefused(message)


def typed(value):
    require(isinstance(value,list) and len(value)==2 and isinstance(value[0],str)
            and isinstance(value[1],str),'Invalid legacy typed value')
    require(value[0]!='pickle','Pickle legacy values are forbidden')
    return value[0],base64.b64decode(value[1],validate=True)


def decode_envelope(raw, *, expected_md5, schema_version):
    require(isinstance(raw,bytes) and 0<len(raw)<=ck.MAX_ENVELOPE_BYTES,'Legacy byte bound refused')
    require(hashlib.md5(raw).hexdigest()==expected_md5,'Legacy source hash differs')
    envelope=json.loads(raw)
    require(isinstance(envelope,dict) and set(envelope)=={
        'envelope_version','schema_version','redacted_fields','storage','writes','blobs'},
        'Legacy envelope fields differ')
    require(envelope['envelope_version']==ck.ENVELOPE_VERSION
            and type(envelope['envelope_version']) is int,'Legacy envelope version differs')
    require(envelope['schema_version']==schema_version
            and type(envelope['schema_version']) is int,'Workflow schema migration is not authorized')
    require(all(isinstance(envelope[k],list) for k in ('storage','writes','blobs','redacted_fields')),
            'Legacy collection shape differs')
    require(all(isinstance(v,str) for v in envelope['redacted_fields']),'Invalid legacy redaction record')
    # No path is opened: hydrate=False and no put/flush/envelope call below.
    source=ck.WorkspaceSaver('.',serde=sz.make_serde(),schema_version=schema_version,hydrate=False)
    seen=set();counts=Counter()
    for rec in envelope['storage']:
        require(isinstance(rec,dict) and set(rec)=={'thread_id','ns','cp_id','checkpoint','metadata','parent_cp_id'},'Invalid checkpoint record')
        key=(rec['thread_id'],rec['ns'],rec['cp_id'])
        require(all(isinstance(v,str) for v in key) and key[0] and key[2] and key not in seen,
                'Invalid or duplicate checkpoint identity')
        require(rec['parent_cp_id'] is None or isinstance(rec['parent_cp_id'],str),'Invalid parent identity')
        seen.add(key);counts[key[:2]]+=1
        source.storage[key[0]][key[1]][key[2]]=(typed(rec['checkpoint']),typed(rec['metadata']),rec['parent_cp_id'])
    require(seen and all(n<=ck.MAX_CHECKPOINTS_PER_THREAD for n in counts.values()),'Legacy retention differs')
    blob_keys=set()
    for rec in envelope['blobs']:
        require(isinstance(rec,dict) and set(rec)=={'key','value'} and isinstance(rec['key'],list)
                and len(rec['key'])==4,'Invalid legacy blob record')
        key=tuple(rec['key']);require(all(isinstance(v,str) for v in key) and key not in blob_keys,'Duplicate or invalid legacy blob')
        blob_keys.add(key);source.blobs[key]=typed(rec['value'])
    write_keys=set()
    for rec in envelope['writes']:
        require(isinstance(rec,dict) and set(rec)=={'key','items'} and isinstance(rec['key'],list)
                and len(rec['key'])==3 and isinstance(rec['items'],list),'Invalid pending-write record')
        key=tuple(rec['key']);require(key in seen and key not in write_keys,'Orphan or duplicate write checkpoint')
        write_keys.add(key)
        for item in rec['items']:
            require(isinstance(item,dict) and set(item)=={'task_id','idx','value'},'Invalid pending-write item')
            value=item['value'];require(isinstance(value,list) and len(value)==4,'Invalid pending-write tuple')
            task,channel,data,path=value;index=item['idx']
            require(isinstance(task,str) and task==item['task_id'] and isinstance(channel,str)
                    and isinstance(path,str) and type(index) is int,'Invalid pending-write identity')
            require((task,index) not in source.writes[key],'Duplicate pending-write index')
            source.writes[key][(task,index)]=(task,channel,typed(data),path)
    source.hydrated_schema_version=schema_version
    source.hydrated_redacted_fields=set(envelope['redacted_fields'])
    rows=[]
    for thread,ns,checkpoint_id in sorted(seen):
        row=source.get_tuple({'configurable':{'thread_id':thread,'checkpoint_ns':ns,'checkpoint_id':checkpoint_id}})
        require(row is not None and row.checkpoint.get('id')==checkpoint_id
                and row.checkpoint.get('v')==4,'Inspected version-4 checkpoint contract required')
        require('_michael' not in row.metadata,'Legacy metadata uses reserved registration field')
        rows.append(row)
    return source,rows


def same_pending(expected, actual):
    remaining=list(actual)
    for item in expected:
        for i,candidate in enumerate(remaining):
            if candidate==item:
                remaining.pop(i);break
        else:return False
    return not remaining


def migrate_bytes(conn, raw, *, expected_md5, scope_id, schema_version):
    source,rows=decode_envelope(raw,expected_md5=expected_md5,schema_version=schema_version)
    target=ScopedPostgresSaver(conn,scope_id,schema_version=schema_version)
    upstream=PostgresSaver(conn,serde=sz.make_serde())
    fields=sorted(source.hydrated_redacted_fields)
    expected_metadata={}
    with target._transaction():
        for table in ('checkpoints','checkpoint_blobs','checkpoint_writes'):
            count=conn.execute('SELECT count(*) AS n FROM '+table+' WHERE left(thread_id,%s)=%s',
                               (len(target.prefix),target.prefix)).fetchone()['n']
            require(count==0,'Migration destination is not empty; do not replay')
        for row in rows:
            cfg=copy.deepcopy(row.parent_config or {'configurable':{
                'thread_id':row.config['configurable']['thread_id'],
                'checkpoint_ns':row.config['configurable']['checkpoint_ns']}})
            mapped=target._internal(cfg)
            checkpoint=copy.deepcopy(row.checkpoint)
            require(sz.sanitize_obj(checkpoint['channel_values'])==checkpoint['channel_values'],
                    'Legacy values require new sanitation; explicit review required')
            metadata=copy.deepcopy(row.metadata)
            require(sz.sanitize_obj(metadata)==metadata,'Legacy metadata requires new sanitation')
            metadata['_michael']={'schema_version':schema_version,'redacted_fields':fields}
            expected_metadata[(row.config['configurable']['thread_id'],row.config['configurable']['checkpoint_ns'],checkpoint['id'])]=get_serializable_checkpoint_metadata(mapped,metadata)
            upstream.put(mapped,checkpoint,metadata,checkpoint['channel_versions'])
        for key,items in source.writes.items():
            for (task,index),(value_task,channel,data,path) in items.items():
                value=source.serde.loads_typed(data)
                require(sz.sanitize_obj(value)==value,'Legacy pending write requires new sanitation')
                # Preserve exact index and task_path, including negative error
                # indices and non-contiguous indices. Do not re-enumerate writes.
                conn.execute('''INSERT INTO checkpoint_writes
                    (thread_id,checkpoint_ns,checkpoint_id,task_id,task_path,idx,channel,type,blob)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)''',
                    (target.prefix+key[0],key[1],key[2],task,path,index,channel,data[0],data[1]))
        for thread in source.thread_ids():target._prune_and_bound(target.prefix+thread)
        for row in rows:
            loaded=target.get_tuple(row.config)
            key=tuple(row.config['configurable'][k] for k in ('thread_id','checkpoint_ns','checkpoint_id'))
            require(loaded is not None and loaded.checkpoint==row.checkpoint,'Checkpoint values changed during migration')
            require(loaded.parent_config==row.parent_config,'Checkpoint parent changed during migration')
            require(loaded.metadata==expected_metadata[key],'Checkpoint metadata changed during migration')
            require(same_pending(row.pending_writes,loaded.pending_writes),'Pending writes changed during migration')
        stored=conn.execute('SELECT count(*) AS n FROM checkpoints WHERE left(thread_id,%s)=%s',
                            (len(target.prefix),target.prefix)).fetchone()['n']
        require(stored==len(rows),'Checkpoint retention changed during migration')
    return {'source_md5':expected_md5,'scope_id':scope_id,'checkpoints':len(rows),
            'threads':source.thread_ids(),'writes':sum(len(items) for items in source.writes.values()),
            'schema_version':schema_version,'redacted_fields':fields,'executed_graph':False}
