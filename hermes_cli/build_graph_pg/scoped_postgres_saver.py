"""Workspace-scoped Postgres saver with explicit sanitation and bounded retention."""
from __future__ import annotations

import copy
import re
import threading
from contextlib import contextmanager

from langgraph.checkpoint.base import CheckpointTuple
from langgraph.checkpoint.postgres import PostgresSaver
from hermes_cli import build_graph_sanitize as sz
from hermes_cli.build_graph_checkpoint import CheckpointTooLarge, _redaction_fields


class ScopeRefused(ValueError):
    pass


class ScopedPostgresSaver(PostgresSaver):
    """One pre-registered workspace scope; caller retains connection ownership.

    The runtime factory supplies registration checks; administrator migration
    and coordinated cleanup own their separate lifecycle boundaries.
    """

    def __init__(self, connection, scope_id, *, schema_version,
                 max_checkpoints=64, max_bytes=8 * 1024 * 1024, scope_guard=None):
        if not isinstance(scope_id, str) or not re.fullmatch(r'[0-9a-f]{32}', scope_id):
            raise ScopeRefused('Invalid registered workspace scope')
        if type(schema_version) is not int or schema_version < 1:
            raise ScopeRefused('Explicit workflow schema version required')
        if type(max_checkpoints) is not int or max_checkpoints < 1:
            raise ScopeRefused('Positive checkpoint retention required')
        if type(max_bytes) is not int or max_bytes < 1:
            raise ScopeRefused('Positive checkpoint byte bound required')
        self.scope_id = scope_id
        self.prefix = scope_id + ':'
        key = int(scope_id[:16], 16)
        self._advisory_key = key if key < 2 ** 63 else key - 2 ** 64
        self.schema_version = schema_version
        self.max_checkpoints = max_checkpoints
        self.max_bytes = max_bytes
        self._scope_lock = threading.RLock()
        self._scope_guard = scope_guard
        super().__init__(connection, serde=sz.make_serde())
        self.hydrated_schema_version = None
        self.hydrated_redacted_fields = set()
        with self._transaction():
            self._hydrate_resume_contract()

    def _hydrate_resume_contract(self):
        # Capture persisted attribution before this process sanitizes anything.
        # Match the old envelope's scope-wide refusal, including other component
        # threads; never infer safety from a fresh serializer's empty history.
        rows = self.conn.execute(
            'SELECT metadata FROM checkpoints WHERE left(thread_id,%s)=%s',
            (len(self.prefix), self.prefix)).fetchall()
        versions = set()
        fields = set()
        for row in rows:
            marker = row['metadata'].get('_michael')
            if not isinstance(marker, dict) or type(marker.get('schema_version')) is not int:
                raise ScopeRefused('Missing persisted workflow schema identity')
            versions.add(marker['schema_version'])
            values = marker.get('redacted_fields')
            if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
                raise ScopeRefused('Invalid persisted redaction attribution')
            fields.update(values)
        if len(versions) > 1:
            raise ScopeRefused('Mixed persisted workflow schema identities')
        self.hydrated_schema_version = next(iter(versions), None)
        self.hydrated_redacted_fields = fields

    def has_thread(self, thread_id):
        mapped = self._internal({'configurable': {'thread_id': thread_id}})
        with self._transaction():
            return self.conn.execute(
                'SELECT 1 FROM checkpoints WHERE thread_id=%s LIMIT 1',
                (mapped['configurable']['thread_id'],)).fetchone() is not None

    def setup(self):
        raise ScopeRefused('Schema setup requires separate administrator delivery')

    def _internal(self, config):
        if not isinstance(config, dict) or not isinstance(config.get('configurable'), dict):
            raise ScopeRefused('Explicit checkpoint configuration required')
        source = config['configurable']
        thread = source.get('thread_id')
        if not isinstance(thread, str) or not thread or len(thread.encode()) > 1024:
            raise ScopeRefused('Bounded external thread identity required')
        namespace = source.get('checkpoint_ns', '')
        if not isinstance(namespace, str) or len(namespace.encode()) > 4096:
            raise ScopeRefused('Bounded checkpoint namespace required')
        mapped = {'thread_id': self.prefix + thread, 'checkpoint_ns': namespace}
        if 'checkpoint_id' in source:
            value = source['checkpoint_id']
            if not isinstance(value, str) or not value or len(value.encode()) > 1024:
                raise ScopeRefused('Invalid checkpoint identity')
            mapped['checkpoint_id'] = value
        # Upstream merges configurable values into JSON metadata. Deliberately
        # pass only its inspected routing fields, never arbitrary caller values.
        return {'configurable': mapped}

    def _external(self, config):
        if config is None:
            return None
        out = copy.deepcopy(config)
        thread = out['configurable']['thread_id']
        if not isinstance(thread, str) or not thread.startswith(self.prefix):
            raise ScopeRefused('Database returned a foreign workspace scope')
        out['configurable']['thread_id'] = thread[len(self.prefix):]
        return out

    def _tuple(self, row):
        if row is None:
            return None
        return CheckpointTuple(self._external(row.config), row.checkpoint,
                               row.metadata, self._external(row.parent_config),
                               row.pending_writes)

    def get_tuple(self, config):
        with self._transaction():
            return self._tuple(super().get_tuple(self._internal(config)))

    def list(self, config, *, filter=None, before=None, limit=None):
        if filter is not None:
            raise ScopeRefused('Metadata-filter history surface is not enabled')
        mapped = self._internal(config)
        bound = self.max_checkpoints if limit is None else limit
        if type(bound) is not int or not 0 <= bound <= self.max_checkpoints:
            raise ScopeRefused('History limit exceeds retention')
        previous = None if before is None else self._internal(before)
        if previous and previous['configurable']['thread_id'] != mapped['configurable']['thread_id']:
            raise ScopeRefused('History cursor belongs to another thread')
        with self._transaction():
            rows = tuple(self._tuple(row) for row in super().list(
                mapped, before=previous, limit=bound))
        yield from rows

    @contextmanager
    def _transaction(self):
        # Candidate uses a single autocommit connection, not a pool. An outer
        # transaction makes the upstream writes and pruning/refusal atomic.
        with self._scope_lock, self.conn.transaction():
            self.conn.execute('SELECT pg_advisory_xact_lock(%s)', (self._advisory_key,))
            if self._scope_guard is not None:
                self._scope_guard(self.conn)
            yield

    def _redactions(self, thread):
        rows = self.conn.execute(
            'SELECT metadata FROM checkpoints WHERE thread_id = %s',
            (thread,)).fetchall()
        fields = set()
        for row in rows:
            marker = row['metadata'].get('_michael')
            if not isinstance(marker, dict) or marker.get('schema_version') != self.schema_version:
                raise ScopeRefused('Missing or incompatible persisted schema marker')
            values = marker.get('redacted_fields')
            if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
                raise ScopeRefused('Invalid persisted redaction attribution')
            fields.update(values)
        return fields

    def put(self, config, checkpoint, metadata, new_versions):
        mapped = self._internal(config)
        thread = mapped['configurable']['thread_id']
        if '_michael' in metadata:
            raise ScopeRefused('Caller supplied reserved checkpoint metadata')
        with self._transaction():
            fields = self._redactions(thread) | _redaction_fields(checkpoint.get('channel_values'))
            cp = copy.deepcopy(checkpoint)
            cp['channel_values'] = sz.sanitize_obj(cp['channel_values'])
            meta = sz.sanitize_obj(copy.deepcopy(metadata))
            meta['_michael'] = {'schema_version': self.schema_version,
                               'redacted_fields': sorted(fields)}
            result = super().put(mapped, cp, meta, new_versions)
            self._prune_and_bound(thread)
            return self._external(result)

    def put_writes(self, config, writes, task_id, task_path=''):
        mapped = self._internal(config)
        with self._transaction():
            super().put_writes(mapped, writes, task_id, task_path)
            self._prune_and_bound(mapped['configurable']['thread_id'])

    def _prune_and_bound(self, thread):
        self.conn.execute('''DELETE FROM checkpoints c USING (
            SELECT thread_id, checkpoint_ns, checkpoint_id,
              row_number() OVER (PARTITION BY checkpoint_ns ORDER BY checkpoint_id DESC) AS rank
            FROM checkpoints WHERE thread_id = %s
        ) old WHERE c.thread_id=old.thread_id AND c.checkpoint_ns=old.checkpoint_ns
            AND c.checkpoint_id=old.checkpoint_id AND old.rank > %s''',
            (thread, self.max_checkpoints))
        self.conn.execute('''DELETE FROM checkpoint_writes w WHERE w.thread_id=%s
            AND NOT EXISTS (SELECT 1 FROM checkpoints c WHERE c.thread_id=w.thread_id
              AND c.checkpoint_ns=w.checkpoint_ns AND c.checkpoint_id=w.checkpoint_id)''', (thread,))
        self.conn.execute('''DELETE FROM checkpoint_blobs b WHERE b.thread_id=%s
            AND NOT EXISTS (SELECT 1 FROM checkpoints c,
              jsonb_each_text(c.checkpoint->'channel_versions') version
              WHERE c.thread_id=b.thread_id AND c.checkpoint_ns=b.checkpoint_ns
                AND version.key=b.channel AND version.value=b.version)''', (thread,))
        # Count logical JSON bytes; TOAST compression must not conceal large
        # decoded values. Index/WAL/physical cluster size is a different bound.
        size = self.conn.execute('''SELECT
            COALESCE((SELECT sum(octet_length(checkpoint::text)+octet_length(metadata::text))
              FROM checkpoints WHERE left(thread_id,%s)=%s),0) +
            COALESCE((SELECT sum(octet_length(blob)) FROM checkpoint_blobs
              WHERE left(thread_id,%s)=%s),0) +
            COALESCE((SELECT sum(octet_length(blob)) FROM checkpoint_writes
              WHERE left(thread_id,%s)=%s),0) AS bytes''',
            (len(self.prefix), self.prefix) * 3).fetchone()['bytes']
        if size > self.max_bytes:
            raise CheckpointTooLarge('Postgres checkpoint payload exceeds workspace bound')

    def get_delta_channel_history(self, *, config, channels):
        with self._transaction():
            mapped = self._internal(config)
            if 'checkpoint_id' not in mapped['configurable']:
                latest = self.get_tuple(config)
                if latest is None:
                    return {}
                mapped['configurable']['checkpoint_id'] = latest.checkpoint['id']
            return super().get_delta_channel_history(config=mapped, channels=channels)

    def delete_thread(self, thread_id):
        mapped = self._internal({'configurable': {'thread_id': thread_id}})
        with self._transaction():
            super().delete_thread(mapped['configurable']['thread_id'])
