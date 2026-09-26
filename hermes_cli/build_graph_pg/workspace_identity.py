"""Board/task/workspace scope identity with protected marker handling."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


class IdentityRefused(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise IdentityRefused(message)


def identity_from_task(conn, task_id, workspace):
    require(isinstance(task_id, str) and 0 < len(task_id.encode()) <= 1024,
            'Bounded task identity required')
    cursor = conn.execute('PRAGMA database_list')
    require(tuple(c[0] for c in cursor.description) == ('seq', 'name', 'file'),
            'SQLite identity contract changed')
    rows = cursor.fetchall()
    main = [row[2] for row in rows if row[1] == 'main']
    require(len(main) == 1 and main[0] and os.path.isabs(main[0]),
            'Durable main board database required')
    board = os.path.realpath(main[0])
    info = os.stat(board, follow_symlinks=False)
    require(stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid(),
            'Board identity ownership differs')
    row = conn.execute('SELECT workspace_kind, workspace_path FROM tasks WHERE id = ?',
                       (task_id,)).fetchone()
    require(row is not None and row[0] in ('scratch', 'worktree', 'dir') and row[1],
            'Task has no supported workspace assignment')
    actual = os.path.realpath(os.fspath(workspace))
    require(actual == os.path.realpath(row[1]), 'Caller workspace differs from task assignment')
    info = os.stat(actual, follow_symlinks=False)
    require(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid(),
            'Workspace identity ownership differs')
    return {'board_database': board, 'task_id': task_id, 'workspace': actual}


def marker_name(identity):
    key = (identity['board_database'] + '\0' + identity['task_id']).encode()
    return 'postgres-' + hashlib.sha256(key).hexdigest() + '.json'


def _read(fd, name):
    source = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
    with os.fdopen(source, 'rb') as stream:
        info = os.fstat(stream.fileno())
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1
                and info.st_uid == os.geteuid() and stat.S_IMODE(info.st_mode) == 0o600
                and info.st_size <= 16384, 'Unsupported scope marker type, owner, mode or size')
        raw = stream.read(16385)
        after = os.fstat(stream.fileno())
        entry = os.stat(name, dir_fd=fd, follow_symlinks=False)
        signature = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_uid,
                               s.st_nlink, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        require(len(raw) == info.st_size and signature(info) == signature(after)
                == signature(entry), 'Marker changed during read')
    return raw


def validate_marker(raw, identity):
    try:
        marker = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise IdentityRefused('Invalid scope marker JSON') from exc
    require(isinstance(marker, dict) and set(marker) == {
        'version', 'scope_id', 'board_database', 'task_id', 'workspace'},
        'Unexpected scope marker fields')
    require(marker['version'] == 1 and type(marker['version']) is int,
            'Unsupported scope marker version')
    token = marker['scope_id']
    require(isinstance(token, str) and len(token) == 32
            and all(c in '0123456789abcdef' for c in token), 'Invalid scope token')
    require(all(marker[k] == v for k, v in identity.items()),
            'Scope marker belongs to another board, task or workspace')
    return marker


@contextmanager
def marker_lock(identity, *, mutation_gate, lock_timeout=5.0):
    """Pin directory descriptors. The caller supplies its authorized write gate.

    Shared-workspace child tasks use distinct board/task marker names. Existing
    directory modes are preserved; new checkpoint directories are private.
    """
    require(type(lock_timeout) in (int, float) and 0 <= lock_timeout <= 30,
            'Scope lock timeout must be between zero and thirty seconds')
    workspace = os.open(identity['workspace'], os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    checkpoint = lock = None
    try:
        info = os.fstat(workspace)
        require(info.st_uid == os.geteuid(), 'Pinned workspace owner differs')
        try:
            checkpoint = os.open('.graph-checkpoints', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                 dir_fd=workspace)
        except FileNotFoundError:
            mutation_gate()
            try:
                os.mkdir('.graph-checkpoints', mode=0o700, dir_fd=workspace)
            except FileExistsError:
                pass
            checkpoint = os.open('.graph-checkpoints', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                 dir_fd=workspace)
        info = os.fstat(checkpoint)
        require(info.st_uid == os.geteuid(), 'Checkpoint directory owner differs')
        mutation_gate()
        lock = os.open('.postgres-scope.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
                       0o600, dir_fd=checkpoint)
        info = os.fstat(lock)
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_uid == os.geteuid()
                and stat.S_IMODE(info.st_mode) == 0o600, 'Unsupported scope lock')
        deadline = time.monotonic() + lock_timeout
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                if time.monotonic() >= deadline:
                    raise IdentityRefused('Scope identity lock is busy; retry later') from exc
                time.sleep(min(0.05, max(0, deadline - time.monotonic())))
        yield checkpoint
    finally:
        if lock is not None:
            os.close(lock)
        if checkpoint is not None:
            os.close(checkpoint)
        os.close(workspace)


def read_marker(fd, identity):
    try:
        raw = _read(fd, marker_name(identity))
    except FileNotFoundError:
        return None
    return validate_marker(raw, identity)


def create_marker(fd, identity, *, mutation_gate, existing_database_scope=False):
    require(not existing_database_scope,
            'Database scope exists without marker; explicit recovery required')
    try:
        os.stat('checkpoints.json', dir_fd=fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        raise IdentityRefused('Legacy checkpoint migration required before Postgres activation')
    marker = {'version': 1, 'scope_id': uuid.uuid4().hex, **identity}
    raw = (json.dumps(marker, sort_keys=True, separators=(',', ':')) + '\n').encode()
    require(len(raw) <= 16384, 'Scope marker exceeds bound')
    name = marker_name(identity)
    temporary = '.postgres-marker-' + uuid.uuid4().hex
    mutation_gate()
    target = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=fd)
    try:
        with os.fdopen(target, 'wb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        mutation_gate()
        # link is an atomic no-replacement publication; an unexpected existing
        # marker cannot be overwritten even if another writer ignores our lock.
        os.link(temporary, name, src_dir_fd=fd, dst_dir_fd=fd, follow_symlinks=False)
        mutation_gate()
        os.unlink(temporary, dir_fd=fd)
        temporary = None
        os.fsync(fd)
        require(_read(fd, name) == raw, 'Published marker differs')
        return marker
    finally:
        if temporary is not None:
            mutation_gate()
            os.unlink(temporary, dir_fd=fd)
