"""Narrow trusted-parent Git exception for the sole A2 run (2026-09-14).

This is a dispatcher/worker guard, not socket authentication. The root executor
still observes real credentials independently. No fake Peer is constructed.
Inventory is checked before and after each operation; this is not an atomic
snapshot across databases and does not close the maintenance race gate.
"""
from contextlib import closing
import importlib.util
import os
from pathlib import Path
import re
import sqlite3
import stat
import subprocess
import sys
import time

ROOT = Path('/home/jetson/.hermes')
MAINTENANCE = Path('/usr/local/lib/michael-maintenance/admission.py')

class Refused(RuntimeError):
    """Never convert a lost binding into permission to park/end a run."""


def gate():
    name = '_michael_parent_admission'
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, MAINTENANCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[name] = module
    if Path(module.__file__) != MAINTENANCE:
        raise Refused('parent admission module collision')
    return module.Gate()


def identity(path, directory=False):
    path = Path(path)
    for parent in (path if directory else path.parent).parents:
        if not stat.S_ISDIR(parent.lstat().st_mode):
            raise Refused('symlink or non-directory ancestor')
    info = path.lstat()
    if ((directory and not stat.S_ISDIR(info.st_mode)) or
            (not directory and (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1))):
        raise Refused('unsafe parent Git path')
    return info.st_dev, info.st_ino


def process():
    if os.getuid() != 1000 or os.geteuid() != 1000 or os.getegid() != 1000:
        raise Refused('parent owner identity')
    status = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
    if (status['Uid'].split() != ['1000'] * 4 or status['Gid'].split() != ['1000'] * 4
            or status['Groups'].split() or status['NoNewPrivs'].strip() != '1'
            or any(int(status[k], 16) for k in ('CapEff','CapPrm','CapInh','CapBnd','CapAmb'))):
        raise Refused('parent confinement changed')
    tick = int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
    return os.getpid(), tick, Path('/proc/self/ns/pid').stat().st_ino


def inventory():
    boards = ROOT / 'kanban/boards'
    identity(boards, True)
    names = sorted(boards.iterdir())
    if not 1 <= len(names) <= 64:
        raise Refused('board inventory bounds')
    paths = [ROOT / 'kanban.db']
    dirs = {}
    for board in names:
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', board.name):
            raise Refused('board name')
        dirs[str(board)] = identity(board, True)
        paths.append(board / 'kanban.db')
    return dirs, {str(p): identity(p) for p in paths}


def filter_overrides(paths):
    """Suppress all configured drivers without executing Git to discover them.

    Global/system configuration is disabled in the child environment. Includes
    are refused: unbounded external configuration is outside this worktree
    contract. Ordinary simple named drivers (including LFS) are disabled only
    for these parent commands. No repository configuration is changed.
    """
    names = set()
    for path in paths:
        if not path.exists():
            continue
        identity(path)
        raw = path.read_bytes()
        if len(raw) > 1024 * 1024:
            raise Refused('Git config bound')
        for line in raw.decode('utf-8').splitlines():
            text = line.strip()
            if not text.startswith('['):
                continue
            match = re.fullmatch(r'\[([A-Za-z0-9.-]+)(?:[ \t]+"([A-Za-z0-9_./-]+)")?\][ \t]*(?:[#;].*)?', text)
            if match is None:
                raise Refused('unsupported Git config section syntax')
            section, name = match.groups()
            section = section.lower()
            if section == 'include' or section == 'includeif' or section.startswith('includeif.'):
                raise Refused('external Git config includes require explicit support')
            if section.startswith('filter.'):
                name = section[7:];section = 'filter'
            if section == 'filter':
                if not name:
                    raise Refused('unnamed Git filter')
                names.add(name)
    result = []
    for name in sorted(names):
        for key, value in (('clean',''),('smudge',''),('process',''),('required','false')):
            result.extend(('-c', 'filter.' + name + '.' + key + '=' + value))
    return result


class ParentGit:
    def __init__(self, task, workspace, board, *, dispatcher=False):
        if (not isinstance(board, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', board)
                or not re.fullmatch(r't_[A-Za-z0-9_-]{1,96}', task.id)
                or type(task.current_run_id) is not int or task.current_run_id <= 0):
            raise Refused('parent run selectors')
        self.task_id, self.run_id, self.board = task.id, task.current_run_id, board
        self.workspace = ROOT / 'kanban/boards' / board / 'workspaces' / task.id
        if workspace != str(self.workspace):
            raise Refused('pre-created canonical task workspace required')
        self.branch = task.branch_name
        if not isinstance(self.branch, str) or not self.branch or self.branch.startswith('-'):
            raise Refused('explicit pre-created branch required')
        self.dispatcher = dispatcher
        self.claim = task.claim_lock
        self.phase = 'baseline'
        self.gate = gate()
        self.binding = None
        self.tree = None
        self.refs = set()
        self.bases = set()
        with self.gate.shared(), self.gate.shared('runtime.lock'):
            self.binding = self.snapshot()
            self.tree = self.tree_identity()
            self.verify()

    def snapshot(self):
        current = process()
        before = inventory()
        target = str(ROOT / 'kanban/boards' / self.board / 'kanban.db')
        active, task = [], None
        for path in before[1]:
            with closing(sqlite3.connect(Path(path).as_uri() + '?mode=ro', uri=True, timeout=2)) as db:
                db.row_factory = sqlite3.Row
                db.execute('PRAGMA trusted_schema=OFF')
                db.execute('PRAGMA query_only=ON')
                active.extend((path, dict(row)) for row in db.execute(
                    'SELECT id,task_id,status,claim_lock,claim_expires,worker_pid,started_at,ended_at,profile FROM task_runs WHERE ended_at IS NULL LIMIT 2'))
                if path == target:
                    row = db.execute('SELECT id,status,current_run_id,claim_lock,claim_expires,worker_pid,workspace_path,workspace_kind,assignee,branch_name,graph_executor FROM tasks WHERE id=?', (self.task_id,)).fetchone()
                    task = dict(row) if row else None
        if before != inventory() or current != process():
            raise Refused('parent inventory or process changed')
        if len(active) != 1 or active[0][0] != target or task is None:
            raise Refused('exact sole active run required')
        run = active[0][1]
        pid = None if self.dispatcher else current[0]
        if (run['id'] != self.run_id or task['current_run_id'] != self.run_id
                or run['task_id'] != self.task_id or task['status'] != 'running' or run['status'] != 'running'
                or not self.claim or run['claim_lock'] != self.claim or task['claim_lock'] != self.claim
                or run['worker_pid'] != pid or task['worker_pid'] != pid
                or not run['profile'] or task['assignee'] != run['profile']
                or type(run['started_at']) is not int or run['started_at'] <= 0
                or task['workspace_kind'] != 'worktree' or task['workspace_path'] != str(self.workspace)
                or task['branch_name'] != self.branch or task['graph_executor'] != 'build_graph'
                or any(type(x) is not int or x <= time.time() for x in (run['claim_expires'], task['claim_expires']))):
            raise Refused('parent assignment or claim mismatch')
        # Claim renewals may advance expiry, but cannot change identity.
        for row in (run, task):
            row.pop('claim_expires')
        return current, before, task, run, identity(self.workspace, True)

    def tree_identity(self):
        link = self.workspace / '.git'
        identity(link)
        raw = link.read_text()
        if not raw.startswith('gitdir: ') or raw.count('\n') != 1:
            raise Refused('linked task worktree required')
        gitdir = Path(raw[8:].strip())
        if not gitdir.is_absolute():
            raise Refused('absolute worktree gitdir required')
        identity(gitdir, True)
        for leaf in ('commondir', 'gitdir', 'HEAD'):
            identity(gitdir / leaf)
        common = (gitdir / 'commondir').read_text().strip()
        common = (gitdir / common).resolve(strict=True)
        identity(common, True)
        if common in (ROOT / '.git', ROOT / 'hermes-agent/.git'):
            raise Refused('config and agent repository Git forbidden')
        if gitdir.parent != common / 'worktrees':
            raise Refused('worktree metadata outside common repository')
        if (gitdir / 'gitdir').read_text().strip() != str(link):
            raise Refused('worktree reciprocal pointer mismatch')
        head = (gitdir / 'HEAD').read_text()
        if head != 'ref: refs/heads/' + self.branch + '\n':
            raise Refused('task branch mismatch')
        configs = (common / 'config', gitdir / 'config.worktree')
        overrides = filter_overrides(configs)
        config_bytes = tuple(p.read_bytes() if p.exists() else None for p in configs)
        return identity(link), raw, identity(gitdir, True), identity(common, True), head, config_bytes, overrides

    def verify(self):
        if self.phase not in ('baseline', 'committed'):
            raise Refused('pending or uncertain payload: no parent Git')
        if self.snapshot() != self.binding or self.tree_identity() != self.tree:
            raise Refused('parent run/worktree binding changed')

    def launch(self):
        with self.gate.shared(), self.gate.shared('runtime.lock'):
            self.verify()
            if self.phase != 'baseline' or self.dispatcher:
                raise Refused('parent launch phase')
            self.phase = 'pending'

    def committed(self):
        if self.phase != 'pending':
            raise Refused('parent receipt phase')
        # Called only after launch_client validated the root committed reply.
        self.phase = 'committed'

    def __call__(self, argv, timeout):
        try:
            return self.command(argv, timeout)
        except Refused:
            raise
        except Exception as exc:
            # build_graph_diff intentionally catches OSError/SubprocessError.
            # Losing a binding must escape that ordinary no-diff park path.
            raise Refused('parent Git operation or binding uncertain') from exc

    def command(self, argv, timeout):
        if argv[:3] != ['git', '-C', str(self.workspace)] or not 0 < timeout <= 60:
            raise Refused('parent Git command scope')
        args = tuple(argv[3:])
        fixed = {('rev-parse','--is-inside-work-tree'), ('rev-parse','--show-toplevel'),
                 ('branch','--show-current'), ('worktree','list','--porcelain'),
                 ('rev-parse','--abbrev-ref','origin/HEAD')}
        allowed = args in fixed
        if not self.dispatcher:
            from hermes_cli.build_graph_diff import _DIFF_EXCLUDES
            allowed |= args == ('add','-A','-N')
            allowed |= len(args) == 4 and args[:3] == ('rev-parse','--verify','--quiet') and args[3] in {r+'^{commit}' for r in self.refs}
            allowed |= len(args) == 3 and args[0] == 'merge-base' and args[1] in self.refs and args[2] == 'HEAD'
            allowed |= any(args == ('diff', *prefix, base, '--', *_DIFF_EXCLUDES)
                           for prefix in ((), ('--numstat',), ('--name-only','-z')) for base in self.bases)
        if not allowed:
            raise Refused('parent Git argv not authorized')
        with self.gate.shared(), self.gate.shared('runtime.lock'):
            self.verify()
            env = {'HOME':'/home/jetson','PATH':'/usr/bin:/bin','LANG':'C.UTF-8',
                   'GIT_CONFIG_NOSYSTEM':'1','GIT_CONFIG_GLOBAL':'/dev/null', 'GIT_TERMINAL_PROMPT':'0'}
            # Override worktree/index redirection from repository config. These
            # paths come from the verified reciprocal linked-worktree metadata.
            gitdir = Path(self.tree[1][8:].strip())
            env.update(GIT_WORK_TREE=str(self.workspace), GIT_DIR=str(gitdir),
                       GIT_COMMON_DIR=str(gitdir.parent.parent),
                       GIT_INDEX_FILE=str(gitdir / 'index'))
            command = ['/usr/bin/git',*self.tree[-1],'-c','core.hooksPath=/dev/null','-c','core.fsmonitor=false',
                       '-c','core.untrackedCache=false','-C',str(self.workspace),*args]
            if args[0] == 'diff':
                command[1:1] = ['--no-pager']
                command.insert(command.index('diff')+1, '--no-ext-diff')
                command.insert(command.index('diff')+1, '--no-textconv')
            result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8',
                                    errors='replace', timeout=timeout, env=env)
            self.verify()
        if result.returncode == 0:
            out = result.stdout.strip()
            if args == ('rev-parse','--abbrev-ref','origin/HEAD') and re.fullmatch(r'[A-Za-z0-9_./-]+',out) and not out.startswith('-'):
                self.refs.add(out)
            if args == ('worktree','list','--porcelain'):
                for line in result.stdout.split('\n\n',1)[0].splitlines():
                    if line.startswith('branch refs/heads/'):
                        self.refs.add(line[7:])
            if args[0] == 'merge-base':
                if not re.fullmatch('[0-9a-f]{40}|[0-9a-f]{64}', out):
                    raise Refused('parent merge-base result')
                self.bases.add(out)
        return result

    def derive(self, workspace):
        if workspace != str(self.workspace):
            raise Refused('parent derivation workspace changed')
        from hermes_cli.build_graph_diff import derive
        return derive(workspace, runner=self)


def precreated(task, *, board):
    context = ParentGit(task, task.workspace_path, board, dispatcher=True)
    for args, expected in ((['rev-parse','--show-toplevel'], task.workspace_path),
                           (['branch','--show-current'], task.branch_name)):
        result = context(['git','-C',task.workspace_path,*args],60)
        if result.returncode or result.stdout.strip() != expected:
            raise Refused('pre-created task worktree mismatch')
    return Path(task.workspace_path), task.branch_name
