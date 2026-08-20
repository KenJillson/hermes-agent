"""D5.2b build-harness graph -- diff derivation (CD-035).

The last thing standing between the graph and a priced edge.

WHY THIS MODULE EXISTS
----------------------
`build_graph.run()` takes `plan` and `diff`, and its own docstring says the
CALLER re-derives them because "the worktree is the truth and a checkpoint is
a stale copy of it". Until now no derivation existed, so CD-034 invoked the
graph ONLY on cards that provably could not reach a priced node -- those with
no parseable `## AC` block. A card WITH one was parked, because
`build_review_prompt` reads `state["diff"]` and an empty diff would have bought
a cloud review of nothing. This module is that derivation, and landing it is
what removes the park.

`plan` is NOT derived here. Producing a plan is the `plan` node's job and that
node is unbuilt (CD-031). CD-035 is diff derivation only; `plan` stays "".

THE BASE REF: MERGE-BASE AGAINST THE ANCHOR'S DEFAULT BRANCH (ruled 2026-08-20)
------------------------------------------------------------------------------
A kanban worktree card is materialized by `kanban_db._ensure_git_worktree`:

    git worktree add -b <branch> <target> HEAD

-- branched from the ANCHOR REPO'S LOCAL `HEAD`. (`cli._resolve_worktree_base`,
with its fetch-the-fresh-remote-tip cascade, is the interactive `hermes -w`
path and is NOT on the kanban dispatch path. Its own docstring records that
this clone "can lag the remote by hundreds of commits", so the branch point and
the remote default branch are routinely different commits.)

So the branch point is discoverable but not assumable, and `merge-base` finds
it regardless of how stale the anchor was. `git diff <merge-base>` -- two dots,
not three -- then covers BOTH commits made on the branch AND uncommitted
working-tree changes in one command, which is the whole work product and
exactly what a reviewer is being paid to read.

NO FETCH, DELIBERATELY. Three reasons, in increasing order of importance:
a network call on the review path inside a worker has unbounded latency; the
one place in this codebase that did fetch on a startup path is documented as
having stalled launches for 30-60s; and -- decisively -- the worktree was cut
from a LOCAL ref, so a freshly-fetched remote tip is not its fork point. A
fetched ref could move the merge-base EARLIER and inflate the diff with commits
this card never touched. Local refs are the correct ones here, not a compromise.

UNTRACKED FILES: INTENT-TO-ADD (ruled 2026-08-20)
-------------------------------------------------
`git diff` reports tracked changes only. A node that writes a NEW file leaves it
untracked and invisible -- and the `implement` node creates new files as its
normal mode, so this is the common case, not a corner. `git add -A -N` records
intent-to-add so new files appear as new-file hunks.

THIS IS THE ONE MUTATING GIT COMMAND IN THIS MODULE, and it is a deliberate
exception to the read-only posture of verification. Its blast radius is stated
rather than assumed: it writes the INDEX of the card's own worktree, which the
worker already owns and writes to; it commits nothing; it pushes nothing; it
touches no file's contents; it respects `.gitignore`; and it cannot reach the
anchor repo or any other worktree. `_assert_allowed_git()` proves at import,
by AST walk, that every git invocation in this module is on a five-entry
allowlist and that the `add` is pinned to exactly `-A -N`, so the exception
stays one command wide instead of becoming a precedent.

THE SIZE CEILING IS DERIVED, NOT CHOSEN
---------------------------------------
`state["diff"]` is a checkpoint channel, so an unbounded diff blows the
checkpointer's own envelope cap -- but only AFTER the run has spent. Refusing
up front is the cheaper failure. The number comes from constants already
anchored in build_graph_checkpoint.py rather than from judgement:

    MAX_ENVELOPE_BYTES        8388608   (8 MiB)
    MAX_CHECKPOINTS_PER_THREAD     64
    per-checkpoint budget     8388608 / 64      = 131072
    measured empty-diff state                   =   3313  (box, 2026-08-18)
    headroom                  131072 - 3313     = 127759
    base64 inflation (4/3)    127759 * 3/4      =  95819
    rounded down to a power of two              =  65536

A REFUSAL THRESHOLD, NOT A TARGET. Over it, the card parks and says the byte
count; it is never truncated. A truncated diff presented to a paid reviewer as
"the change" is the failure this whole module exists to prevent.

EVERY FAILURE PARKS. NOTHING RAISES.
------------------------------------
Same posture as an unbuilt node (CD-031) and a refused resume (CD-032): a raise
under a checkpointer loses state and may re-spend cloud calls already paid for,
while a parked card is bounded and visible to a human. `derive()` returns a
verdict dict; the caller turns `reason` into a `block_task` reason.
"""

from __future__ import annotations

import ast
import os
import subprocess
from typing import Callable, Optional

# Anchored in build_graph_checkpoint.py; see the docstring for the arithmetic.
MAX_DIFF_BYTES = 64 * 1024

# Per-git-command wall clock. A diff is local work on a small tree; a command
# that takes longer than this is wedged, not slow, and a wedged git under a
# worker would burn the card's whole runtime budget silently.
GIT_TIMEOUT = 60

# Every git invocation this module is allowed to make, as an argv PREFIX.
#
# An ALLOWLIST, not a deny-list of dangerous verbs. A deny-list has to
# enumerate every way to write, and it gets that wrong the first time someone
# adds a verb nobody thought of. This says instead: these five shapes, nothing
# else, and anything new must be added here on purpose.
#
# It is also why the check is keyed on a PREFIX rather than a bare verb. The
# first draft used bare verbs and listed `worktree` as forbidden -- and then
# fired on this module's own `git worktree list`, which is read-only. That is
# the ninth-plus occurrence of the standing lesson: a guard that fires on your
# own correct text measures the wrong thing, and the fix is to NARROW it (verb
# + subcommand) rather than to loosen it (drop the verb).
_ALLOWED_GIT = (
    ("rev-parse",),
    ("merge-base",),
    ("diff",),
    ("worktree", "list"),
    ("add",),                 # the one mutating shape; argv pinned below
)

# The mutating command, pinned exactly. `add` in the allowlist above permits
# the verb; this permits precisely one argv and nothing else, so `git add -A`
# (which STAGES CONTENT) can never be reached by dropping a flag.
_PINNED_ADD_ARGV = ("add", "-A", "-N")


class DerivationError(RuntimeError):
    """Only ever raised by the import-time self-guard, never on a card path."""


def _git_argvs(source: str) -> list:
    """Every literal argv passed to `_git` in this module, as tuples.

    An AST walk, not a substring scan. This module's own docstring names
    `git checkout`, `git commit` and `git fetch` while explaining why it must
    never run them -- prose a substring guard would trip on. An AST walk cannot
    see a docstring, which is exactly why it is the right instrument.

    A non-literal element (a variable such as `base` or `ref`) is recorded as
    None: it is an ARGUMENT, never a verb, and the guard only reads the
    leading literal tokens.
    """
    out = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = getattr(fn, "id", None) or getattr(fn, "attr", None)
        if name != "_git":
            continue
        for arg in node.args:
            if isinstance(arg, (ast.List, ast.Tuple)):
                argv = []
                for el in arg.elts:
                    if isinstance(el, ast.Constant) and isinstance(el.value, str):
                        argv.append(el.value)
                    else:
                        argv.append(None)
                out.append(tuple(argv))
                break
    return out


def _assert_allowed_git(path: Optional[str] = None) -> list:
    """Prove at import that every git call here is on the allowlist.

    Fails CLOSED. A module that silently grew a `git checkout` on the review
    path would be discovered by its effects, which in this codebase means
    discovered on a card that already spent.
    """
    path = path or os.path.abspath(__file__)
    with open(path, "r", encoding="utf-8") as fh:
        argvs = _git_argvs(fh.read())
    if not argvs:
        # A guard that inspects nothing passes everything. The first draft did
        # exactly this: call sites used a local alias, the walk matched only
        # `_git`, and the check was vacuous while reporting success.
        raise DerivationError(
            "build_graph_diff's git-allowlist guard found NO git call sites to "
            "check. Either every call site was renamed away from `_git`, or "
            "the walk is broken. Both mean this guard is inspecting nothing.")
    for argv in argvs:
        ok = False
        for prefix in _ALLOWED_GIT:
            if argv[:len(prefix)] == prefix:
                ok = True
                break
        if not ok:
            raise DerivationError(
                "build_graph_diff runs a git command that is not on its "
                "allowlist: %r. This module holds the graph's ONLY write bit "
                "and it is scoped to `git add -A -N`; widening it needs its "
                "own ruling, not an edit." % (list(argv),))
        if argv[0] == "add" and argv != _PINNED_ADD_ARGV:
            raise DerivationError(
                "build_graph_diff runs %r; the only permitted mutating "
                "command is exactly %r. `git add -A` without -N STAGES "
                "CONTENT in the card's worktree, which is a different and "
                "much larger blast radius." % (list(argv), list(_PINNED_ADD_ARGV)))
    return argvs


def _default_runner(argv: list, timeout: int):
    return subprocess.run(argv, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)


def _git(workspace: str, args: list, *, timeout: int = GIT_TIMEOUT, runner=None):
    """Run one git command in `workspace`. Never raises on a nonzero rc.

    Injection happens at the RUNNER (the subprocess layer), not by swapping
    this function out for a local alias. That is not a style choice: the
    import-time guard reads git argvs by walking calls to `_git`, and the
    first draft threaded a local alias `g` through every call site -- so the
    walk found ZERO argvs and the guard passed everything it was written to
    catch. Caught 2026-08-20 by a test asserting the guard finds a non-empty
    argv set. Keeping the name at every call site is what makes the guard real,
    and injecting one layer lower keeps argv CONSTRUCTION under test rather
    than mocked away.
    """
    argv = ["git", "-C", workspace] + list(args)
    return (runner or _default_runner)(argv, timeout)


def _out(proc) -> str:
    return (proc.stdout or "").strip()


def is_work_tree(workspace: str, *, runner=None) -> bool:
    try:
        p = _git(workspace, ["rev-parse", "--is-inside-work-tree"], runner=runner)
    except (OSError, subprocess.SubprocessError):
        return False
    return p.returncode == 0 and _out(p) == "true"


def main_worktree_branch(workspace: str, *, runner=None) -> Optional[str]:
    """The branch checked out in the ANCHOR (main) worktree, or None.

    `git worktree list --porcelain` emits the main worktree FIRST; each record
    is `worktree <path>` then `HEAD <sha>` then `branch <ref>`. Only the first
    record is read, which is the anchor this card's worktree was cut from.
    """
    try:
        p = _git(workspace, ["worktree", "list", "--porcelain"], runner=runner)
    except (OSError, subprocess.SubprocessError):
        return None
    if p.returncode != 0:
        return None
    for line in (p.stdout or "").splitlines():
        line = line.strip()
        if not line:
            break                       # end of the FIRST record
        if line.startswith("branch "):
            ref = line.split(" ", 1)[1].strip()
            return ref or None
    return None


def resolve_default_ref(workspace: str, *, runner=None):
    """(ref, how) for the anchor's default branch, or (None, reason).

    NO NETWORK. See the module docstring for why a fetch here would be both
    slow and wrong.
    """
    def _resolves(ref: str) -> bool:
        try:
            p = _git(workspace, ["rev-parse", "--verify", "--quiet", ref + "^{commit}"],
                     runner=runner)
        except (OSError, subprocess.SubprocessError):
            return False
        return p.returncode == 0

    # 1. The remote's recorded default branch, read from LOCAL refs only.
    try:
        p = _git(workspace, ["rev-parse", "--abbrev-ref", "origin/HEAD"],
                 runner=runner)
        if p.returncode == 0:
            ref = _out(p)
            if ref and ref != "origin/HEAD" and _resolves(ref):
                return ref, "origin/HEAD"
    except (OSError, subprocess.SubprocessError):
        pass

    # 2. The anchor worktree's own checked-out branch -- which is literally the
    #    commit `_ensure_git_worktree` cut this card's branch from.
    ref = main_worktree_branch(workspace, runner=runner)
    if ref and _resolves(ref):
        return ref, "main-worktree-branch"

    return None, "no_default_ref"


def _numstat(text: str):
    """(files, added_lines) from `git diff --numstat`.

    A BINARY file is reported as `-\\t-\\t<path>`: it counts as a FILE but
    contributes no added lines. Treating the dash as an integer is a crash, and
    treating the row as absent under-reports `diff_files` -- which feeds the
    `large_diff` selection signal, so it is a spend input, not a statistic.
    """
    files = 0
    added = 0
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        files += 1
        if parts[0].isdigit():
            added += int(parts[0])
    return files, added


def derive(workspace: str, *, runner=None) -> dict:
    """Derive the card's work product. Returns a verdict dict; never raises.

    Keys: ok, reason, diff, diff_files, diff_added_lines, base, base_ref,
          base_how, diff_bytes.
    """
    def _park(reason: str, **extra) -> dict:
        out = {"ok": False, "reason": reason, "diff": "", "diff_files": 0,
               "diff_added_lines": 0, "base": "", "base_ref": "",
               "base_how": "", "diff_bytes": 0}
        out.update(extra)
        return out

    if not workspace or not os.path.isdir(workspace):
        return _park("graph_no_diff_source:workspace_missing")

    if not is_work_tree(workspace, runner=runner):
        return _park("graph_no_diff_source:not_a_git_worktree")

    ref, how = resolve_default_ref(workspace, runner=runner)
    if ref is None:
        return _park("graph_no_diff_source:%s" % how)

    try:
        p = _git(workspace, ["merge-base", ref, "HEAD"], runner=runner)
    except (OSError, subprocess.SubprocessError) as exc:
        return _park("graph_no_diff_source:merge_base_error:%s" % type(exc).__name__)
    if p.returncode != 0 or not _out(p):
        return _park("graph_no_diff_source:no_merge_base:%s" % how)
    base = _out(p)

    # The one mutating command. See the module docstring for its blast radius.
    try:
        p = _git(workspace, ["add", "-A", "-N"], runner=runner)
    except (OSError, subprocess.SubprocessError) as exc:
        return _park("graph_no_diff_source:intent_to_add_error:%s" % type(exc).__name__)
    if p.returncode != 0:
        return _park("graph_no_diff_source:intent_to_add_failed")

    try:
        p = _git(workspace, ["diff", "--numstat", base], runner=runner)
    except (OSError, subprocess.SubprocessError) as exc:
        return _park("graph_no_diff_source:numstat_error:%s" % type(exc).__name__)
    if p.returncode != 0:
        return _park("graph_no_diff_source:numstat_failed")
    files, added = _numstat(p.stdout or "")

    try:
        p = _git(workspace, ["diff", base], runner=runner)
    except (OSError, subprocess.SubprocessError) as exc:
        return _park("graph_no_diff_source:diff_error:%s" % type(exc).__name__)
    if p.returncode != 0:
        return _park("graph_no_diff_source:diff_failed")
    text = p.stdout or ""

    nbytes = len(text.encode("utf-8"))
    if nbytes > MAX_DIFF_BYTES:
        # Never truncated. See the module docstring.
        return _park("graph_diff_too_large:%d>%d" % (nbytes, MAX_DIFF_BYTES),
                     diff_bytes=nbytes, base=base, base_ref=ref, base_how=how)

    if not text.strip():
        # An empty diff is the CD-034 spend hazard by another route: the card
        # has a `## AC` block, so cheap_gate can reach a priced node, and there
        # is nothing to review. Park rather than buy a review of nothing.
        return _park("graph_no_work_product:empty_diff",
                     base=base, base_ref=ref, base_how=how)

    return {"ok": True, "reason": None, "diff": text, "diff_files": files,
            "diff_added_lines": added, "base": base, "base_ref": ref,
            "base_how": how, "diff_bytes": nbytes}


_assert_allowed_git()
