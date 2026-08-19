"""D5.2a build-harness graph -- the card-workspace checkpointer (CD-032).

Part 1 of D5.2. D5.2b (the four `delegate_task` nodes and the
GATE_PASS_TARGET revert) is BLOCKED and deliberately not here: `delegate_task`
requires a `parent_agent` handle, fork ruling 1 puts this graph worker-side
holding a board connection and a workspace, and where a worker-side node
obtains a parent_agent is an open design question, not a signature lookup.

WHAT THIS ADDS
--------------
CD-031 compiles the graph against `InMemorySaver`, which fork ruling 4 accepted
for D5.1 while recording that D5.1 does NOT claim the cross-run-resume
acceptance criterion. This module supplies the durable half: the same saver
semantics, persisted under the card workspace, plus an explicit resume path.

Postgres is D5.3 and stays D5.3. This is not a step toward it that must be
undone -- the seam is the same `checkpointer=` argument either way.

WHY A SUBCLASS OF InMemorySaver AND NOT A FRESH BaseCheckpointSaver
-------------------------------------------------------------------
BaseCheckpointSaver's sync surface is five methods -- put, put_writes,
get_tuple, list, get_next_version -- and reimplementing them means
reimplementing channel versioning, blob storage and pending-write bookkeeping.
That is roughly 350 lines of subtle logic that InMemorySaver already gets
right, and a subtle bug there fails as "resume quietly lost half the state".

So this subclasses and adds persistence at put/put_writes. The cost is a
coupling to three PRIVATE attributes -- .storage, .writes, .blobs. That is the
same trade build_graph.verdict_passed already makes against
`core._verdict_passed`, made loud the same way: _assert_shape() proves all
three exist with the expected container types at construction and FAILS CLOSED
if an upgrade moved them. A checkpointer that silently persists nothing is
strictly worse than one that refuses to start.

NEVER PICKLE
------------
`langgraph.checkpoint.memory` also exports `PersistentDict`, and
InMemorySaver.__init__ takes a public `factory:` parameter that enters the
three stores as context managers -- which reads exactly like a supported
persistence seam. It was rejected, and the reason is recorded here because the
next reader will find that seam in about ninety seconds and it looks like the
answer:

  * PersistentDict is PICKLE (`pickle.dump` / `pickle.load`). That puts an
    unbounded pickle load on the RESUME path, reading a file inside a directory
    WORKERS WRITE TO. A worker that drops a file in its own workspace would get
    code execution in the graph process. That is the same failure class that
    got langgraph-checkpoint-sqlite and -redis barred; importing it by the back
    door while carefully avoiding those two packages would be absurd.
  * Its `load()` is never called from `__init__`, so it would not rehydrate.
  * Its `sync()` runs only on `close()`. Write-at-close is not a checkpoint.

This module writes a JSON envelope with base64'd blobs. A spike leaf-census
over all three stores (box, 2026-08-18) found exactly {str, bytes, int,
NoneType} and asserted the stray set empty, so that is sufficient. The read
path is `serde.loads_typed`, which is the bounded one;
`JsonPlusSerializer.with_msgpack_allowlist` can bound it further at D5.3.

TWO RESUME SEMANTICS, AND THEY ARE NOT THE SAME
------------------------------------------------
Verified on the box (spike Q1/Q2, langgraph 1.2.10 / checkpoint 4.1.1):

  * On an INTERRUPTED thread (a node raised), `invoke(None, cfg)` CONTINUES
    mid-graph. Only the crashed node re-ran; the completed one did not.
  * On a TERMINAL thread (reached END), a PARTIAL input dict RE-ENTERS FROM
    START with the named channels merged over the restored ones. Unnamed
    channels survive untouched.

So "resume" here means RE-ENTER WITH STATE PRESERVED, not CONTINUE AT THE
INTERRUPTED NODE, except in the interrupted case where it means exactly that.
Stated plainly because a reader will otherwise assume node-level resume
everywhere and be wrong half the time. For CD-031's topology the re-entry
semantics are the correct ones: the graph is a loop (cheap_gate -> review ->
fix -> cheap_gate), and re-entering at cheap_gate with `rung_attempts`
preserved is precisely what makes the section 2.2 rung caps bind across runs
instead of resetting every dispatch.

THE CHECKPOINT IS LOSSY, AND RESUME IS BUILT KNOWING THAT
----------------------------------------------------------
SanitizingSerde (CD-031) redacts on the way out and does NOT restore on the way
in -- deliberately, and its docstring says so. Its own rationale for redacting
the checkpoint COPY rather than live state is "live state keeps the truth".
After a cross-run resume there is no live state left to keep it.

RULED 2026-08-19 (Ken): re-derive plus fail closed.

  * The checkpoint is authoritative for CONTROL state only -- rung,
    rung_attempts, cloud_review_calls, iteration, the objection sets, the
    recurrence fields, holdout, classify_log_failed.
  * WORK PRODUCT (`plan`, `diff`) is re-derived by the caller, which holds the
    worktree, and passed as the partial input. This is correct even setting
    redaction aside: the worktree is the truth and a checkpoint is a stale copy
    of it, so restoring a diff from a checkpoint older than the worktree is
    wrong whether or not the bytes round-trip.
  * Any redaction recorded against a field that is NOT re-derived parks the
    card at `human` rather than resuming from a lossy value.

build_graph_sanitize.py IS NOT MODIFIED BY CD-032. The first design had
SanitizingSerde record what it redacted, which would have meant editing
security-critical CD-031 code. It is not necessary and it would not have
worked: dumps_typed is called PER CHANNEL VALUE and never sees a field name.
Attribution lives in WorkspaceSaver.put, which receives Checkpoint.
channel_values -- documented as "mapping from channel name to deserialized
channel snapshot value" -- and detects a redaction by asking the UNMODIFIED
sanitizer whether it would change the value. One definition of
credential-shaped, no second copy, and the security module keeps the exact
bytes that were reviewed and byte-verified for CD-031.

WORKSPACE LIFETIME -- WHY THIS DIRECTORY IS THE RIGHT ONE
----------------------------------------------------------
Read at source (kanban_db.py, box 2026-08-18): `_cleanup_workspace` has exactly
one call site, inside `complete_task`, after the DB transaction commits. It
removes ONLY `workspace_kind == 'scratch'`; `worktree` and `dir` are
intentionally preserved; it defers while any child task is non-terminal.

So a card that blocks keeps its workspace, a card that crashes keeps its
workspace, a re-dispatched card keeps its workspace, and the only event that
destroys it is the one after which there is nothing left to resume. Checkpoints
therefore self-GC with the workspace, and the directory is a strict descendant
of the board's `workspaces/` root so `_is_managed_scratch_path` treats it as
managed. That is a property of the lifecycle, not a coincidence, and it is why
the plan's "checkpointed to the card workspace" is the right target.
"""

from __future__ import annotations

import base64
import json
import os
import tempfile
from typing import Any, Iterable, Optional

from langgraph.checkpoint.memory import InMemorySaver

from hermes_cli import build_graph_sanitize as sz

# Envelope format version. Bumped when the ON-DISK shape changes, which is a
# different axis from WorkflowState's SCHEMA_VERSION -- one describes the
# container, the other the payload. Conflating them means a container fix
# invalidates every payload.
ENVELOPE_VERSION = 1

CHECKPOINT_DIRNAME = ".graph-checkpoints"
ENVELOPE_NAME = "checkpoints.json"

# Retention. A single run is bounded by recursion_limit (40 in build_graph),
# but cross-run resume accumulates on the SAME thread across dispatches, so
# without a cap the envelope grows without bound. Dropping the OLDEST
# checkpoints only truncates history: get_tuple with no checkpoint_id returns
# the LATEST, which is what resume reads, so the newest is never at risk.
MAX_CHECKPOINTS_PER_THREAD = 64

# A refusal threshold, not a target. If the envelope ever exceeds this the
# saver stops writing and says so, rather than quietly consuming the card
# workspace. Real measurement: one full WorkflowState checkpoint is 3313 bytes
# of msgpack on the box, so 8 MiB is several thousand of them.
MAX_ENVELOPE_BYTES = 8 * 1024 * 1024


def _redaction_fields(channel_values) -> set:
    """Which channels would the sanitizer alter? Attributed BY NAME.

    Attribution has to happen HERE and NOT inside SanitizingSerde. The serde's
    dumps_typed is called PER CHANNEL VALUE and receives a bare object with no
    field name attached -- a recording sanitizer inside it can only report
    "<root>". Observed in testing 2026-08-19, and it would have left
    resume_refusal unable to tell a redaction in `diff` (legitimate, re-derived,
    harmless) from one in `objections_current` (lossy, must park). The gate
    would then have refused every card whose diff legitimately contains a
    `password = "..."` line -- which build_graph_sanitize's own docstring names
    as the common case that must NOT be treated as a secret.

    Checkpoint.channel_values is documented as "Mapping from channel name to
    deserialized channel snapshot value", so the names exist exactly here, and
    only here.

    This reads the CD-031 sanitizer without modifying it: a field is redacted
    iff sanitize_obj changes it. That keeps the security module byte-identical
    and keeps ONE definition of what counts as credential-shaped.
    """
    hit = set()
    for name, value in (channel_values or {}).items():
        try:
            if sz.sanitize_obj(value) == value:
                continue
        except Exception:
            # sz.redact never raises by design; if something upstream changes
            # that, a logging-class failure must not kill the card. The HALT
            # guard at node return is what actually stops dangerous payloads.
            continue

        if name.startswith("__") and isinstance(value, dict):
            # LangGraph pseudo-channels. `__start__` carries the INPUT DICT,
            # whose keys are the real field names. Attributing a hit to
            # "__start__" would name something that is never a re-derived
            # field, so EVERY redacted input would park the card -- including
            # the `password = "..."` config diff that build_graph_sanitize
            # explicitly calls a legitimate shape rather than a secret. Caught
            # by test 2026-08-19; the first version named the pseudo-channel
            # and refused every such card on resume.
            for inner_name, inner in value.items():
                try:
                    if sz.sanitize_obj(inner) != inner:
                        hit.add(inner_name)
                except Exception:
                    continue
            continue

        hit.add(name)
    return hit


class CheckpointShapeError(RuntimeError):
    """InMemorySaver's internals moved. Fail closed rather than persist nothing."""


class CheckpointTooLarge(RuntimeError):
    """The envelope hit MAX_ENVELOPE_BYTES. Loud, never silent growth."""


# --------------------------------------------------------------------------
# typed-blob <-> JSON
# --------------------------------------------------------------------------

def _enc_typed(pair) -> Optional[list]:
    """Encode a serde ('type', bytes) pair for JSON.

    The type string is NOT always 'msgpack': the box spike observed 'empty'
    with a zero-length body for unset channels. Round-tripping the type
    verbatim is what keeps an empty channel distinguishable from an absent one.
    """
    if pair is None:
        return None
    typ, blob = pair
    return [typ, base64.b64encode(blob).decode("ascii")]


def _dec_typed(item) -> Optional[tuple]:
    if item is None:
        return None
    typ, b64 = item
    return (typ, base64.b64decode(b64.encode("ascii")))


class WorkspaceSaver(InMemorySaver):
    """InMemorySaver whose three stores are persisted to the card workspace.

    Flushes on EVERY put/put_writes. Debouncing was considered and rejected:
    a checkpointer that batches writes is not crash-safe, and crash-safety is
    the entire deliverable. The cost is real but small -- the box spike measured
    7 put/put_writes calls for a two-node graph, and a full ladder run against
    a recursion_limit of 40 writes an envelope of a few hundred KB a few hundred
    times, against a run whose every step makes cloud calls taking seconds.
    """

    def __init__(self, root: str, *, serde=None,
                 schema_version: Optional[int] = None,
                 max_checkpoints_per_thread: int = MAX_CHECKPOINTS_PER_THREAD,
                 max_envelope_bytes: int = MAX_ENVELOPE_BYTES,
                 hydrate: bool = True):
        super().__init__(serde=serde)
        self._assert_shape()
        self.root = os.path.abspath(root)
        self.schema_version = schema_version
        self.max_checkpoints_per_thread = max_checkpoints_per_thread
        self.max_envelope_bytes = max_envelope_bytes
        self.flush_count = 0
        self.hydrated_from = None
        self.hydrated_schema_version = None
        self.hydrated_redacted_fields = set()
        self.redacted_fields = set()
        self.refused_reason = None
        if hydrate:
            self.hydrate()

    # -- the private-attribute coupling, made loud -------------------------

    def _assert_shape(self) -> None:
        """Prove InMemorySaver still stores what this module persists.

        Not decoration. If an upgrade renames or restructures these, a subclass
        that keeps overriding put() would run happily and persist an empty
        envelope -- and the failure would surface as "resume lost everything",
        far from its cause.
        """
        for attr in ("storage", "writes", "blobs"):
            if not hasattr(self, attr):
                raise CheckpointShapeError(
                    "InMemorySaver has no .%s -- its internals changed under "
                    "this module. Re-read langgraph.checkpoint.memory before "
                    "trusting WorkspaceSaver; it is currently persisting "
                    "against an interface that no longer exists." % attr)
        for attr in ("storage", "writes", "blobs"):
            store = getattr(self, attr)
            if not hasattr(store, "items"):
                raise CheckpointShapeError(
                    ".%s is a %s, which has no .items() -- expected a "
                    "dict-like store." % (attr, type(store).__name__))

    @property
    def envelope_path(self) -> str:
        return os.path.join(self.root, ENVELOPE_NAME)

    # -- write path --------------------------------------------------------

    def put(self, config, checkpoint, metadata, new_versions):
        # Attribute BEFORE serialization, while channel names still exist.
        self.redacted_fields |= _redaction_fields(checkpoint.get("channel_values"))
        out = super().put(config, checkpoint, metadata, new_versions)
        self.flush()
        return out

    def put_writes(self, config, writes, task_id, task_path=""):
        out = super().put_writes(config, writes, task_id, task_path)
        self.flush()
        return out

    def _prune(self) -> None:
        """Drop the oldest checkpoints per (thread, ns) beyond the cap.

        Blobs are keyed by (thread, ns, channel, version) and are not
        reference-counted here. Dropping old checkpoints can therefore orphan
        a few blobs, which wastes a little space and corrupts nothing. Building
        reference tracking would mean parsing checkpoint msgpack to find live
        channel versions -- a second authority on what a checkpoint references,
        for a few KB. Recorded rather than solved.
        """
        cap = self.max_checkpoints_per_thread
        if cap <= 0:
            return
        for _thread_id, by_ns in self.storage.items():
            for _ns, by_cp in by_ns.items():
                if len(by_cp) <= cap:
                    continue
                # Checkpoint ids are uuid6: lexicographic order IS time order,
                # which is why the base class's own list() sorts on them.
                for stale in sorted(by_cp)[:len(by_cp) - cap]:
                    by_cp.pop(stale, None)

    def envelope(self) -> dict:
        """The full on-disk structure, as data.

        Every store is a LIST OF RECORDS, never a nested JSON object: the box
        spike showed the top-level keys of all three stores are TUPLES, and
        `writes` inner keys are (str, int) tuples. JSON object keys can only be
        strings, so an object here would either stringify the tuples lossily or
        fail outright.
        """
        self._prune()

        storage = []
        for thread_id, by_ns in self.storage.items():
            for ns, by_cp in by_ns.items():
                for cp_id, val in by_cp.items():
                    cp, meta, parent = val
                    storage.append({
                        "thread_id": thread_id, "ns": ns, "cp_id": cp_id,
                        "checkpoint": _enc_typed(cp),
                        "metadata": _enc_typed(meta),
                        "parent_cp_id": parent,
                    })

        writes = []
        for key, inner in self.writes.items():
            items = []
            for ik, iv in inner.items():
                task_id, channel, typed, path = iv
                items.append({
                    "task_id": ik[0], "idx": ik[1],
                    "value": [task_id, channel, _enc_typed(typed), path],
                })
            writes.append({"key": list(key), "items": items})

        blobs = []
        for key, typed in self.blobs.items():
            blobs.append({"key": list(key), "value": _enc_typed(typed)})

        return {
            "envelope_version": ENVELOPE_VERSION,
            "schema_version": self.schema_version,
            # The redaction record MUST be persisted, not read off the live
            # serde at resume time. The redaction happens in the run that
            # WRITES the checkpoint; the run that RESUMES constructs a fresh
            # SanitizingSerde whose set is empty. Reading the live one would
            # make the lossy gate pass unconditionally -- a guard that is
            # present, tested against a same-process fixture, and dead in the
            # only situation it exists for.
            "redacted_fields": sorted(self._live_redacted()),
            "storage": storage,
            "writes": writes,
            "blobs": blobs,
        }

    def _live_redacted(self) -> set:
        """Redactions attributed by this process, plus any inherited.

        Union, not replacement: a resumed run's checkpoint must still declare
        the redactions its ANCESTOR run made, or a second resume would see a
        clean record and adopt state its predecessor was refused.
        """
        return set(self.redacted_fields) | set(self.hydrated_redacted_fields or ())

    def flush(self) -> None:
        """Write the envelope atomically. Never partially observable.

        tempfile + os.replace in the SAME directory, so the rename is atomic on
        the filesystem and a crash mid-write leaves the previous good envelope
        intact rather than a truncated one. A truncated checkpoint that still
        parses is the worst possible outcome here.
        """
        os.makedirs(self.root, exist_ok=True)
        body = json.dumps(self.envelope(), separators=(",", ":"),
                          sort_keys=True).encode("utf-8")
        if len(body) > self.max_envelope_bytes:
            self.refused_reason = (
                "envelope %d bytes exceeds max_envelope_bytes %d"
                % (len(body), self.max_envelope_bytes))
            raise CheckpointTooLarge(
                "%s -- refusing to write. The card workspace is not an "
                "unbounded store; raise the cap deliberately or lower "
                "max_checkpoints_per_thread." % self.refused_reason)
        fd, tmp = tempfile.mkstemp(prefix=".checkpoints-", suffix=".tmp",
                                   dir=self.root)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(body)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.envelope_path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        self.flush_count += 1

    # -- read path ---------------------------------------------------------

    def hydrate(self) -> bool:
        """Load a previous envelope into the three stores. True if one existed.

        A malformed or wrong-version envelope RAISES rather than starting
        empty. Silently beginning a fresh run when a checkpoint was supposed to
        be adopted is the failure that makes the whole feature untrustworthy:
        the card would re-spend cloud calls already paid for while reporting
        success.
        """
        path = self.envelope_path
        if not os.path.isfile(path):
            return False
        with open(path, "rb") as fh:
            raw = fh.read()
        if not raw.strip():
            return False
        try:
            env = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise CheckpointShapeError(
                "checkpoint envelope at %s is not valid JSON (%s). Refusing to "
                "start empty: that would silently discard a checkpoint the "
                "caller asked to resume from." % (path, exc)) from exc

        got = env.get("envelope_version")
        if got != ENVELOPE_VERSION:
            raise CheckpointShapeError(
                "checkpoint envelope version %r != %r at %s. No migration path "
                "is defined, and inventing one at read time would be a second "
                "authority on what a checkpoint means."
                % (got, ENVELOPE_VERSION, path))

        for rec in env.get("storage", []):
            self.storage[rec["thread_id"]][rec["ns"]][rec["cp_id"]] = (
                _dec_typed(rec["checkpoint"]),
                _dec_typed(rec["metadata"]),
                rec["parent_cp_id"],
            )
        for rec in env.get("writes", []):
            key = tuple(rec["key"])
            inner = self.writes[key]
            for item in rec["items"]:
                task_id, channel, typed, path_ = item["value"]
                inner[(item["task_id"], item["idx"])] = (
                    task_id, channel, _dec_typed(typed), path_)
        for rec in env.get("blobs", []):
            self.blobs[tuple(rec["key"])] = _dec_typed(rec["value"])

        self.hydrated_from = path
        self.hydrated_schema_version = env.get("schema_version")
        self.hydrated_redacted_fields = set(env.get("redacted_fields") or ())
        return True

    # -- introspection for the resume gates --------------------------------

    def thread_ids(self) -> list:
        return sorted(self.storage.keys())

    def has_thread(self, thread_id: str) -> bool:
        by_ns = self.storage.get(thread_id)
        if not by_ns:
            return False
        return any(by_cp for by_cp in by_ns.values())

    def checkpoint_count(self, thread_id: str) -> int:
        return sum(len(by_cp) for by_cp in self.storage.get(thread_id, {}).values())


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------

def checkpoint_root(workspace: str) -> str:
    """`<workspace>/.graph-checkpoints`.

    A strict descendant of the board's `workspaces/` root, so
    kanban_db._is_managed_scratch_path treats it as managed and it is removed
    with the workspace when complete_task fires. Checkpoints self-GC; there is
    no separate reaper to write or forget.
    """
    return os.path.join(workspace, CHECKPOINT_DIRNAME)


def _default_schema_version() -> Optional[int]:
    try:
        from hermes_cli.build_graph_state import SCHEMA_VERSION
        return SCHEMA_VERSION
    except Exception:
        return None


def make_workspace_checkpointer(workspace: str, *, schema_version=None,
                                hydrate: bool = True):
    """WorkspaceSaver with checkpoint sanitation installed. Returns (saver, serde).

    Mirrors build_graph_sanitize.make_checkpointer's contract exactly so
    build() can swap one for the other with no other change. The serde is the
    SAME SanitizingSerde wrapper -- D5.2 inherits checkpoint sanitation through
    the seam rather than reimplementing it, which is what the CD-031 docstring
    said D5.2 and D5.3 would do.
    """
    root = checkpoint_root(workspace)
    if schema_version is None:
        schema_version = _default_schema_version()
    probe = InMemorySaver()
    serde = sz.SanitizingSerde(probe.serde)
    saver = WorkspaceSaver(root, serde=serde, schema_version=schema_version,
                           hydrate=hydrate)
    return saver, serde


# --------------------------------------------------------------------------
# resume gates
# --------------------------------------------------------------------------

# Fields the caller re-derives from the worktree and passes as partial input.
# A redaction recorded against one of these is harmless: the checkpointed value
# is discarded before it is ever read.
REDERIVED_FIELDS = ("plan", "diff")


def resume_refusal(saver: WorkspaceSaver, thread_id: str, *,
                   schema_version: Optional[int],
                   redacted_fields: Optional[Iterable[str]] = None) -> Optional[str]:
    """Return a terminal_reason if resume must be refused, else None.

    Every refusal PARKS the card. None of them raise. An unbuilt node parks and
    never raises (CD-031); a refused resume is the same class of event and gets
    the same posture -- a crash under a checkpointer loses state and may
    re-spend cloud calls already paid for.

    `redacted_fields` defaults to what was HYDRATED FROM DISK, never to the
    live serde. Passing the live serde's set is the shape that makes this gate
    dead: the resuming process has a fresh SanitizingSerde that has redacted
    nothing yet, so the check would pass every time while looking correct in a
    same-process test. The parameter exists for tests to inject; production
    should leave it None.
    """
    if redacted_fields is None:
        redacted_fields = saver.hydrated_redacted_fields

    if not saver.has_thread(thread_id):
        return "resume_no_checkpoint:%s" % thread_id

    got = saver.hydrated_schema_version
    if schema_version is not None and got is not None and got != schema_version:
        # Never migrate. SCHEMA_VERSION went 1 -> 2 last session with three
        # fields added; a migration path here would be a second authority on
        # what a checkpoint means, and it would run on the resume path where a
        # wrong answer costs real money.
        return "resume_schema_mismatch:%s!=%s" % (got, schema_version)

    for field in sorted(redacted_fields or ()):
        head = field.split(":", 1)[0].split(".", 1)[0]
        if head not in REDERIVED_FIELDS:
            return "resume_lossy:%s" % field
    return None
