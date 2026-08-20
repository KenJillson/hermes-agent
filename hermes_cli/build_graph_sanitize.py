"""D5.1 build-harness graph -- checkpoint sanitation (design v2 section 4.1).

CD-031, part 1 of 2. The security-critical half, kept in its own module so it
carries its own test group and can be reviewed without the topology.

WHAT SECTION 4.1 ASKS FOR, AND WHAT THIS DOES INSTEAD
-----------------------------------------------------
Section 4.1 says "the sanitizer is applied at checkpoint write". Taken
literally -- redacting workflow_state in place -- that corrupts the artifact the
ladder operates on: `diff` is what fix_rung* rewrites and what assemble lands,
and SECRET_ASSIGN_RE fires on `password = "..."` with a six-character value
floor, which is a shape a config diff contains LEGITIMATELY and is often the
very thing under review. In-place redaction would hand a fix node `[REDACTED]`
and let assemble land it.

RULED 2026-08-18 (Ken): sanitize the CHECKPOINT COPY only. Live state keeps the
truth; the persisted copy is redacted. This matches what section 4.1 actually
names as the threat -- "the checkpoint is an unsanitized egress surface (F4)" --
and it is the persisted copy, not the in-memory dict, that F4 is about. D5.2
(workspace) and D5.3 (Postgres) inherit the property through the same seam
rather than each having to remember.

Verified behaviourally on the box before this module was written (spike
2026-08-18, langgraph 1.2.10 / langgraph-checkpoint 4.1.1):

  * InMemorySaver.__init__(self, *, serde: SerializerProtocol | None = None, ...)
    -- the seam exists.
  * SerializerProtocol is exactly TWO methods, dumps_typed and loads_typed.
    Not the four-method dumps/loads interface it is easy to assume.
  * The default is JsonPlusSerializer emitting MSGPACK, not JSON. The checkpoint
    is bytes, so "read it back and grep" is not a text operation -- the driver
    scans raw stored bytes.
  * With this serde installed: live state kept the planted credential, the raw
    stored blobs did not, and did contain [REDACTED].

THE TWO TIERS RUN IN TWO PLACES, DELIBERATELY
----------------------------------------------
A checkpointer is not a node and has no edges, so it cannot route. Under the
2026-08-18 ruling a HALT must route to `human` and write no checkpoint. So:

  * HALT detection  -> halt_reason(), called explicitly at node return, where
                       the graph can route.
  * REDACTION       -> SanitizingSerde, at serialization.

The graph must NOT use sanitize(text, redact=False) for the halt check: that
raises on credential-shaped content too, which would send every ordinary config
diff to `human`. halt_reason() tests the three HALT predicates and nothing else.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Callable, Optional

# The canonical sanitizer lives in the OTHER repo, at
# <hermes-config>/lib/model_call/sanitizer.py -- which is NOT always $HERMES_HOME;
# see hermes_home() below. Reaching across the repo boundary at
# runtime is a real coupling and is recorded as such -- but the alternative is
# duplicating three security predicates into hermes-agent, and two copies of a
# security rule in two repos drift silently. The coupling is the lesser harm and
# is made loud rather than quiet: _load_sanitizer() asserts the file it loaded.
_DEFAULT_HERMES_HOME = "/home/jetson/.hermes"
_SANITIZER_RELPATH = os.path.join("lib", "model_call", "sanitizer.py")

_sanitizer_mod = None


def _derived_config_root() -> str:
    """The config root implied by this module's own location on disk.

    hermes-agent is checked out AT <hermes-config>/hermes-agent (CLAUDE.md 2),
    so this file sits at <config-root>/hermes-agent/hermes_cli/ and three
    dirnames up is the root. Derived rather than configured, so it stays correct
    with no env at all -- and it is only ever a CANDIDATE: the caller still
    proves the file it loaded.
    """
    here = os.path.abspath(__file__)
    return os.path.dirname(os.path.dirname(os.path.dirname(here)))


def hermes_home() -> str:
    """The hermes-config repo root. NOT necessarily $HERMES_HOME.

    A kanban WORKER is spawned with HERMES_HOME pointed at its PROFILE root
    (kanban_db._default_spawn calls resolve_profile_env, deliberately, so the
    worker reads profile-scoped config). lib/model_call/ lives in the CONFIG
    repo and not under any profile, so in a worker $HERMES_HOME/lib/model_call
    is always absent -- and _load_sanitizer() therefore failed closed on every
    graph run inside a worker. Invisible until the graph first ran in one
    (2026-08-19); every offline test had the two roots coinciding.

    Resolution is ORDERED, and the first candidate that actually CONTAINS the
    sanitizer wins:

      1. HERMES_CONFIG_ROOT   explicit override
      2. HERMES_HOME          correct for gateway/CLI, wrong for a worker
      3. _derived_config_root()
      4. _DEFAULT_HERMES_HOME

    This widens WHERE we look. It does not widen WHAT we accept: _load_sanitizer()
    still asserts the realpath of the module it imported and still checks the
    attribute shape, so a wrong file on the path fails closed exactly as before.

    When no candidate holds the file, the FIRST candidate is returned so the
    error names the most likely intended root rather than a fallback.
    """
    seen = []
    for cand in (os.environ.get("HERMES_CONFIG_ROOT"),
                 os.environ.get("HERMES_HOME"),
                 _derived_config_root(),
                 _DEFAULT_HERMES_HOME):
        if not cand or cand in seen:
            continue
        seen.append(cand)
        if os.path.isfile(os.path.join(cand, _SANITIZER_RELPATH)):
            return cand
    return seen[0] if seen else _DEFAULT_HERMES_HOME


def expected_sanitizer_path() -> str:
    return os.path.join(hermes_home(), _SANITIZER_RELPATH)


def _load_sanitizer():
    """Import lib/model_call's sanitizer and PROVE it is the right file.

    Route: sys.path.APPEND, never insert(0). Insert-at-zero is the shape that
    caused the 2026-08-17 defect where a driver shadowed installed modules with
    its own directory's copies; appending cannot shadow site-packages.

    The __file__ assertion is the point. "It imported" is not evidence that the
    canonical sanitizer was loaded -- asking what a green import actually
    exercised is the same lesson as verifying through the package rather than
    trusting sys.path. If the wrong file is on the path, this fails closed here
    rather than silently redacting with someone else's rules.
    """
    global _sanitizer_mod
    if _sanitizer_mod is not None:
        return _sanitizer_mod

    want = expected_sanitizer_path()
    if not os.path.isfile(want):
        raise RuntimeError(
            "canonical sanitizer not found at %s -- checkpoint sanitation "
            "cannot be established, so the graph must not run. Set "
            "HERMES_CONFIG_ROOT (note: a kanban worker's HERMES_HOME is its "
            "PROFILE root, not the config repo) or check the hermes-config "
            "checkout." % want)

    libdir = os.path.join(hermes_home(), "lib")
    if libdir not in sys.path:
        sys.path.append(libdir)

    import model_call.sanitizer as mod  # noqa: E402  (path set above)

    got = os.path.realpath(getattr(mod, "__file__", "") or "")
    if got != os.path.realpath(want):
        raise RuntimeError(
            "loaded the WRONG sanitizer: %s (expected %s). Refusing to run: a "
            "checkpoint redactor that is not the canonical one is worse than "
            "none, because it reports success having applied unknown rules."
            % (got, want))

    for attr in ("GENOMICS_RE", "HONCHO_MARKERS", "PRIVKEY_RE", "SanitizerHalt",
                 "sanitize"):
        if not hasattr(mod, attr):
            raise RuntimeError(
                "sanitizer at %s is missing %r -- its shape changed under us; "
                "re-read it before trusting this module." % (got, attr))

    _sanitizer_mod = mod
    return mod


# --------------------------------------------------------------------------
# HALT tier -- explicit, at node return, so the graph can route to `human`
# --------------------------------------------------------------------------

def halt_reason(text: Any) -> Optional[str]:
    """Name the HALT rule this text trips, or None.

    The three tiers from sanitizer.py: genomics tree paths, Honcho context
    markers, private-key blocks. Each means the payload must never ship and no
    redaction is meaningful.

    Returns a RULE NAME, never the match. Echoing the matched text into a log or
    a card body would defeat the point of the check.

    Non-str input returns None: only strings can carry these.
    """
    if not isinstance(text, str) or not text:
        return None
    mod = _load_sanitizer()
    if mod.GENOMICS_RE.search(text):
        return "genomics_path"
    low = text.lower()
    for marker in mod.HONCHO_MARKERS:
        if marker in low:
            return "honcho_marker"
    if mod.PRIVKEY_RE.search(text):
        return "private_key"
    return None


def state_halt_reason(state) -> Optional[str]:
    """First (field, rule) that trips a HALT anywhere in `state`, or None.

    Walks nested dicts/lists because objections_* carry raw model output and
    gate_summary carries the gate's own note strings.
    """
    for key, value in state.items():
        rule = _walk_halt(value)
        if rule:
            return "%s:%s" % (key, rule)
    return None


def _walk_halt(value, depth: int = 0) -> Optional[str]:
    if depth > 8:
        return None
    if isinstance(value, str):
        return halt_reason(value)
    if isinstance(value, dict):
        for v in value.values():
            r = _walk_halt(v, depth + 1)
            if r:
                return r
        return None
    if isinstance(value, (list, tuple)):
        for v in value:
            r = _walk_halt(v, depth + 1)
            if r:
                return r
    return None


# --------------------------------------------------------------------------
# REDACT tier -- at serialization, via the checkpointer's serde seam
# --------------------------------------------------------------------------

# Cheap pre-filter. The spike measured 29 serde dump calls for a TWO-node graph,
# so the redactor runs on every channel write, not once per superstep -- against
# a large diff that is 29 passes of a VERBOSE alternation regex.
#
# The token list is DERIVED FROM THE SANITIZER'S OWN CONSTANTS where it can be.
# A hand-written list is how this goes wrong: the first draft of this filter
# listed "honcho" while HONCHO_MARKERS is ("honcho_context", "peer_representation",
# "working_representation"), so a payload carrying only `peer_representation`
# would have skipped the expensive path and the HALT would never have fired --
# a security check silently disabled by an optimisation. Correctness is asserted
# by test, not by inspection: for a fixture corpus, _might_match() must be True
# whenever sanitize() changes the text or raises.
_REGEX_FAMILY_TOKENS = (
    # SECRET_ASSIGN_RE key alternation: api_key/apikey/secret/password/passwd/
    # token/access_key/auth -- "key" and "auth" cover the compounds.
    "key", "secret", "password", "passwd", "token", "auth",
    "bearer",          # BEARER_RE (matched case-insensitively by the filter)
    "akia", "asia",    # AWS_KEY_RE
    "://",             # CONNSTR_RE
    "/home/data", "genomics-platform",   # GENOMICS_RE
    "begin",           # PRIVKEY_RE ("-----BEGIN ... PRIVATE KEY-----")
)


def _filter_tokens() -> tuple:
    mod = _load_sanitizer()
    return tuple(t.lower() for t in _REGEX_FAMILY_TOKENS) + tuple(
        m.lower() for m in mod.HONCHO_MARKERS)


def _might_match(text: str) -> bool:
    """Fast reject. MUST NOT return False for anything sanitize() would touch."""
    low = text.lower()
    return any(tok in low for tok in _filter_tokens())


def redact(text: str) -> str:
    """Redact credential-shaped content. NEVER raises, NEVER halts.

    A serde must not decide policy: HALT routing is the graph's job (see the
    module docstring). If the canonical sanitizer raises SanitizerHalt here --
    which it will for genomics/Honcho/private-key content -- this returns the
    text unchanged and leaves the decision to the node-return guard, which is
    what stops the run and routes to `human` before a checkpoint is written.
    """
    if not isinstance(text, str) or not text:
        return text
    if not _might_match(text):
        return text
    mod = _load_sanitizer()
    try:
        return mod.sanitize(text)
    except mod.SanitizerHalt:
        return text
    except Exception:
        # A redactor that raises into the checkpoint path would kill a card for
        # a logging-class failure. Fail safe for the CARD; the HALT guard and
        # the driver's assertions are what keep this from being a silent hole.
        return text


def sanitize_obj(obj: Any, depth: int = 0) -> Any:
    """Structure-preserving redaction of every string in `obj`."""
    if depth > 8:
        return obj
    if isinstance(obj, str):
        return redact(obj)
    if isinstance(obj, dict):
        return {k: sanitize_obj(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_obj(v, depth + 1) for v in obj]
    if isinstance(obj, tuple):
        return tuple(sanitize_obj(v, depth + 1) for v in obj)
    return obj


class SanitizingSerde:
    """Wraps a SerializerProtocol; redacts strings on the way OUT only.

    Implements exactly the two methods the protocol declares (spike-confirmed:
    dumps_typed / loads_typed). dumps/loads are provided as passthroughs because
    some call sites use them, but they are NOT part of the protocol.

    Asymmetric by design: dumps redacts, loads does not "unredact" -- there is
    nothing to restore. A checkpoint is a lossy record of state, and D5.2's
    resume must be built knowing that. Recorded here because a future reader
    will otherwise assume round-trip fidelity.
    """

    def __init__(self, inner, on_redact: Optional[Callable[[], None]] = None):
        self.inner = inner
        self.on_redact = on_redact
        self.dump_calls = 0

    def dumps_typed(self, obj):
        self.dump_calls += 1
        return self.inner.dumps_typed(sanitize_obj(obj))

    def loads_typed(self, data):
        return self.inner.loads_typed(data)

    def dumps(self, obj):
        self.dump_calls += 1
        return self.inner.dumps(sanitize_obj(obj))

    def loads(self, data):
        return self.inner.loads(data)


def make_checkpointer():
    """InMemorySaver with checkpoint sanitation installed.

    Fork ruling 4: in-memory at D5.1, Postgres at D5.3. Section 8 deferral 1
    records the accepted risk -- a mid-run crash loses all state, the card
    restarts from the beginning, and it MAY RE-SPEND cloud calls already paid
    for, bounded only by the CD-025 per-card cap.
    """
    from langgraph.checkpoint.memory import InMemorySaver

    serde = SanitizingSerde(InMemorySaver().serde)
    return InMemorySaver(serde=serde), serde
