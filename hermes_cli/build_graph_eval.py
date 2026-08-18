"""D5.1 build-harness graph -- the evaluation apparatus (design v2 section 6).

CD-030. Section 6 opens by saying what this module is: "This section specifies
what to build. It is not commentary, and it is not optional." The comparator is
a judgement made ahead of evidence, so it ships with the means to check itself.
Without it, "is the classification doing anything?" is unanswerable and the
fork-2 ruling calcifies by default.

Contents:
  * the classifier prompt builder (section 5.4) -- a PURE function of the two
    objection sets and nothing else
  * the response parser, fail-closed (section 5.5)
  * author-independence detection (section 5.3)
  * decide_recurrence(), which combines them with the climbing default
  * the config-gated forced-climb holdout (section 6.3)
  * the graph-classify.jsonl writer (section 6.1)

Standard library only, plus an OPTIONAL late import of hermes_cli.kanban_db for
the board logs directory. That import is deferred into the writer so every pure
function in this module stays importable and testable off-box.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Optional

# Line-format version for graph-classify.jsonl. Bump when the FIELD SET changes.
LOG_SCHEMA_VERSION = 1

# Version of the classifier prompt. Bump on ANY change to build_classify_prompt.
#
# Section 6.1 is emphatic about why this is not bookkeeping. Section 5.4
# constrains the prompt to receive the two objection sets and NOTHING ELSE, and
# that constraint is the entire reason the signal is decorrelated from both the
# reviewer and the implementer. If the prompt is widened or reworded mid-corpus,
# every threshold in section 6.4 is silently computed across two different
# instruments and nothing in the log would reveal it. The field costs one string
# now and is unrecoverable later -- lines already written cannot be retro-labelled.
CLASSIFIER_PROMPT_VERSION = "1"

HOLDOUT_ENV = "HERMES_GRAPH_CLASSIFY_HOLDOUT"

LOG_FILENAME = "graph-classify.jsonl"


# --------------------------------------------------------------------------
# section 5.4 -- the classifier prompt, a pure function of the objections
# --------------------------------------------------------------------------

_PROMPT_HEADER = (
    "You are comparing two sets of code-review objections.\n"
    "\n"
    "Answer exactly one question: do these two sets describe the SAME "
    "underlying objection?\n"
    "\n"
    "Reply with exactly one JSON object and nothing else:\n"
    '  {"same": true} or {"same": false}\n'
)


def _render_objections(label: str, objections: list) -> str:
    if not objections:
        return "%s: (none)\n" % label
    lines = ["%s:" % label]
    for i, obj in enumerate(objections, 1):
        if isinstance(obj, dict):
            text = obj.get("text")
            if text is None:
                text = json.dumps(obj.get("raw"), sort_keys=True)
        else:
            text = str(obj)
        lines.append("  %d. %s" % (i, text))
    return "\n".join(lines) + "\n"


def build_classify_prompt(objections_prior: list, objections_current: list) -> str:
    """Build the local `classify` activity's prompt.

    Receives THE TWO OBJECTION SETS AND NOTHING ELSE -- no card body, no plan,
    no diff, no rung state, no spend state. It is not asked what to do about the
    answer and is not told what its answer routes to.

    That is not tidiness. It is what keeps the signal decorrelated from both the
    reviewer and the implementer, and what prevents the Goodhart problem that
    sank the `failure_class` proposal in fork 2: a model told that a field
    controls routing can optimise for it.

    The edge stays model-free. A free local classifier is permitted INSIDE the
    comparator because the edge itself reads a boolean off typed state and the
    GRAPH decides the route. What is forbidden is asking a model "should we skip
    the second rung?" -- the distinction is between model compliance (abolished
    by D5.1) and a model-derived signal written to typed state (permitted).
    """
    return (
        _PROMPT_HEADER
        + "\n"
        + _render_objections("PRIOR OBJECTIONS", objections_prior)
        + "\n"
        + _render_objections("CURRENT OBJECTIONS", objections_current)
    )


def parse_classify_response(text: Optional[str]) -> Optional[bool]:
    """Parse the classifier reply. Returns True/False, or None if unusable.

    Fail-closed to None, which decide_recurrence() turns into the climbing
    default. Mirrors _parse_verdict's posture in lib/model_call/core.py, which
    returns None rather than raising when a response is not parseable JSON.

    Never re-call to chase a parseable answer. That is the unbounded-loop
    pitfall the router skill names explicitly.
    """
    if not text:
        return None
    candidate = text.strip()
    # Tolerate a fenced block, which models emit even when told not to.
    if candidate.startswith("```"):
        parts = candidate.split("```")
        if len(parts) >= 2:
            candidate = parts[1]
            if candidate.startswith("json"):
                candidate = candidate[4:]
            candidate = candidate.strip()
    # Tolerate prose around a single object.
    if not candidate.startswith("{"):
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end < start:
            return None
        candidate = candidate[start : end + 1]
    try:
        obj = json.loads(candidate)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    value = obj.get("same")
    if isinstance(value, bool):
        return value
    return None


# --------------------------------------------------------------------------
# section 5.3 -- author independence is a precondition, not a detail
# --------------------------------------------------------------------------


def is_confounded(implementer_model: Optional[str], reviewer_model: Optional[str]) -> bool:
    """True when recurrence cannot be treated as evidence.

    Recurrence is evidence only if the review of the first fix was produced by a
    model that did not author it. A model reviewing its own rewrite is
    confounded: persistence and self-consistency are indistinguishable.

    Establishable structurally, and checked rather than assumed -- ModelResult
    surfaces .model from the typed Resolution, so the graph compares the fix
    call's model against the review call's. It correctly catches the confounded
    case too: under a --directive pinning both to the same model, the two match.

    Either value missing is ALSO confounded. If independence cannot be
    established, it has not been established, and section 5.3 requires going
    straight to the default rather than treating an unknown as a pass.
    """
    if not implementer_model or not reviewer_model:
        return True
    return implementer_model == reviewer_model


# --------------------------------------------------------------------------
# section 5.5 -- the default climbs, and the two error directions differ
# --------------------------------------------------------------------------


def decide_recurrence(
    classifier_answer: Optional[bool],
    *,
    implementer_model: Optional[str] = None,
    reviewer_model: Optional[str] = None,
) -> dict:
    """Return {objection_recurred, recurrence_tier, recurrence_confounded,
    dispute_class}.

    If the classifier errored, timed out, or returned anything unparseable -- or
    if the comparison is confounded -- then objection_recurred is False and the
    ladder climbs normally to rung 2.

    The two error directions are NOT symmetric, which is the whole argument:

      * Wrongly calling it approach-disputed skips rung 2 permanently. If rung 3
        then fails, the card reaches `human` having never attempted rung 2. The
        error is invisible and unrecoverable within the run.
      * Wrongly calling it impl-bug spends one rung-2 attempt that fails, and
        the ladder proceeds to rung 3 anyway. The error is bounded,
        self-correcting, visible in the ledger, and already capped by both
        limiters in section 2.2.

    So the uncertain case climbs. It does mean the rung-2 skip is forgone in
    exactly the ambiguous cases -- the correct trade, because the skip is an
    efficiency and the rung is a safety net, and efficiency yields to safety
    under uncertainty.
    """
    confounded = is_confounded(implementer_model, reviewer_model)
    if confounded or classifier_answer is None:
        recurred = False
        tier = "default"
    else:
        recurred = bool(classifier_answer)
        tier = "classifier"
    return {
        "objection_recurred": recurred,
        "recurrence_tier": tier,
        "recurrence_confounded": confounded,
        "dispute_class": "approach_disputed" if recurred else "impl_bug",
    }


# --------------------------------------------------------------------------
# section 6.3 -- the config-gated forced-climb holdout
# --------------------------------------------------------------------------


def holdout_n(env: Optional[dict] = None) -> Optional[int]:
    """Read HERMES_GRAPH_CLASSIFY_HOLDOUT. None = OFF (the default).

    An unparseable or non-positive value is treated as OFF, because the failure
    posture must not be "spend a rung-2 attempt on every card because someone
    typed a letter". But OFF-by-typo is exactly how a corpus ends up with no
    counterfactual and nobody noticing, so holdout_config_invalid is recorded on
    every logged line -- the misconfiguration is visible in the data rather than
    only in someone's shell.
    """
    source = os.environ if env is None else env
    raw = source.get(HOLDOUT_ENV)
    if raw is None or str(raw).strip() == "":
        return None
    try:
        n = int(str(raw).strip())
    except (TypeError, ValueError):
        return None
    return n if n >= 1 else None


def holdout_config_invalid(env: Optional[dict] = None) -> bool:
    """True when the env var is SET but does not yield a usable N."""
    source = os.environ if env is None else env
    raw = source.get(HOLDOUT_ENV)
    if raw is None or str(raw).strip() == "":
        return False
    return holdout_n(env=source) is None


def is_holdout(card_id: str, env: Optional[dict] = None) -> bool:
    """Deterministic 1-in-N forced-climb selection, keyed on card_id.

    Section 6.3: the skipped rung is never observed. When the graph routes
    approach_disputed and skips rung 2, there is no counterfactual telling you
    whether rung 2 would have worked, so per-arm outcome comparison is
    unanswerable from ordinary traffic -- you only ever see rung-2 outcomes from
    the arm the classifier already judged favourable, which is a selection
    effect rather than a measurement.

    DETERMINISTIC, not random, deliberately. A stable hash of card_id means the
    same card always lands in the same arm, so the assignment is reproducible
    and auditable from the log alone long after the run. Random selection would
    need a seed that nobody records, leaving "why was that card held out?"
    permanently unanswerable -- which is the same class of problem this
    apparatus exists to solve. The cost is that it is not a fresh draw per card;
    with hex card ids the distribution is uniform enough for a 1-in-N gate.
    """
    n = holdout_n(env=env)
    if not n:
        return False
    digest = hashlib.sha256(card_id.encode("utf-8")).hexdigest()[:16]
    return int(digest, 16) % n == 0


# --------------------------------------------------------------------------
# section 6.1 -- the graph-classify.jsonl writer
# --------------------------------------------------------------------------


def build_classify_record(
    *,
    card_id: str,
    run_id: Any,
    component: str,
    recurrence_tier: Optional[str],
    recurrence_confounded: bool,
    objections_prior: list,
    objections_current: list,
    dispute_class: Optional[str],
    rung_taken: Optional[str],
    rung_outcome: Optional[str],
    holdout: bool,
    findings_prior: Any = None,
    findings_current: Any = None,
    env: Optional[dict] = None,
) -> dict:
    """Build one graph-classify.jsonl line. Pure -- no I/O, so it is testable.

    Field set is section 6.1's, plus the raw findings arrays AS RECEIVED. Those
    are not decoration: they are the instrument that decides section 8's
    deferred contract question, by measuring what fraction of findings items
    VOLUNTARILY carry a file path, a symbol or a severity with no clause
    demanding it. That number is currently unknown and is the only thing that
    would justify reopening VERDICT_CONTRACT (section 6.4, last row).
    """
    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "event": "graph_classify",
        "schema_version": LOG_SCHEMA_VERSION,
        "classifier_prompt_version": CLASSIFIER_PROMPT_VERSION,
        "card_id": card_id,
        "run_id": run_id,
        "component": component,
        "recurrence_tier": recurrence_tier,
        "recurrence_confounded": recurrence_confounded,
        "objections_prior": objections_prior,
        "objections_current": objections_current,
        "dispute_class": dispute_class,
        "rung_taken": rung_taken,
        "rung_outcome": rung_outcome,
        "holdout": holdout,
        "holdout_n": holdout_n(env=env),
        "holdout_config_invalid": holdout_config_invalid(env=env),
        "findings_prior": findings_prior,
        "findings_current": findings_current,
    }


def append_classify_line(record: dict, logs_dir=None) -> bool:
    """Append one line to graph-classify.jsonl. Returns True on success.

    Same shape as _darklaunch_log_autoaccept / _darklaunch_log_arbiter in
    kanban_db.py -- board logs dir, mkdir parents, append, sort_keys, one JSON
    object per line -- and the same best-effort posture: any error is swallowed
    so a logging failure can never affect a card.

    DIVERGENCE FROM THAT PRECEDENT, deliberate: this returns a BOOLEAN. The two
    dark-launch writers are observational side-channels where a lost line costs
    nothing. This one is the measurement instrument for a ruling that section
    6.4 revisits after 20 LOGGED evaluations. A silently dropped line is
    indistinguishable from an evaluation that never happened, so the corpus
    would develop holes that move every threshold with nothing saying so. The
    caller records the failure in workflow_state.classify_log_failed, which
    makes silence unambiguous without ever raising into the card's path.
    """
    try:
        if logs_dir is None:
            from hermes_cli.kanban_db import worker_logs_dir  # late, optional

            logs_dir = worker_logs_dir()
        logs_dir.mkdir(parents=True, exist_ok=True)
        with open(logs_dir / LOG_FILENAME, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
        return True
    except Exception:
        return False
