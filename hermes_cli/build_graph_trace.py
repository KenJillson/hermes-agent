"""D5.1/5.5 build-harness graph -- the per-node trace emitter (CD-038).

A `graph-trace.jsonl` sibling to `graph-classify.jsonl`: per-node elapsed, the
routing decision each edge returned, and the model-call `run_id` that joins to
the Overlord ledger.

WHY THIS AND NOT LANGFUSE FIRST
-------------------------------
Handoff 2026-08-20 section 6: the span emitter is a prerequisite for Langfuse
either way, satisfies Phase 5.5's joinability and ledger-authoritative criteria
outright, closes AC-3's measurement gap, and swaps its transport for OTel later
behind the same call site. AC-3 was demonstrated on 2026-08-20 by timing a
graph node from OUTSIDE -- a task_runs row and five 50s sleeps. That worked,
but the graph could not say how long its own node took. This is that.

THE SAFETY CONSTRAINT IS THE DESIGN
-----------------------------------
The self-cert disables LangSmith because "a live tracer is a second egress path
that never passes [the sanitizer]". A trace FILE has exactly the same property.
``WorkflowState`` carries ``plan``, ``diff``, ``objections_prior`` and
``objections_current`` -- all payload, and the reason build_graph_sanitize.py
and the section 4.1 HALT guard exist at all.

So this module NEVER dumps state. It copies a fixed ALLOWLIST of scalars
(``_STATE_FIELDS``), and payload-bearing fields appear only as COUNTS. A count
is not payload. There is no "include everything" switch and no way to widen it
from a card body, a worker or an env var -- widening it is a source edit that
shows up in a diff, which is the point.

Two fields are allowlisted that look like exceptions and are not:
  * ``halt_reason`` is by construction ``field:rule`` -- a rule NAME and a field
    NAME, never the matched text (build_graph.guard, section 4.1).
  * ``ac_record_path`` is a path the ac-execution record already occupies, and
    it is the join key from a trace line to that record.
``directive`` is NOT allowlisted; only whether one was set.

TRANSPARENCY IS ENFORCED, NOT INTENDED
--------------------------------------
The wrappers return the wrapped callable's value unchanged and RE-RAISE its
exceptions. A tracer that swallowed a node exception would convert a crash into
a silent wrong answer -- and under a checkpointer that means re-spending cloud
calls already paid for (design section 8 deferral 1). Emission itself is
best-effort and never raises into the card's path; dropped lines are COUNTED
and reported in a final summary line, so silence is unambiguous the way
``classify_log_failed`` makes it unambiguous for the classify writer.

DISABLED MEANS IDENTICAL
------------------------
``HERMES_GRAPH_TRACE=0`` (or ``false``/``off``/``no``) makes every wrapper the
IDENTITY function -- ``build()`` then registers exactly the callables it
registers today, byte-for-byte the same objects. Not "a tracer that writes
nothing": no wrapper at all. That is what makes the kill switch trustworthy.

Standard library only, plus the same OPTIONAL late import of
hermes_cli.kanban_db for the board logs dir that build_graph_eval uses.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Optional

# Bump when the FIELD SET changes, same contract as build_graph_eval's.
TRACE_SCHEMA_VERSION = 1

LOG_FILENAME = "graph-trace.jsonl"

TRACE_ENV = "HERMES_GRAPH_TRACE"
_OFF = ("0", "false", "off", "no")

# THE ALLOWLIST. Scalars only. Adding a field here is a source edit, on
# purpose -- see the module docstring.
_STATE_FIELDS = (
    "schema_version",
    "card_id",
    "component",
    "iteration",
    "rung",
    "cloud_review_calls",
    "diff_files",
    "diff_added_lines",
    "objection_recurred",
    "recurrence_tier",
    "recurrence_confounded",
    "dispute_class",
    "implementer_model",
    "reviewer_model",
    "classify_log_failed",
    "holdout",
    "halt_reason",
    "terminal_reason",
    "last_verdict_passed",
    "ac_record_path",
    "security_sensitive",
    "financial_sensitive",
    "novel_architecture",
    "re_review_after_severe",
    "author_independence",
    "component_count",
)

# Payload-bearing fields recorded ONLY as lengths.
_COUNT_FIELDS = ("plan", "diff", "objections_prior", "objections_current")

# gate_summary is a dict of counts, but it is built by another module and could
# grow a message field, so it is allowlisted key-by-key rather than copied.
_GATE_FIELDS = ("verdict", "checks_total", "checks_passed", "checks_failed",
                "checks_unrunnable", "judgment_count", "all_clean")

# ModelResult keys that are safe. `text`, `rationale` and `envelope_path`
# are NOT here: text is model output and rationale can quote it.
_MODEL_FIELDS = ("ok", "lane", "model", "activity", "task_class", "run_id",
                 "cost_usd", "rejected")


def enabled(env: Optional[dict] = None) -> bool:
    """Trace on unless explicitly switched off.

    ON by default, matching graph-classify.jsonl, which is written
    unconditionally from make_classify_failure. A trace nobody turns on is a
    trace nobody has when they need it. The kill switch exists because an
    always-on writer with no escape is worse.
    """
    source = os.environ if env is None else env
    raw = str(source.get(TRACE_ENV, "")).strip().lower()
    return raw not in _OFF if raw else True


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _scalar(value):
    """Keep only JSON scalars. Anything else becomes its type name.

    A belt-and-braces second line of defence behind the allowlist: if a field
    named in _STATE_FIELDS ever starts holding a dict or a list, this records
    that it did WITHOUT recording what was in it.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return "<%s>" % type(value).__name__


def redact_state(state) -> dict:
    """The allowlisted projection of workflow_state. NEVER the state itself."""
    out = {}
    if not isinstance(state, dict):
        return out
    for key in _STATE_FIELDS:
        if key in state:
            out[key] = _scalar(state[key])
    for key in _COUNT_FIELDS:
        value = state.get(key)
        if value is None:
            continue
        try:
            out["%s_len" % key] = len(value)
        except TypeError:
            out["%s_len" % key] = None
    gate = state.get("gate_summary")
    if isinstance(gate, dict):
        out["gate"] = {k: _scalar(gate[k]) for k in _GATE_FIELDS if k in gate}
    # rung_attempts is a dict of ints -- counts, not payload.
    attempts = state.get("rung_attempts")
    if isinstance(attempts, dict):
        out["rung_attempts"] = {str(k): _scalar(v) for k, v in attempts.items()}
    return out


def redact_model_result(out) -> dict:
    """The allowlisted projection of one call_model() return."""
    rec = {}
    if not isinstance(out, dict):
        return rec
    rec["rc"] = _scalar(out.get("rc"))
    rec["klass"] = _scalar(out.get("klass"))
    result = out.get("result")
    if isinstance(result, dict):
        for key in _MODEL_FIELDS:
            if key in result:
                rec[key] = _scalar(result[key])
        err = result.get("error")
        if isinstance(err, dict):
            # STAGE ONLY. `detail` can quote the model or the prompt.
            rec["error_stage"] = _scalar(err.get("stage"))
        elif err:
            rec["error_stage"] = "<present>"
    return rec


class Tracer:
    """Wraps node, edge and model callables. Disabled -> identity wrappers."""

    def __init__(self, *, thread_id: str = "", logs_dir=None,
                 env: Optional[dict] = None, sink: Optional[Callable] = None):
        self.thread_id = thread_id
        self.logs_dir = logs_dir
        self.env = env
        self.enabled = enabled(env)
        self.seq = 0
        self.dropped = 0
        self.written = 0
        self.pid = os.getpid()
        # `sink` exists so the offline driver can capture lines without
        # touching a board logs directory. Production leaves it None.
        self._sink = sink

    # -- emission -------------------------------------------------------
    def _emit(self, record: dict) -> None:
        """Best-effort. NEVER raises into the card's path."""
        try:
            self.seq += 1
            record["seq"] = self.seq
            record["ts"] = _now()
            record["pid"] = self.pid
            record["thread_id"] = self.thread_id
            record["schema_version"] = TRACE_SCHEMA_VERSION
            line = json.dumps(record, sort_keys=True)
            if self._sink is not None:
                self._sink(line)
            else:
                logs_dir = self.logs_dir
                if logs_dir is None:
                    from hermes_cli.kanban_db import worker_logs_dir
                    logs_dir = worker_logs_dir()
                logs_dir.mkdir(parents=True, exist_ok=True)
                with open(logs_dir / LOG_FILENAME, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            self.written += 1
        except Exception:
            self.dropped += 1

    def summary(self) -> None:
        """Final line. A dropped-line count makes silence unambiguous."""
        if not self.enabled:
            return
        self._emit({"event": "graph_trace_summary",
                    "lines_written": self.written, "lines_dropped": self.dropped})

    # -- wrappers -------------------------------------------------------
    def node(self, name: str, fn: Callable) -> Callable:
        if not self.enabled:
            return fn

        def _wrapped(state):
            t0 = time.perf_counter()
            try:
                update = fn(state)
            except BaseException as exc:
                # RE-RAISE. A tracer that swallowed this would turn a crash
                # into a silent wrong answer, and under a checkpointer that
                # means re-spending calls already paid for.
                self._emit({"event": "graph_node", "node": name,
                            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
                            "raised": type(exc).__name__,
                            "state_in": redact_state(state)})
                raise
            self._emit({"event": "graph_node", "node": name,
                        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
                        "raised": None,
                        "state_in": redact_state(state),
                        "update": redact_state(update)})
            return update

        _wrapped.__name__ = getattr(fn, "__name__", name)
        _wrapped.__doc__ = getattr(fn, "__doc__", None)
        # Preserve markers the topology or its tests may read off a node.
        for attr in ("unbuilt",):
            if hasattr(fn, attr):
                setattr(_wrapped, attr, getattr(fn, attr))
        return _wrapped

    def edge(self, from_node: str, fn: Callable) -> Callable:
        if not self.enabled:
            return fn

        def _wrapped(state):
            t0 = time.perf_counter()
            decision = fn(state)
            self._emit({"event": "graph_edge", "from": from_node,
                        "decision": _scalar(decision),
                        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
                        "state_in": redact_state(state)})
            return decision

        _wrapped.__name__ = getattr(fn, "__name__", "route_%s" % from_node)
        return _wrapped

    def model(self, fn: Callable) -> Callable:
        """Wrap deps.model. This is where run_id comes from.

        Wrapping the CALL rather than reading state is deliberate: run_id is
        returned by call_model and is not a workflow_state field, so getting it
        from state would mean a schema change. It does not need one.
        """
        if not self.enabled:
            return fn

        if getattr(fn, "_traced", False):
            # Already wrapped. build() assigns deps.model = tr.model(deps.model),
            # and a second build() on the same Deps would otherwise nest the
            # wrappers -- two lines per call and an elapsed_ms that includes the
            # inner wrapper. Idempotent by marker, not by hoping build() is
            # called once.
            return fn

        def _wrapped(*args, **kwargs):
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            rec = {"event": "graph_model_call",
                   "activity": _scalar(kwargs.get("activity")),
                   "card_id": _scalar(kwargs.get("card_id")),
                   "component": _scalar(kwargs.get("component")),
                   "mode": _scalar(kwargs.get("mode")),
                   "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)}
            rec.update(redact_model_result(out))
            self._emit(rec)
            return out

        _wrapped.__name__ = getattr(fn, "__name__", "model")
        _wrapped._traced = True
        return _wrapped
