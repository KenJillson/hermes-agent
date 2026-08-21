"""D5.1 build-harness graph -- the topology (design v2 sections 1, 2, 5).

CD-031, part 2 of 2. Part 1 is build_graph_sanitize.py.

SCOPE -- THE SPINE, AND WHY THE LINE IS THERE
----------------------------------------------
Design v2's node table has 13 nodes. This module builds the ones whose wrapped
primitive has been READ AT SOURCE this session:

  built     cheap_gate (ac_check_runner.gate), cloud_review, cloud_re_review,
            classify_failure, fix_rung1, fix_rung2 (all model-call), assemble,
            human, and fix_rung3 as an auth-deferred node (section 8 deferral 6)
  unbuilt   plan, plan_review, local_review
  CD-042    implement -- BUILT AND REACHABLE. START routes to it when the
            card has no work product yet; see route_entry().

The line is not arbitrary. `plan` and `local_review` wrap `delegate_task`
and fix_rung3's real body wraps `terminal()`. CD-041 READ `delegate_task`
at source and built `implement` on it, so that primitive is no longer
unread; `plan` and `local_review` stay unbuilt because each needs a work
product `delegate_task` does not return -- a plan, and a verdict, the
latter being `local_review`'s own deliverable. `terminal()` is still
unread, and writing a node on an unread primitive would mean guessing a
signature, which rule 1 forbids.

The unbuilt nodes are present in the topology with their edges wired.

CD-042 REWIRED THE ENTRY. START is no longer a static edge to cheap_gate;
it is a conditional edge that reads whether the card already HAS a work
product. A card with a diff goes straight to cheap_gate, which is exactly
the CD-033/034/035 path and is unchanged. A card with an EMPTY diff goes
to `implement` first.

That second branch is the point of the CD, and it is a SPEND REDUCTION
rather than a new cost. Before it, a from-scratch card ran its AC checks
against an unimplemented workspace, failed them, and the arbiter returned
ESCALATE -- so `fix_rung1`, a METERED CLOUD CALL, was doing the
implementing. Routing through `implement` puts a free local child in front
of that. Measured claim, not a hope: verify it on the ledger after the
first live card.

AN UNBUILT NODE ROUTES TO `human`. IT DOES NOT RAISE.
A raise inside a node kills the run, and under the in-memory checkpointer that
loses all state and may re-spend cloud calls already paid for (section 8
deferral 1). Parking the card for a human is bounded and visible; crashing is
neither. Same posture as the sandbox ruling -- unrunnable means a human looks,
not a silent wrong answer.

THE ACCEPTANCE CRITERION
------------------------
The plan's criterion is that "approach-disputed -> skip the second same-family
fix rung" is verifiable FROM THE GRAPH DEFINITION rather than from model
compliance. It lives in route_after_classify() below, registered through
add_conditional_edges with an explicit path_map. It reads a BOOLEAN off typed
state. No model is asked what to do; section 5.4's classifier computes an input,
the input is written to state, and the graph enforces the rule on it.

TWO ORTHOGONAL LIMITERS (section 2.2)
-------------------------------------
Rung caps (attempt counts) and the spend gate (dollars) are independent, can
disagree in both directions, and BOTH terminate to `human`. Neither overrides
the other; whichever binds first, binds. This module deliberately does NOT
reconcile them -- a reconciliation function would become a third authority.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from typing import Any, Callable, Optional

from hermes_cli import build_graph_checks as ck
from hermes_cli import build_graph_eval as ev
from hermes_cli import build_graph_sanitize as sz
from hermes_cli import build_graph_trace as tl
from hermes_cli.build_graph_state import (
    SCHEMA_VERSION,
    WorkflowState,
    new_workflow_state,
    selection_signals,
    validate,
)

MODEL_CALL_EXE = os.environ.get(
    "HERMES_MODEL_CALL", "/home/jetson/.local/bin/model-call")

# Per-rung attempt cap and the per-component cloud-review cap (section 2.2).
RUNG_ATTEMPT_CAP = 1
CLOUD_REVIEW_CAP = 2

# Ladder order. rung3 is in the topology but auth-deferred (section 8/6).
LADDER = ("rung1", "rung2", "rung3")

# Where the arbiter's `pass` decision goes.
#
# Design section 2.1 routes it to `local_review`. At D5.1 local_review is
# UNBUILT (its primitive, delegate_task, has not been read), and an unbuilt node
# parks the card at `human`. Routing `pass` through it therefore ORPHANS the
# entire cloud_review -> cloud_re_review -> classify_failure chain: those nodes
# would be unreachable from START, and the acceptance-criterion edge would exist
# as a function the compiled graph can never traverse. D5.1's whole job is to
# make that criterion demonstrable, so a topology that cannot reach it fails the
# phase while looking green.
#
# Caught 2026-08-18 by asserting reachability rather than node presence: every
# node was present and the edge list showed classify_failure with NO edges.
#
# So at D5.1 `pass` goes straight to cloud_review, and local_review is inserted
# between them when its primitive lands at D5.2. Recorded as a deliberate
# deviation from section 2.1 rather than left as an accident.
GATE_PASS_TARGET = "cloud_review"

# CD-041a CORRECTION. CD-041 shipped ("ok", "error", "timeout", "failed").
# THAT SET WAS WRONG IN BOTH DIRECTIONS and the node could never succeed:
# "ok" is NEVER a task status, and "completed" -- the SUCCESS value -- was
# absent, so every successful child parked as implement_unknown_status.
# "interrupted" was missing too. It failed closed, which is why nothing
# broke visibly; it was simply inert.
#
# The error was reading a grep instead of the assignment site. "ok" appears
# at delegate_tool.py:2360 inside result_meta for TOOL_TRACE entries -- a
# per-tool-call status, not the task status. The two dict families are
# told apart by a sibling key: a per-task result carries "task_index" and
# a tool_trace entry does not.
#
# DERIVED BY AST from the two functions that actually produce task results
# (_run_single_child, _execute_and_aggregate), not transcribed:
#   status = "..."   -> interrupted / completed / failed   (2324/2329/2331)
#   dict literals    -> timeout / error                    (2279/2532/3040/3052/3077)
# `unknown` is excluded deliberately: it lives only in
# _subagent_stop_tool_call_history, which is tool_trace, not a task result.
#
# The driver now RE-DERIVES this set from the installed delegate_tool.py by
# AST and asserts equality, so this constant can never again be a claim
# about the contract rather than a reading of it.
IMPLEMENT_STATUSES = ("completed", "failed", "interrupted", "timeout", "error")

# The ONE value that means the child did the work. summary present, not
# interrupted, and not the "(empty)" sentinel run_agent.py emits when it
# gives up after repeated empty-LLM-response retries.
IMPLEMENT_OK = "completed"

# CD-042: how many times `implement` may run for one card, ACROSS
# DISPATCHES. Separate from RUNG_ATTEMPT_CAP, which governs the fix ladder
# and does not apply here.
#
# Without a cap the loop is UNBOUNDED: a failed child leaves the diff
# empty, so the next dispatch routes to `implement` again, forever. The
# board's BLOCK_RECURRENCE_LIMIT breaker would eventually send the card to
# triage, but that is a backstop for a misbehaving card, not a budget for
# this node.
#
# The count lives in the EXISTING `rung_attempts` dict channel, so there is
# NO SCHEMA CHANGE -- SCHEMA_VERSION stays 2 and every checkpoint written
# since CD-032 still resumes. LADDER does not contain "implement", so
# next_available_rung ignores the key and the fix ladder is unaffected.
IMPLEMENT_ATTEMPT_CAP = 1

# model-call staged exit codes, from lib/model_call/cli.py (read 2026-08-18).
RC_OK = 0
RC_LEDGER_UNRECORDED = 9          # call SUCCEEDED, ledger append FAILED
RC_SANITIZE_HALT = 3
_RETRYABLE = (4, 6, 8)            # transfer / fetch / transport


def rc_class(rc: int) -> str:
    """Name the outcome class of a model-call exit code.

    RC 9 IS NOT A GENERIC FAILURE. cli.py: "ledger append failed AFTER a
    successful call -- spend may be unrecorded; reconcile before retrying (the
    one state that must never pass silently)". A node that lumps it in with
    other failures and retries would spend twice and record neither. It routes
    to `human` and says why.
    """
    if rc == RC_OK:
        return "ok"
    if rc == RC_LEDGER_UNRECORDED:
        return "ledger_unrecorded"
    if rc == RC_SANITIZE_HALT:
        return "sanitize_halt"
    if rc in _RETRYABLE:
        return "retryable"
    return "failed"


# --------------------------------------------------------------------------
# model-call invocation (design section 0.1: the node invokes ONE CLI command)
# --------------------------------------------------------------------------

def call_model(
    *,
    activity: str,
    prompt: str,
    workspace: str,
    card_id: str,
    task: str = "-",
    component: str = "-",
    task_class: str = "-",
    directive: Optional[str] = None,
    signals: Optional[dict] = None,
    cwd: Optional[str] = None,
    mode: str = "none",
    timeout: int = 900,
    exe: str = None,
) -> dict:
    """Invoke model-call and return {"rc", "klass", "result"}.

    The prompt goes through --prompt-file, NEVER argv. cli.py's docstring is
    explicit that prompt text on a command line is "the improvised-invocation
    failure mode this primitive retires".

    `result` is the ModelResult as_dict() parsed from stdout: ok, lane, model,
    activity, task_class, run_id, rationale, rejected, text, verdict,
    envelope_path, ledger_line, cost_usd, error. It is JSON by construction,
    which is what makes it safe to park in workflow_state (section 4.1).

    NO RETRIES AT D5.1, deliberately, even for the retryable classes. The router
    skill names unbounded re-calling as an explicit pitfall, and a retry that
    crosses the rc-9 case double-spends. Retry policy is a D5.4 question.
    """
    exe = exe or MODEL_CALL_EXE
    fd, path = tempfile.mkstemp(prefix="graph-prompt-", suffix=".txt",
                                dir=workspace if os.path.isdir(workspace) else None)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(prompt)
        argv = [exe, "--activity", activity, "--prompt-file", path,
                "--card-id", card_id, "--task", task, "--component", component,
                "--task-class", task_class, "--workspace", workspace,
                "--mode", mode]
        if directive:
            argv += ["--directive", directive]
        if cwd:
            argv += ["--cwd", cwd]
        for name, fired in (signals or {}).items():
            if fired:
                argv += ["--signal", name]
        try:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
            rc, out = p.returncode, p.stdout
        except subprocess.TimeoutExpired:
            return {"rc": None, "klass": "failed",
                    "result": {"ok": False,
                               "error": {"stage": "timeout",
                                         "detail": "no response in %ss" % timeout}}}
        except OSError as e:
            return {"rc": None, "klass": "failed",
                    "result": {"ok": False,
                               "error": {"stage": "exec", "detail": str(e)}}}
        try:
            result = json.loads(out) if out.strip() else {}
        except json.JSONDecodeError:
            # Fail closed: an unparseable envelope is not an excuse to guess.
            result = {"ok": False,
                      "error": {"stage": "parse",
                                "detail": "stdout was not JSON"}}
        return {"rc": rc, "klass": rc_class(rc), "result": result}
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def verdict_passed(verdict) -> bool:
    """Did a review verdict pass? Delegates to the PRIMITIVE's own normalizer.

    Deliberately not reimplemented here. core.py records that five live cards
    produced six improvised vocabularies ("passed", {"passed": true}, "PASS",
    "needs_changes", "PASS_WITH_NOTES", "approve"), which is why
    _verdict_passed's prefix normalizer exists at all. A second copy in this repo
    would drift from it, and then the graph and the primitive would disagree
    about whether a review passed -- silently, and in the direction of spending.

    Coupling to a private symbol is the cost, and it is the lesser one. Asserted
    by the driver against the observed vocabularies.
    """
    core = _load_core()
    return bool(core._verdict_passed(verdict))


_core_mod = None


def _load_core():
    """Load lib/model_call/core.py, proving the file, same posture as the
    sanitizer loader in part 1 (append to sys.path, never insert(0); assert
    __file__; fail closed rather than fall back)."""
    global _core_mod
    if _core_mod is not None:
        return _core_mod
    import sys
    want = os.path.join(sz.hermes_home(), "lib", "model_call", "core.py")
    if not os.path.isfile(want):
        raise RuntimeError("model_call core not found at %s" % want)
    libdir = os.path.join(sz.hermes_home(), "lib")
    if libdir not in sys.path:
        sys.path.append(libdir)
    import model_call.core as mod
    got = os.path.realpath(getattr(mod, "__file__", "") or "")
    if got != os.path.realpath(want):
        raise RuntimeError("loaded the WRONG model_call core: %s" % got)
    if not hasattr(mod, "_verdict_passed"):
        raise RuntimeError(
            "model_call.core has no _verdict_passed -- its shape changed; "
            "re-read it before trusting the review edges.")
    _core_mod = mod
    return mod


# --------------------------------------------------------------------------
# HALT guard -- section 4.1, ruled 2026-08-18: route to human, no checkpoint
# --------------------------------------------------------------------------

def guard(update: dict, state) -> dict:
    """Screen a node's update for HALT-tier content BEFORE it enters state.

    "Write no checkpoint" cannot be implemented as "suppress a checkpoint" --
    LangGraph checkpoints every superstep and a node cannot veto that. It IS
    implementable as "the payload never enters state": the offending fields are
    dropped, a rule name is recorded, and the graph routes to `human`. Nothing
    carrying the payload is ever handed to the serializer, so no checkpoint can
    contain it. That is the same guarantee by a different route, and it is worth
    saying plainly because the two are easy to conflate.

    The recorded value is a RULE NAME and a field name, never the match.
    """
    reason = sz.state_halt_reason(update)
    if reason is None:
        return update
    return {"halt_reason": reason,
            "terminal_reason": "sanitizer_halt:%s" % reason}


# --------------------------------------------------------------------------
# limiters (section 2.2)
# --------------------------------------------------------------------------

def rung_available(state, rung: str) -> bool:
    return state["rung_attempts"].get(rung, 0) < RUNG_ATTEMPT_CAP


def next_available_rung(state, *, skip=()) -> Optional[str]:
    for r in LADDER:
        if r in skip:
            continue
        if rung_available(state, r):
            return r
    return None


def bump_rung(state, rung: str) -> dict:
    attempts = dict(state["rung_attempts"])
    attempts[rung] = attempts.get(rung, 0) + 1
    return attempts


# --------------------------------------------------------------------------
# nodes
# --------------------------------------------------------------------------

class Deps:
    """Injected primitives. Defaults are the real ones; tests pass doubles.

    Fork ruling 1 requires run(conn, task_id, workspace, ...) to be
    "offline-testable against a temp DB in the CD-025/026 harness pattern".
    That is only true if the primitives are injectable -- otherwise every test
    needs a live agent, a model and a ledger.
    """

    def __init__(self, *, gate=None, arbiter=None, model=None, parse_ac=None,
                 agent=None, delegate=None, derive=None,
                 conn=None, task_id="", workspace="", body="",
                 changed_files=None):
        self.gate = gate
        self.arbiter = arbiter
        self.model = model or call_model
        self.parse_ac = parse_ac
        self.conn = conn
        self.task_id = task_id
        self.workspace = workspace
        self.body = body
        # CD-036 (D5.5): the files this card changed, for the synthetic check
        # specs. Empty is the honest default -- it narrows coverage to the
        # AI-instruction carve-out rather than silently passing.
        self.changed_files = list(changed_files or [])
        # CD-041 (D5.2b-iv). `agent` is the LIVE AIAgent the worker already
        # holds; delegate_task REFUSES without one (tools/delegate_tool.py:
        # "delegate_task requires a parent agent context"). `delegate` and
        # `derive` are injection seams for the offline driver.
        #
        # resolve_real() deliberately does NOT bind these two. Binding
        # `delegate` there would make run() import tools.delegate_tool --
        # and therefore the whole agent -- for every offline test that
        # injects a gate double, destroying the "offline-testable against a
        # temp DB" property fork ruling 1 requires. They are bound LAZILY
        # inside the node, so a test that injects them never touches the
        # agent and a run that never reaches `implement` never imports it.
        self.agent = agent
        self.delegate = delegate
        self.derive = derive

    def resolve_real(self):
        """Bind the real primitives. Called only when defaults are needed, so
        importing this module never requires the agent to be installed."""
        if self.gate is None:
            from hermes_cli import ac_check_runner
            self.gate = ac_check_runner.gate
        if self.arbiter is None:
            from hermes_cli.spend_accounting import arbiter_decision
            self.arbiter = arbiter_decision
        if self.parse_ac is None:
            from hermes_cli.kanban_db import parse_ac_text
            self.parse_ac = parse_ac_text
        return self


def unbuilt(name: str):
    """A node whose primitive has not been read. Parks the card, never raises."""

    def _node(state):
        return {"terminal_reason": "node_not_implemented:%s" % name}

    _node.__name__ = "unbuilt_%s" % name
    _node.unbuilt = True
    return _node


def make_implement(deps: Deps):
    """Design section 1: `implement` wraps delegate_task -> local, worktree write.

    UNREACHABLE AT THIS CD. build() registers this node with its design section
    1 OUTBOUND edge (implement -> cheap_gate) and NO inbound edge, so it cannot
    be entered from START. Wiring the inbound side is the topology rewire, and
    that is where the first spend happens: delegate_task spawns a child agent
    and there is no CD-034-style free-path split for it.

    FOUR THINGS READ AT SOURCE, each of which would be wrong if recalled:

      * `parent_agent` is MANDATORY. Without it delegate_task returns
        tool_error() immediately. That is what deps.agent carries.
      * A CALLER-SUPPLIED `max_iterations` IS IGNORED -- delegate_task logs it
        at debug and substitutes delegation.max_iterations from config. This
        node therefore CANNOT bound its child's iteration budget from the call
        site, and passing the argument would only look like it could. The child
        timeout is config-side too (_get_child_timeout).
      * `background=True` IS NOT AVAILABLE HERE. async_delivery_supported() is
        false for one-shot runners -- delegate_tool names Kanban workers
        explicitly -- and the call silently falls back to synchronous execution
        with a note appended. Passing background=False states that rather than
        depending on a fallback.
      * THE CHILD DOES NOT INHERIT THE WORKTREE. _resolve_workspace_hint is
        best-effort and PROMPT-ONLY: it reads TERMINAL_CWD or parent-agent
        attributes and injects a path into the child's prompt. So the goal
        NAMES deps.workspace explicitly. Relying on inheritance would have the
        child edit some other tree, derive() return an empty diff, and the card
        park with a reason pointing at git rather than at the real cause.

    THE DIFF IS RE-DERIVED HERE, AND THAT IS NOT HOUSEKEEPING. cloud_review's
    prompt reads state["diff"] and this node writes the worktree. Without the
    re-derive, the moment the inbound edge lands, cloud_review buys a review of
    the diff as it stood BEFORE the implementer ran -- for a fresh card, the
    empty one. That is exactly the spend hazard CD-034 made structurally
    unreachable, rebuilt on the other side of the graph.

    NO PAYLOAD FROM THE CHILD EVER REACHES terminal_reason. Not the summary,
    not `error`, not an exception string. terminal_reason goes to the board via
    block_task(reason=...), and the vocabulary below is closed and
    code-authored -- same posture as cli.py's graph_invoke_failed:<TypeName>.

    NO SCHEMA CHANGE. Every field written already exists in WorkflowState, so
    SCHEMA_VERSION stays 2 and resume_refusal keeps accepting every checkpoint
    written since CD-032. Carrying the child's summary in state would have
    forced a bump; it is not worth one.
    """

    def _node(state):
        if deps.agent is None:
            return {"terminal_reason": "implement_no_agent"}

        delegate = deps.delegate
        if delegate is None:
            try:
                from tools.delegate_tool import delegate_task as delegate
            except Exception:
                # An import failure must PARK, not raise. A raise inside a node
                # kills the run and, under a checkpointer, can re-spend cloud
                # calls already paid for (CD-031 posture).
                return {"terminal_reason": "implement_delegate_unavailable"}

        # CD-042 THE ATTEMPT CAP, checked HERE rather than in route_entry so
        # the refusal carries a REASON. A router cannot write state, so a cap
        # enforced there would park the card as a bare "human_review" with
        # nothing saying why. Entering the node to refuse costs nothing: no
        # child, no subprocess, no dollars.
        if state["rung_attempts"].get("implement", 0) >= IMPLEMENT_ATTEMPT_CAP:
            return {"terminal_reason": "implement_attempt_cap"}

        # Bumped BEFORE the call, so the attempt counts even if the child never
        # returns cleanly. EVERY return below carries it -- an attempt that is
        # not recorded is an attempt that repeats forever.
        attempts = bump_rung(state, "implement")

        def _park(reason):
            return {"terminal_reason": reason, "rung_attempts": attempts}

        raw = delegate(
            goal=build_implement_goal(state, deps.workspace),
            context=build_implement_context(state),
            role="leaf",
            background=False,
            parent_agent=deps.agent,
        )

        try:
            payload = json.loads(raw) if isinstance(raw, str) else None
        except ValueError:
            payload = None
        if not isinstance(payload, dict):
            return _park("implement_unparseable")

        # A REFUSAL IS ALSO VALID JSON. tool_error() returns
        # {"error": "..."} with NO `results` key, so "it parsed" is not "it
        # ran". The absence of `results` is the discriminator.
        results = payload.get("results")
        if not isinstance(results, list):
            return _park("implement_refused")
        if not results or not isinstance(results[0], dict):
            return _park("implement_no_result")

        first = results[0]
        status = first.get("status")
        if status not in IMPLEMENT_STATUSES:
            return _park("implement_unknown_status")
        if status != IMPLEMENT_OK:
            return _park("implement_%s" % status)

        model = first.get("model")
        update = {"implementer_model": model if isinstance(model, str) else None,
                  "rung_attempts": attempts}

        derive = deps.derive
        if derive is None:
            try:
                from hermes_cli import build_graph_diff as _bgd
            except Exception:
                return _park("implement_derive_unavailable")
            derive = _bgd.derive

        verdict = derive(deps.workspace)
        if not isinstance(verdict, dict) or not verdict.get("ok"):
            reason = (verdict or {}).get("reason") if isinstance(verdict, dict) else None
            return _park(reason or "graph_no_diff_source:unknown")

        # changed_files lives on Deps, NOT in state, so this is a Deps mutation
        # and is easy to miss. Without it the next cheap_gate builds its D5.5
        # synthetic check specs over the PRE-implement file list.
        deps.changed_files = list(verdict.get("changed_files") or [])
        update["diff"] = verdict["diff"]
        update["diff_files"] = verdict["diff_files"]
        update["diff_added_lines"] = verdict["diff_added_lines"]
        return guard(update, state)

    _node.__name__ = "implement"
    return _node


def make_cheap_gate(deps: Deps):
    """ac_check_runner.gate alone (fork ruling 5).

    F3 INVARIANT: this call site passes NO evidence_hook. gate()'s signature
    defaults it to None, and graph iterations recording evidence would pollute
    the per-profile verification_evidence.db that the D4.3 stamp -- and the
    LOCKED D4.4-write enable -- depend on. Only the terminal board gate records
    evidence. Asserted by an AST test over this file.

    F2/F9: `path=` IS passed, so per-iteration records do not overwrite each
    other and the terminal record wins by construction rather than by ordering.
    """

    def _node(state):
        rec = os.path.join(
            deps.workspace,
            "ac-execution-%s-%s-%d.json" % (state["card_id"], state["component"],
                                            state["iteration"]))
        summary = deps.gate(
            state["card_id"], deps.body, deps.workspace,
            parse_fn=ck.wrap_parse_fn(deps.parse_ac, deps.workspace,
                                      deps.changed_files),
            path=rec,
        )
        return guard({"gate_summary": summary, "ac_record_path": rec,
                      "iteration": state["iteration"] + 1}, state)

    return _node


def make_cloud_review(deps: Deps, *, activity: str, node_name: str):
    """cloud_review / cloud_re_review. --mode read_only, no --cwd (note 2)."""

    def _node(state):
        if state["cloud_review_calls"] >= CLOUD_REVIEW_CAP:
            return {"terminal_reason": "cloud_review_cap"}
        prompt = build_review_prompt(state)
        out = deps.model(
            activity=activity, prompt=prompt, workspace=deps.workspace,
            card_id=state["card_id"], component=state["component"],
            directive=state["directive"], signals=selection_signals(state),
            mode="read_only")
        res = out.get("result") or {}
        update = {"cloud_review_calls": state["cloud_review_calls"] + 1,
                  "reviewer_model": res.get("model")}
        if out["klass"] != "ok":
            update["terminal_reason"] = "model_call_%s:%s" % (out["klass"], activity)
            return guard(update, state)
        update["objections_prior"] = state["objections_current"]
        update["objections_current"] = extract_objections(res)
        update["last_verdict_passed"] = verdict_passed(res.get("verdict"))
        return guard(update, state)

    _node.__name__ = node_name
    return _node


def make_classify_failure(deps: Deps):
    """Section 5.4: the FREE LOCAL `classify` activity, on the objections only.

    Two things that would be wrong if recalled rather than read:

      * The prompt receives the two objection sets AND NOTHING ELSE. That is
        what decorrelates the signal from both reviewer and implementer.
        build_classify_prompt (CD-030) enforces it; do not widen it here.
      * The local lane returns EARLY from model_call with verdict=None. A node
        reading .verdict here would get None every time and fall silently to the
        default tier -- indistinguishable from a working classifier. The answer
        is on .text.
    """

    def _node(state):
        prompt = ev.build_classify_prompt(state["objections_prior"],
                                          state["objections_current"])
        out = deps.model(activity="classify", prompt=prompt,
                         workspace=deps.workspace, card_id=state["card_id"],
                         component=state["component"], mode="none")
        res = out.get("result") or {}
        answer = (ev.parse_classify_response(res.get("text"))
                  if out["klass"] == "ok" else None)
        decided = ev.decide_recurrence(
            answer,
            implementer_model=state["implementer_model"],
            reviewer_model=state["reviewer_model"])

        holdout = ev.is_holdout(state["card_id"])
        record = ev.build_classify_record(
            card_id=state["card_id"], run_id=res.get("run_id"),
            component=state["component"],
            recurrence_tier=decided["recurrence_tier"],
            recurrence_confounded=decided["recurrence_confounded"],
            objections_prior=state["objections_prior"],
            objections_current=state["objections_current"],
            dispute_class=decided["dispute_class"],
            rung_taken=state["rung"], rung_outcome=None, holdout=holdout)
        logged = ev.append_classify_line(record)

        update = dict(decided)
        update["holdout"] = holdout
        update["classify_log_failed"] = not logged
        return guard(update, state)

    return _node


def make_fix(deps: Deps, *, rung: str, activity: str):
    """fix_rung1 / fix_rung2. --mode write, --cwd the worktree (note 2)."""

    def _node(state):
        out = deps.model(
            activity=activity, prompt=build_fix_prompt(state),
            workspace=deps.workspace, card_id=state["card_id"],
            component=state["component"], directive=state["directive"],
            signals=selection_signals(state), mode="write", cwd=deps.workspace)
        res = out.get("result") or {}
        update = {"rung": rung, "rung_attempts": bump_rung(state, rung),
                  "implementer_model": res.get("model")}
        if out["klass"] != "ok":
            update["terminal_reason"] = "model_call_%s:%s" % (out["klass"], activity)
        return guard(update, state)

    _node.__name__ = "fix_%s" % rung
    return _node


def make_fix_rung3(deps: Deps):
    """Cross-provider tie-break. AUTH DEFERRED (section 8 deferral 6).

    Present in the topology so the graph does not need reshaping when
    `codex login` is run as the sandbox account. Until then the ladder is
    local -> rung1 -> rung2 -> human, and this node says so rather than
    pretending to try.
    """

    def _node(state):
        return {"rung": "rung3", "rung_attempts": bump_rung(state, "rung3"),
                "terminal_reason": "fix_rung3_auth_deferred"}

    return _node


def node_assemble(state):
    return {"terminal_reason": "assembled"}


def node_human(state):
    return {"terminal_reason": state.get("terminal_reason") or "human_review"}


# --------------------------------------------------------------------------
# prompts + objection extraction
# --------------------------------------------------------------------------

def build_implement_goal(state, workspace: str) -> str:
    """The child's goal. NAMES THE WORKTREE -- see make_implement's docstring.

    No commit, no push, no history rewrite: build_graph_diff.derive() computes
    the card's work product as a diff against the merge-base, and its ONE
    mutating command is pinned to `git add -A -N` (record intent, stage no
    content). A child that committed would move HEAD out from under that.
    """
    return ("Implement component %r of card %s.\n\n"
            "Work in this directory and nowhere else:\n  %s\n\n"
            "Make the change on disk. Do NOT commit, push, create branches, or "
            "otherwise alter git history -- the harness derives the diff "
            "itself.\n\nPLAN:\n%s\n"
            % (state["component"], state["card_id"], workspace, state["plan"]))


def build_implement_context(state):
    """Optional context. Returns None rather than an empty string when there is
    no directive, so the child's prompt carries no empty section."""
    return state.get("directive") or None


def build_review_prompt(state) -> str:
    return ("Review the following change for component %r of card %s.\n\n"
            "PLAN:\n%s\n\nDIFF:\n%s\n"
            % (state["component"], state["card_id"], state["plan"], state["diff"]))


def build_fix_prompt(state) -> str:
    return ("Address the following review objections for component %r.\n\n"
            "OBJECTIONS:\n%s\n\nDIFF:\n%s\n"
            % (state["component"],
               json.dumps(state["objections_current"], indent=2, sort_keys=True),
               state["diff"]))


def extract_objections(result: dict) -> list:
    """Objections out of a ModelResult dict, as Objection records.

    VERDICT_CONTRACT asks for a `findings` array. Nothing validates it, items may
    legitimately be bare strings, and no locator is carried (section 5.1/5.2).
    So `raw` keeps the item EXACTLY as received -- that is the instrument section
    6.1 uses to measure how many findings voluntarily carry a path or a symbol,
    which is the only thing that would justify reopening the contract.
    """
    verdict = result.get("verdict")
    items = []
    if isinstance(verdict, dict):
        found = verdict.get("findings")
        if isinstance(found, list):
            items = found
    out = []
    for item in items:
        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            text = item.get("text") or item.get("summary") or json.dumps(
                item, sort_keys=True)
        else:
            text = str(item)
        out.append({"text": text, "review_id": result.get("run_id"), "raw": item})
    return out


# --------------------------------------------------------------------------
# EDGES (section 2.1)
# --------------------------------------------------------------------------

def route_after_gate(state, deps: Deps):
    """The arbiter edge. arbiter_decision returns a TUPLE (decision, reason).

    Read at source 2026-08-18 -- it is NOT a bare string, and taking it as one
    would compare a tuple against "escalate" and fall through to the else branch
    on every card. Its four values are the module constants ESCALATE / OVER_CAP
    / PASS / NO_ARBITER.
    """
    if state.get("terminal_reason"):
        return "human"
    decision, _reason = deps.arbiter(deps.conn, deps.task_id, state["gate_summary"])
    if decision == "pass":
        return GATE_PASS_TARGET
    if decision == "escalate":
        rung = next_available_rung(state)
        return ("fix_%s" % rung) if rung else "human"
    return "human"          # over_cap and no_arbiter both terminate (section 2.2)


def route_after_review(state):
    if state.get("terminal_reason"):
        return "human"
    if state.get("last_verdict_passed"):
        return "assemble"
    # "revise" and verdict-is-None are the SAME edge. _parse_verdict fail-closes
    # to None and the graph must never re-call to chase a parseable verdict --
    # a reviewer produces new findings each pass, so a clean verdict is not
    # reachable by re-calling.
    return "cloud_re_review"


def route_after_re_review(state):
    if state.get("terminal_reason"):
        return "human"
    return "assemble" if state.get("last_verdict_passed") else "classify_failure"


def route_after_classify(state):
    """THE ACCEPTANCE-CRITERION EDGE.

    approach-disputed (objection_recurred True) SKIPS the second same-family
    rung and goes to rung 3. Verifiable from the graph definition: it is a pure
    function of typed state, registered with an explicit path_map, and no model
    is consulted at this point.

    The holdout (section 6.3) forces a normal climb regardless, which is the
    only source of a counterfactual for the skipped rung.
    """
    if state.get("terminal_reason"):
        return "human"
    if state.get("holdout"):
        rung = next_available_rung(state)
        return ("fix_%s" % rung) if rung else "human"
    if state.get("objection_recurred") is True:
        return "fix_rung3" if rung_available(state, "rung3") else "human"
    return "fix_rung2" if rung_available(state, "rung2") else "human"


def route_entry(state):
    """START. CD-042: does this card already HAVE a work product?

    THE PREDICATE IS THE DIFF, AND IT IS A PROXY -- say so rather than pretend
    otherwise. "diff is empty" is not identical to "nothing has been built". A
    card whose change already exists upstream routes to `implement` and asks a
    child to build something already built; the child no-ops, the diff stays
    empty, and the attempt cap bounds it. That is the accepted cost of a
    predicate that is cheap, has no schema footprint, and is correct on the two
    cases that matter:

      * A FRESH card has no diff -> implement. Before CD-042 this card ran its
        AC checks against an unimplemented workspace, failed, and the arbiter
        escalated to fix_rung1 -- a METERED CLOUD CALL doing the implementing.
      * A RESUMED card HAS a diff, because run() re-derives it from the
        worktree and passes it in the partial input. So it re-enters at
        cheap_gate and does NOT re-implement. Resume correctness falls out of
        the predicate rather than needing its own flag.

    The CD-033/034/035 path -- a card created to review an EXISTING change --
    is byte-for-byte unchanged: non-empty diff, straight to cheap_gate.
    """
    if state.get("terminal_reason"):
        return "human"
    if (state.get("diff") or "").strip():
        return "cheap_gate"
    return "implement"


def route_after_implement(state):
    """Post-implement. Same shape as route_after_fix.

    Replaces CD-041's STATIC implement -> cheap_gate edge. A static edge would
    run cheap_gate even when implement had already parked the card, executing
    the AC checks for nothing before route_after_gate sent it to `human`.
    """
    if state.get("terminal_reason"):
        return "human"
    return "cheap_gate"


def route_after_fix(state):
    """Post-fix verification, or `human` when the ladder is spent."""
    if state.get("terminal_reason"):
        return "human"
    return "cheap_gate"


# --------------------------------------------------------------------------
# graph construction
# --------------------------------------------------------------------------

def build(deps: Deps, *, checkpointer=None):
    """Compile the graph. Returns (app, serde).

    `checkpointer` is an optional pre-made (saver, serde) pair. Absent, the
    D5.1 in-memory checkpointer is built exactly as before -- so every existing
    caller is unaffected and the default behaviour of this module does not
    change. run() supplies the pair when it needs a reference to the SAVER
    itself, which the (app, serde) return does not carry.
    """
    from langgraph.graph import StateGraph, START, END

    g = StateGraph(WorkflowState)

    # CD-038: the trace emitter, wired AT REGISTRATION so that no node body,
    # router, prompt builder or state field changes. The tracer's wrappers are
    # transparent -- they return the wrapped callable's value unchanged and
    # RE-RAISE its exceptions -- and with HERMES_GRAPH_TRACE=0 every wrapper is
    # the identity function, so the callables registered below are the SAME
    # OBJECTS as before this CD.
    #
    # deps.model is wrapped because run_id is a call_model RETURN VALUE, not a
    # workflow_state field. Getting the ledger join key any other way would
    # mean a state schema change; it does not need one. The wrap is idempotent
    # by marker, so a second build() on the same Deps cannot nest it.
    tracer = tl.Tracer(thread_id=getattr(deps, "task_id", "") or "")
    deps.tracer = tracer
    deps.model = tracer.model(deps.model)

    def _add(name, fn):
        g.add_node(name, tracer.node(name, fn))

    def _cond(source, fn, path_map):
        g.add_conditional_edges(source, tracer.edge(source, fn), path_map)

    _add("cheap_gate", make_cheap_gate(deps))
    _add("cloud_review", make_cloud_review(
        deps, activity="code_review", node_name="cloud_review"))
    _add("cloud_re_review", make_cloud_review(
        deps, activity="re_review", node_name="cloud_re_review"))
    _add("classify_failure", make_classify_failure(deps))
    _add("fix_rung1", make_fix(deps, rung="rung1", activity="fix_sonnet"))
    _add("fix_rung2", make_fix(deps, rung="rung2", activity="fix_opus"))
    _add("fix_rung3", make_fix_rung3(deps))
    _add("assemble", node_assemble)
    _add("human", node_human)

    # Unbuilt: wired, honest, terminal-to-human.
    for name in ("plan", "plan_review", "local_review"):
        _add(name, unbuilt(name))
        g.add_edge(name, "human")

    # CD-041: `implement` is BUILT. Its OUTBOUND edge is the design section 1
    # edge (implement -> cheap_gate). Its INBOUND edge is DELIBERATELY ABSENT:
    # START goes to cheap_gate, and the only router that could name this node
    # is route_after_gate, whose path_map does not contain it and whose four
    # arbiter values are pass / escalate / over_cap / no_arbiter. So the node
    # is unreachable from START and this CD is INERT IN PRODUCTION -- a
    # property of the compiled graph, asserted by the driver rather than
    # claimed here.
    #
    # The outbound edge is cheap_gate rather than human because whether
    # langgraph compiles a node with NO outgoing edge is not something this CD
    # needs to find out by guessing. Giving it the edge it will keep removes
    # the question and leaves the rewire as inbound-side work only.
    _add("implement", make_implement(deps))
    _cond("implement", route_after_implement,
          {"cheap_gate": "cheap_gate", "human": "human"})

    # CD-042: conditional entry. NOTE FOR ANY FUTURE TOPOLOGY ASSERTION --
    # a START branch registers under builder.branches["__start__"] and does
    # NOT appear in builder.edges. An assertion that looks only at
    # builder.edges will report START as unwired. Verified against
    # langgraph 1.2.10.
    _cond(START, route_entry,
          {"implement": "implement", "cheap_gate": "cheap_gate",
           "human": "human"})

    _cond(
        "cheap_gate", lambda s: route_after_gate(s, deps),
        {"cloud_review": "cloud_review", "local_review": "local_review",
         "fix_rung1": "fix_rung1", "fix_rung2": "fix_rung2",
         "fix_rung3": "fix_rung3", "human": "human"})

    _cond(
        "cloud_review", route_after_review,
        {"assemble": "assemble", "cloud_re_review": "cloud_re_review",
         "human": "human"})

    _cond(
        "cloud_re_review", route_after_re_review,
        {"assemble": "assemble", "classify_failure": "classify_failure",
         "human": "human"})

    _cond(
        "classify_failure", route_after_classify,
        {"fix_rung1": "fix_rung1", "fix_rung2": "fix_rung2",
         "fix_rung3": "fix_rung3", "human": "human"})

    for fix in ("fix_rung1", "fix_rung2", "fix_rung3"):
        _cond(
            fix, route_after_fix,
            {"cheap_gate": "cheap_gate", "human": "human"})

    g.add_edge("assemble", END)
    g.add_edge("human", END)

    if checkpointer is None:
        saver, serde = sz.make_checkpointer()
    else:
        saver, serde = checkpointer
    return g.compile(checkpointer=saver), serde


def run(conn, task_id, workspace, *, body="", component="main", plan="", diff="",
        diff_files=0, diff_added_lines=0, changed_files=None,
        directive=None, deps=None, thread_id=None, recursion_limit=40,
        agent=None,
        checkpoint="memory", resume=False):
    """Entry point (fork ruling 1: worker-side, offline-testable).

    NO HTTP SURFACE, structurally (section 4.1). This module exposes none.

    checkpoint="memory"     D5.1 behaviour. InMemorySaver, no durability. This
                            is the DEFAULT, so nothing that calls run() today
                            changes behaviour.
              "workspace"   CD-032. Persisted under the card workspace.

    resume=True adopts an existing checkpoint for this thread. It requires
    checkpoint="workspace", and it is OPT-IN rather than automatic: thread_id
    defaults to "<task_id>:<component>", which is STABLE ACROSS DISPATCHES, so
    an automatic resume would silently re-enter a card that already finished.
    Explicit opt-in is the same posture as CD-028's Ken-only routing column.

    RESUME SEMANTICS -- verified on the box 2026-08-19, and the two cases are
    not the same:

      * INTERRUPTED thread (a node raised): invoke(None) CONTINUES at the
        crashed node; completed supersteps are not repeated.
      * TERMINAL thread (reached END): a PARTIAL input RE-ENTERS FROM START
        with the named channels merged over the restored ones, and channels
        not named survive untouched.

    This function takes the SECOND form deliberately. `plan` and `diff` are
    RE-DERIVED by the caller, which holds the worktree, and passed as the
    partial input; control state (rung, rung_attempts, cloud_review_calls, the
    objection sets, the recurrence fields) comes off the checkpoint. That is
    correct independently of the sanitizer being lossy: the worktree is the
    truth and a checkpoint is a stale copy of it, so restoring a diff from a
    checkpoint older than the worktree is wrong even with perfect fidelity.

    Re-entering at cheap_gate with rung_attempts preserved is precisely what
    makes the section 2.2 rung caps bind ACROSS dispatches instead of resetting
    on every one.
    """
    deps = deps or Deps(conn=conn, task_id=task_id, workspace=workspace, body=body)
    deps.conn, deps.task_id, deps.workspace, deps.body = conn, task_id, workspace, body
    # CD-041: default None, so every existing caller -- cli.py included --
    # is byte-for-byte unaffected. Only an explicit agent= populates it.
    if agent is not None:
        deps.agent = agent
    deps.changed_files = list(changed_files or [])
    if deps.gate is None or deps.arbiter is None or deps.parse_ac is None:
        deps.resolve_real()

    if checkpoint not in ("memory", "workspace"):
        raise ValueError(
            "checkpoint must be 'memory' or 'workspace', got %r. This is a "
            "caller error at entry, before any card state exists, so it raises "
            "rather than parking." % (checkpoint,))

    if resume == "auto" and checkpoint != "workspace":
        raise ValueError(
            'resume="auto" requires checkpoint="workspace", got %r. Same '
            "posture as the validation above: a caller error at entry, "
            "before any card state exists, so it raises rather than "
            "parking -- there is no card state to lose yet." % (checkpoint,))

    tid = thread_id or "%s:%s" % (task_id, component)
    cfg = {"configurable": {"thread_id": tid},
           "recursion_limit": recursion_limit}

    ckmod = None
    pair = None
    if checkpoint == "workspace":
        from hermes_cli import build_graph_checkpoint as _ck
        ckmod = _ck
        pair = ckmod.make_workspace_checkpointer(workspace)

    app, serde = build(deps, checkpointer=pair)

    def _parked(reason):
        """A refusal PARKS the card; it does not raise.

        Same posture as an unbuilt node (CD-031): a raise under a checkpointer
        loses state and may re-spend cloud calls already paid for, while a
        parked card is bounded and visible to a human.
        """
        parked = new_workflow_state(task_id, component, plan=plan, diff=diff,
                                    diff_files=diff_files,
                                    diff_added_lines=diff_added_lines,
                                    directive=directive)
        parked["terminal_reason"] = reason
        return parked

    def _invoke(payload):
        try:
            if ckmod is None:
                return app.invoke(payload, cfg)
            try:
                return app.invoke(payload, cfg)
            except ckmod.CheckpointTooLarge:
                # A checkpointer is not a node and has no edges, so it cannot
                # route. This is the only place the size refusal can become a
                # parked card instead of a crashed run.
                return _parked("checkpoint_too_large")
        finally:
            # CD-038: one summary line per invocation, carrying the trace's own
            # dropped-line count. In a FINALLY deliberately -- it is written
            # even when a node raises, and a crashed run is exactly when the
            # trace is worth having. A dropped line that is not counted is
            # indistinguishable from a node that never ran.
            _tracer = getattr(deps, "tracer", None)
            if _tracer is not None:
                _tracer.summary()

    if resume == "auto":
        # CD-040. The criterion is that the worker must not be TOLD to
        # look, so the presence of a checkpoint for THIS thread is the
        # whole decision, and run() already holds the saver.
        #
        # This cannot re-enter a finished card. complete_task removes the
        # workspace, and checkpoint_root is a managed descendant of it, so
        # the checkpoint goes with it (checkpoint_root docstring). A card
        # that finished has nothing to resume from; a card that PARKED
        # keeps both, and is exactly the card whose rung_attempts must
        # carry. RUNG_ATTEMPT_CAP is 1 -- without this the ladder resets
        # every dispatch and the cap does almost no work.
        resume = bool(pair[0].has_thread(tid))

    if resume:
        if checkpoint != "workspace":
            return _parked("resume_requires_workspace_checkpoint")
        refusal = ckmod.resume_refusal(pair[0], tid, schema_version=SCHEMA_VERSION)
        if refusal:
            return _parked(refusal)
        # Partial input: re-derived work product only. Everything else is
        # restored from the checkpoint.
        return _invoke({"plan": plan, "diff": diff,
                        "diff_files": diff_files,
                        "diff_added_lines": diff_added_lines,
                        "terminal_reason": None,
                        "halt_reason": None})

    # new_workflow_state populates EVERY field, including the three CD-031
    # additions. No field is patched in afterwards: a partially-populated
    # TypedDict defeats the point of pinning the schema, and LangGraph silently
    # drops updates naming keys that are not channels.
    state = new_workflow_state(task_id, component, plan=plan, diff=diff,
                               diff_files=diff_files,
                               diff_added_lines=diff_added_lines,
                               directive=directive)
    validate(state)
    return _invoke(state)
