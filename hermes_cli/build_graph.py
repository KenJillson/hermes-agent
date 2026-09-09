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
  built     implement -- owned local subprocess for empty/whitespace diffs;
            existing terminal reasons still park at human; see route_entry().

The line is not arbitrary. `plan` and `local_review` still need work-product
contracts that do not exist -- a plan and a verdict -- and fix_rung3's real
body wraps the still-unread `terminal()` primitive. A1 replaces implement's
former in-process `delegate_task` call with an owned subprocess, but does not
invent either missing deliverable or widen fix_rung3.

The unbuilt nodes are present in the topology with their edges wired.

A1 ENABLES THE OWNED IMPLEMENTER AT EMPTY-DIFF ENTRY. START remains a
conditional edge: a card with a diff goes straight to cheap_gate, while an
empty or whitespace-only diff enters implement. A prior terminal reason
always parks at human. Successful implementation re-derives the work product
before the cheap gate; failed implementation parks without entering the gate.

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

# Closed A1 parent-runner vocabulary. Unknown values park without copying any
# child-controlled payload into the board reason.
IMPLEMENT_STATUSES = (
    "completed", "failed", "timeout", "interrupted", "protocol_error",
    "exec_error",
)

# The ONE value that means the child did the work. summary present, not
# interrupted, and not the "(empty)" sentinel run_agent.py emits when it
# gives up after repeated empty-LLM-response retries.
IMPLEMENT_OK = "completed"

# CD-042: how many times `implement` may run for one card, ACROSS
# DISPATCHES. Separate from RUNG_ATTEMPT_CAP, which governs the fix ladder
# and does not apply here.
#
# When `implement` is reachable, without a cap the loop is UNBOUNDED: a failed
# child leaves the diff empty, so the next dispatch routes to it again. The
# board's BLOCK_RECURRENCE_LIMIT breaker would eventually send the card to
# triage, but that is a backstop for a misbehaving card, not a budget for
# this node.
#
# The count lives in the EXISTING `rung_attempts` dict channel, so there is
# NO SCHEMA CHANGE -- SCHEMA_VERSION stays 2 and every checkpoint written
# since CD-032 still resumes. LADDER does not contain "implement", so
# next_available_rung ignores the key and the fix ladder is unaffected.
IMPLEMENT_ATTEMPT_CAP = 1

# A1 owns both budgets at the graph boundary.  These do not consult the
# delegation config and therefore cannot be widened by a card or child.
IMPLEMENT_MAX_ITERATIONS = 50
IMPLEMENT_TIMEOUT_SECONDS = 900

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
                 agent=None, implement_runner=None, derive=None, over_cap=None,
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
        # A1. `agent` is retained only as an interrupt source for the owned
        # subprocess. `implement_runner` and `derive` are injection seams for
        # offline behavior tests.
        #
        # resolve_real() deliberately does NOT bind these two. Binding
        # `delegate` there would make run() import tools.delegate_tool --
        # and therefore the whole agent -- for every offline test that
        # injects a gate double, destroying the "offline-testable against a
        # temp DB" property fork ruling 1 requires. They are bound LAZILY
        # inside the node, so a test that injects them never touches the
        # agent and a run that never reaches `implement` never imports it.
        self.agent = agent
        self.implement_runner = implement_runner
        self.derive = derive
        # CD-045: the per-card dollar ceiling predicate. Same seam shape and
        # same reason as the two above -- bound LAZILY inside the gate helper
        # so that resolve_real() never makes importing this module require a
        # live ledger, an ssh path, or a board connection.
        self.over_cap = over_cap

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
    """Design section 1: run one owned local subprocess, then derive its diff.

    START enters this node for empty or whitespace-only diffs without a prior
    terminal reason. Its outbound conditional route enters cheap_gate only
    after successful implementation and diff derivation; failures park at human.

    The runner owns PID/PGID, cwd, prompt-file transport, iteration and wall
    budgets, provider-locality checks, interrupt propagation, and mandatory
    reap. The graph accepts only its closed status vocabulary and never copies
    child output or exception text into terminal_reason.

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
        # CD-044: FAIL CLOSED ON AN EMPTY SPECIFICATION.
        #
        # A child given no statement of intent does NOT no-op. Measured on
        # t_6ea5ea7f 2026-08-21: given a worktree containing a two-line README,
        # it wrote a Hello World main.py. That satisfies derive(), makes
        # route_entry's diff predicate true, and routes an INVENTION onward as
        # a legitimate work product -- on a card with a ## AC block, into the
        # metered region, where it is paid for. A no-op would have parked
        # visibly; this failed OPEN.
        #
        # Placed with the no-agent park above, and ABOVE the attempt cap,
        # deliberately: this cannot succeed on retry -- the card body will not
        # have grown -- so it must not consume an attempt. No child, no
        # subprocess, no dollars.
        spec = (deps.body or "").strip()
        if not spec:
            return {"terminal_reason": "implement_no_spec"}

        runner = deps.implement_runner
        if runner is None:
            try:
                from hermes_cli.build_graph_implementer import run_implementer as runner
            except Exception:
                return {"terminal_reason": "implement_runner_unavailable"}

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

        def _interrupted():
            agent = deps.agent
            if agent is None:
                return False
            if getattr(agent, "_interrupt_requested", False) is True:
                return True
            event = getattr(agent, "_hard_interrupt_requested", None)
            return bool(event is not None and hasattr(event, "is_set") and event.is_set())

        raw = runner(
            goal=build_implement_goal(state, deps.workspace, spec),
            workspace=deps.workspace,
            max_iterations=IMPLEMENT_MAX_ITERATIONS,
            timeout_seconds=IMPLEMENT_TIMEOUT_SECONDS,
            interrupt_check=_interrupted,
        )
        if not isinstance(raw, dict):
            return _park("implement_protocol_error")
        status = raw.get("status")
        if status not in IMPLEMENT_STATUSES:
            return _park("implement_unknown_status")
        if status != IMPLEMENT_OK:
            return _park("implement_%s" % status)

        identity = raw.get("identity")
        model = identity.get("model") if isinstance(identity, dict) else None
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


def spend_gate(deps: Deps, state):
    """CD-045. Consult the per-card dollar ceiling BEFORE a metered call.

    Returns a terminal update when the card must not spend, else None.

    WHY IN THE NODE AND NOT AT A ROUTER. A router cannot write state, so a
    refusal enforced there parks the card as a bare human_review with nothing
    saying why. CD-042 already ruled this exact trade for the implement cap:
    entering the node to refuse costs nothing -- no call, no subprocess, no
    dollars -- and the refusal carries a REASON.

    WHY NOT arbiter_decision. That primitive is AC-VERDICT-SHAPED: it reads
    gate_summary and returns PASS on `all_pass` BEFORE it ever consults the
    ledger. Called mid-walk on a card that passed its gate it would return PASS
    every time -- a check that structurally CANNOT FIRE, and one that would
    look correct in review. The ceiling predicate is called directly instead.

    FAILS CLOSED, per the 2026-08-20 ruling. The ledger is read over ssh to
    another host, so transport failure is routine rather than exotic, and an
    unreadable ledger means the ceiling cannot be enforced. The failure of the
    control that exists to prevent spending must not itself be a decision to
    spend. The two refusals carry DIFFERENT reasons because they mean different
    things to whoever reads the parked card: one says the card is out of money,
    the other says nobody knows.

    THE FREE LOCAL LANE IS NOT GATED, and that asymmetry is the point. The
    classify activity resolves to the local lane and costs nothing; gating it
    would make a provably free path depend on an ssh to another host and, under
    the rule above, park free cards on a transport failure. An AST assertion
    over this module pins which call sites carry this gate and which do not.
    """
    cap = deps.over_cap
    if cap is None:
        try:
            from hermes_cli import spend_accounting as _sa
        except Exception:
            return {"terminal_reason": "spend_cap_unavailable"}
        cap = _sa.over_spend_cap
    try:
        blocked = cap(deps.conn, deps.task_id)
    except Exception as exc:
        return {"terminal_reason": "spend_unknown:%s" % type(exc).__name__}
    if blocked:
        return {"terminal_reason": "over_spend_cap"}
    return None


def make_cloud_review(deps: Deps, *, activity: str, node_name: str):
    """cloud_review / cloud_re_review. --mode read_only, no --cwd (note 2)."""

    def _node(state):
        if state["cloud_review_calls"] >= CLOUD_REVIEW_CAP:
            return {"terminal_reason": "cloud_review_cap"}
        # After the count cap deliberately: that check is free, this one costs
        # a ledger read over ssh.
        refusal = spend_gate(deps, state)
        if refusal:
            return refusal
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


FIX_CONTRACT = (
    "\n\n=== RESPONSE CONTRACT (mandatory) ===\n"
    "Do NOT edit any file yourself. You have no write access and no working "
    "directory; the harness applies your answer on the machine that owns the "
    "worktree.\n"
    "Your FINAL message must be EXACTLY ONE JSON object and nothing else - no "
    "prose before or after it.\n"
    'The object MUST have a "files" key whose value is a NON-EMPTY array. '
    "Each element MUST be an object with exactly two keys:\n"
    '  "path"    - the file\'s path exactly as it appears in the DIFF above\n'
    '  "content" - that file\'s COMPLETE new contents. NOT a patch, NOT an '
    "excerpt, NOT an elision such as \"... unchanged ...\". The harness "
    "replaces the whole file with this string.\n"
    "Return ONLY the files you are changing. A path that does not appear in "
    "the DIFF above is REFUSED and the card parks unfixed.\n")


def _parse_fix(text):
    """Extract the fix envelope from a model's free-form final message.

    TWO CANDIDATES, deliberately: the whole stripped text, then the span from
    the first '{' to the last '}'. The second covers a fenced ```json block
    and any preamble the model adds, without needing `re`.

    Fails closed: anything that is not a dict returns None, and the caller
    parks. lib/model_call/core.py's _parse_verdict does the same job for
    reviews; it is NOT reused because the agent venv cannot import model_call
    (measured, CD-045: ModuleNotFoundError, and the sys.path workaround was
    rejected on CD-034 grounds).
    """
    if not isinstance(text, str):
        return None
    cands = [text.strip()]
    i, j = text.find("{"), text.rfind("}")
    if 0 <= i < j:
        cands.append(text[i:j + 1])
    for c in cands:
        if not c:
            continue
        try:
            d = json.loads(c)
        except ValueError:
            continue
        if isinstance(d, dict):
            return d
    return None


def apply_fix(workspace, text, allowed):
    """Apply a fix response to the card's worktree. Returns (written, reason).

    `reason` is None on success and a CLOSED, CODE-AUTHORED terminal reason
    otherwise. NEVER RAISES, and NO MODEL PAYLOAD EVER REACHES IT -- same
    posture as make_implement, because terminal_reason goes to the board via
    block_task(reason=...).

    THIS IS THE GRAPH'S SECOND WRITE INTO THE CARD'S WORKTREE AND THE ONLY ONE
    THAT WRITES FILE CONTENTS. build_graph_diff holds the first -- `git add -A
    -N`, pinned by an AST guard to exactly that argv, with a docstring saying
    the exception must stay one command wide rather than become a precedent.
    This earns the same narrowness:

      * A returned path must ALREADY BE IN THE CARD'S WORK PRODUCT (`allowed`
        is deps.changed_files, which comes from derive()). A fix that wants to
        create a NEW file is REFUSED and parks visibly. This rung exists to
        address objections about the change under review; a cloud model that
        can create arbitrary paths in a git worktree is a different capability
        needing a different ruling.
      * The realpath must stay inside the workspace, so a symlink already in
        the work product cannot be used to write outside it.
      * WHOLE FILES, never patches. A model-authored diff that does not apply
        is an ambiguity, and ambiguity while holding a write bit means
        guessing. A whole file is written or refused.

    ALL-OR-NOTHING. Every path is validated, then every original is read into
    memory, and only then is anything written; a failure part-way restores
    every file already written. A half-applied fix is worse than a refused one
    -- the gate would then run against a state no one authored.
    """
    payload = _parse_fix(text)
    if payload is None:
        return 0, "fix_unparseable"
    files = payload.get("files")
    if not isinstance(files, list):
        return 0, "fix_no_files"
    if not files:
        # An empty array is a well-formed way of saying "I changed nothing",
        # which is not a fix. Distinguished from fix_no_files so the parked
        # card says which one happened.
        return 0, "fix_empty_files"
    allowed_set = set(allowed or ())
    if not allowed_set:
        # Nothing is in scope, so nothing can be legally written. Fail closed
        # rather than widen the scope to "anything in the workspace".
        return 0, "fix_no_work_product"

    root = os.path.realpath(workspace)
    planned = []
    seen = set()
    for item in files:
        if not isinstance(item, dict):
            return 0, "fix_bad_item"
        rel = item.get("path")
        content = item.get("content")
        if not isinstance(rel, str) or not rel:
            return 0, "fix_bad_path"
        if not isinstance(content, str):
            return 0, "fix_content_not_text"
        if rel in seen:
            # Two entries for one path: the later would silently win.
            return 0, "fix_duplicate_path"
        seen.add(rel)
        if os.path.isabs(rel) or ".." in rel.replace("\\", "/").split("/"):
            return 0, "fix_unsafe_path"
        if rel not in allowed_set:
            return 0, "fix_path_not_in_work_product"
        full = os.path.realpath(os.path.join(root, rel))
        if full != root and not full.startswith(root + os.sep):
            return 0, "fix_path_outside_workspace"
        if not os.path.isfile(full):
            return 0, "fix_path_not_a_file"
        planned.append((full, content))

    originals = []
    changed = 0
    for full, content in planned:
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as fh:
                before = fh.read()
        except OSError:
            return 0, "fix_read_failed"
        originals.append((full, before))
        if before != content:
            changed += 1
    if not changed:
        # The model returned the file unchanged. Writing would produce an
        # identical diff, the gate would fail identically, and the next rung
        # would fire -- burning the ladder on a no-op. Park instead.
        return 0, "fix_no_change"

    done = []
    for full, content in planned:
        try:
            with open(full, "w", encoding="utf-8") as fh:
                fh.write(content)
        except OSError:
            for prev_full, prev_text in originals:
                if prev_full in done:
                    try:
                        with open(prev_full, "w", encoding="utf-8") as fh:
                            fh.write(prev_text)
                    except OSError:
                        # Restoration itself failed. Say so distinctly: the
                        # worktree is now in a state nobody authored and a
                        # human must look.
                        return 0, "fix_restore_failed"
            return 0, "fix_write_failed"
        done.append(full)
    return changed, None


def make_fix(deps: Deps, *, rung: str, activity: str):
    """fix_rung1 / fix_rung2. PROMPT-ONLY; the harness applies the answer.

    CD-051. THIS NODE USED TO SEND A LOCAL PATH TO ANOTHER MACHINE. It passed
    the card's Jetson worktree as the remote working directory, into a
    parameter lib/model_call/cli.py documents as "remote working dir for
    lane_a (a disposable worktree)". `cd` therefore failed on
    coder@10.10.40.2, no model ran, and every rung died at
    model_call_failed:<activity> for $0.00. Measured 2026-08-26 on
    t_ae6332d7: rc 5, which is _STAGE_RC["invoke"], retryable False.

    THE D5.1 DESIGN'S NOTE 2 SAID SO, AND THIS FUNCTION CITED IT WHILE
    BREAKING IT: "--cwd is a remote worktree path, never a local one, and is
    passed only on write-mode nodes. Review nodes take their source in the
    prompt and pass no --cwd." The remote worktree was never built. Ruled
    2026-08-26: do not build it -- shipping files to the other host invents a
    SECOND EGRESS PATH the sanitizer never sees (it runs on prompts, in
    core.model_call, not on transferred files), which is the objection that
    also keeps LangSmith disabled. Note 2's second sentence is generalized to
    fix nodes instead; its first is retired, in the D5.1 rulings addendum.

    THE ANSWER WAS ALREADY ARRIVING HOME AND THIS NODE THREW IT AWAY.
    lanes.lane_a_call writes the envelope to
    <workspace>/<card_id>-<run_id>-result.json -- `workspace` is a SEPARATE
    PARAMETER from `cwd` and it is the Jetson path -- and core.model_call
    returns that envelope's .result as ModelResult.text. call_model parks the
    whole as_dict() in out["result"]. So res["text"] was in this node's hands
    the entire time; it read only res["model"]. Option A needed no transport
    built, which is why it was ruled over shipping the worktree.

    THE RE-DERIVE IS NOT HOUSEKEEPING. make_implement's docstring makes the
    identical argument for the identical reason: cloud_review's prompt reads
    state["diff"], so a node that writes the worktree and does not re-derive
    hands the next reviewer the PREVIOUS work product. Without it the ladder
    would gate unchanged code and burn every remaining rung against it -- the
    silent failure that made repairing --cwd alone the wrong fix.

    NO SCHEMA CHANGE. Every field written here already exists in
    WorkflowState -- the same set make_implement writes -- so SCHEMA_VERSION
    stays 2 and every checkpoint since CD-032 still resumes. The count of
    files written is deliberately NOT recorded in state; it would have forced
    a bump and the trace already carries the node's span.
    """

    def _node(state):
        refusal = spend_gate(deps, state)
        if refusal:
            return refusal
        out = deps.model(
            activity=activity, prompt=build_fix_prompt(state),
            workspace=deps.workspace, card_id=state["card_id"],
            component=state["component"], directive=state["directive"],
            signals=selection_signals(state))
        res = out.get("result") or {}
        update = {"rung": rung, "rung_attempts": bump_rung(state, rung),
                  "implementer_model": res.get("model")}
        if out["klass"] != "ok":
            update["terminal_reason"] = "model_call_%s:%s" % (out["klass"], activity)
            return guard(update, state)

        # The attempt is already recorded above, so a refusal here still burns
        # the rung. An attempt that is not recorded is an attempt that repeats.
        written, reason = apply_fix(deps.workspace, res.get("text"),
                                    deps.changed_files)
        if reason:
            update["terminal_reason"] = reason
            return guard(update, state)

        # Same lazy seam, same reason, as make_implement: binding derive in
        # resolve_real() would make importing this module require the diff
        # module for every offline test that injects a gate double.
        derive = deps.derive
        if derive is None:
            try:
                from hermes_cli import build_graph_diff as _bgd
            except Exception:
                update["terminal_reason"] = "fix_derive_unavailable"
                return guard(update, state)
            derive = _bgd.derive

        verdict = derive(deps.workspace)
        if not isinstance(verdict, dict) or not verdict.get("ok"):
            why = (verdict or {}).get("reason") if isinstance(verdict, dict) else None
            update["terminal_reason"] = why or "graph_no_diff_source:unknown"
            return guard(update, state)

        # changed_files lives on Deps, NOT in state, so this is a Deps mutation
        # and is easy to miss -- make_implement's comment says the same. Without
        # it the next cheap_gate builds its D5.5 synthetic check specs over the
        # PRE-fix file list, and apply_fix's scope set would be stale too.
        deps.changed_files = list(verdict.get("changed_files") or [])
        update["diff"] = verdict["diff"]
        update["diff_files"] = verdict["diff_files"]
        update["diff_added_lines"] = verdict["diff_added_lines"]
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

def build_implement_goal(state, workspace: str, spec: str) -> str:
    """The child's goal. NAMES THE WORKTREE -- see make_implement's docstring.

    No commit, no push, no history rewrite: build_graph_diff.derive() computes
    the card's work product as a diff against the merge-base, and its ONE
    mutating command is pinned to `git add -A -N` (record intent, stage no
    content). A child that committed would move HEAD out from under that.

    CD-044: `spec` IS THE CARD BODY, carried on Deps by the caller. It is NOT
    the `plan` channel, and the four reasons are worth keeping because the
    obvious fix is the other one:

      * PROVENANCE. `plan` is the (unbuilt) `plan` node's output slot -- this
        module's own docstring lists it under `unbuilt`. Filling it from the
        card body destroys the ability to tell "no plan node ran" from "a plan
        node ran and produced this", permanently, in checkpointed state.
      * COUPLING THROUGH A PAID PROMPT. build_review_prompt reads that same
        channel. Routing the body through it would mean every future change to
        what the FREE implementer sees silently changes what a METERED reviewer
        is sent.
      * THE RESUME CLOBBER. run()'s resume partial input NAMES that channel,
        and cli.py always passes "". A terminal-thread resume merges named
        channels over the restored ones, so a plan threaded at the call site is
        erased on the second dispatch. The card body on Deps is re-set from the
        live card row on every run() call and is never checkpointed, so it
        cannot have that failure.
      * CHECKPOINT BYTES. `plan` is a checkpointed channel and card bodies are
        unbounded operator text. CD-043 was a checkpointed channel compounding
        past MAX_DIFF_BYTES; run() already has a CheckpointTooLarge park.

    THIS IS THE INTERIM CONTRACT, NOT THE END STATE. The card body is a
    specification, not a plan. When the `plan` node is built it slots ABOVE
    this and the body becomes the fallback. Said here so a future session
    does not discover the layering by surprise.
    """
    return ("Implement component %r of card %s.\n\n"
            "Work in this directory and nowhere else:\n  %s\n\n"
            "Make the change on disk. Do NOT commit, push, create branches, or "
            "otherwise alter git history -- the harness derives the diff "
            "itself.\n\nSPECIFICATION:\n%s\n"
            % (state["component"], state["card_id"], workspace, spec))


def build_implement_context(state):
    """Optional context. Returns None rather than an empty string when there is
    no directive, so the child's prompt carries no empty section."""
    return state.get("directive") or None


def build_review_prompt(state) -> str:
    return ("Review the following change for component %r of card %s.\n\n"
            "PLAN:\n%s\n\nDIFF:\n%s\n"
            % (state["component"], state["card_id"], state["plan"], state["diff"]))


MAX_FAILED_CHECKS_IN_PROMPT = 6
MAX_CHECK_OUTPUT_CHARS = 800


def failing_checks(state) -> list:
    """The gate's FAILING check verdicts, or []. Never raises.

    WHY THIS EXISTS -- MEASURED, NOT SUPPOSED. A fix rung reached from a gate
    `exception` has NEVER seen a cloud review, and `make_cloud_review` is the
    only thing that writes `objections_current`. graph-trace.jsonl seq 7,
    card t_ae6332d7, 2026-08-26:

        node fix_rung1  objections_current_len 0  objections_prior_len 0
                        gate {verdict: exception, checks_failed: 2}

    with the control on the same file, seq 15, t_52340522: objections
    3 and 2 on a card that DID review. So the zero is real and not a missing
    field. build_fix_prompt rendered `OBJECTIONS: []` and the rung was asked to
    repair something it was never told about. CD-051 fixed how a fix RETURNS;
    this is the same rung's INPUT.

    WHY THE RECORD AND NOT state["gate_summary"]. The summary carries COUNTS
    only -- checks_failed, checks_passed, verdict -- because `gate()` returns
    `summary` and discards the per-check `verdicts` list. The detail lives in
    the durable record `persist_execution` writes, whose path CD-A (CD-027) put
    into state as `ac_record_path` precisely so per-iteration records do not
    overwrite each other. Confirmed as a live state channel in the same trace
    line above, not inferred.

    NO SCHEMA CHANGE. Nothing new is written to state; this only READS a channel
    that has existed since CD-027, so SCHEMA_VERSION stays 2.

    FAILS SOFT, DELIBERATELY. A missing or malformed record must not park a card
    that has a real diff and a real objection to fix -- the caller renders a
    short "detail unavailable" note instead. Same posture as CD-036's
    changed_files best-effort: losing the detail narrows the prompt, it does not
    invalidate it.
    """
    path = state.get("ac_record_path")
    if not isinstance(path, str) or not path:
        return []
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            rec = json.load(fh)
    except (OSError, ValueError):
        return []
    if not isinstance(rec, dict):
        return []
    out = []
    for v in (rec.get("verdicts") or []):
        if not isinstance(v, dict):
            continue
        if v.get("kind") != "check":
            continue
        # 'failed' AND 'unrunnable' both drive the gate to `exception`
        # (run_ac_checks sets the verdict on either), so both are things the
        # fixer needs to see. An unrunnable check is usually a BROKEN check
        # rather than broken code, and saying so is what lets the model decline
        # to "fix" it instead of inventing a change.
        if v.get("status") in ("failed", "unrunnable"):
            out.append(v)
    return out


def render_failing_checks(verdicts) -> str:
    """Render failing verdicts for the fix prompt. Pure; no I/O."""
    if not verdicts:
        return ""
    shown = verdicts[:MAX_FAILED_CHECKS_IN_PROMPT]
    lines = ["FAILING ACCEPTANCE CHECKS (from the gate that escalated this "
             "card):"]
    for v in shown:
        tail = v.get("stdout_tail") or ""
        if len(tail) > MAX_CHECK_OUTPUT_CHARS:
            tail = tail[-MAX_CHECK_OUTPUT_CHARS:]
        lines.append("")
        lines.append("  check %s: %s" % (v.get("index"), v.get("status")))
        lines.append("    command:   %s" % (v.get("command"),))
        lines.append("    expected:  %s:%s" % (v.get("expect_kind"),
                                               v.get("expect_value")))
        lines.append("    exit code: %s" % (v.get("exit_code"),))
        if v.get("run_error"):
            lines.append("    run error: %s" % (v.get("run_error"),))
        lines.append("    output:    %s" % (tail.strip() or "(none)",))
    dropped = len(verdicts) - len(shown)
    if dropped > 0:
        # NEVER A SILENT CAP. CD-036's lesson: a gate that quietly stopped
        # covering things reads as "covered everything" in the record.
        lines.append("")
        lines.append("  [%d further failing check(s) not shown]" % dropped)
    return "\n".join(lines) + "\n"


def build_fix_prompt(state) -> str:
    """Objections + the diff + the gate's failing checks, plus the contract.

    THE RESPONSE CONTRACT LIVES HERE, IN THIS REPO, and the choice was made
    against the obvious alternative. core.py appends its verdict contract to
    every review prompt inside the primitive because -- its own comment -- "five
    live cards produced six verdict vocabularies ... worker prompt discipline
    cannot pin the contract, so the primitive does". That argument is about MANY
    authors writing prompts. This is ONE code-authored function, and the parser
    that reads the response sits just above it. Decisively: the agent venv
    CANNOT import model_call (CD-045 measured ModuleNotFoundError on the box,
    and the sys.path workaround was rejected on CD-034 grounds), so a contract
    in core.py with a parser here would put ONE format under TWO owners across a
    subprocess boundary -- the CD-036/037 marker defect.

    CD-052 ADDS THE FAILING CHECKS, AND THE REASON IS MEASURED. A rung reached
    from a gate `exception` has never seen a cloud review, so
    `objections_current` is EMPTY and this prompt used to say `OBJECTIONS: []`
    and nothing else. graph-trace.jsonl seq 7 (t_ae6332d7, 2026-08-26):
    objections_current_len 0 with gate verdict `exception`, checks_failed 2 --
    against the control at seq 15 (t_52340522) showing 3 and 2 on a card that
    DID review. The rung was being asked to repair something it was never told
    about. CD-051 fixed how a fix RETURNS; this fixes what it is GIVEN.

    KEYED ON THE RECORD HAVING FAILURES, NOT ON OBJECTIONS BEING EMPTY. Two
    unrelated conditions should not be coupled: on the review route the gate
    passed, so there are no failing verdicts and the section is naturally empty.
    One rule, no hidden dependency between the two halves of the prompt.
    """
    return ("Address the following review objections for component %r.\n\n"
            "OBJECTIONS:\n%s\n\nDIFF:\n%s\n"
            % (state["component"],
               json.dumps(state["objections_current"], indent=2, sort_keys=True),
               state["diff"])) + render_failing_checks(
                   failing_checks(state)) + FIX_CONTRACT


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
    """START. Empty work products enter the owned local implementer.

    A prior terminal reason parks at human. A non-empty diff enters cheap_gate.
    Empty and whitespace-only diffs enter make_implement, which enforces the
    specification and persistent attempt cap before spawning its child.
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

    # The owned implementer is reachable from START; terminal failures park
    # before the cheap gate can evaluate an absent or stale work product.
    _add("implement", make_implement(deps))
    _cond("implement", route_after_implement,
          {"cheap_gate": "cheap_gate", "human": "human"})

    # A1: conditional entry. NOTE FOR ANY FUTURE TOPOLOGY ASSERTION --
    # a START branch registers under builder.branches["__start__"] and does
    # NOT appear in builder.edges. An assertion that looks only at
    # builder.edges will report START as unwired. Verified against
    # langgraph 1.2.10.
    _cond(START, route_entry,
          {"implement": "implement", "cheap_gate": "cheap_gate", "human": "human"})

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
