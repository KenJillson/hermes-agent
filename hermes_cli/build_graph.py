"""D5.1 build-harness graph -- the topology (design v2 sections 1, 2, 5).

CD-031, part 2 of 2. Part 1 is build_graph_sanitize.py.

SCOPE -- THE SPINE, AND WHY THE LINE IS THERE
----------------------------------------------
Design v2's node table has 13 nodes. This module builds the ones whose wrapped
primitive has been READ AT SOURCE this session:

  built     cheap_gate (ac_check_runner.gate), cloud_review, cloud_re_review,
            classify_failure, fix_rung1, fix_rung2 (all model-call), assemble,
            human, and fix_rung3 as an auth-deferred node (section 8 deferral 6)
  unbuilt   plan, plan_review, implement, local_review

The line is not arbitrary. plan / implement / local_review wrap `delegate_task`
and fix_rung3's real body wraps `terminal()`, and NEITHER primitive has been
read. Writing them now would mean guessing a signature, which rule 1 forbids.
They are present in the topology with their edges wired, so the graph does not
need reshaping when they land at D5.2.

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

    g.add_node("cheap_gate", make_cheap_gate(deps))
    g.add_node("cloud_review", make_cloud_review(
        deps, activity="code_review", node_name="cloud_review"))
    g.add_node("cloud_re_review", make_cloud_review(
        deps, activity="re_review", node_name="cloud_re_review"))
    g.add_node("classify_failure", make_classify_failure(deps))
    g.add_node("fix_rung1", make_fix(deps, rung="rung1", activity="fix_sonnet"))
    g.add_node("fix_rung2", make_fix(deps, rung="rung2", activity="fix_opus"))
    g.add_node("fix_rung3", make_fix_rung3(deps))
    g.add_node("assemble", node_assemble)
    g.add_node("human", node_human)

    # Unbuilt: wired, honest, terminal-to-human.
    for name in ("plan", "plan_review", "implement", "local_review"):
        g.add_node(name, unbuilt(name))
        g.add_edge(name, "human")

    g.add_edge(START, "cheap_gate")

    g.add_conditional_edges(
        "cheap_gate", lambda s: route_after_gate(s, deps),
        {"cloud_review": "cloud_review", "local_review": "local_review",
         "fix_rung1": "fix_rung1", "fix_rung2": "fix_rung2",
         "fix_rung3": "fix_rung3", "human": "human"})

    g.add_conditional_edges(
        "cloud_review", route_after_review,
        {"assemble": "assemble", "cloud_re_review": "cloud_re_review",
         "human": "human"})

    g.add_conditional_edges(
        "cloud_re_review", route_after_re_review,
        {"assemble": "assemble", "classify_failure": "classify_failure",
         "human": "human"})

    g.add_conditional_edges(
        "classify_failure", route_after_classify,
        {"fix_rung1": "fix_rung1", "fix_rung2": "fix_rung2",
         "fix_rung3": "fix_rung3", "human": "human"})

    for fix in ("fix_rung1", "fix_rung2", "fix_rung3"):
        g.add_conditional_edges(
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
    deps.changed_files = list(changed_files or [])
    if deps.gate is None or deps.arbiter is None or deps.parse_ac is None:
        deps.resolve_real()

    if checkpoint not in ("memory", "workspace"):
        raise ValueError(
            "checkpoint must be 'memory' or 'workspace', got %r. This is a "
            "caller error at entry, before any card state exists, so it raises "
            "rather than parking." % (checkpoint,))

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
        if ckmod is None:
            return app.invoke(payload, cfg)
        try:
            return app.invoke(payload, cfg)
        except ckmod.CheckpointTooLarge:
            # A checkpointer is not a node and has no edges, so it cannot
            # route. This is the only place the size refusal can become a
            # parked card instead of a crashed run.
            return _parked("checkpoint_too_large")

    if resume:
        if checkpoint != "workspace":
            return _parked("resume_requires_workspace_checkpoint")
        refusal = ckmod.resume_refusal(pair[0], tid, schema_version=SCHEMA_VERSION)
        if refusal:
            return _parked(refusal)
        # Partial input: re-derived work product only. Everything else is
        # restored from the checkpoint.
        return _invoke({"plan": plan, "diff": diff})

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
