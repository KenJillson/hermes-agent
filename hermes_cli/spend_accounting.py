#!/usr/bin/env python3
# spend_accounting.py -- D4.5 Sub-CD A: per-card cloud-spend accounting (dollars,
# Lane A). The arbiter (B2) reads these to decide whether an AC-failure escalation
# stays under the card's dollar ceiling before it spends. ACCOUNTING PRIMITIVE
# ONLY -- nothing here changes live behaviour; B2 wires the predicate into the
# escalation path. See docs/d4.5-arbiter-design-v1.md sec 3.
#
# ONE ledger axis: Lane A (cloud dollars, metered cost_usd). Codex/Plus quota is
# bounded by the ladder rung cap, not a dollar figure (design sec 1 fork 2).
#
# The ledger read reuses reconcile's env contract (ROUTER_LEDGER_SSH /
# ROUTER_LEDGER_PATH) so there is ONE way to reach the Overlord ledger. It is
# read-only (ssh cat) and independent of the local llama-server, so spend
# accounting works even when Qwen inference is down. `ledger_lines` injection
# makes the sum fixture-testable offline (no ssh, no spend). Pure + stdlib-only.

import os
import json
import shlex
import subprocess

# Config default dollar ceiling, read from the board process environment
# (profile .env). A card body / ## AC cannot set it -- but that is a statement
# about card authorship, NOT containment: this name is not in the set
# scrub_kanban_env removes, so a delegated child's subprocess inherits it, and
# any process holding a `terminal` tool can set it for a subprocess it spawns.
# See docs/root-causes-v1.2.md RC-005 -- OPEN.
_DEFAULT_CAP_ENV = "HERMES_KANBAN_MAX_CARD_SPEND"
_DEFAULT_CAP_USD = 5.0

# Same env contract reconcile uses (ONE way to reach the ledger).
_LEDGER_SSH = os.environ.get(
    "ROUTER_LEDGER_SSH",
    "ssh -o BatchMode=yes -o ConnectTimeout=10 coder@10.10.40.2")
_LEDGER_PATH = os.environ.get("ROUTER_LEDGER_PATH", "/work/coder/router-usage.log")

_LANE_A = "lane_a"


def default_cap_usd():
    """Config-default dollar ceiling (env override, else 5.0). Fail-closed: a
    malformed env value falls back to the built-in default, never raises."""
    raw = os.environ.get(_DEFAULT_CAP_ENV)
    if raw is None or raw.strip() == "":
        return _DEFAULT_CAP_USD
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_CAP_USD
    return v if (v == v and 0 <= v < 1e9) else _DEFAULT_CAP_USD


def _load_ledger_lines(ledger_file=None):
    """Raw ledger text lines. ledger_file: local path (test). None: ssh-cat the
    remote Overlord ledger (read-only). Transport failure raises to the caller,
    which decides how to gate spend on it."""
    if ledger_file:
        with open(ledger_file) as fh:
            return fh.read().splitlines()
    argv = shlex.split(_LEDGER_SSH) + ["cat %s" % _LEDGER_PATH]
    p = subprocess.run(argv, capture_output=True, timeout=30)
    if p.returncode != 0:
        raise RuntimeError(
            "ledger fetch failed rc=%d: %s"
            % (p.returncode, p.stderr.decode(errors="replace")[-200:]))
    return p.stdout.decode(errors="replace").splitlines()


def card_cloud_spend_usd(task_id, *, ledger_lines=None, ledger_file=None):
    """Sum cost_usd over Lane-A ledger lines whose task_id matches. `ledger_lines`
    (dicts or raw json strings) short-circuits all I/O -- the offline fixture path.
    Unparseable / non-Lane-A / other-task lines are skipped; a line missing
    cost_usd contributes 0. Returns a float (0.0 if none)."""
    if ledger_lines is None:
        ledger_lines = _load_ledger_lines(ledger_file)
    total = 0.0
    for ln in ledger_lines:
        if isinstance(ln, str):
            ln = ln.strip()
            if not ln:
                continue
            try:
                d = json.loads(ln)
            except json.JSONDecodeError:
                continue
        else:
            d = ln
        if d.get("task_id") != task_id:
            continue
        if d.get("lane") != _LANE_A:
            continue
        c = d.get("cost_usd")
        if isinstance(c, (int, float)):
            total += float(c)
    return total


def resolved_cap(conn, task_id):
    """Effective dollar ceiling: the per-card override (tasks.max_card_spend_usd,
    written only by the Ken-only `set-spend-cap` verb) if set, else the config
    default. On the reach of that default see the note at _DEFAULT_CAP_ENV.
    Missing column (pre-migration) or unknown id -> default."""
    try:
        row = conn.execute(
            "SELECT max_card_spend_usd FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
    except Exception:
        return default_cap_usd()
    if row is None:
        return default_cap_usd()
    v = row[0]
    if v is None:
        return default_cap_usd()
    try:
        return float(v)
    except (TypeError, ValueError):
        return default_cap_usd()


def over_spend_cap(conn, task_id, *, ledger_lines=None, ledger_file=None):
    """Pre-call gate for the arbiter (B2): True when this card's Lane-A cloud spend
    has reached/exceeded its resolved cap. Checked BEFORE each escalation, so
    overshoot is bounded by at most one in-flight call's cost (design sec 3)."""
    spend = card_cloud_spend_usd(
        task_id, ledger_lines=ledger_lines, ledger_file=ledger_file)
    return spend >= resolved_cap(conn, task_id)


# ============= D4.5 Sub-CD B1: objective arbiter decision (CD-026) ==============
# The escalation TRIGGER is the board's objective AC verdict (ac_check_runner.gate)
# -- honest-broker: a mechanical `exception` (a check failed or was unrunnable),
# never a model's opinion of its own family's output. Judgment / no-check cards
# have no mechanical arbiter and route to the human exception gate (D4.4), never
# auto-escalate. Interpretation ONLY: does not run checks (gate() did) and does not
# spend (over_spend_cap is a read-only pre-call gate).

ESCALATE = "escalate"       # AC failed AND under the card's Lane-A $ cap
OVER_CAP = "over_cap"       # AC failed but at/over the cap -> D3.4 review-required block
PASS = "pass"               # all_pass -> nothing to escalate
NO_ARBITER = "no_arbiter"   # needs_human / no_checks / no_workspace -> human, not auto


def arbiter_decision(conn, task_id, gate_summary, *, ledger_lines=None,
                     ledger_file=None):
    """Return (decision, reason) from the board AC-gate summary + the spend-gate.
    Mapping (design sec 2): all_pass->PASS; exception->ESCALATE (under cap) or
    OVER_CAP (at/over cap); needs_human/no_checks/no_workspace/unknown->NO_ARBITER.
    A ledger-read failure FAILS CLOSED (RULED 2026-08-21): it returns
    OVER_CAP with a spend_unknown reason, which route_after_gate sends to
    `human`. An unreadable ledger means the cap CANNOT BE ENFORCED, and the
    ledger is read over ssh to coder@10.10.40.2, so transport failure is a
    routine event rather than an exotic one.

    The previous behaviour returned ESCALATE, which reaches fix_rung1 and
    SPENDS -- the failure of the control that exists to prevent spending was
    itself a decision to spend. Its stated justification, that "B2 fails
    closed at the real spend seam", is UNSUPPORTED: over_spend_cap has
    exactly one call site (this function), resolved_cap only one (that), and
    nothing downstream reads the cap. Measured 2026-08-20."""
    verdict = (gate_summary or {}).get("verdict")
    if verdict == "all_pass":
        return PASS, "all_pass"
    if verdict != "exception":
        return NO_ARBITER, "no-mechanical-arbiter:%s" % verdict
    try:
        over = over_spend_cap(conn, task_id, ledger_lines=ledger_lines,
                              ledger_file=ledger_file)
    except Exception as e:
        return OVER_CAP, "ac_failure;spend_unknown:%s" % type(e).__name__
    if over:
        return OVER_CAP, "ac_failure;over_cap"
    return ESCALATE, "ac_failure;under_cap"
