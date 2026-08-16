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

# Config default dollar ceiling. Author-UNREACHABLE: sourced only from the board
# process environment (profile .env), never from a card body / ## AC / worker.
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
    default. Author-unreachable by construction. Missing column (pre-migration) or
    unknown id -> default."""
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
