#!/usr/bin/env python3
# ac_check_runner.py — D4.2 board-side AC check execution (standalone, offline-
# testable). Runs the D4.1-parsed `ac_results` checks in a card workspace,
# evaluates pass/fail/unrunnable, and writes the durable verdict record
# `ac-execution-<card_id>.json` (the emit reads this to flip ac_results status +
# recompute ac_passed). NOTHING here dispatches or touches the board; the
# review-seam hook (separate unit) calls run_ac_checks() and persist_execution().
#
# Isolation (D-2): _detect_sandbox() prefers bwrap (ro-bind / , rw workspace,
# --unshare-net), then unshare, else a hardened subprocess fallback (hard
# timeout, scrubbed env — NO inherited secrets/sudo, no stdin, cwd=workspace,
# ulimit, output cap) with a recorded no-network caveat. Untrusted author-authored
# commands => the fallback's caveat matters; confirm bwrap/unshare on the box
# before untrusted worker-authored checks run live.

import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone

_OUTPUT_CAP = 64 * 1024          # bytes of stdout/stderr retained per check
_DEFAULT_TIMEOUT = 60            # seconds per check (caller overrides per budget)
# Minimal env: no secrets, tokens, or SUDO_PASSWORD leak into an author command.
_SAFE_ENV_KEYS = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TERM", "TZ")


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _detect_sandbox():
    """Return 'bwrap' | 'unshare' | 'none' based on what's installed."""
    if shutil.which("bwrap"):
        return "bwrap"
    if shutil.which("unshare"):
        return "unshare"
    return "none"


def _scrubbed_env(workspace):
    env = {k: os.environ[k] for k in _SAFE_ENV_KEYS if k in os.environ}
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    env["HOME"] = str(workspace)          # no access to the real HOME/dotfiles
    return env


_EXTRA_RO_BINDS_ENV = "HERMES_KANBAN_EXTRA_RO_BINDS"


def _extra_ro_bind_denied(realpath, home):
    """True if `realpath` (already realpath'd) must NOT be ro-bound into the
    sandbox: the filesystem root, a system-sensitive tree, a secret directory or
    anything under it, or an ANCESTOR of a secret directory (binding it would
    pull the secret back in). This deny-list defends the config-only
    extra_ro_binds hook so a bad config line cannot re-expose ~/.ssh or
    profiles/*/.env."""
    if realpath == os.sep:
        return True
    for r in ("/etc", "/proc", "/sys", "/dev", "/root", "/boot"):
        if realpath == r or realpath.startswith(r + os.sep):
            return True
    for s in (os.path.join(home, ".ssh"),
              os.path.join(home, ".hermes"),
              os.path.join(home, ".gnupg")):
        if realpath == s or realpath.startswith(s + os.sep):
            return True          # candidate is / lives under a secret dir
        if s == realpath or s.startswith(realpath + os.sep):
            return True          # candidate is an ancestor of a secret dir
    return False


def _resolve_extra_ro_binds():
    """Validated extra ro-bind paths, sourced ONLY from the board process env
    (HERMES_KANBAN_EXTRA_RO_BINDS, ':'-separated), which is populated from the
    profile .env — NEVER from a card body (author checks run under _scrubbed_env
    and cannot set it). Each entry is realpath'd and deny-list checked; a
    rejected or missing path is dropped with a stderr note, not mounted. Empty by
    default, so absent config == the minimal bind set."""
    raw = os.environ.get(_EXTRA_RO_BINDS_ENV, "").strip()
    if not raw:
        return []
    home = os.path.expanduser("~")
    out = []
    for entry in raw.split(":"):
        p = entry.strip()
        if not p:
            continue
        rp = os.path.realpath(p)
        if _extra_ro_bind_denied(rp, home):
            sys.stderr.write(
                "[ac_check_runner] refusing extra_ro_bind (deny-list): %s\n" % p)
            continue
        if not os.path.exists(rp):
            sys.stderr.write(
                "[ac_check_runner] skipping extra_ro_bind (not present): %s\n" % p)
            continue
        out.append(rp)
    return out


def _wrap(cmd, workspace, isolation):
    """Return the argv list that runs `cmd` (a shell string) under `isolation`."""
    ws = str(workspace)
    if isolation == "bwrap":
        # Selective binds (CD-021 hardening): do NOT --ro-bind / /, which would
        # expose host secrets (~/.ssh, profiles/*/.env) to a check's persisted
        # stdout. Bind only the runtime prefixes a check legitimately needs. On
        # this usrmerge box /bin,/sbin,/lib are symlinks into /usr and /lib64 is
        # absent, so we bind /usr once and recreate the merge symlinks. /run stays
        # a tmpfs so /run/docker.sock (a root escape --unshare-net does not stop)
        # is masked. Optional config-only extra ro-binds (deny-list validated,
        # never author-supplied) cover off-prefix runtimes (e.g. ~/.nvm, ~/.local).
        argv = ["bwrap",
                "--ro-bind", "/usr", "/usr",
                "--symlink", "usr/bin", "/bin",
                "--symlink", "usr/sbin", "/sbin",
                "--symlink", "usr/lib", "/lib",
                "--ro-bind-try", "/etc/alternatives", "/etc/alternatives",
                "--ro-bind-try", "/etc/ssl", "/etc/ssl",
                "--ro-bind-try", "/etc/ca-certificates", "/etc/ca-certificates",
                "--dev", "/dev", "--proc", "/proc",
                "--tmpfs", "/run", "--tmpfs", "/tmp"]
        for _p in _resolve_extra_ro_binds():
            argv += ["--ro-bind-try", _p, _p]
        argv += ["--bind", ws, ws, "--chdir", ws,
                 "--unshare-net", "--unshare-pid", "--die-with-parent",
                 "--", "/bin/sh", "-c", cmd]
        return argv
    if isolation == "unshare":
        # userns net+pid isolation; filesystem stays host (cwd pins to workspace)
        return ["unshare", "--user", "--map-root-user", "--net", "--pid", "--fork",
                "/bin/sh", "-c", cmd]
    return ["/bin/sh", "-c", cmd]         # fallback: no namespace isolation


def run_check(command, workspace, *, timeout=_DEFAULT_TIMEOUT, isolation="auto"):
    """Run one check command. Returns a dict; never raises for command failure."""
    iso = _detect_sandbox() if isolation == "auto" else isolation
    argv = _wrap(command, workspace, iso)
    result = {"isolation_used": iso, "timed_out": False, "run_error": None,
              "exit_code": None, "stdout": "", "stderr": ""}
    try:
        p = subprocess.run(
            argv, cwd=str(workspace), env=_scrubbed_env(workspace),
            stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout,
            preexec_fn=_limit_resources)
        result["exit_code"] = p.returncode
        result["stdout"] = p.stdout.decode("utf-8", "replace")[:_OUTPUT_CAP]
        result["stderr"] = p.stderr.decode("utf-8", "replace")[:_OUTPUT_CAP]
    except subprocess.TimeoutExpired as e:
        result["timed_out"] = True
        result["run_error"] = f"timeout after {timeout}s"
        if e.stdout:
            result["stdout"] = e.stdout.decode("utf-8", "replace")[:_OUTPUT_CAP]
    except FileNotFoundError as e:
        result["run_error"] = f"command/sandbox not found: {e}"
    except OSError as e:
        result["run_error"] = f"exec error: {e}"
    return result


def _limit_resources():
    # best-effort resource caps (CPU seconds, file size); ignored if unsupported
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
        resource.setrlimit(resource.RLIMIT_FSIZE, (256 << 20, 256 << 20))
    except Exception:
        pass


def evaluate(check, run):
    """Map a check spec + a run() result to 'passed'|'failed'|'unrunnable'.

    'unrunnable' = the check could not be meaningfully run: sandbox/exec error,
    timeout, an un-runnable authored regex, OR the command itself was not found /
    not executable (exit 126/127) — a BROKEN CHECK, distinct from a genuine test
    failure. An explicit `expect=exit:126|127` is honored first, so a check that
    deliberately asserts that code still passes."""
    if run.get("run_error") or run.get("timed_out"):
        return "unrunnable"
    ek = check.get("expect_kind", "exit")
    ev = check.get("expect_value", "0")
    ec = run.get("exit_code")
    out = run.get("stdout", "")
    if ek == "exit":
        try:
            if ec == int(ev):
                return "passed"          # honor explicit expect (incl. 126/127)
        except (TypeError, ValueError):
            return "unrunnable"
    if ec in (126, 127):
        return "unrunnable"              # broken check: command missing/not exec
    if ek == "exit":
        return "failed"
    if ek == "stdout_regex":
        try:
            return "passed" if re.search(ev, out, re.MULTILINE) else "failed"
        except re.error:
            return "unrunnable"          # a bad authored regex can't be run
    if ek == "stdout_substr":
        return "passed" if ev in out else "failed"
    return "unrunnable"


def run_ac_checks(ac_results, workspace, *, timeout=_DEFAULT_TIMEOUT,
                  isolation="auto"):
    """Execute every check-kind AC; judgment ACs are recorded needs-human, never
    run. Returns (verdicts, summary)."""
    verdicts = []
    for r in (ac_results or []):
        base = {"index": r.get("index"), "kind": r.get("kind")}
        if r.get("kind") == "judgment":
            verdicts.append({**base, "status": "needs_human",
                             "judgment": r.get("judgment")})
            continue
        run = run_check(r["check"]["command"], workspace,
                        timeout=timeout, isolation=isolation)
        status = evaluate(r["check"], run)
        verdicts.append({**base, "status": status,
                         "command": r["check"]["command"],
                         "expect_kind": r["check"]["expect_kind"],
                         "expect_value": r["check"]["expect_value"],
                         "exit_code": run["exit_code"],
                         "timed_out": run["timed_out"],
                         "run_error": run["run_error"],
                         "isolation_used": run["isolation_used"],
                         "stdout_tail": run["stdout"][-2000:]})
    checks = [v for v in verdicts if v["kind"] == "check"]
    passed = sum(1 for v in checks if v["status"] == "passed")
    summary = {
        "ac_total": len(verdicts),
        "checks_total": len(checks),
        "checks_passed": passed,
        "checks_failed": sum(1 for v in checks if v["status"] == "failed"),
        "checks_unrunnable": sum(1 for v in checks if v["status"] == "unrunnable"),
        "judgment_count": sum(1 for v in verdicts if v["kind"] == "judgment"),
        # "clean" = every check passed AND nothing needs a human (judgment/unrunnable)
        "all_clean": (len(checks) > 0
                      and passed == len(checks)
                      and not any(v["kind"] == "judgment" for v in verdicts)),
        "verdict": None,   # set below
    }
    if summary["checks_unrunnable"] or summary["checks_failed"]:
        summary["verdict"] = "exception"          # -> Ken (D4.4)
    elif summary["judgment_count"]:
        summary["verdict"] = "needs_human"         # judgment-only -> Ken
    elif summary["all_clean"]:
        summary["verdict"] = "all_pass"            # candidate auto-accept (D4.4)
    else:
        summary["verdict"] = "no_checks"           # no runnable ACs at all
    return verdicts, summary


def persist_execution(card_id, workspace, verdicts, summary, *, path=None):
    """Write the durable verdict record. Atomic (temp + os.replace). Returns path.
    D4.2 records ONLY; it does not accept/complete (that's D4.4, config-gated)."""
    rec = {"card_id": card_id, "executed_at": _now(),
           "workspace": str(workspace), "summary": summary, "verdicts": verdicts,
           "schema": "ac-execution/1"}
    out = path or os.path.join(str(workspace), f"ac-execution-{card_id}.json")
    d = os.path.dirname(out) or "."
    tmp = os.path.join(d, f".{os.path.basename(out)}.tmp")
    with open(tmp, "w") as fh:
        json.dump(rec, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, out)
    return out


# ---- board-side accept gate (D4.2, option B) --------------------------------
# Orchestration used by the `hermes kanban gate` CLI verb, which accept-card.sh
# calls before `complete`. Kept dependency-free here (the shared parser is
# injected as parse_fn so this module needs no hermes/DB import and stays
# standalone-testable); the CLI verb passes kb.parse_ac_text and a
# record_terminal_result hook.

def gate(card_id, body, workspace, *, parse_fn, isolation="auto",
         timeout=_DEFAULT_TIMEOUT, evidence_hook=None):
    """Run the card's authored checks in its workspace and persist the verdict.
    RECORDS ONLY — never accepts/completes (that decision is the caller's, from
    summary['verdict']). Returns the summary dict. A card with no workspace or no
    runnable ## AC yields a non-blocking verdict so the gate never refuses a card
    that simply has nothing to check."""
    if not body:
        return {"verdict": "no_checks", "ac_total": 0, "checks_total": 0,
                "checks_passed": 0, "checks_failed": 0, "checks_unrunnable": 0,
                "judgment_count": 0, "all_clean": False, "note": "no body"}
    ac_results = parse_fn(body)[3]
    if not ac_results:
        return {"verdict": "no_checks", "ac_total": 0, "checks_total": 0,
                "checks_passed": 0, "checks_failed": 0, "checks_unrunnable": 0,
                "judgment_count": 0, "all_clean": False, "note": "no ## AC"}
    if not workspace or not os.path.isdir(str(workspace)):
        return {"verdict": "no_workspace", "ac_total": len(ac_results),
                "checks_total": 0, "checks_passed": 0, "checks_failed": 0,
                "checks_unrunnable": 0, "judgment_count": 0, "all_clean": False,
                "note": f"workspace missing: {workspace}"}
    verdicts, summary = run_ac_checks(ac_results, workspace,
                                      isolation=isolation, timeout=timeout)
    try:
        persist_execution(card_id, workspace, verdicts, summary)
    except OSError as e:
        summary["persist_error"] = str(e)     # verdict still returned to caller
    if evidence_hook:
        for v in verdicts:
            if v.get("kind") == "check":
                try:
                    evidence_hook(v.get("command"), str(workspace),
                                  v.get("exit_code"), v.get("stdout_tail", ""))
                except Exception:
                    pass                        # evidence is best-effort (D4.3)
    return summary


def gate_exit_code(summary):
    """Accept decision for accept-card.sh. Only a mechanical failure/unrunnable
    (verdict 'exception') REFUSES accept (rc=3). all_pass / needs_human (judgment
    is Ken's call, and he is the one accepting) / no_checks / no_workspace all
    allow accept (rc=0) — the gate blocks broken cards, it does not usurp Ken's
    judgment on non-mechanizable ACs."""
    return 3 if summary.get("verdict") == "exception" else 0
