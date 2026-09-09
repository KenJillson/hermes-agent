"""Owned subprocess runner for the Phase-5 build-graph implement node.

The parent side resolves and proves a local inference identity, writes a
mode-0600 request, launches one isolated process group in the card worktree,
and owns deadline/interrupt/reap behavior.  The child side resolves the same
identity again, runs a narrowly tooled AIAgent, and emits one JSON envelope.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import uuid
import math
from typing import Any, Callable, Mapping, Optional


PROTOCOL_VERSION = 1
REQUEST_KEYS = frozenset({
    "protocol", "goal", "workspace", "max_iterations",
    "deadline_monotonic", "expected_identity",
})
IDENTITY_KEYS = frozenset({"provider", "model", "base_url", "api_mode"})
RESULT_KEYS = frozenset({"protocol", "status", "identity", "summary"})
CHILD_STATUSES = frozenset({"completed", "failed", "interrupted", "protocol_error"})
PARENT_STATUSES = frozenset({
    "completed", "failed", "timeout", "interrupted",
    "protocol_error", "exec_error",
})
MAX_REQUEST_BYTES = 256 * 1024
MAX_RESULT_BYTES = 64 * 1024
MAX_STDERR_BYTES = 64 * 1024
DEFAULT_POLL_SECONDS = 0.2
DEFAULT_TERM_GRACE_SECONDS = 5.0
EVIDENCE_KEEP = 40


class ProtocolError(ValueError):
    pass


def _closed(status: str, *, identity: Optional[dict[str, str]] = None,
            evidence_path: Optional[str] = None) -> dict[str, Any]:
    if status not in PARENT_STATUSES:
        status = "protocol_error"
    return {
        "status": status,
        "identity": identity if isinstance(identity, dict) else None,
        "evidence_path": evidence_path,
    }


def _read_bounded(path: Path, limit: int) -> tuple[bytes, bool]:
    with path.open("rb") as fh:
        data = fh.read(limit + 1)
    return data[:limit], len(data) > limit


def _write_private(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())


def _identity() -> dict[str, str]:
    from agent.model_metadata import is_local_endpoint
    from hermes_cli.config import load_config_readonly
    from hermes_cli.runtime_provider import resolve_runtime_provider

    cfg = load_config_readonly()
    model_cfg = cfg.get("model") if isinstance(cfg, dict) else None
    if not isinstance(model_cfg, dict):
        raise ProtocolError("model_config_missing")
    requested = str(model_cfg.get("provider") or "").strip()
    model = str(model_cfg.get("default") or "").strip()
    if not requested or not model:
        raise ProtocolError("local_identity_incomplete")
    runtime = resolve_runtime_provider(requested=requested, target_model=model)
    provider = str(runtime.get("provider") or "").strip().lower()
    base_url = str(runtime.get("base_url") or "").strip().rstrip("/")
    api_mode = str(runtime.get("api_mode") or "").strip()
    resolved_model = str(runtime.get("model") or model).strip()
    if provider != "custom" or not is_local_endpoint(base_url):
        raise ProtocolError("provider_not_local")
    identity = {
        "provider": provider,
        "model": resolved_model,
        "base_url": base_url,
        "api_mode": api_mode,
    }
    if set(identity) != IDENTITY_KEYS or not all(identity.values()):
        raise ProtocolError("local_identity_incomplete")
    return identity


def _validate_request(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != REQUEST_KEYS:
        raise ProtocolError("request_shape")
    if raw.get("protocol") != PROTOCOL_VERSION:
        raise ProtocolError("request_version")
    goal = raw.get("goal")
    workspace = raw.get("workspace")
    iterations = raw.get("max_iterations")
    deadline = raw.get("deadline_monotonic")
    expected = raw.get("expected_identity")
    if not isinstance(goal, str) or not goal.strip():
        raise ProtocolError("request_goal")
    if not isinstance(workspace, str) or not os.path.isabs(workspace):
        raise ProtocolError("request_workspace")
    if not isinstance(iterations, int) or isinstance(iterations, bool) or not 1 <= iterations <= 500:
        raise ProtocolError("request_iterations")
    if (not isinstance(deadline, (int, float)) or isinstance(deadline, bool)
            or not math.isfinite(float(deadline))):
        raise ProtocolError("request_deadline")
    if not isinstance(expected, dict) or set(expected) != IDENTITY_KEYS:
        raise ProtocolError("request_identity")
    return raw


def _parse_result(data: bytes, expected_identity: dict[str, str],
                  returncode: int) -> dict[str, Any]:
    try:
        raw = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("result_json") from exc
    if not isinstance(raw, dict) or set(raw) != RESULT_KEYS:
        raise ProtocolError("result_shape")
    if raw.get("protocol") != PROTOCOL_VERSION:
        raise ProtocolError("result_version")
    status = raw.get("status")
    identity = raw.get("identity")
    summary = raw.get("summary")
    if status not in CHILD_STATUSES:
        raise ProtocolError("result_status")
    if not isinstance(identity, dict) or identity != expected_identity:
        raise ProtocolError("result_identity")
    if not isinstance(summary, str):
        raise ProtocolError("result_summary")
    expected_rc = 0 if status == "completed" else 2 if status == "protocol_error" else 1
    if returncode != expected_rc:
        raise ProtocolError("result_exit_mismatch")
    if status == "completed" and (not summary.strip() or summary.strip() == "(empty)"):
        raise ProtocolError("result_empty_summary")
    return raw


def _terminate_group(proc: subprocess.Popen[Any], grace_seconds: float) -> None:
    """Terminate the owned group even if its leader exits before descendants."""
    pgid = None
    if os.name != "posix" or not hasattr(os, "killpg"):
        proc.terminate()
    else:
        pgid = os.getpgid(proc.pid)
        if pgid != proc.pid or pgid == os.getpgrp():
            raise RuntimeError("child_process_group_not_owned")
        os.killpg(pgid, signal.SIGTERM)
    deadline = time.monotonic() + max(0.0, grace_seconds)
    while time.monotonic() < deadline:
        # Keep a POSIX leader unreaped until group signalling is complete,
        # reserving its PID/PGID throughout the grace period.
        if pgid is None and proc.poll() is not None:
            break
        # Preserve the grace period for descendants after the leader exits.
        time.sleep(min(DEFAULT_POLL_SECONDS, max(0.0, deadline - time.monotonic())))
    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass  # The whole group exited during the grace period.
    elif proc.poll() is None:
        proc.kill()
    proc.wait()


def _evidence_dir(explicit: Optional[str]) -> Path:
    if explicit:
        root = Path(explicit)
    else:
        from hermes_constants import get_hermes_home
        root = get_hermes_home() / "logs" / "build-graph-implementer"
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        os.chmod(root, 0o700)
    except OSError:
        pass
    return root


def _persist_evidence(root: Path, record: dict[str, Any]) -> str:
    path = root / ("run-%s.json" % uuid.uuid4().hex)
    encoded = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
    _write_private(path, encoded)
    candidates = sorted(root.glob("run-*.json"), key=lambda p: p.stat().st_mtime,
                        reverse=True)
    for stale in candidates[EVIDENCE_KEEP:]:
        try:
            stale.unlink()
        except OSError:
            pass
    return str(path)


def run_implementer(*, goal: str, workspace: str, max_iterations: int,
                    timeout_seconds: float, interrupt_check: Optional[Callable[[], bool]] = None,
                    python_exe: Optional[str] = None, child_argv: Optional[list[str]] = None,
                    evidence_dir: Optional[str] = None,
                    poll_seconds: float = DEFAULT_POLL_SECONDS,
                    term_grace_seconds: float = DEFAULT_TERM_GRACE_SECONDS,
                    env: Optional[Mapping[str, str]] = None) -> dict[str, Any]:
    """Run one implementer child. Test-only seams select a fixture argv/evidence root."""
    started = time.monotonic()
    try:
        workspace_real = os.path.realpath(workspace)
        if not os.path.isabs(workspace) or not os.path.isdir(workspace_real):
            return _closed("protocol_error")
        identity = _identity()
        timeout = float(timeout_seconds)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ProtocolError("timeout_invalid")
        deadline = started + timeout
        request = {
            "protocol": PROTOCOL_VERSION,
            "goal": goal,
            "workspace": workspace_real,
            "max_iterations": max_iterations,
            "deadline_monotonic": deadline,
            "expected_identity": identity,
        }
        _validate_request(request)
        evidence_root = _evidence_dir(evidence_dir)
    except Exception:
        return _closed("protocol_error")

    status = "exec_error"
    returncode: Optional[int] = None
    result_bytes = b""
    result_oversized = False
    stderr_bytes = b""
    stderr_oversized = False
    proc: Optional[subprocess.Popen[Any]] = None
    try:
        with tempfile.TemporaryDirectory(prefix="build-graph-implementer-") as tmp:
            tmp_path = Path(tmp)
            request_path = tmp_path / "request.json"
            stdout_path = tmp_path / "result.json"
            stderr_path = tmp_path / "stderr.log"
            _write_private(request_path, json.dumps(request, separators=(",", ":")).encode())
            _write_private(stdout_path, b"")
            _write_private(stderr_path, b"")
            argv = child_argv or [
                python_exe or sys.executable,
                "-m", "hermes_cli.build_graph_implementer",
                "--request-file", str(request_path),
            ]
            argv = [
                str(request_path) if value == "{request_file}" else value
                for value in argv
            ]
            from agent.delegation_context import scrub_kanban_env
            child_env = scrub_kanban_env(env if env is not None else os.environ)
            child_env = {
                key: value for key, value in child_env.items()
                if not key.startswith("HERMES_KANBAN_")
            }
            child_env["TERMINAL_CWD"] = workspace_real
            with stdout_path.open("wb") as stdout_fh, stderr_path.open("wb") as stderr_fh:
                proc = subprocess.Popen(
                    argv,
                    cwd=workspace_real,
                    env=child_env,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_fh,
                    stderr=stderr_fh,
                    start_new_session=True,
                )
                while proc.poll() is None:
                    if interrupt_check is not None and interrupt_check():
                        status = "interrupted"
                        _terminate_group(proc, term_grace_seconds)
                        break
                    if time.monotonic() >= deadline:
                        status = "timeout"
                        _terminate_group(proc, term_grace_seconds)
                        break
                    time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))
                if proc.poll() is None:
                    proc.wait()
                returncode = proc.returncode
            result_bytes, result_oversized = _read_bounded(stdout_path, MAX_RESULT_BYTES)
            stderr_bytes, stderr_oversized = _read_bounded(stderr_path, MAX_STDERR_BYTES)
            if status not in {"timeout", "interrupted"}:
                if result_oversized:
                    status = "protocol_error"
                else:
                    try:
                        envelope = _parse_result(result_bytes, identity, int(returncode))
                        status = str(envelope["status"])
                    except ProtocolError:
                        status = "protocol_error"
    except Exception:
        status = "exec_error"
        if proc is not None and proc.poll() is None:
            try:
                _terminate_group(proc, term_grace_seconds)
            except Exception:
                try:
                    proc.kill()
                    proc.wait()
                except Exception:
                    pass

    record = {
        "protocol": PROTOCOL_VERSION,
        "status": status,
        "identity": identity,
        "pid": proc.pid if proc is not None else None,
        "returncode": returncode,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "result_oversized": result_oversized,
        "stderr_oversized": stderr_oversized,
        "stderr": stderr_bytes.decode("utf-8", errors="replace"),
    }
    try:
        evidence_path = _persist_evidence(evidence_root, record)
    except Exception:
        return _closed("exec_error", identity=identity)
    return _closed(status, identity=identity, evidence_path=evidence_path)


def _child_envelope(request: dict[str, Any]) -> tuple[dict[str, Any], int]:
    request = _validate_request(request)
    if os.path.realpath(os.getcwd()) != os.path.realpath(request["workspace"]):
        raise ProtocolError("child_cwd_mismatch")
    if time.monotonic() >= float(request["deadline_monotonic"]):
        raise ProtocolError("child_deadline_expired")
    identity = _identity()
    if identity != request["expected_identity"]:
        raise ProtocolError("child_identity_mismatch")

    from hermes_cli.runtime_provider import resolve_runtime_provider
    runtime = resolve_runtime_provider(
        requested=identity["provider"], target_model=identity["model"])
    try:
        from agent.delegation_context import delegated_child_context
        from run_agent import AIAgent
        from tools.terminal_tool import set_approval_callback
        set_approval_callback(lambda *args, **kwargs: "deny")
        with delegated_child_context():
            agent = AIAgent(
                base_url=identity["base_url"],
                api_key=runtime.get("api_key"),
                provider=identity["provider"],
                requested_provider=identity["provider"],
                api_mode=identity["api_mode"],
                model=identity["model"],
                max_iterations=request["max_iterations"],
                max_tokens=runtime.get("max_output_tokens"),
                request_overrides=dict(runtime.get("request_overrides") or {}),
                enabled_toolsets=["file", "terminal"],
                disabled_toolsets=["kanban", "delegation", "clarify", "memory", "cronjob"],
                quiet_mode=True,
                platform="subagent",
                skip_memory=True,
                clarify_callback=None,
            )
        with delegated_child_context(str(getattr(agent, "session_id", "") or "")):
            result = agent.run_conversation(user_message=request["goal"])
    except Exception as exc:
        print("build-graph implementer agent failed: %s" % type(exc).__name__,
              file=sys.stderr)
        return {
            "protocol": PROTOCOL_VERSION,
            "status": "failed",
            "identity": identity,
            "summary": "",
        }, 1
    result = result if isinstance(result, dict) else {}
    summary = result.get("final_response")
    if not isinstance(summary, str):
        summary = ""
    if result.get("interrupted"):
        status = "interrupted"
    elif result.get("failed"):
        status = "failed"
    else:
        status = "completed"
    if not summary.strip() or summary.strip() == "(empty)":
        status = "failed"
    return {
        "protocol": PROTOCOL_VERSION,
        "status": status,
        "identity": identity,
        "summary": summary[:MAX_RESULT_BYTES // 2],
    }, 0 if status == "completed" else 1


def child_main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-file", required=True)
    args = parser.parse_args(argv)
    stdout = sys.stdout
    try:
        path = Path(args.request_file)
        data, oversized = _read_bounded(path, MAX_REQUEST_BYTES)
        if oversized or (path.stat().st_mode & 0o077):
            raise ProtocolError("request_file")
        request = json.loads(data.decode("utf-8"))
        with contextlib.redirect_stdout(sys.stderr):
            envelope, rc = _child_envelope(request)
    except Exception as exc:
        print("build-graph implementer child failed: %s" % type(exc).__name__, file=sys.stderr)
        try:
            identity = _identity()
        except Exception:
            identity = {k: "unavailable" for k in IDENTITY_KEYS}
        envelope = {
            "protocol": PROTOCOL_VERSION,
            "status": "protocol_error",
            "identity": identity,
            "summary": "",
        }
        rc = 2
    stdout.write(json.dumps(envelope, separators=(",", ":")) + "\n")
    stdout.flush()
    return rc


if __name__ == "__main__":
    raise SystemExit(child_main())
