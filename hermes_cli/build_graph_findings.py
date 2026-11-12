"""CD-106: a review park records WHY, as an inert, bounded, sanitized event.

Handoff 2026-11-11 4.2, Ken ruling option 1. Before this, a card parked by a
review (`a2_stage_review_required:<stage>`, `cloud_plan_review_rejected`)
carried only the terminal reason; the findings lived in the checkpointer and
were lost to every harvest (Test14e t_ccb5e9ab). This writes them as ONE
task_events row on the parked run, never as a comment: build_worker_context
feeds comments into worker prompts, and events it does not. Harvests commit
card.json with its events, so every finding is bounded, redacted, and withheld
by rule name when it trips a sanitizer HALT rule.

Only fresh worker processes import this module (cli.py's graph park path); the
dispatcher does not. Nothing here may change a park: the callers guard it.
"""
import re

EVENT = "graph_review_findings"
MAX_FINDINGS = 8
MAX_CHARS = 400
REVIEW_PARK_PREFIXES = ("a2_stage_review_required:",)
REVIEW_PARK_EXACT = ("cloud_plan_review_rejected",)
_RULE = re.compile(r"[a-z][a-z0-9_]{0,63}")


def is_review_park(terminal):
    return isinstance(terminal, str) and (
        terminal.startswith(REVIEW_PARK_PREFIXES) or terminal in REVIEW_PARK_EXACT)


def payload(terminal, state, sanitize=None):
    """The findings event payload for a review park, or None for any other park."""
    if not is_review_park(terminal):
        return None
    if sanitize is None:
        from hermes_cli import build_graph_sanitize as sanitize
    items = state.get("objections_current") if isinstance(state, dict) else None
    texts = [item["text"] for item in (items if isinstance(items, list) else [])
             if isinstance(item, dict) and isinstance(item.get("text"), str) and item["text"].strip()]
    findings, cut = [], False
    for text in texts[:MAX_FINDINGS]:
        try:
            halt = sanitize.halt_reason(text)
        except Exception:
            halt = "sanitizer_unavailable"
        if halt:
            # A rule name only, never the matched text.
            findings.append("[withheld: %s]" % (halt if isinstance(halt, str) and _RULE.fullmatch(halt) else "halt"))
            continue
        try:
            clean = sanitize.redact(text)
        except Exception:
            clean = None
        if not isinstance(clean, str):
            findings.append("[withheld: sanitizer_unavailable]")
            continue
        if len(clean) > MAX_CHARS:
            clean, cut = clean[:MAX_CHARS - 1] + "…", True
        findings.append(clean)
    return {"terminal": terminal, "count": len(texts),
            "truncated": cut or len(texts) > MAX_FINDINGS, "findings": findings}


def record(kb, task_id, run_id, value, log=None):
    """Append the findings event after a successful park. Never raises."""
    if value is None:
        return False
    try:
        conn = kb.connect()
        try:
            with kb.write_txn(conn):
                kb._append_event(conn, task_id, EVENT, value, run_id=run_id)
        finally:
            try:
                conn.close()
            except Exception:
                pass
        return True
    except Exception as exc:
        if log is not None:
            # The park already stands. Type only: the payload is model text.
            log.error("build-graph review findings NOT recorded for %s: %s", task_id, type(exc).__name__)
        return False
