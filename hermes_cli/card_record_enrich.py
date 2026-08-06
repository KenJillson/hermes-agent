#!/usr/bin/env python3
# card_record_enrich.py — shared task_class classifier + structured-LESSON parser.
#
# Destination on box: ~/.hermes/hermes-agent/hermes_cli/card_record_enrich.py
# (native file on branch hermes-local; part of CD-009, pushed to fork).
#
# ONE implementation of the two record-enrichment primitives, imported by BOTH:
#   * hermes_cli.kanban_db._emit_card_record  — the AT-EMIT route (records are
#     born classified + lesson-structured; D2.4 on-card + D2.6 at-emit, CD-009),
#   * scripts/classify-task-records.py / scripts/structure-lessons.py — the
#     RECORD-ONLY backfill route (repairs records emitted before CD-009).
#
# Keeping both routes on one module honours the arch §20 vocabulary rule
# ("task_class lives in ONE place; never rename") and the scripts' own "single
# swap point" design, so the at-emit and backfill routes can never diverge.
#
# Pure + stdlib-only (re). Deterministic and offline — no model / llama-server
# dependency. classify() is the swap point for a future qwen `classify` activity.
#
# The emit-facing helpers (classify_at_emit / structure_lesson_at_emit) are
# FAIL-SOFT: on ANY internal error they return the pre-enrichment GATED form, so
# a bug here degrades a record to "backfillable later" and NEVER blocks emit or
# loses the record. (emit is best-effort/try-except at its call sites; a raise
# from here would cost the whole record — the one thing D2.8 coverage forbids.)

import re

# ============================ task_class (D2.4) ==============================

CLASSIFIER_VERSION = "rules-v1"

# Closed vocabulary (the G4 axis). ONE place. Add rarely; NEVER rename — a rename
# breaks the G4 time series and strands frozen tags in immutable records (§20).
TASK_CLASSES = ("infra", "test", "feature", "fix", "docs", "research")
DEFAULT_CLASS = "feature"                 # generic build/implement work
# Tie-break priority when two classes score equal (most-specific first).
CLASS_ORDER = ("test", "fix", "docs", "research", "infra", "feature")

# Heuristic keyword signals — auditable/tunable, scored case-insensitively
# against title+body. A qwen classify can replace classify() wholesale later.
CLASS_KEYWORDS = {
    "test":     ["probe", "verification", "verify", "controlled test",
                 "regression test", "gate-test", "acceptance", "test the"],
    "fix":      ["fix", "bug", "broken", "hotfix", "repair", "defect",
                 "regression", "root-cause"],
    "docs":     ["readme", "documentation", "changelog", "handoff",
                 "write-up", "docs", "version-instruction"],
    "research": ["research", "investigate", "spike", "explore", "evaluate",
                 "scoping", "analysis", "compare", "feasibility"],
    "infra":    ["cron", "deploy", "pipeline", "backup", "monitor", "sweep",
                 "install", "provision", "session-start", "wiring", "script",
                 "ledger", "snapshot", "harness"],
    "feature":  ["implement", "build", "add ", "create", "support", "enable"],
}

_GATED_TASK_CLASS = "gated:D2.4"


def classify(title, body):
    """Return (task_class, reason). Scored keyword hits; highest wins,
    CLASS_ORDER breaks ties; no signal -> DEFAULT_CLASS. Verbatim from
    classify-task-records.py (the swap point for a future qwen backend)."""
    text = f"{title or ''}\n{body or ''}".lower()
    scores, hits = {}, {}
    for cls, kws in CLASS_KEYWORDS.items():
        matched = [k.strip() for k in kws if k.lower() in text]
        if matched:
            scores[cls] = len(matched)
            hits[cls] = matched
    if not scores:
        return DEFAULT_CLASS, "default:no-keyword-signal"
    top = max(scores.values())
    winners = [c for c in CLASS_ORDER if scores.get(c) == top]
    chosen = winners[0]
    return chosen, f"rules:{'+'.join(hits[chosen])}"


def classify_at_emit(title, body):
    """Emit-path helper. Returns (task_class, task_class_source, classified_by).

    Populated => source null (D2.8 null-discipline; provenance lives in
    meta.classified_by, not the source tag). FAIL-SOFT to the gated form so a
    classifier bug degrades to backfillable, never loses the record."""
    try:
        cls, _reason = classify(title, body)
        if cls not in TASK_CLASSES:            # defensive: never emit an off-axis tag
            return None, _GATED_TASK_CLASS, None
        return cls, None, CLASSIFIER_VERSION
    except Exception:
        return None, _GATED_TASK_CLASS, None


# ============================ LESSON (D2.6) ==================================

_GATED_LESSON = "gated:D2.6"
_CONFIDENCE = {"high", "medium", "low"}
_CARD_RE = re.compile(r"^t_[0-9a-f]{8}$")
_MARKER_RE = re.compile(r"\[(\w+)=([^\]]*)\]")


def parse_lesson(text):
    """Return (clean_text, scope, confidence, supersedes, evidence_ref,
    n_markers). Only the four known markers are consumed; unknown markers are
    left in text. Fail-closed on bad values. Verbatim from structure-lessons.py."""
    scope = confidence = supersedes = evidence = None
    n = 0
    consumed_spans = []
    for m in _MARKER_RE.finditer(text or ""):
        key, val = m.group(1).lower(), m.group(2).strip()
        if key == "scope":
            scope = val or None
        elif key == "confidence":
            confidence = val.lower() if val.lower() in _CONFIDENCE else None
        elif key == "supersedes":
            supersedes = val if _CARD_RE.match(val) else None
        elif key == "evidence":
            evidence = val or None
        else:
            continue                            # unknown marker: leave in text
        n += 1
        consumed_spans.append(m.span())
    clean = text or ""
    for a, b in sorted(consumed_spans, reverse=True):
        clean = clean[:a] + clean[b:]
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean, scope, confidence, supersedes, evidence, n


def structure_lesson_at_emit(raw_text):
    """Emit-path helper. One fully-structured lessons[] entry from one raw LESSON
    body (caller has already stripped the 'LESSON:' prefix).

    Uses the emit's SHARED detail_source convention (one tag for the entry, not
    per-field *_source siblings) — so it stays OUTSIDE the D2.8 sibling-pair
    null-discipline exactly like the current emit and the record-only pass; NO
    integrity change is required. FAIL-SOFT to the bare gated entry."""
    try:
        clean, scope, conf, sup, ev, n = parse_lesson(raw_text)
        return {
            "text": clean,
            "scope": scope,
            "confidence": conf,
            "supersedes": sup,               # field the pre-CD-009 emit did not carry
            "evidence_ref": ev,
            "detail_source": None if n else "null:unstructured-lesson",
        }
    except Exception:
        return {
            "text": raw_text,
            "scope": None,
            "confidence": None,
            "supersedes": None,
            "evidence_ref": None,
            "detail_source": _GATED_LESSON,
        }


# Self-test: `python3 card_record_enrich.py` runs a couple of assertions offline.
if __name__ == "__main__":
    c, s, _by = classify_at_emit("Fix broken regression in the parser", "")
    assert c == "fix" and s is None, (c, s)
    e = structure_lesson_at_emit("[scope=parser][confidence=high] validate first")
    assert e["scope"] == "parser" and e["confidence"] == "high", e
    assert e["detail_source"] is None and e["text"] == "validate first", e
    bare = structure_lesson_at_emit("just a plain note")
    assert bare["detail_source"] == "null:unstructured-lesson", bare
    dft, dsrc, _ = classify_at_emit("", "")
    assert dft == "feature" and dsrc is None, (dft, dsrc)
    print("card_record_enrich self-test OK")
