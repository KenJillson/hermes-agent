"""D5.1 build-harness graph -- the typed workflow_state (design v2 section 4).

CD-030. Pure data plus helpers. Imports NOTHING outside the standard library:
no langgraph, no hermes_cli, no board connection. That is deliberate -- it keeps
this module offline-testable without the agent, and it is what lets the graph's
state contract be verified independently of the topology that consumes it
(CD-031).

Design section 4 pins this schema at D5.1. The checkpoint BACKEND is separable
and deferred (in-memory now, Postgres at D5.3, fork ruling 4); the SCHEMA is
part of the topology and is fixed here.

Two constraints from section 4.1 are enforced rather than documented:

  * JSON-SERIALIZABILITY, from the first node. Every field must survive
    json.dumps at all times. A node that parks a live object, a connection or a
    file handle in state passes every D5.1 test and fails only when a real
    checkpoint backend is attached -- late, in the phase that adds durability,
    where the failure reads as a checkpointer bug. assert_serializable() exists
    so that failure happens here instead.

  * NO STATE-HISTORY ENDPOINT. This module exposes no HTTP surface. That is a
    property of the module, not a setting to get right.

The state is a flat TypedDict with NO Annotated reducers. That was verified
behaviourally against langgraph 1.2.10 / langgraph-checkpoint 4.1.1 on the box
before this module was written: nodes replace fields wholesale rather than
accumulating into them, so reducers are not required.
"""

from __future__ import annotations

import json
from typing import Any, Literal, Optional, TypedDict

SCHEMA_VERSION = 1

# Ladder positions (design section 1 flow, section 2.2 rung caps).
RUNGS = ("none", "rung1", "rung2", "rung3")

# How objection_recurred was decided (design section 5, section 6.2).
# The structural tier is DROPPED at D5.1 (section 5.2): a conformant verdict
# carries neither a bucket dimension nor a locator, so a structural tier would
# reach its own indeterminate branch every time. The distribution is therefore
# classifier vs default, and a high `default` rate is the primary health signal
# -- it means the comparator is not deciding anything and the ladder is simply
# climbing.
RECURRENCE_TIERS = ("classifier", "default")

DISPUTE_CLASSES = ("approach_disputed", "impl_bug")


class Objection(TypedDict):
    """One objection as carried between review passes.

    Thinner than D5.1 v1 assumed, and deliberately so. The source read on
    2026-08-17 against lib/model_call/core.py established that there is NO
    verdict schema: ModelResult.verdict is `dict | None` holding whatever
    _parse_verdict extracted from free-form text, and the only shape assertion
    anywhere is VERDICT_CONTRACT, a prose string requiring a `findings` array
    with no locator of any kind. Items may legitimately be bare strings.

    `raw` therefore holds the item exactly as received, which is what section
    6.1 measures to decide whether reopening VERDICT_CONTRACT is ever justified.
    """

    text: str
    review_id: Optional[str]
    raw: Any


class WorkflowState(TypedDict):
    """Design section 4, implemented field-for-field."""

    schema_version: int

    # identity
    card_id: str
    component: str
    iteration: int

    # work product
    plan: str
    diff: str
    gate_summary: dict
    ac_record_path: str

    # model-call selection inputs (section 1, note 4). Read by cloud nodes,
    # never re-derived per call. policy.py promotes on ANY fired signal and
    # records rationale=promotion:<signals>; omitting a signal under-reviews and
    # inventing one overspends, so these live in typed state precisely so a
    # promotion cannot be reintroduced or lost by omission.
    component_count: int
    diff_files: int
    diff_added_lines: int
    security_sensitive: bool
    financial_sensitive: bool
    novel_architecture: bool
    re_review_after_severe: bool
    author_independence: bool
    directive: Optional[str]

    # ladder position
    rung: Literal["none", "rung1", "rung2", "rung3"]
    rung_attempts: dict
    cloud_review_calls: int

    # comparator state (section 5)
    objections_prior: list
    objections_current: list
    objection_recurred: Optional[bool]
    recurrence_tier: Optional[str]
    recurrence_confounded: bool
    dispute_class: Optional[str]

    # provenance for the author-independence assertion (section 5.3)
    implementer_model: Optional[str]
    reviewer_model: Optional[str]

    # evaluation apparatus bookkeeping (section 6). NOT in the section 4 list.
    # Added because the graph-classify writer is best-effort by necessity -- a
    # card must never die because a log write failed -- and a silently dropped
    # line is indistinguishable from an evaluation that never happened. Section
    # 6.4 revisits the fork-2 ruling after 20 LOGGED evaluations, so a corpus
    # with invisible holes would move those thresholds without anything saying
    # so. Recording the failure makes silence unambiguous.
    classify_log_failed: bool
    holdout: bool


def new_workflow_state(
    card_id: str,
    component: str,
    *,
    plan: str = "",
    diff: str = "",
    directive: Optional[str] = None,
    component_count: int = 1,
    security_sensitive: bool = False,
    financial_sensitive: bool = False,
    novel_architecture: bool = False,
) -> WorkflowState:
    """Build a fully-populated state. Every field is present from creation.

    No field is left absent to be filled in later: a partially-populated
    TypedDict defeats the point of pinning the schema, and a KeyError inside a
    node is a much worse failure than a wrong default.
    """
    return WorkflowState(
        schema_version=SCHEMA_VERSION,
        card_id=card_id,
        component=component,
        iteration=0,
        plan=plan,
        diff=diff,
        gate_summary={},
        ac_record_path="",
        component_count=component_count,
        diff_files=0,
        diff_added_lines=0,
        security_sensitive=security_sensitive,
        financial_sensitive=financial_sensitive,
        novel_architecture=novel_architecture,
        re_review_after_severe=False,
        author_independence=False,
        directive=directive,
        rung="none",
        rung_attempts={},
        cloud_review_calls=0,
        objections_prior=[],
        objections_current=[],
        objection_recurred=None,
        recurrence_tier=None,
        recurrence_confounded=False,
        dispute_class=None,
        implementer_model=None,
        reviewer_model=None,
        classify_log_failed=False,
        holdout=False,
    )


def assert_serializable(state: WorkflowState) -> None:
    """Raise TypeError naming the offending field if state cannot be checkpointed.

    Section 4.1's constraint, enforced from the first node rather than
    retrofitted at D5.3. Checked per-field so the error names the culprit --
    json.dumps on the whole dict reports only that something failed, which in a
    thirty-field state is close to useless.
    """
    for key, value in state.items():
        try:
            json.dumps({key: value})
        except TypeError as exc:
            raise TypeError(
                "workflow_state field %r is not JSON-serializable (%s). "
                "Section 4.1: state must be checkpointable at all times; a live "
                "object here passes every D5.1 test and fails at D5.3." % (key, exc)
            ) from exc


def selection_signals(state: WorkflowState) -> dict:
    """The honest signal set a cloud node passes to model-call.

    Read off typed state, never re-derived per call (section 1, note 4).
    Returns only the signals that are TRUE, since policy.py promotes on any
    fired signal -- passing False entries would be noise at best and, if the
    receiving end is ever truthy-tested, a silent over-promotion.
    """
    candidates = {
        "many_components": state["component_count"] > 1,
        "novel_architecture": state["novel_architecture"],
        "security_sensitive": state["security_sensitive"],
        "financial_sensitive": state["financial_sensitive"],
        "large_diff": state["diff_files"] > 1 or state["diff_added_lines"] > 0,
        "re_review_after_severe": state["re_review_after_severe"],
        "author_independence": state["author_independence"],
    }
    return {k: True for k, v in candidates.items() if v}


def validate(state: WorkflowState) -> None:
    """Structural invariants. Raises ValueError on the first violation."""
    if state["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            "schema_version %r != %r" % (state["schema_version"], SCHEMA_VERSION)
        )
    if state["rung"] not in RUNGS:
        raise ValueError("rung %r not in %r" % (state["rung"], RUNGS))
    tier = state["recurrence_tier"]
    if tier is not None and tier not in RECURRENCE_TIERS:
        raise ValueError("recurrence_tier %r not in %r" % (tier, RECURRENCE_TIERS))
    dc = state["dispute_class"]
    if dc is not None and dc not in DISPUTE_CLASSES:
        raise ValueError("dispute_class %r not in %r" % (dc, DISPUTE_CLASSES))
    if not state["card_id"]:
        raise ValueError("card_id is empty")
    assert_serializable(state)
