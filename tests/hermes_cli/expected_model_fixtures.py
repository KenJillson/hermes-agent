"""Explicit protected classifications for graph behavior tests."""
import hashlib
import json


def contract_kwargs(task_id="t_a1", component="main", *, classification="unconstrained",
                    expected=None):
    mode = "unconstrained" if classification == "unconstrained" else "model"
    activities = {name: ({"mode": mode} if expected is None else
                         {"mode": mode, "expected": dict(expected)})
                  for name in ("plan_review", "code_review", "re_review")}
    contract = {
        "board_database": "/isolated/test-board.db",
        "task_id": task_id,
        "component": component,
        "workflow_classification": classification,
        "contract_id": "offline-test",
        "contract_version": 1,
        "authorization_ref": None if classification == "unconstrained" else "test-only",
        "protocol_sha256": "b" * 64,
        "activity_constraints": activities,
    }
    raw = json.dumps(contract, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode()
    return {"review_contract": contract,
            "review_contract_sha256": hashlib.sha256(raw).hexdigest()}
