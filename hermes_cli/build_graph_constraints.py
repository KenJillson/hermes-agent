"""Protected classification and expected-model contracts for build_graph."""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path

REGISTRY_PATH = Path("/etc/michael-maintenance/review-constraints.json")
MAX_REGISTRY_BYTES = 1024 * 1024
REGISTRY_OWNER = (0, 0)
REGISTRY_MODE = 0o644
ACTIVITIES = ("plan_review", "code_review", "re_review")
ENTRY_KEYS = {"board_database", "task_id", "component",
              "workflow_classification", "contract_id", "contract_version",
              "authorization_ref", "protocol_sha256", "activity_constraints"}


class ConstraintRefused(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise ConstraintRefused(message)


def _pairs(items):
    out = {}
    for key, value in items:
        require(key not in out, "Duplicate registry JSON key")
        out[key] = value
    return out


def _bounded_text(value, name, limit=1024):
    require(isinstance(value, str) and value, "Invalid bounded %s" % name)
    try:
        bounded = len(value.encode("utf-8")) <= limit
    except UnicodeError as exc:
        raise ConstraintRefused("Invalid bounded %s" % name) from exc
    require(bounded, "Invalid bounded %s" % name)
    return value


def _expected(value):
    require(isinstance(value, dict)
            and set(value) == {"schema", "model", "lane", "invoke"},
            "Invalid expected-model fields")
    require(type(value["schema"]) is int and value["schema"] == 1,
            "Unsupported expected-model schema")
    for key in ("model", "lane", "invoke"):
        _bounded_text(value[key], "expected-model %s" % key, 256)
    return value


def parse_registry(raw, *, board_database, task_id, component):
    """Parse exact protected bytes and return (contract, canonical_sha256)."""
    require(isinstance(raw, bytes) and len(raw) <= MAX_REGISTRY_BYTES,
            "Registry exceeds byte bound")
    try:
        document = json.loads(raw, object_pairs_hook=_pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(
                ConstraintRefused("Nonfinite registry JSON")))
    except (ValueError, UnicodeError) as exc:
        if isinstance(exc, ConstraintRefused):
            raise
        raise ConstraintRefused("Invalid registry JSON") from exc
    require(isinstance(document, dict) and set(document) == {"version", "entries"},
            "Unexpected registry document fields")
    require(type(document["version"]) is int and document["version"] == 1,
            "Unsupported registry version")
    entries = document["entries"]
    require(isinstance(entries, list) and len(entries) <= 4096,
            "Invalid registry entry inventory")
    matches = []
    identities = set()
    for entry in entries:
        require(isinstance(entry, dict) and set(entry) == ENTRY_KEYS,
                "Unexpected registry entry fields")
        identity = tuple(_bounded_text(entry[k], k) for k in
                         ("board_database", "task_id", "component"))
        require(identity not in identities, "Duplicate registry identity")
        identities.add(identity)
        classification = entry["workflow_classification"]
        require(classification in ("constrained", "unconstrained"),
                "Unknown workflow classification")
        _bounded_text(entry["contract_id"], "contract_id")
        require(type(entry["contract_version"]) is int
                and 1 <= entry["contract_version"] <= 2 ** 31 - 1,
                "Invalid contract version")
        if classification == "constrained":
            _bounded_text(entry["authorization_ref"], "authorization_ref", 4096)
        else:
            require(entry["authorization_ref"] is None,
                    "Unconstrained classification cannot claim authorization")
        require(isinstance(entry["protocol_sha256"], str)
                and re.fullmatch(r"[0-9a-f]{64}", entry["protocol_sha256"]),
                "Invalid protocol SHA-256")
        activities = entry["activity_constraints"]
        require(isinstance(activities, dict) and set(activities) == set(ACTIVITIES),
                "Every review activity requires explicit classification")
        for activity, rule in activities.items():
            require(isinstance(rule, dict) and rule.get("mode") in
                    ("model", "forbidden", "unconstrained"),
                    "Invalid activity constraint")
            mode = rule["mode"]
            require(set(rule) == ({"mode", "expected"} if mode == "model" else {"mode"}),
                    "Unexpected activity constraint fields")
            if mode == "model":
                _expected(rule["expected"])
            if classification == "constrained":
                require(mode in ("model", "forbidden"),
                        "Constrained entry cannot contain unconstrained activity")
            else:
                require(mode == "unconstrained",
                        "Unconstrained entry must classify every activity unconstrained")
        if identity == (board_database, task_id, component):
            matches.append(entry)
    require(len(matches) == 1, "Explicit protected workflow classification required")
    canonical = json.dumps(matches[0], sort_keys=True, separators=(",", ":"),
                           allow_nan=False).encode()
    return json.loads(canonical), hashlib.sha256(canonical).hexdigest()


def _supported_registry_file(info):
    return (stat.S_ISREG(info.st_mode) and info.st_nlink == 1
            and (info.st_uid, info.st_gid) == REGISTRY_OWNER
            and stat.S_IMODE(info.st_mode) == REGISTRY_MODE
            and info.st_size <= MAX_REGISTRY_BYTES)


def _supported_registry_directory(info):
    return (stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode)
            and info.st_uid == 0 and not info.st_mode & 0o022
            and info.st_mode & stat.S_IXOTH)


def _protected_bytes():
    path = REGISTRY_PATH
    require(path.is_absolute(), "Registry path must be fixed and absolute")
    current = Path("/")
    for part in path.parts[1:-1]:
        current /= part
        info = current.lstat()
        require(_supported_registry_directory(info),
                "Unprotected registry directory")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        # The graph caller runs as the unprivileged jetson identity.  Exact
        # root:root 0644 makes the registry runtime-readable while retaining
        # root-only mutation authority.
        require(_supported_registry_file(info),
                "Unsupported registry file identity")
        raw = stream.read(MAX_REGISTRY_BYTES + 1)
        after = os.fstat(stream.fileno())
        entry = os.stat(path, follow_symlinks=False)
        signature = lambda value: (value.st_dev, value.st_ino, value.st_mode,
            value.st_uid, value.st_gid, value.st_nlink, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)
        require(len(raw) == info.st_size
                and signature(info) == signature(after) == signature(entry),
                "Registry changed during read")
    return raw


def load_review_contract(*, board_database, task_id, component):
    """Production entrypoint: fixed protected source, no caller path override."""
    contract, digest = parse_registry(
        _protected_bytes(), board_database=board_database,
        task_id=task_id, component=component)
    # The protected contract constrains policy vocabulary; it never supplies
    # policy output.  Prove the validator came from the canonical config root.
    import sys
    from hermes_constants import get_hermes_home
    lib = get_hermes_home() / "lib"
    if str(lib) not in sys.path:
        sys.path.append(str(lib))
    from model_call import constraints as model_constraints
    expected_origin = os.path.realpath(lib / "model_call" / "constraints.py")
    require(os.path.realpath(getattr(model_constraints, "__file__", "")) == expected_origin,
            "Canonical expected-model validator identity differs")
    for rule in contract["activity_constraints"].values():
        if rule["mode"] == "model":
            model_constraints.validate_expected_model(rule["expected"])
    return contract, digest


def activity_rule(contract, activity):
    require(isinstance(contract, dict), "Review contract missing")
    require(activity in ACTIVITIES, "Unknown review activity")
    try:
        return contract["activity_constraints"][activity]
    except (KeyError, TypeError) as exc:
        raise ConstraintRefused("Review activity classification missing") from exc
