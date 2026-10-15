import contextlib
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "hermes_cli" / "kanban_db.py"
MAINTENANCE_SOURCE = ROOT / "hermes_cli" / "maintenance_gate.py"
LEDGER_MD5 = "54f97f847ccb285a55f4f2f2af8d21e0"


def load_candidate():
    package = types.ModuleType("hermes_cli")
    package.__path__ = []
    gate_module = types.ModuleType("hermes_cli.maintenance_gate")
    admission_calls = []
    gate_module.admitted = lambda fn: fn
    gate_module.exclusive_admission = lambda: None
    gate_module.hold_runtime_for_process = lambda: None
    sqlite_module = types.ModuleType("hermes_cli.sqlite_util")
    sqlite_module.add_column_if_missing = lambda *args, **kwargs: None
    tools = types.ModuleType("toolsets")
    tools.get_toolset_names = lambda: []
    name = "qualified_prelaunch_kanban"
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {
        "hermes_cli": package,
        "hermes_cli.maintenance_gate": gate_module,
        "hermes_cli.sqlite_util": sqlite_module,
        "toolsets": tools,
        name: module,
    }):
        spec.loader.exec_module(module)
    module._qualification_admission_calls = admission_calls
    return module


class Fixture(unittest.TestCase):
    def setUp(self):
        self.m = load_candidate()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.db_path = self.root / "kanban.db"
        self.db_path = Path(os.path.realpath(self.db_path))
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.requests = self.root / "requests"
        self.requests.mkdir(mode=0o700)
        self.db = sqlite3.connect(self.db_path)
        self.addCleanup(self.db.close)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            CREATE TABLE tasks(
                id TEXT PRIMARY KEY,status TEXT,assignee TEXT,current_run_id INTEGER,
                claim_lock TEXT,worker_pid INTEGER,workspace_path TEXT,
                graph_executor TEXT,last_failure_error TEXT
            );
            CREATE TABLE task_runs(
                id INTEGER PRIMARY KEY,task_id TEXT,status TEXT,outcome TEXT,
                started_at INTEGER,ended_at INTEGER,worker_pid INTEGER,
                error TEXT,metadata TEXT
            );
            CREATE TABLE task_events(
                id INTEGER PRIMARY KEY AUTOINCREMENT,task_id TEXT,run_id INTEGER,
                kind TEXT,payload TEXT,created_at INTEGER
            );
            CREATE TABLE task_comments(body TEXT,task_id TEXT,created_at INTEGER);
        """)
        self.db.execute(
            "INSERT INTO tasks VALUES (?,?,?,?,?,?,?,?,?)",
            ("t_smoke", "blocked", None, None, None, None,
             str(self.workspace), "build_graph", None),
        )
        metadata = {"failures": 1, "trigger_outcome": "spawn_failed",
                    "effective_limit": 1, "limit_source": "task"}
        self.db.execute(
            "INSERT INTO task_runs VALUES (?,?,?,?,?,?,?,?,?)",
            (220, "t_smoke", "gave_up", "gave_up", 10, 10, None,
             "workspace: parent confinement changed", json.dumps(metadata)),
        )
        self.event("claimed", {"lock": "fixture:1", "expires": 20,
                               "run_id": 220}, 220)
        self.event("gave_up", {"failures": 1, "effective_limit": 1,
                               "limit_source": "task",
                               "error": "workspace: parent confinement changed",
                               "trigger_outcome": "spawn_failed"}, 220)
        self.db.commit()
        self.m._MODEL_REQUEST_ROOT = self.requests
        self.m._MODEL_REQUEST_UID = os.geteuid()
        self.m._MODEL_REQUEST_GID = os.getegid()
        self.m._check_file_length_invariant = lambda conn: None
        self.m.kanban_db_path = lambda board=None: self.db_path
        self.idle_calls = 0
        self.active = False
        self.exclusive_entries = 0
        self.admission_lock = self.root / "admission.lock"
        self.admission_lock.write_bytes(b"")
        @contextlib.contextmanager
        def exclusive_admission():
            with self.admission_lock.open("rb") as lock:
                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise RuntimeError("exclusive Michael admission is busy") from exc
                self.exclusive_entries += 1
                def check():
                    self.idle_calls += 1
                    if self.active:
                        raise RuntimeError("active run exists")
                check()
                yield check
                check()
        self.m._maintenance_exclusive_admission = exclusive_admission
        self.lock_calls = 0
        @contextlib.contextmanager
        def dispatch_lock(path):
            self.lock_calls += 1
            self.assertEqual(Path(path), self.db_path)
            yield True
        self.m._dispatch_tick_lock = dispatch_lock
        self.pg_scope_count = 0
        self.pg_available = True
        self.postgres_probe = self.m._prelaunch_postgres_evidence
        def postgres(board_database, task_id):
            if not self.pg_available:
                raise RuntimeError("PostgreSQL checkpoint evidence unavailable")
            return {"role": "michael_checkpoint", "database": "michael_checkpoints",
                    "postgres_scopes": self.pg_scope_count}
        self.m._prelaunch_postgres_evidence = postgres
        self.provider_envelopes = []
        self.provider_available = True
        self.ledger_md5 = LEDGER_MD5
        self.provider_probe = self.m._prelaunch_provider_evidence
        def provider(task_id, expected_ledger_md5):
            if not self.provider_available:
                raise RuntimeError("provider evidence unavailable")
            if self.provider_envelopes or expected_ledger_md5 != self.ledger_md5:
                raise RuntimeError("provider envelope or ledger evidence differs")
            return {"provider_envelopes": [], "ledger_md5": self.ledger_md5}
        self.m._prelaunch_provider_evidence = provider

    def event(self, kind, payload, run_id):
        self.db.execute(
            "INSERT INTO task_events(task_id,run_id,kind,payload,created_at) "
            "VALUES (?,?,?,?,?)",
            ("t_smoke", run_id, kind, json.dumps(payload), 10),
        )

    def reconcile(self):
        return self.m.reconcile_prelaunch_no_worker(
            self.db, "t_smoke", 220, board="default",
            expected_ledger_md5=LEDGER_MD5,
        )

    def count_records(self):
        return self.db.execute(
            "SELECT count(*) FROM task_events WHERE kind=?",
            (self.m._PRELAUNCH_RECONCILIATION_KIND,),
        ).fetchone()[0]

    def test_success_idempotence_and_run_unchanged(self):
        before = dict(self.db.execute(
            "SELECT * FROM task_runs WHERE id=220").fetchone())
        first = self.reconcile()
        second = self.reconcile()
        self.assertEqual(first, second)
        self.assertEqual(self.count_records(), 1)
        self.assertEqual(dict(self.db.execute(
            "SELECT * FROM task_runs WHERE id=220").fetchone()), before)
        self.assertEqual(first["board_database"], str(self.db_path))
        self.assertEqual(first["task_id"], "t_smoke")
        self.assertEqual(first["run_id"], 220)
        self.assertEqual(first["classification"], "prelaunch_no_worker")
        self.assertEqual(first["evidence_sha256"], hashlib.sha256(
            self.m._canonical_json(first["evidence"])).hexdigest())
        self.assertIsNone(self.m.check_respawn_guard(self.db, "t_smoke"))
        self.assertGreaterEqual(self.idle_calls, 8)
        self.assertEqual(self.lock_calls, 2)
        self.assertEqual(self.exclusive_entries, 2)
        self.assertEqual(self.m._qualification_admission_calls, [])

    def test_wrong_run_and_task_binding_refuse(self):
        self.db.execute(
            "INSERT INTO tasks VALUES (?,?,?,?,?,?,?,?,?)",
            ("t_other", "blocked", None, None, None, None,
             str(self.workspace), "build_graph", None),
        )
        self.db.commit()
        with self.assertRaisesRegex(RuntimeError, "proven pre-launch"):
            self.m.reconcile_prelaunch_no_worker(
                self.db, "t_smoke", 221, board="default",
                expected_ledger_md5=LEDGER_MD5)
        with self.assertRaisesRegex(RuntimeError, "proven pre-launch"):
            self.m.reconcile_prelaunch_no_worker(
                self.db, "t_other", 220, board="default",
                expected_ledger_md5=LEDGER_MD5)
        self.assertEqual(self.count_records(), 0)

    def test_wrong_board_and_nonblocked_state_refuse(self):
        other = self.root / "other.db"
        original = self.m.kanban_db_path
        self.m.kanban_db_path = lambda board=None: (
            other if board == "other" else original(board)
        )
        with self.assertRaisesRegex(RuntimeError, "canonical board"):
            self.m.reconcile_prelaunch_no_worker(
                self.db, "t_smoke", 220, board="other",
                expected_ledger_md5=LEDGER_MD5)
        self.db.execute("UPDATE tasks SET status='ready' WHERE id='t_smoke'")
        self.db.commit()
        with self.assertRaisesRegex(RuntimeError, "blocked, unassigned"):
            self.reconcile()
        self.assertEqual(self.count_records(), 0)

    def test_changed_failure_evidence_refuses(self):
        self.db.execute("UPDATE task_runs SET error='workspace: another failure' WHERE id=220")
        self.db.commit()
        with self.assertRaisesRegex(RuntimeError, "proven pre-launch"):
            self.reconcile()
        self.assertEqual(self.count_records(), 0)

    def test_worker_evidence_refuses(self):
        self.event("spawned", {"pid": 1234}, 220)
        self.db.commit()
        with self.assertRaisesRegex(RuntimeError, "event path"):
            self.reconcile()
        self.assertEqual(self.count_records(), 0)

    def test_worker_pid_evidence_refuses(self):
        self.db.execute("UPDATE task_runs SET worker_pid=1234 WHERE id=220")
        self.db.commit()
        with self.assertRaisesRegex(RuntimeError, "proven pre-launch"):
            self.reconcile()
        self.assertEqual(self.count_records(), 0)

    def test_request_lock_contention_refuses(self):
        lock_path = self.requests / (hashlib.sha256(b"t_smoke").hexdigest() + ".lock")
        with lock_path.open("a+b") as held:
            os.chmod(lock_path, 0o600)
            fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "task request evidence is busy"):
                self.reconcile()
        self.assertEqual(self.count_records(), 0)

    def test_exclusive_admission_contention_refuses(self):
        with self.admission_lock.open("rb") as held:
            fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "exclusive Michael admission is busy"):
                self.reconcile()
        self.assertEqual(self.lock_calls, 0)
        self.assertEqual(self.count_records(), 0)

    def test_request_evidence_refuses(self):
        key = hashlib.sha256(b"t_smoke").hexdigest()
        (self.requests / (key + "-1234abcd.json")).write_text("{}")
        with self.assertRaisesRegex(RuntimeError, "durable model request"):
            self.reconcile()
        self.assertEqual(self.count_records(), 0)

    def test_filesystem_checkpoint_evidence_refuses(self):
        cp = self.workspace / ".graph-checkpoints"
        cp.mkdir()
        identity = (str(self.db_path) + "\0t_smoke").encode()
        (cp / ("postgres-" + hashlib.sha256(identity).hexdigest() + ".json")).write_text("{}")
        with self.assertRaisesRegex(RuntimeError, "filesystem checkpoint"):
            self.reconcile()
        self.assertEqual(self.count_records(), 0)

    def test_postgres_checkpoint_evidence_refuses(self):
        self.pg_scope_count = 1
        with self.assertRaisesRegex(RuntimeError, "PostgreSQL checkpoint"):
            self.reconcile()
        self.assertEqual(self.count_records(), 0)

    def test_postgres_probe_drops_identity_and_validates_peer(self):
        good = types.SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"role": "michael_checkpoint",
                               "database": "michael_checkpoints",
                               "postgres_scopes": 0}).encode(),
        )
        with patch.object(self.m.subprocess, "run", return_value=good) as run:
            self.assertEqual(
                self.postgres_probe(str(self.db_path), "t_smoke"),
                {"role": "michael_checkpoint", "database": "michael_checkpoints",
                 "postgres_scopes": 0},
            )
            command = run.call_args.args[0]
            self.assertTrue(command[0].endswith("/setpriv"))
            self.assertIn("--reuid=1000", command)
            self.assertIn("--no-new-privs", command)
        wrong_peer = types.SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"role": "postgres", "database": "michael_checkpoints",
                               "postgres_scopes": 0}).encode(),
        )
        with patch.object(self.m.subprocess, "run", return_value=wrong_peer):
            with self.assertRaisesRegex(RuntimeError, "evidence invalid"):
                self.postgres_probe(str(self.db_path), "t_smoke")
        unavailable = types.SimpleNamespace(returncode=2, stdout=b"")
        with patch.object(self.m.subprocess, "run", return_value=unavailable):
            with self.assertRaisesRegex(RuntimeError, "evidence unavailable"):
                self.postgres_probe(str(self.db_path), "t_smoke")

    def test_provider_envelope_ledger_and_unavailable_evidence_refuse(self):
        self.provider_envelopes = ["t_smoke-deadbeef-result.json"]
        with self.assertRaisesRegex(RuntimeError, "provider envelope or ledger"):
            self.reconcile()
        self.provider_envelopes = []
        self.ledger_md5 = "0" * 32
        with self.assertRaisesRegex(RuntimeError, "provider envelope or ledger"):
            self.reconcile()
        self.ledger_md5 = LEDGER_MD5
        self.provider_available = False
        with self.assertRaisesRegex(RuntimeError, "provider evidence unavailable"):
            self.reconcile()
        self.assertEqual(self.count_records(), 0)

    def test_provider_probe_strict_result_validation(self):
        good = types.SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"provider_envelopes": [],
                               "ledger_md5": LEDGER_MD5}).encode(),
        )
        with patch.object(self.m.subprocess, "run", return_value=good) as run:
            self.assertEqual(
                self.provider_probe("t_smoke", LEDGER_MD5),
                {"provider_envelopes": [], "ledger_md5": LEDGER_MD5},
            )
            self.assertTrue(run.call_args.args[0][0].endswith("/setpriv"))
        envelope = types.SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"provider_envelopes": ["t_smoke-a-result.json"],
                               "ledger_md5": LEDGER_MD5}).encode(),
        )
        with patch.object(self.m.subprocess, "run", return_value=envelope):
            with self.assertRaisesRegex(RuntimeError, "provider envelope or ledger"):
                self.provider_probe("t_smoke", LEDGER_MD5)
        unavailable = types.SimpleNamespace(returncode=255, stdout=b"")
        with patch.object(self.m.subprocess, "run", return_value=unavailable):
            with self.assertRaisesRegex(RuntimeError, "provider evidence unavailable"):
                self.provider_probe("t_smoke", LEDGER_MD5)

    def test_active_run_refuses_before_lock_or_write(self):
        self.active = True
        with self.assertRaisesRegex(RuntimeError, "active run"):
            self.reconcile()
        self.assertEqual(self.lock_calls, 0)
        self.assertEqual(self.count_records(), 0)

    def test_additional_unreconciled_run_still_blocks(self):
        self.reconcile()
        self.db.execute(
            "INSERT INTO task_runs VALUES (?,?,?,?,?,?,?,?,?)",
            (221, "t_smoke", "crashed", "crashed", 30, 40, 123,
             "later uncertain run", json.dumps({"trigger_outcome": "crashed"})),
        )
        self.db.commit()
        self.assertEqual(
            self.m.check_respawn_guard(self.db, "t_smoke"),
            "a2_prior_run_requires_reconciliation",
        )

    def test_tampered_record_does_not_exempt_run(self):
        self.reconcile()
        row = self.db.execute(
            "SELECT id,payload FROM task_events WHERE kind=?",
            (self.m._PRELAUNCH_RECONCILIATION_KIND,),
        ).fetchone()
        value = json.loads(row["payload"])
        value["run_id"] = 999
        self.db.execute("UPDATE task_events SET payload=? WHERE id=?",
                        (json.dumps(value), row["id"]))
        self.db.commit()
        self.assertEqual(
            self.m.check_respawn_guard(self.db, "t_smoke"),
            "a2_prior_run_requires_reconciliation",
        )

    def test_historical_run_changes_after_reconciliation_restore_guard(self):
        cases = [
            ("error", "workspace: changed after reconciliation"),
            ("worker_pid", 1234),
            ("metadata", json.dumps({"trigger_outcome": "changed"})),
        ]
        for column, value in cases:
            with self.subTest(column=column):
                self.reconcile()
                original = self.db.execute(
                    "SELECT " + column + " FROM task_runs WHERE id=220"
                ).fetchone()[0]
                self.db.execute(
                    "UPDATE task_runs SET " + column + "=? WHERE id=220", (value,)
                )
                self.db.commit()
                self.assertEqual(
                    self.m.check_respawn_guard(self.db, "t_smoke"),
                    "a2_prior_run_requires_reconciliation",
                )
                self.db.execute(
                    "UPDATE task_runs SET " + column + "=? WHERE id=220", (original,)
                )
                self.db.commit()

    def test_historical_event_changes_and_launch_evidence_restore_guard(self):
        self.reconcile()
        claimed = self.db.execute(
            "SELECT id,payload FROM task_events WHERE task_id='t_smoke' "
            "AND run_id=220 AND kind='claimed'"
        ).fetchone()
        compact = json.dumps(json.loads(claimed["payload"]), separators=(",", ":"))
        self.assertNotEqual(compact, claimed["payload"])
        self.db.execute("UPDATE task_events SET payload=? WHERE id=?",
                        (compact, claimed["id"]))
        self.db.commit()
        self.assertEqual(self.m.check_respawn_guard(self.db, "t_smoke"),
                         "a2_prior_run_requires_reconciliation")
        self.db.execute("UPDATE task_events SET payload=? WHERE id=?",
                        (json.dumps({"run_id": 220, "changed": True}), claimed["id"]))
        self.db.commit()
        self.assertEqual(self.m.check_respawn_guard(self.db, "t_smoke"),
                         "a2_prior_run_requires_reconciliation")
        self.db.execute("UPDATE task_events SET payload=? WHERE id=?",
                        (claimed["payload"], claimed["id"]))
        self.event("spawned", {"pid": 1234}, 220)
        self.db.commit()
        self.assertEqual(self.m.check_respawn_guard(self.db, "t_smoke"),
                         "a2_prior_run_requires_reconciliation")

    def test_historical_metadata_byte_change_restores_guard(self):
        self.reconcile()
        original = self.db.execute(
            "SELECT metadata FROM task_runs WHERE id=220"
        ).fetchone()[0]
        compact = json.dumps(json.loads(original), separators=(",", ":"))
        self.assertNotEqual(compact, original)
        self.db.execute("UPDATE task_runs SET metadata=? WHERE id=220", (compact,))
        self.db.commit()
        self.assertEqual(self.m.check_respawn_guard(self.db, "t_smoke"),
                         "a2_prior_run_requires_reconciliation")

    def test_later_task_state_and_external_artifacts_do_not_invalidate_history(self):
        self.reconcile()
        self.db.execute("UPDATE tasks SET status='ready',assignee='builder' "
                        "WHERE id='t_smoke'")
        key = hashlib.sha256(b"t_smoke").hexdigest()
        (self.requests / (key + "-later123.json")).write_text("{}")
        cp = self.workspace / ".graph-checkpoints"
        cp.mkdir()
        (cp / "checkpoints.json").write_text("{}")
        self.pg_rows = [{"scope_id": "b" * 32, "state": "active",
                         "workflow_schema": 3}]
        self.assertIsNone(self.m.check_respawn_guard(self.db, "t_smoke"))

    def test_malformed_record_nested_values_fail_closed(self):
        self.reconcile()
        row = self.db.execute(
            "SELECT id,payload FROM task_events WHERE kind=?",
            (self.m._PRELAUNCH_RECONCILIATION_KIND,),
        ).fetchone()
        original = json.loads(row["payload"])
        variants = []
        for path, value in (
            (("evidence",), []),
            (("evidence", "run"), []),
            (("evidence", "run", "metadata"), []),
            (("evidence", "events"), {}),
            (("evidence", "events", 0, "payload"), []),
            (("evidence", "external"), []),
            (("evidence", "external", "provider_envelopes"), "none"),
        ):
            value_copy = json.loads(json.dumps(original))
            target = value_copy
            for part in path[:-1]:
                target = target[part]
            target[path[-1]] = value
            variants.append(value_copy)
        for value in variants:
            with self.subTest(value=value):
                self.db.execute("UPDATE task_events SET payload=? WHERE id=?",
                                (json.dumps(value), row["id"]))
                self.db.commit()
                self.assertEqual(
                    self.m.check_respawn_guard(self.db, "t_smoke"),
                    "a2_prior_run_requires_reconciliation",
                )
        self.db.execute("UPDATE task_events SET payload=? WHERE id=?",
                        (row["payload"], row["id"]))
        self.db.execute("UPDATE task_runs SET metadata='[]' WHERE id=220")
        self.db.commit()
        self.assertEqual(self.m.check_respawn_guard(self.db, "t_smoke"),
                         "a2_prior_run_requires_reconciliation")

    def test_malformed_authoritative_event_payload_fails_closed(self):
        self.reconcile()
        self.db.execute("UPDATE task_events SET payload='[]' WHERE task_id='t_smoke' "
                        "AND run_id=220 AND kind='claimed'")
        self.db.commit()
        self.assertEqual(self.m.check_respawn_guard(self.db, "t_smoke"),
                         "a2_prior_run_requires_reconciliation")


class ExclusiveAdmissionFixture(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location(
            "qualified_prelaunch_maintenance", MAINTENANCE_SOURCE)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.lock_path = self.root / "admission.lock"
        self.lock_path.write_bytes(b"")
        outer = self
        class Gate:
            active = False
            idle_calls = 0
            def _directory(self):
                return os.open(outer.root, os.O_RDONLY | os.O_DIRECTORY)
            def _file(self, directory, name):
                return os.open(name, os.O_RDONLY, dir_fd=directory)
            def _state(self, directory):
                return {"version": 1, "mode": "open", "transaction": ""}
            def _open(self, state):
                if state["mode"] != "open":
                    raise RuntimeError("closed")
            def _assert_no_worker_inhibits(self, directory):
                return None
            def no_active_runs(self):
                self.idle_calls += 1
                if self.active:
                    raise RuntimeError("active run exists")
        self.gate = Gate()
        protected = types.SimpleNamespace(Gate=lambda: self.gate)
        self.protected = patch.object(
            self.module, "_protected_module", return_value=protected)
        self.root_identity = patch.object(self.module.os, "geteuid", return_value=0)
        self.protected.start()
        self.root_identity.start()
        self.addCleanup(self.protected.stop)
        self.addCleanup(self.root_identity.stop)

    def test_real_exclusive_lock_and_checks(self):
        with self.module.exclusive_admission() as check:
            check()
            with self.lock_path.open("rb") as contender:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(
                        contender.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.assertGreaterEqual(self.gate.idle_calls, 3)

    def test_real_lock_contention_refuses(self):
        with self.lock_path.open("rb") as held:
            fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "exclusive Michael admission is busy"):
                with self.module.exclusive_admission():
                    pass

    def test_active_run_refuses_and_releases_lock(self):
        self.gate.active = True
        with self.assertRaisesRegex(RuntimeError, "active run"):
            with self.module.exclusive_admission():
                pass
        self.gate.active = False
        with self.module.exclusive_admission():
            pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
