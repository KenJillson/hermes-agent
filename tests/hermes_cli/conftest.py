"""Fixtures shared across hermes_cli kanban tests."""

from __future__ import annotations

from pathlib import Path
from contextlib import contextmanager
import pytest


@pytest.fixture(autouse=True)
def _canonical_graph_config_root(monkeypatch):
    """Keep graph security modules bound to the colocated config checkout.

    The repository-wide fixture deliberately replaces HERMES_HOME per test.
    Graph tests require the separately authoritative config repository and must
    name it explicitly, matching the production worker contract.
    """
    checkout = Path(__file__).resolve().parents[3]
    for candidate in (checkout / "config", checkout):
        if (candidate / "lib/model_call/sanitizer.py").is_file():
            monkeypatch.setenv("HERMES_CONFIG_ROOT", str(candidate))
            break
    protected = Path("/usr/local/lib/michael-maintenance")
    if not protected.exists():
        from hermes_cli import maintenance_gate

        @contextmanager
        def admitted():
            yield

        class FixtureAdmission:
            admit = staticmethod(admitted)

            @staticmethod
            @contextmanager
            def runtime():
                yield

        monkeypatch.setattr(maintenance_gate, "_protected_module",
                            lambda: FixtureAdmission)
        monkeypatch.setattr(maintenance_gate, "hold_runtime_for_process",
                            lambda: None)


@pytest.fixture
def all_assignees_spawnable(monkeypatch):
    """Pretend every assignee maps to a real Hermes profile.

    Most dispatcher tests use synthetic assignees ("alice", "bob") that
    don't correspond to actual profile directories on disk. Without this
    patch, the dispatcher's profile-exists guard (PR #20105) routes
    those tasks into ``skipped_nonspawnable`` instead of spawning, which
    would break tests that assert spawn behavior.
    """
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)


@pytest.fixture(autouse=True)
def _suppress_concurrent_hermes_gate(request, monkeypatch):
    """Default ``_detect_concurrent_hermes_instances`` to ``[]`` for every test.

    The Windows update path now refuses to proceed when another
    ``hermes.exe`` is detected (issue #26670). On a developer's Windows
    machine running the test suite via ``hermes`` itself, this would
    flag the running agent as a concurrent instance and abort every
    ``cmd_update`` test. Tests that want to exercise the gate explicitly
    re-patch ``_detect_concurrent_hermes_instances`` with their own
    return value — autouse here gives a clean default without touching
    the rest of the suite.

    Tests that need to call the REAL function (e.g. unit tests for the
    helper itself) opt out with ``@pytest.mark.real_concurrent_gate``.
    """
    if request.node.get_closest_marker("real_concurrent_gate"):
        return
    try:
        from hermes_cli import main as _cli_main
    except Exception:
        return
    # raising=False: under pytest's per-test spawn isolation, a concurrent
    # xdist worker importing a module that transitively touches hermes_cli.main
    # can briefly expose a partially-initialized module object here — one where
    # _detect_concurrent_hermes_instances isn't defined yet. A bare setattr
    # would raise AttributeError and error the (unrelated) test. The attribute
    # always exists once main.py finishes importing, so a no-op when it's
    # transiently absent is the correct, race-free default.
    monkeypatch.setattr(
        _cli_main,
        "_detect_concurrent_hermes_instances",
        lambda *_a, **_k: [],
        raising=False,
    )
