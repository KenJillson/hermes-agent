"""Tracked Hermes adapter for the protected admission module; local candidate."""
import atexit
import functools
import importlib.util
import os
from pathlib import Path
import stat
import threading

_module = None
_load_lock = threading.Lock()
_runtime_context = None
_runtime_pid = None


def _protected_module():
    global _module
    with _load_lock:
        if _module is not None:
            return _module
        source = Path('/usr/local/lib/michael-maintenance/admission.py')
        for path in [*reversed(source.parents), source]:
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022:
                raise RuntimeError('Unprotected Michael admission module')
        spec = importlib.util.spec_from_file_location('_michael_protected_admission', source)
        if spec is None or spec.loader is None:
            raise RuntimeError('Michael admission module unavailable')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _module = module
        return module


def hold_runtime_for_process():
    """Enter before mutable runtime imports; retain until this process exits."""
    global _runtime_context, _runtime_pid
    if _runtime_pid == os.getpid() and _runtime_context is not None:
        return
    context = _protected_module().runtime()
    context.__enter__()
    _runtime_context, _runtime_pid = context, os.getpid()
    atexit.register(context.__exit__, None, None, None)


def admitted(function):
    @functools.wraps(function)
    def guarded(*args, **kwargs):
        with _protected_module().admit():
            return function(*args, **kwargs)
    return guarded
