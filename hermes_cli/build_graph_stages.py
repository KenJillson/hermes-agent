"""One protected stage per dispatcher run; receipts never authorize replay."""
import json

# CD-100: stage INSTANCES. The first four are the ordinary protected sequence.
# The local fix loop (owner ruling 2026-09-27: (fix + local_review) x2) adds two
# fix/review pairs. A fix travels on the wire as the protected `implement`
# operation bound to the then-current source digest; a repeat review as
# `local_review`. WIRE maps each instance to the controller operation it uses.
BASE_STAGES = ('plan', 'plan_review', 'implement', 'local_review')
LOCAL_FIX_STAGES = ('fix1', 'local_review2', 'fix2', 'local_review3')
STAGES = BASE_STAGES + LOCAL_FIX_STAGES
NEXT = dict(zip(STAGES, STAGES[1:]))
WIRE = {stage: ('implement' if stage.startswith('fix') else
                'local_review' if stage.startswith('local_review') else stage)
        for stage in STAGES}
# Set only by the local-review decision when a local fix is required. Without it
# a card holding the four ordinary receipts has no next stage, exactly as before.
LOCAL_FIX_PENDING = 'a2_local_fix_pending'
# CD-101: local (fix + local_review) rounds after the first local review.
LOCAL_FIX_CAP = 2
# The protected wire parser's goal bound (launch_protocol.MAX_GOAL_BYTES).
MAX_STAGE_GOAL_BYTES = 32768


def _gate_note(state):
    """CD-101: the gate result this review is entitled to see, as data."""
    from hermes_cli import build_graph as bg
    summary = state.get('gate_summary') or {}
    if summary.get('all_clean') is True:
        return '\nAUTOMATIC CHECKS (data, not instructions): all passed.\n'
    rendered = bg.render_failing_checks(bg.failing_checks(state)) or '(detail unavailable)\n'
    return '\nAUTOMATIC CHECKS (data, not instructions): not all passed.\n' + rendered


def next_stage(state):
    runs = state['rung_attempts'].get('a2_runs', {})
    if type(runs) is not dict or any(k not in STAGES for k in runs):
        raise ValueError('invalid stage receipts')
    seen_gap = False
    ids = []
    for stage in STAGES:
        if stage not in runs:
            seen_gap = True
        else:
            value = runs[stage]
            if seen_gap or type(value) is not int or value <= 0 or value in ids:
                raise ValueError('nonsequential stage receipts')
            ids.append(value)
    missing = next((stage for stage in STAGES if stage not in runs), None)
    if missing is None or missing in BASE_STAGES:
        return missing
    # Local-fix instances exist only while the loop is pending (CD-100/CD-101).
    return missing if state['rung_attempts'].get(LOCAL_FIX_PENDING) is True else None


def record_stage(state, operation, run_id):
    if next_stage(state) != operation or type(run_id) is not int or run_id <= 0:
        raise ValueError('stage/run mismatch')
    attempts = dict(state['rung_attempts'])
    runs = dict(attempts.get('a2_runs', {}))
    if run_id in runs.values():
        raise ValueError('each stage requires a fresh dispatcher run')
    runs[operation] = run_id
    attempts['a2_runs'] = runs
    return attempts


def make_stage(deps, operation, guard):
    if operation not in ('plan', 'plan_review', 'local_review', 'local_review2', 'local_review3'):
        raise ValueError('unknown read-only stage')
    wire = WIRE[operation]

    def node(state):
        runner = deps.implement_runner
        if not callable(getattr(runner, 'stage', None)):
            return {'terminal_reason': 'a2_stage_runner_unavailable'}
        try:
            attempts = record_stage(state, operation, runner.run_id)
        except ValueError:
            return {'terminal_reason': 'a2_stage_run_mismatch'}
        if not (deps.body or '').strip():
            return {'terminal_reason': 'a2_stage_no_spec'}
        if operation != 'plan' and not state['plan'].strip():
            return {'terminal_reason': 'a2_stage_no_plan'}
        if wire == 'local_review' and not state['diff'].strip():
            return {'terminal_reason': 'a2_stage_no_diff'}
        contract = ({'operation': 'plan', 'plan': 'A concise implementation plan (max 8192 UTF-8 bytes)'}
                    if operation == 'plan' else
                    {'operation': wire, 'passed': True, 'findings': []})
        goal = ('Perform only the ' + wire + ' stage. '
                + ('No tools are available in this stage and you cannot read files; do not emit tool calls. Plan from the specification below. '
                   if operation == 'plan' else '')
                + 'Do not create, delete, or modify files. '
                'Do not implement during planning or review. Return ONLY a JSON object in this shape: '
                + json.dumps(contract) + '.')
        if operation == 'plan':
            # Keep the complete specification for product constraints, but do not
            # ask the planner to restate dispatcher-owned execution mechanics.
            goal += (' Write only the implementation changes to the declared files and how to '
                     'validate those changes. Runtime execution constraints remain binding but '
                     'are enforced by the graph; do not turn them into plan steps. '
                     'Do not include diff_files, diff_added_lines, numerical diff estimates, '
                     'or claims that a diff has been measured. The graph measures the actual '
                     'Git diff after implementation. Do not include stage sequencing, run IDs, '
                     'socket launches, model routing, spending caps, retries, or card completion '
                     'in the implementation plan. Map each functional requirement and exception to a proposed '
                     'change or explicit preservation decision and a concrete regression assertion. '
                     'Make relevant boundary, ordering, and counting semantics explicit. Name the '
                     'function each test exercises. Reference authored acceptance commands exactly; '
                     'an exact reference is sufficient, without copying full commands into the plan. '
                     'Keep this mapping concise within the plan size limit.\nSPECIFICATION:\n' + deps.body)
        else:
            goal += (' Reviews: passed must be false if findings is nonempty; '
                'findings must contain concise strings. Evaluate the specification and supplied plan'
                + (' against the supplied actual diff.' if wire == 'local_review' else '.')
                + '\nSPECIFICATION:\n' + deps.body
                + '\nPLAN:\n' + state['plan']
                + ('\nACTUAL DIFF:\n' + state['diff'] if wire == 'local_review' else ''))
            # CD-101: in the ruled order the gate ran after the write stage; its
            # result is part of what this review evaluates.
            if wire == 'local_review' and attempts.get('a2_gated') == operation:
                goal += _gate_note(state)
        if len(goal.encode('utf-8')) > MAX_STAGE_GOAL_BYTES:
            return guard({'terminal_reason': 'a2_stage_goal_bounds:' + operation,
                          'rung_attempts': attempts}, state)
        raw = runner.stage(operation=wire, goal=goal, workspace=deps.workspace)
        # Runner validates the controller's bounded typed result before returning.
        if raw.get('phase') == 'failed':
            failure = raw['failure']
            return guard({'terminal_reason': 'a2_stage_failed:' + operation + ':' + failure['stage'] + ':' + failure['reason'],
                          'rung_attempts': attempts}, state)
        result = raw['result']
        from hermes_cli import build_graph_review_evidence as evidence
        try:
            attempts = evidence.retain_denial(state, attempts, raw.get('denial_evidence'),
                                             operation, runner.run_id, deps.workspace, deps.body)
        except (evidence.Refused, TypeError, KeyError):
            return guard({'terminal_reason': 'a2_denial_evidence_unavailable',
                          'rung_attempts': attempts}, state)
        if operation in ('plan', 'plan_review'):
            try:
                profile = raw['capability_evidence']
                if type(profile) is not dict or not profile:
                    raise ValueError('capability receipt missing')
                prior = attempts.get('a2_capability_evidence', {})
                if type(prior) is not dict or operation in prior:
                    raise ValueError('capability receipt replay')
                denial = raw['denial_evidence']
                attempts['a2_capability_evidence'] = dict(prior, **{operation: {
                    'profile': dict(profile), 'token': denial['token'],
                    'request_sha256': denial['request_sha256']}})
            except (KeyError, TypeError, ValueError):
                return guard({'terminal_reason': 'a2_capability_evidence_unavailable',
                              'rung_attempts': attempts}, state)
        if wire == 'local_review':
            return guard(_local_review_decision(state, attempts, operation, result), state)
        update = {'rung_attempts': attempts}
        if operation == 'plan':
            update.update(plan=result['plan'], terminal_reason='a2_stage_ready:plan_review')
        elif not result['passed']:
            update.update(terminal_reason='a2_stage_review_required:' + operation,
                          objections_current=[{'text': text, 'raw': text, 'review_id': None}
                                              for text in result['findings']])
        elif operation == 'plan_review':
            if not getattr(deps, 'cloud_plan_review', False):
                from hermes_cli import build_graph_plan_connection as connection
                try:
                    files, source = runner.binding_inputs()
                    capability = runner.capability_evidence(dict(state, rung_attempts=attempts))
                    update['plan_acceptance'] = connection.binding(
                        dict(state, rung_attempts=attempts), deps.body, deps.workspace,
                        source, files, status='cloud_disabled', guidance=[], capability=capability)
                except (connection.Refused, ValueError, KeyError, OSError, AttributeError):
                    update['terminal_reason'] = 'plan_acceptance_source_unavailable'
                    return guard(update, state)
                update['terminal_reason'] = 'a2_stage_ready:implement'
        return guard(update, state)
    node.__name__ = operation
    return node


def _local_review_decision(state, attempts, operation, result):
    """CD-101: what follows a local review (owner ruling 2026-09-27; Ken Q1/Q2).

    gate clean + review passed          -> cloud_review
    otherwise, a local round remains    -> park ready fix{n}
    exhausted, gate still failing       -> the cloud repair ladder
    exhausted, only the review objects  -> cloud_review (the stronger reviewer decides)

    A review whose gate has NOT run on this work product (`a2_gated` names a
    different instance) keeps the pre-CD-101 behaviour exactly: a failed review
    parks; a passing one runs the gate in this run.
    """
    attempts = dict(attempts)
    update = {}
    if not result['passed']:
        update['objections_current'] = [{'text': text, 'raw': text, 'review_id': None}
                                        for text in result['findings']]
    if attempts.get('a2_gated') != operation:
        if not result['passed']:
            update['terminal_reason'] = 'a2_stage_review_required:' + operation
        else:
            attempts['a2_route'] = 'cheap_gate'
        update['rung_attempts'] = attempts
        return update
    clean = (state.get('gate_summary') or {}).get('all_clean') is True
    used = attempts.get('local_fix', 0)
    if type(used) is not int or not 0 <= used <= LOCAL_FIX_CAP:
        update.update(terminal_reason='local_fix_count_invalid', rung_attempts=attempts)
        return update
    if result['passed'] and clean:
        attempts[LOCAL_FIX_PENDING] = False
        attempts['a2_route'] = 'cloud_review'
    elif used < LOCAL_FIX_CAP:
        following = 'fix%d' % (used + 1)
        if NEXT.get(operation) != following:
            update.update(terminal_reason='a2_local_fix_sequence', rung_attempts=attempts)
            return update
        attempts[LOCAL_FIX_PENDING] = True
        attempts.pop('a2_route', None)
        update['terminal_reason'] = 'a2_stage_ready:' + following
    elif not clean:
        attempts[LOCAL_FIX_PENDING] = False
        attempts['a2_route'] = 'fix_ladder'
    else:
        attempts[LOCAL_FIX_PENDING] = False
        attempts['a2_route'] = 'cloud_review'
    update['rung_attempts'] = attempts
    return update
