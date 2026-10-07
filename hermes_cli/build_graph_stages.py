"""One protected stage per dispatcher run; receipts never authorize replay."""
import json

STAGES = ('plan', 'plan_review', 'implement', 'local_review')
NEXT = dict(zip(STAGES, STAGES[1:]))


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
    return next((stage for stage in STAGES if stage not in runs), None)


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
    if operation not in ('plan', 'plan_review', 'local_review'):
        raise ValueError('unknown read-only stage')

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
        if operation == 'local_review' and not state['diff'].strip():
            return {'terminal_reason': 'a2_stage_no_diff'}
        contract = ({'operation': 'plan', 'plan': 'A concise implementation plan (max 8192 UTF-8 bytes)'}
                    if operation == 'plan' else
                    {'operation': operation, 'passed': True, 'findings': []})
        goal = ('Perform only the ' + operation + ' stage. Do not create, delete, or modify files. '
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
                     'in the implementation plan.\nSPECIFICATION:\n' + deps.body)
        else:
            goal += (' Reviews: passed must be false if findings is nonempty; '
                'findings must contain concise strings. Evaluate the specification and supplied plan'
                + (' against the supplied actual diff.' if operation == 'local_review' else '.')
                + '\nSPECIFICATION:\n' + deps.body
                + '\nPLAN:\n' + state['plan']
                + ('\nACTUAL DIFF:\n' + state['diff'] if operation == 'local_review' else ''))
        raw = runner.stage(operation=operation, goal=goal, workspace=deps.workspace)
        # Runner validates the controller's bounded typed result before returning.
        if raw.get('phase') == 'failed':
            failure = raw['failure']
            return guard({'terminal_reason': 'a2_stage_failed:' + operation + ':' + failure['stage'] + ':' + failure['reason'],
                          'rung_attempts': attempts}, state)
        result = raw['result']
        update = {'rung_attempts': attempts}
        if operation == 'plan':
            update.update(plan=result['plan'], terminal_reason='a2_stage_ready:plan_review')
        elif not result['passed']:
            update.update(terminal_reason='a2_stage_review_required:' + operation,
                          objections_current=[{'text': text, 'raw': text, 'review_id': None}
                                              for text in result['findings']])
        elif operation == 'plan_review':
            update['terminal_reason'] = 'a2_stage_ready:implement'
        return guard(update, state)
    node.__name__ = operation
    return node
