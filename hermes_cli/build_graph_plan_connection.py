"""Bind fixed-Qwen implementation to retained planning and cloud evidence."""
import hashlib
import json
import os
from pathlib import Path
import re
import stat


class Refused(ValueError):
    pass


def canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'),
                          ensure_ascii=True, allow_nan=False).encode()
    except (TypeError, ValueError, UnicodeError) as exc:
        raise Refused('noncanonical_evidence') from exc


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def cloud_result(value, files):
    """Validate shape and file references; semantic agreement remains a review judgment."""
    if type(value) is not dict or set(value) != {'verdict', 'feasibility', 'findings', 'guidance',
                                                 'summary', 'approach_preserved', 'requirements_preserved'}:
        raise Refused('cloud_plan_review_shape')
    if value['verdict'] not in ('passed', 'revise') or value['feasibility'] not in ('feasible', 'infeasible', 'unknown'):
        raise Refused('cloud_plan_review_vocabulary')
    if type(value['approach_preserved']) is not bool or type(value['requirements_preserved']) is not bool:
        raise Refused('cloud_plan_review_preservation_shape')
    for key, limit in (('findings', 12), ('guidance', 12)):
        if type(value[key]) is not list or len(value[key]) > limit:
            raise Refused('cloud_plan_review_bounds')
    if type(value['summary']) is not str or len(value['summary'].encode('utf-8')) > 2048:
        raise Refused('cloud_plan_review_bounds')
    if any(type(x) is not str or not x.strip() or len(x.encode('utf-8')) > 1024 for x in value['findings']):
        raise Refused('cloud_plan_review_findings')
    for item in value['guidance']:
        if (type(item) is not dict or set(item) != {'file', 'action'}
                or type(item['file']) is not str or item['file'] not in files
                or type(item['action']) is not str or not item['action'].strip()
                or len(item['action'].encode('utf-8')) > 1024):
            raise Refused('cloud_plan_review_guidance_file')
    if len(canonical(value)) > 16384:
        raise Refused('cloud_plan_review_bounds')
    if value['verdict'] == 'passed' and (value['feasibility'] != 'feasible' or value['findings']
                                          or not value['approach_preserved']
                                          or not value['requirements_preserved']):
        raise Refused('cloud_plan_review_contradiction')
    return value


def planning_receipt_sha(state, workspace, spec):
    # Existing authenticated stage receipts, including exact task/spec/run and
    # sealed-journal accounting, are checked by the established renderer.
    from hermes_cli import build_graph_review_evidence as evidence
    try:
        return digest(evidence.render_plan_denials(state, workspace, spec))
    except (evidence.Refused, TypeError, KeyError, ValueError) as exc:
        raise Refused('planning_receipt_unavailable') from exc


def _envelope(path, workspace, card_id, run_id):
    root = (Path(workspace) / '.graph-checkpoints').resolve()
    candidate = Path(path)
    if (not candidate.is_absolute() or candidate.name != f'{card_id}-{run_id}-result.json'
            or not re.fullmatch(r'model-call-[^/]{6,}', candidate.parent.name)
            or candidate.parent.parent.resolve() != root):
        raise Refused('cloud_envelope_path')
    try:
        fd = os.open(candidate, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, 'rb') as fh:
            before = os.fstat(fh.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > 1024 * 1024:
                raise Refused('cloud_envelope_type_or_bounds')
            raw = fh.read(1024 * 1024 + 1)
            after = os.fstat(fh.fileno())
        if (len(raw) > 1024 * 1024 or
                (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
            raise Refused('cloud_envelope_changed')
        envelope = json.loads(raw)
        if type(envelope) is not dict or type(envelope.get('result')) is not str:
            raise Refused('cloud_envelope_shape')
        return hashlib.sha256(raw).hexdigest(), envelope['result']
    except (OSError, ValueError, UnicodeError) as exc:
        raise Refused('cloud_envelope_unavailable') from exc


def cloud_receipt(result, workspace, card_id, files):
    """Reuse model-call's retained envelope, request identity and ledger read-back."""
    ident = result.get('request_identity')
    run_id = ident.get('run_id') if type(ident) is dict and ident.get('card_id') == card_id else None
    if (result.get('ok') is not True or result.get('activity') != 'plan_review'
            or type(run_id) is not str or not re.fullmatch(r'[0-9a-f]{8}', run_id)
            or type(result.get('model')) is not str or not result['model']
            or type(result.get('ledger_line')) is not str or not result['ledger_line'].strip()
            or type(result.get('envelope_path')) is not str):
        raise Refused('cloud_attribution_unavailable')
    envelope_sha, text = _envelope(result['envelope_path'], workspace, card_id, run_id)
    if text != result.get('text'):
        raise Refused('cloud_result_envelope_drift')
    try:
        parsed = json.loads(text)
    except (ValueError, UnicodeError) as exc:
        raise Refused('cloud_result_not_exact_json') from exc
    if parsed != result.get('verdict'):
        raise Refused('cloud_verdict_envelope_drift')
    verdict = cloud_result(parsed, files)
    try:
        ledger = json.loads(result['ledger_line'])
    except (ValueError, UnicodeError) as exc:
        raise Refused('cloud_ledger_readback_invalid') from exc
    if (type(ledger) is not dict or ledger.get('run_id') != run_id
            or ledger.get('task_id') != card_id or ledger.get('activity') != 'plan_review'
            or ledger.get('model') != result['model']
            or ledger.get('verdict_passed') != (verdict['verdict'] == 'passed')):
        raise Refused('cloud_ledger_attribution_mismatch')
    return ({'run_id': run_id, 'model': result['model'], 'activity': 'plan_review',
             'envelope_path': result['envelope_path'], 'envelope_sha256': envelope_sha,
             'ledger_line_sha256': digest(result['ledger_line']), 'verdict_sha256': sha(verdict)}, verdict)


def binding(state, spec, workspace, source_digest, files, *, status, guidance,
            cloud=None, capability=None):
    if status not in ('cloud_accepted', 'cloud_disabled') or not state['plan'].strip():
        raise Refused('plan_acceptance_missing')
    if type(source_digest) is not str or re.fullmatch(r'[0-9a-f]{64}', source_digest) is None:
        raise Refused('plan_source_unavailable')
    if type(files) is not list or type(guidance) is not list:
        raise Refused('plan_guidance_invalid')
    if (status == 'cloud_disabled' and (guidance or cloud is not None)) or (status == 'cloud_accepted' and type(cloud) is not dict):
        raise Refused('plan_review_mode_invalid')
    if type(capability) is not dict:
        raise Refused('plan_capability_binding_invalid')
    return {'version': 1, 'card_id': state['card_id'], 'component': state['component'],
            'plan_sha256': digest(state['plan']), 'spec_sha256': digest(spec),
            'source_sha256': source_digest, 'files_sha256': sha(files),
            'review_contract_sha256': state['review_contract_sha256'],
            'planning_receipt_sha256': planning_receipt_sha(state, workspace, spec),
            'status': status, 'guidance': guidance, 'guidance_sha256': sha(guidance),
            'cloud_receipt': cloud, 'capability_sha256': sha(capability)}


def verify(record, state, spec, workspace, source_digest, files, *, cloud_required,
           capability=None):
    if type(record) is not dict or set(record) != {
            'version', 'card_id', 'component', 'plan_sha256', 'spec_sha256', 'source_sha256',
            'files_sha256', 'review_contract_sha256', 'planning_receipt_sha256',
            'status', 'guidance', 'guidance_sha256', 'cloud_receipt', 'capability_sha256'}:
        raise Refused('plan_acceptance_missing')
    if (cloud_required and record['status'] != 'cloud_accepted') or (not cloud_required and record['status'] != 'cloud_disabled'):
        raise Refused('plan_review_mode_drift')
    attempts = state['rung_attempts']
    if record['status'] == 'cloud_accepted':
        if type(record['cloud_receipt']) is not dict:
            raise Refused('cloud_receipt_shape')
        expected_identity = {key: record['cloud_receipt'].get(key) for key in (
            'run_id', 'envelope_sha256', 'verdict_sha256', 'ledger_line_sha256')}
        expected_identity['guidance_sha256'] = record['guidance_sha256']
        if (attempts.get('cloud_plan_review_calls') != 1
                or attempts.get('cloud_plan_review_evidence') != expected_identity):
            raise Refused('cloud_review_identity_drift')
    elif attempts.get('cloud_plan_review_calls', 0) or attempts.get('cloud_plan_review_evidence'):
        raise Refused('cloud_disabled_with_review_evidence')
    expected = binding(state, spec, workspace, source_digest, files,
                       status=record['status'], guidance=record['guidance'],
                       cloud=record['cloud_receipt'], capability=capability)
    if record != expected:
        raise Refused('plan_acceptance_stale')
    if record['status'] == 'cloud_accepted':
        receipt = record['cloud_receipt']
        if type(receipt) is not dict or set(receipt) != {
                'run_id', 'model', 'activity', 'envelope_path', 'envelope_sha256',
                'ledger_line_sha256', 'verdict_sha256'}:
            raise Refused('cloud_receipt_shape')
        envelope_sha, text = _envelope(receipt['envelope_path'], workspace,
                                       state['card_id'], receipt['run_id'])
        try:
            verdict = cloud_result(json.loads(text), files)
        except (ValueError, UnicodeError) as exc:
            raise Refused('cloud_result_unavailable') from exc
        if (envelope_sha != receipt['envelope_sha256'] or sha(verdict) != receipt['verdict_sha256']
                or verdict['verdict'] != 'passed' or verdict['feasibility'] != 'feasible'
                or verdict['guidance'] != record['guidance'] or sha(record['guidance']) != record['guidance_sha256']):
            raise Refused('cloud_approval_or_guidance_drift')
    return record
