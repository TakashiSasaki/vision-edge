"""Validate that native observations describe the exact comparison experiment."""
from capture_native_reference import LIBRARY_SHA256
from verify_checkpoint import ARTIFACTS, REVISION

OPTIONS = {'sections': ['objects', 'text'], 'fields': ['bbox', 'score', 'mask'],
           'mask_format': 'polygons', 'conf': 0.3, 'text_conf': 0.3,
           'max_objects': 64, 'max_text_lines': 32}


def validate_native_evidence(report, expected_inputs):
    if report.get('completed') is not True:
        raise ValueError('incomplete native evidence')
    expected_model = dict(zip(('size', 'sha256'), ARTIFACTS['mirage.cact']))
    if report.get('model') != expected_model or report.get('checkpoint_revision') != REVISION:
        raise ValueError('native checkpoint mismatch')
    if report.get('library', {}).get('sha256') != LIBRARY_SHA256:
        raise ValueError('native library mismatch')
    if report.get('options') != OPTIONS or report.get('language') != 'en':
        raise ValueError('native options/language differ from candidate experiment')
    cases = report['cases']
    names = [case['name'] for case in cases]
    if len(names) != len(set(names)) or set(names) != expected_inputs.keys():
        raise ValueError('native fixture set contains duplicates or differs')
    for case in cases:
        if any(case.get(k) != value for k, value in expected_inputs[case['name']].items()):
            raise ValueError('native input identity mismatch')
        if case['status'] not in ('OBSERVED', 'FAIL'):
            raise ValueError('incomplete native observation')
        if case['status'] == 'OBSERVED' and (not isinstance(case.get('output'), dict) or
                not all(isinstance(case['output'].get(k), list) for k in ('objects', 'text'))):
            raise ValueError('unexpected native output structure')
    expected_status = 'FAIL' if any(c['status'] == 'FAIL' for c in cases) else 'OBSERVED'
    if report.get('status') != expected_status:
        raise ValueError('native aggregate contradicts detailed cases')
    return {case['name']: case for case in cases}
