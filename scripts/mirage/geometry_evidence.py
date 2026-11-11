"""Explicit geometry-only native scope. It cannot stand in for score parity."""
from capture_native_reference import LIBRARY_SHA256
from verify_checkpoint import ARTIFACTS, REVISION

GEOMETRY_OPTIONS = {'sections': ['objects', 'text'], 'fields': ['bbox', 'mask'],
                    'mask_format': 'polygons', 'conf': 0.3, 'text_conf': 0.3,
                    'max_objects': 64, 'max_text_lines': 32}


def validate(report, expected_inputs):
    if report.get('completed') is not True or report.get('scope') != 'GEOMETRY_ONLY_NO_SCORE_PARITY':
        raise ValueError('incomplete or incorrect geometry scope')
    if report.get('options') != GEOMETRY_OPTIONS or report.get('language') != 'en':
        raise ValueError('geometry capture contract mismatch')
    if report.get('checkpoint_revision') != REVISION or report.get('model') != dict(zip(('size', 'sha256'), ARTIFACTS['mirage.cact'])):
        raise ValueError('geometry checkpoint mismatch')
    if report.get('library', {}).get('sha256') != LIBRARY_SHA256:
        raise ValueError('geometry library mismatch')
    cases = report['cases']
    names = [c['name'] for c in cases]
    if len(names) != len(set(names)) or set(names) != expected_inputs.keys():
        raise ValueError('geometry fixture set mismatch or duplicate')
    for case in cases:
        if any(case.get(k) != v for k, v in expected_inputs[case['name']].items()):
            raise ValueError('geometry input identity mismatch')
        if case.get('status') not in ('OBSERVED', 'FAIL'):
            raise ValueError('incomplete geometry observation')
        if case['status'] == 'OBSERVED':
            output = case.get('output')
            if not isinstance(output, dict) or not all(isinstance(output.get(k), list) for k in ('objects', 'text')):
                raise ValueError('unexpected geometry structure')
            if any('score' in obj for kind in ('objects', 'text') for obj in output[kind]):
                raise ValueError('unexpected score in geometry-only capture')
    expected_status = 'FAIL' if any(c['status']=='FAIL' for c in cases) else 'OBSERVED'
    if report['status'] != expected_status:
        raise ValueError('geometry aggregate contradicts cases')
    return {c['name']: c for c in cases}
