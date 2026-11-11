"""Capture a supported geometry-only native request without altering raw responses."""
import argparse
import ctypes
import json
from pathlib import Path

from capture_native_reference import LIBRARY_SHA256, ENGINE_REVISION
from encoder_fixtures import image_identity
from geometry_evidence import GEOMETRY_OPTIONS
from refinement_inputs import images
from verify_checkpoint import ARTIFACTS, REVISION, identity, read_json, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cact', 'library', 'photo', 'output-dir'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    model_id = verify(args.cact, ARTIFACTS['mirage.cact'])
    library_id = identity(args.library)
    if library_id['sha256'] != LIBRARY_SHA256:
        raise ValueError('unverified native library')
    inputs = images(args.photo)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    lib = ctypes.CDLL(str(args.library.resolve()))
    text, integer = ctypes.c_char_p, ctypes.c_int
    for name, (parameters, result) in {
        'load': ([text, ctypes.c_uint64], integer), 'models': ([], integer),
        'last_error': ([], text), 'set_threads': ([integer], None),
        'describe': ([text, integer, integer, text, text, text, text, integer], integer),
    }.items():
        method = getattr(lib, 'cactus_'+name)
        method.argtypes, method.restype = parameters, result
    lib.cactus_set_threads(2)
    data = args.cact.read_bytes()
    if lib.cactus_load(data, len(data)) < 0 or not lib.cactus_models() & 4:
        raise RuntimeError(lib.cactus_last_error().decode())
    sources = ('capture_refinement_geometry.py', 'geometry_evidence.py', 'refinement_inputs.py',
               'encoder_fixtures.py', 'capture_native_reference.py', 'fixtures/refinement-images.json')
    report = {'scope': 'GEOMETRY_ONLY_NO_SCORE_PARITY', 'model': model_id, 'library': library_id,
              'checkpoint_revision': REVISION, 'engine_revision': ENGINE_REVISION,
              'source_identities': {name: identity(Path(__file__).parent/name) for name in sources},
              'options': GEOMETRY_OPTIONS, 'language': 'en', 'status': 'RUNNING', 'completed': False, 'cases': []}
    for name, image in inputs.items():
        width, height = image.size
        buffer = ctypes.create_string_buffer(16*1024*1024)
        code = lib.cactus_describe(image.tobytes(), width, height, json.dumps(GEOMETRY_OPTIONS).encode(), b'en', None, buffer, len(buffer))
        raw_path = args.output_dir / (name+'-raw.txt')
        raw_path.write_bytes(buffer.value)
        case = {'name': name, **image_identity(image), 'native_code': code, 'raw_file': raw_path.name, 'raw_identity': identity(raw_path)}
        try:
            if code < 0:
                raise ValueError(lib.cactus_last_error().decode())
            result = read_json(buffer.value)
            if not isinstance(result, dict) or not all(isinstance(result.get(k), list) for k in ('objects', 'text')):
                raise ValueError('unexpected native output structure')
            if any('score' in item for group in ('objects', 'text') for item in result[group]):
                raise ValueError('score unexpectedly present in geometry-only request')
            case.update(status='OBSERVED', output=result)
        except (ValueError, UnicodeError) as error:
            case.update(status='FAIL', error=f'{type(error).__name__}: {error}')
        report['cases'].append(case)
        print(name, case['status'], flush=True)
        (args.output_dir/'geometry.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    report['completed'] = True
    report['status'] = 'FAIL' if any(c['status']=='FAIL' for c in report['cases']) else 'OBSERVED'
    (args.output_dir/'geometry.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    raise SystemExit(1 if report['status']=='FAIL' else 0)


if __name__ == '__main__':
    main()
