"""Capture pinned native observations for the encoder fixtures. Invalid JSON remains FAIL."""
import argparse
import ctypes
import json
from pathlib import Path

from capture_native_reference import LIBRARY_SHA256, ENGINE_REVISION
from encoder_fixtures import fixtures, image_identity
from verify_checkpoint import ARTIFACTS, REVISION, identity, verify, read_json
from native_evidence_contract import OPTIONS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cact', 'library', 'photo', 'output-dir'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    model_id = verify(args.cact, ARTIFACTS['mirage.cact'])
    library_id = identity(args.library)
    if library_id['sha256'] != LIBRARY_SHA256:
        raise ValueError('unverified native library')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    lib = ctypes.CDLL(str(args.library.resolve()))
    text, integer = ctypes.c_char_p, ctypes.c_int
    for name, (inputs, result) in {
        'load': ([text, ctypes.c_uint64], integer), 'models': ([], integer),
        'last_error': ([], text), 'set_threads': ([integer], None),
        'describe': ([text, integer, integer, text, text, text, text, integer], integer),
    }.items():
        method = getattr(lib, 'cactus_' + name)
        method.argtypes, method.restype = inputs, result
    lib.cactus_set_threads(2)
    data = args.cact.read_bytes()
    if lib.cactus_load(data, len(data)) < 0 or not lib.cactus_models() & 4:
        raise RuntimeError(lib.cactus_last_error().decode())
    options = OPTIONS
    report = {'checkpoint_revision': REVISION, 'model': model_id, 'library': library_id,
              'engine_revision': ENGINE_REVISION, 'source': identity(Path(__file__)),
              'fixture_sources': {name: identity(Path(__file__).parent / name) for name in ('encoder_fixtures.py', 'capture_native_reference.py', 'fixtures/encoder-images.json', 'native_evidence_contract.py')},
              'options': options, 'language': 'en', 'cases': [], 'status': 'RUNNING', 'completed': False}
    for name, image in fixtures(args.photo).items():
        case = {'name': name, **image_identity(image)}
        width, height = image.size
        buffer = ctypes.create_string_buffer(16 * 1024 * 1024)
        code = lib.cactus_describe(image.tobytes(), width, height, json.dumps(options).encode(), b'en', None, buffer, len(buffer))
        path = args.output_dir / f'{name}-raw.txt'
        path.write_bytes(buffer.value)
        case.update(native_code=code, raw_file=path.name, raw_identity=identity(path))
        try:
            if code < 0:
                raise ValueError(lib.cactus_last_error().decode())
            output = read_json(buffer.value)
            if not isinstance(output, dict) or not all(isinstance(output.get(k), list) for k in ('objects', 'text')):
                raise ValueError('unexpected native output structure')
            case.update(status='OBSERVED', output=output)
        except (ValueError, UnicodeError) as error:
            case.update(status='FAIL', error=f'{type(error).__name__}: {error}')
            report['status'] = 'FAIL'
        report['cases'].append(case)
        (args.output_dir / 'native.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        print(name, case['status'], flush=True)
    report['completed'] = True
    report['status'] = 'FAIL' if any(c['status'] == 'FAIL' for c in report['cases']) else 'OBSERVED'
    (args.output_dir / 'native.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    raise SystemExit(1 if report['status'] == 'FAIL' else 0)


if __name__ == '__main__':
    main()
