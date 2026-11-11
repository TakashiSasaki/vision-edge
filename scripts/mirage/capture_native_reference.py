"""Capture official .cact observations on deterministic images, without ONNX parity claims.

Requires a library extracted from the checksum-verified official 4.1.0 Linux
x86_64 runtime wheel. The C ABI is documented by the pinned Cactus frontend.
No downloads, Python Cactus import, or alternate execution provider occur here.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

import verify_checkpoint as cp

LIBRARY_SHA256 = '2a4c9afa68aad5122dacb0adc60773988622debbffe4572f535404a676fb0e6a'
FRONTEND_REVISION = 'bafcf7348607c89f30ff9fde8e59d20b76baa86f'
ENGINE_REVISION = '4e822c9036be2eb0e47ef556df32a818bb1572ef'


def image_fixtures():
    images = {}
    for name, width, height in [('square_blank', 512, 512), ('landscape_blank', 512, 256),
                                ('portrait_blank', 256, 512), ('small_objects', 512, 512),
                                ('boundary_objects', 512, 256), ('overlap_shapes', 512, 512),
                                ('text_regions', 512, 256)]:
        x = np.full((height, width, 3), 255, np.uint8)
        if name == 'small_objects':
            for px, py, size, color in [(64, 64, 3, (255, 0, 0)), (256, 128, 8, (0, 0, 255)), (400, 400, 16, (0, 128, 0))]:
                x[py:py + size, px:px + size] = color
        elif name == 'boundary_objects':
            x[:100, :80] = (255, 0, 0)
            x[-100:, -80:] = (0, 0, 255)
        elif name == 'overlap_shapes':
            x[80:330, 80:300] = (255, 0, 0)
            x[200:450, 200:420] = (0, 128, 0)
            yy, xx = np.indices((height, width))
            x[(xx - 256) ** 2 + (yy - 256) ** 2 < 90 ** 2] = (0, 0, 255)
        elif name == 'text_regions':
            # Exact bitmap glyphs for '123'; no host font or rendering dependency.
            glyphs = ['010110010010111', '110001010100111', '110001010001110']
            for k, glyph in enumerate(glyphs):
                for i, value in enumerate(glyph):
                    if value == '1':
                        row, col = divmod(i, 3)
                        x[64 + 12 * row:76 + 12 * row, 120 + 60 * k + 12 * col:132 + 60 * k + 12 * col] = 0
        images[name] = x
    return images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cact', required=True, type=Path)
    parser.add_argument('--library', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path, help='must not exist')
    args = parser.parse_args()
    model_identity = cp.verify(args.cact, cp.ARTIFACTS['mirage.cact'])
    library_identity = cp.identity(args.library)
    if library_identity['sha256'] != LIBRARY_SHA256:
        raise ValueError('not the pinned official Linux x86_64 library')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    lib = ctypes.CDLL(str(args.library.resolve()))
    text, integer = ctypes.c_char_p, ctypes.c_int
    for name, (inputs, result) in {
        'load': ([text, ctypes.c_uint64], integer), 'models': ([], integer),
        'last_error': ([], text), 'set_threads': ([integer], None),
        'describe': ([text, integer, integer, text, text, text, text, integer], integer),
    }.items():
        function = getattr(lib, 'cactus_' + name)
        function.argtypes, function.restype = inputs, result
    lib.cactus_set_threads(2)
    data = args.cact.read_bytes()
    if lib.cactus_load(data, len(data)) < 0 or not lib.cactus_models() & 4:
        raise RuntimeError(lib.cactus_last_error().decode())
    options = {'sections': ['objects', 'text'], 'mask_format': 'polygons', 'conf': 0.3}
    report = {'scope': 'official native observations only; no ONNX image model exists',
              'checkpoint_revision': cp.REVISION, 'model': model_identity,
              'engine_version': '4.1.0', 'engine_revision': ENGINE_REVISION,
              'frontend_revision': FRONTEND_REVISION, 'library': library_identity,
              'capture_source': cp.identity(Path(__file__)), 'options': options, 'language': 'en',
              'fixtures_kind': 'synthetic geometry and bitmap text; no model accuracy ground truth',
              'semantic_category_coverage': 'NOT_COVERED: natural multi-category images still required',
              'onnx_comparison': {'status': 'NOT_RUN', 'bbox_iou': None, 'mask_iou': None,
                                  'text_agreement': None, 'reason': 'Isolated HConvGLU export produces no perception outputs'},
              'cases': [], 'availability_gate': 'MIRAGE_ONNX_NOT_VALIDATED'}
    report['status'] = 'OBSERVATIONS_CAPTURED'
    for name, pixels in image_fixtures().items():
        height, width, _ = pixels.shape
        rgb = pixels.tobytes()
        image_path = args.output_dir / f'{name}.ppm'
        image_path.write_bytes(f'P6\n{width} {height}\n255\n'.encode() + rgb)
        # Verify that the saved fixture contains the exact bytes supplied to the native ABI.
        if Image.open(image_path).convert('RGB').tobytes() != rgb:
            raise ValueError('fixture image does not round-trip')
        buffer = ctypes.create_string_buffer(16 * 1024 * 1024)
        code = lib.cactus_describe(rgb, width, height, json.dumps(options).encode(), b'en', None, buffer, len(buffer))
        raw_path = args.output_dir / f'{name}-raw-response.json'
        raw_path.write_bytes(buffer.value)
        case = {'name': name, 'shape': list(pixels.shape), 'image': cp.identity(image_path),
                'rgb_sha256': hashlib.sha256(rgb).hexdigest(), 'native_return_code': code,
                'raw_response': raw_path.name, 'raw_response_identity': cp.identity(raw_path)}
        if code < 0:
            case.update(status='FAIL', error=lib.cactus_last_error().decode())
        else:
            try:
                result = cp.read_json(buffer.value)
                if not isinstance(result, dict) or not all(isinstance(result.get(k), list) for k in ('objects', 'text')):
                    raise ValueError('unexpected native output structure')
                case.update(output=result, status='OBSERVED', object_count=len(result['objects']), text_count=len(result['text']))
            except (ValueError, UnicodeError) as error:
                case.update(status='FAIL', error=f'{type(error).__name__}: {error}')
        if case['status'] == 'FAIL':
            report['status'] = 'FAIL'
        report['cases'].append(case)
        print(f"{name}: {case['status']} objects={case.get('object_count')} text={case.get('text_count')} error={case.get('error')}", flush=True)
        # Incremental capture retains completed observations if a later native call fails.
        (args.output_dir / 'native-observations.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    raise SystemExit(1 if report['status'] == 'FAIL' else 0)


if __name__ == '__main__':
    main()
