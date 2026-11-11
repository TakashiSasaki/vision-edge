"""Provision and verify the experimental partial Mirage ONNX artifact and browser verification fixtures.

Never overwrites an existing target file, never retries restricted HTTP 403 downloads,
and never relaxes numerical tolerances (atol = rtol = 2e-4).
"""
import argparse
import base64
import hashlib
import io
import json
import struct
import urllib.request
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from archive_encoder_weights import decode_cq
from corpus_fixtures import cases as corpus_cases_manifest, verify_image
from corpus_postprocessing import decode_policy
from encoder_candidate import output_names, required_shapes, validate_weights
from encoder_fixtures import PHOTO_SHA256, PHOTO_URL, image_identity
from encoder_reference_order import ReferenceOrderEncoder
from oasis_candidate_reference import OASIS_SHA256
from perception_diagnostics import compare_native, decode
from refinement_inputs import images as refinement_images, preprocess
from verify_checkpoint import ARTIFACTS, REVISION, identity, verify

VERIFIED_ONNX_RELATIVE_PATH = Path(
    '.cache/mirage-refinement/numerical-cact-dequantized-pixel-0-r3/cact-dequantized-reference-order.onnx'
)
VERIFIED_ONNX_IDENTITY = (
    34281503,
    '52d52d2918160404cd806702a270736819712025cf3c6aa8e907a06bc3cbdd07',
)
DEFAULT_PHOTO_PATH = Path('.cache/mirage-encoder/coco-231049.jpg')
DEFAULT_CORPUS_CACHE = Path('.cache/mirage-corpus/images')
RESERVED_CORPUS_CASES = (
    'coco_000000458054',
    'coco_000000331352',
    'coco_000000296649',
    'coco_000000386912',
)
SYNTHETIC_AND_REFINEMENT_CASES = (
    'photo_original_portrait',
    'overlap_shapes',
    'text_regions',
    'square_blank',
)
FINAL_HEADS = ('detections', 'prototypes', 'coefficients')


def ensure_verified_photo(photo_path: Path = DEFAULT_PHOTO_PATH) -> Path:
    if photo_path.exists():
        digest = hashlib.sha256(photo_path.read_bytes()).hexdigest()
        if digest != PHOTO_SHA256:
            raise ValueError(f'existing photo SHA-256 mismatch at {photo_path}: {digest}')
        return photo_path
    photo_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(PHOTO_URL, headers={'User-Agent': 'MirageResearch/1.0'})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != PHOTO_SHA256:
        raise ValueError(f'downloaded photo SHA-256 mismatch: {digest}')
    with photo_path.open('xb') as f:
        f.write(raw)
    return photo_path


def ensure_verified_corpus_image(name: str, cache_dir: Path = DEFAULT_CORPUS_CACHE):
    records = corpus_cases_manifest()
    record = records[name]
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / record['file']
    blocked = cache_dir / f'{name}-http403.json'
    if not target.exists():
        if blocked.exists():
            raise ValueError(f'recorded HTTP 403 for {name} is not retried')
        urls = [record['url']]
        if record['url'].startswith('https://images.cocodataset.org/'):
            urls.append('http://' + record['url'][len('https://'):])
        data = None
        last_err = None
        for u in urls:
            try:
                req = urllib.request.Request(u, headers={'User-Agent': 'MirageResearch/1.0'})
                with urllib.request.urlopen(req, timeout=20) as resp:
                    data = resp.read(10 * 1024 * 1024 + 1)
                break
            except Exception as err:
                last_err = err
        if data is None:
            raise RuntimeError(f'failed to fetch {name}: {last_err}')
        if len(data) != record['size'] or hashlib.sha256(data).hexdigest() != record['sha256']:
            raise ValueError(f'download identity mismatch for {name}')
        with target.open('xb') as f:
            f.write(data)
    return verify_image(target, record), target


def load_archive_weights_from_mapping(cact_path: Path, mapping_path: Path):
    native_id = verify(cact_path, ARTIFACTS['mirage.cact'])
    data = cact_path.read_bytes()
    fields = struct.unpack_from('<48I', data)
    if fields[0] != 0x05E12A84 or fields[1] != 806 or fields[2] != 28:
        raise ValueError('unexpected pinned archive header')
    codebook = np.frombuffer(data, '<f4', count=28, offset=196).astype(np.float64)
    mapping_rows = json.loads(mapping_path.read_text(encoding='utf-8'))
    shapes = required_shapes()
    archive_weights = {}
    for row in mapping_rows:
        name = row['name']
        entry = row['entry']
        shape = shapes[name]
        start = int(entry['offset'])
        length = int(entry['bytes'])
        raw = data[start:start + length]
        if entry['dtype'] == 3:
            arr = decode_cq(raw, tuple(entry['shape']), entry['group'], entry['bits'], codebook).T.copy()
        elif entry['dtype'] in (1, 2):
            numpy_dtype = '<f4' if entry['dtype'] == 2 else '<f2'
            arr = np.frombuffer(raw, numpy_dtype).astype(np.float32).reshape(shape).copy()
        else:
            raise ValueError(f'unsupported archive dtype: {entry["dtype"]}')
        archive_weights[name] = torch.from_numpy(arr)
    validate_weights(archive_weights)
    return archive_weights, native_id


def ensure_verified_onnx(
    target_path: Path = VERIFIED_ONNX_RELATIVE_PATH,
    cact_path: Path = Path('.cache/container-models/mirage.cact'),
    mapping_path: Path = Path('docs/verification/mirage-encoder-archive-mapping.json'),
):
    if target_path.exists():
        verified = verify(target_path, VERIFIED_ONNX_IDENTITY)
        return {'status': 'present_verified', 'path': str(target_path), **verified}

    if not cact_path.exists():
        raise FileNotFoundError(
            f'Required cached archive not found at {cact_path}; cannot reconstruct verified ONNX.'
        )

    archive_weights, native_id = load_archive_weights_from_mapping(cact_path, mapping_path)
    torch.set_num_threads(2)
    model = ReferenceOrderEncoder(archive_weights, 'symmetric').eval()
    x = np.full((1, 3, 512, 512), 240.0 / 127.5 - 1.0, dtype=np.float32)
    hw = np.array([512, 512], dtype=np.int64)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix('.onnx.tmp')
    if temp_path.exists():
        temp_path.unlink()

    names = output_names()
    torch.onnx.export(
        model,
        (torch.from_numpy(x), torch.from_numpy(hw)),
        str(temp_path),
        input_names=['image', 'valid_hw'],
        output_names=names,
        opset_version=17,
        dynamo=False,
    )
    graph = onnx.load(str(temp_path))
    converter_id = identity(Path(__file__).parent / 'encoder_reference_order.py')
    onnx.helper.set_model_props(
        graph,
        {
            'scope': 'PARTIAL_CANDIDATE_ENCODER_AND_HEADS',
            'arithmetic': 'unfused-reference-order',
            'weights_mode': 'cact-dequantized',
            'checkpoint_revision': REVISION,
            'weights_sha256': native_id['sha256'],
            'converter_sha256': converter_id['sha256'],
            'reference_sha256': OASIS_SHA256,
            'availability_gate': 'MIRAGE_ONNX_NOT_VALIDATED',
        },
    )
    onnx.checker.check_model(graph, full_check=True)
    onnx.save(graph, str(temp_path))
    del graph

    verified = verify(temp_path, VERIFIED_ONNX_IDENTITY)
    temp_path.replace(target_path)
    return {'status': 'reconstructed_verified', 'path': str(target_path), **verified}


def serialize_proposals(items, include_mask=False):
    result = []
    for item in items:
        entry = {
            'cell': [int(item['cell'][0]), int(item['cell'][1])],
            'score': float(item['score']),
            'bbox': [int(v) for v in item['bbox']],
        }
        if include_mask and '_mask' in item:
            mask_u8 = np.ascontiguousarray(item['_mask'].astype(np.uint8))
            entry['mask_active_pixels'] = int(mask_u8.sum())
            entry['mask_sha256'] = hashlib.sha256(mask_u8.tobytes()).hexdigest()
            entry['mask_packed_base64'] = base64.b64encode(np.packbits(mask_u8).tobytes()).decode('ascii')
        result.append(entry)
    return result


def dump_browser_fixtures(
    output_dir: Path,
    onnx_path: Path = VERIFIED_ONNX_RELATIVE_PATH,
    photo_path: Path = DEFAULT_PHOTO_PATH,
    native_corpus_path: Path = Path('docs/verification/mirage-native-corpus-results.json'),
    native_geometry_path: Path = Path('docs/verification/mirage-refinement-native-geometry.json'),
):
    verified_onnx = verify(onnx_path, VERIFIED_ONNX_IDENTITY)
    ensure_verified_photo(photo_path)
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
    session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=['CPUExecutionProvider'])
    if session.get_providers() != ['CPUExecutionProvider']:
        raise ValueError(f'unexpected provider: {session.get_providers()}')

    ref_imgs = refinement_images(photo_path)
    corpus_report = json.loads(native_corpus_path.read_text(encoding='utf-8'))
    reserved_native = {
        c['name']: c['requests']['geometry-only']
        for c in corpus_report['native_reserved']['cases']
    }
    geometry_report = json.loads(native_geometry_path.read_text(encoding='utf-8'))
    refinement_native = {c['name']: c for c in geometry_report['cases']}

    case_sources = {}
    for name in RESERVED_CORPUS_CASES:
        im, _ = ensure_verified_corpus_image(name)
        case_sources[name] = ('reserved_validation', im, reserved_native.get(name))
    for name in SYNTHETIC_AND_REFINEMENT_CASES:
        case_sources[name] = ('refinement_fixture', ref_imgs[name], refinement_native.get(name))

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        'schema_version': 1,
        'onnx': verified_onnx,
        'rule': 'ceil32',
        'tolerance': {'atol': 2e-4, 'rtol': 2e-4, 'near_zero_denominator_floor': 1e-6},
        'reserved_cases': list(RESERVED_CORPUS_CASES),
        'refinement_cases': list(SYNTHETIC_AND_REFINEMENT_CASES),
        'cases': {},
    }

    for name, (group_kind, im, native_entry) in case_sources.items():
        x, hw, meta = preprocess(im, 'ceil32')
        buf = io.BytesIO()
        im.save(buf, format='PNG')
        png_bytes = buf.getvalue()
        png_path = output_dir / f'{name}.png'
        png_path.write_bytes(png_bytes)

        input_bin_path = output_dir / f'{name}.image.f32.bin'
        input_bin_path.write_bytes(x.astype('<f4').tobytes())

        outputs = dict(zip(FINAL_HEADS, session.run(list(FINAL_HEADS), {'image': x, 'valid_hw': hw})))
        head_files = {}
        for head_name, arr in outputs.items():
            head_bin = output_dir / f'{name}.{head_name}.f32.bin'
            raw_bytes = np.ascontiguousarray(arr, dtype='<f4').tobytes()
            head_bin.write_bytes(raw_bytes)
            head_files[head_name] = {
                'file': head_bin.name,
                'shape': list(arr.shape),
                'sha256': hashlib.sha256(raw_bytes).hexdigest(),
            }

        decoded_default = decode(outputs, meta)
        decoded_nms = decode_policy(outputs, meta, 0.5)
        native_obs = (
            native_entry['output']
            if native_entry and native_entry.get('status') == 'OBSERVED'
            else None
        )
        native_eval_default = (
            compare_native(decoded_default, native_obs, compare_scores=False)
            if native_obs is not None
            else None
        )
        native_eval_nms = (
            compare_native(decoded_nms, native_obs, compare_scores=False)
            if native_obs is not None
            else None
        )

        manifest['cases'][name] = {
            'name': name,
            'group': group_kind,
            'image_identity': image_identity(im),
            'png_file': png_path.name,
            'png_sha256': hashlib.sha256(png_bytes).hexdigest(),
            'input_tensor_file': input_bin_path.name,
            'input_tensor_sha256': hashlib.sha256(x.astype('<f4').tobytes()).hexdigest(),
            'valid_hw': [int(hw[0]), int(hw[1])],
            'metadata': meta,
            'heads': head_files,
            'decoded_default': {
                'object_count': len(decoded_default['objects']),
                'text_count': len(decoded_default['text']),
                'objects': serialize_proposals(decoded_default['objects'], include_mask=True),
                'text': serialize_proposals(decoded_default['text'], include_mask=False),
                'native_evaluation': native_eval_default,
            },
            'decoded_nms_0p50': {
                'object_count': len(decoded_nms['objects']),
                'text_count': len(decoded_nms['text']),
                'objects': serialize_proposals(decoded_nms['objects'], include_mask=True),
                'text': serialize_proposals(decoded_nms['text'], include_mask=False),
                'native_evaluation': native_eval_nms,
            },
            'native_observation': native_obs,
        }

    manifest_path = output_dir / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=Path, default=VERIFIED_ONNX_RELATIVE_PATH)
    parser.add_argument('--cact', type=Path, default=Path('.cache/container-models/mirage.cact'))
    parser.add_argument(
        '--mapping',
        type=Path,
        default=Path('docs/verification/mirage-encoder-archive-mapping.json'),
    )
    parser.add_argument('--dump-browser-fixtures', type=Path, default=None)
    args = parser.parse_args()

    result = ensure_verified_onnx(args.target, args.cact, args.mapping)
    if args.dump_browser_fixtures is not None:
        manifest_path = dump_browser_fixtures(args.dump_browser_fixtures, args.target)
        result['browser_fixtures_manifest'] = str(manifest_path)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
