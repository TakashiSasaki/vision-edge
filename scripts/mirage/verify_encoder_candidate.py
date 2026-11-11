"""Export/measure a partial candidate encoder and heads; native equivalence stays gated."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from safetensors.torch import load_file
import torch

from archive_encoder_weights import load_archive_weights
from encoder_candidate import EncoderCandidate, output_names, validate_weights
from encoder_fixtures import fixtures, preprocess, image_identity, PHOTO_URL, PHOTO_SHA256
from hconvglu import compare, reference as mixer_reference
from oasis_candidate_reference import load_reference, capture, OASIS_REVISION, OASIS_SHA256
from perception_diagnostics import decode, compare_native
from verify_checkpoint import ARTIFACTS, REVISION, identity, verify, inventory, read_json
from acquire_reference import CONFIG_IDENTITY
from native_evidence_contract import validate_native_evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'config', 'photo', 'oasis-reference', 'output-dir', 'native-report'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--cact', type=Path)
    parser.add_argument('--weights', choices=['fp32', 'cact-dequantized'], required=True)
    parser.add_argument('--padding', choices=['symmetric', 'same'], required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    checkpoint_id = verify(args.checkpoint, ARTIFACTS['mirage.safetensors'])
    config_id = verify(args.config, CONFIG_IDENTITY)
    weights = load_file(args.checkpoint)
    required = validate_weights(weights)
    native = read_json(args.native_report.read_bytes())
    mapping = None
    if args.weights == 'cact-dequantized':
        if not args.cact:
            parser.error('--cact required for cact-dequantized diagnostics')
        weights, mapping = load_archive_weights(args.cact, weights)
    ref, namespace = load_reference(args.oasis_reference, weights, args.padding)
    cases = fixtures(args.photo)
    native_cases = validate_native_evidence(native, {name: image_identity(image) for name, image in cases.items()})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    model = EncoderCandidate(weights, args.padding).eval()
    names = output_names()
    x, hw, _ = preprocess(next(iter(cases.values())))
    path = args.output_dir / 'partial-encoder-and-heads.onnx'
    torch.onnx.export(model, (torch.from_numpy(x), torch.from_numpy(hw)), str(path),
                      opset_version=17, input_names=['image', 'valid_hw'], output_names=names, dynamo=False)
    graph = onnx.load(str(path))
    source_names = ('encoder_candidate.py', 'encoder_fixtures.py', 'fixtures/encoder-images.json',
                    'oasis_candidate_reference.py', 'archive_encoder_weights.py', 'perception_diagnostics.py',
                    'verify_encoder_candidate.py', 'hconvglu.py', 'verify_checkpoint.py', 'acquire_reference.py', 'native_evidence_contract.py', 'capture_native_reference.py')
    sources = {name: identity(Path(__file__).parent / name) for name in source_names}
    onnx.helper.set_model_props(graph, {'scope': 'PARTIAL_CANDIDATE_ENCODER_AND_DENSE_HEADS; NO_LANGUAGE_OR_OCR',
        'checkpoint_revision': REVISION, 'checkpoint_sha256': checkpoint_id['sha256'],
        'weights_mode': args.weights, 'padding': args.padding,
        'weights_sha256': ARTIFACTS['mirage.cact'][1] if args.weights == 'cact-dequantized' else checkpoint_id['sha256'],
        'reference_sha256': OASIS_SHA256, 'reference_revision': OASIS_REVISION,
        'converter_sha256': sources['encoder_candidate.py']['sha256'],
        'availability_gate': 'MIRAGE_ONNX_NOT_VALIDATED'})
    onnx.checker.check_model(graph, full_check=True)
    onnx.save(graph, str(path))
    del graph
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
    if session.get_providers() != ['CPUExecutionProvider'] or [o.name for o in session.get_outputs()] != names:
        raise ValueError('unexpected provider or ONNX output contract')
    report = {'schema_version': 1, 'scope': 'partial candidate encoder and detection/prototype/coefficient maps; no language or OCR',
              'status': 'RUNNING', 'completed': False, 'checkpoint_revision': REVISION, 'checkpoint': checkpoint_id, 'config': config_id,
              'weights_mode': args.weights, 'padding': args.padding, 'source_identities': sources,
              'reference': {'kind': 'THIRD_PARTY_CANDIDATE_NOT_OFFICIAL', 'revision': OASIS_REVISION, 'sha256': OASIS_SHA256},
              'candidate_assumptions': {'resize': 'Pillow bilinear', 'size_rounding': 'nearest integer', 'canvas_fill_normalized': 0,
                  'placement': 'top-left', 'rope': 'rows first, pairs16, registers unrotated', 'level_gates': 'raw'},
              'onnx': identity(path), 'onnx_checker': 'PASS', 'provider': session.get_providers(),
              'versions': {k: importlib.metadata.version(k) for k in ('torch', 'numpy', 'onnx', 'onnxruntime', 'safetensors', 'Pillow')},
              'mapped_tensors': len(required), 'checkpoint_total_tensors': 583, 'out_of_scope_tensors': 583-len(required),
              'native_evidence': identity(args.native_report), 'photo': {'source_url': PHOTO_URL, 'sha256': PHOTO_SHA256},
              'native_equivalence': 'NOT_ESTABLISHED', 'webgpu': 'NOT_RUN: CPU/native acceptance not established',
              'availability_gate': 'MIRAGE_ONNX_NOT_VALIDATED', 'cases': [], 'first_divergent_operation': None,
              'first_local_mixer_failure': None,
              'divergence_semantics': 'First threshold failure in captured checkpoint traversal, not the first primitive operation inside a block'}
    if args.cact:
        report['cact'] = verify(args.cact, ARTIFACTS['mirage.cact'])
    if mapping is not None:
        (args.output_dir / 'archive-weight-mapping.json').write_text(json.dumps(mapping, indent=2) + '\n')
    rows = inventory(args.checkpoint)
    for row in rows:
        row['mapping_status'] = 'MAPPED_CANDIDATE_ENCODER_OR_HEAD' if row['name'] in required else 'OUT_OF_SCOPE_UNMAPPED'
        row['reference_operation'] = row['name'] if row['name'] in required else None
    (args.output_dir / 'tensor-mapping.json').write_text(json.dumps({'tensors': rows}, indent=2) + '\n')
    with torch.inference_mode():
        for case_name, image in cases.items():
            x, hw, meta = preprocess(image)
            ref_input, ref_meta = ref.letterbox(image)
            if not np.array_equal(x, ref_input.numpy()) or meta != ref_meta:
                raise ValueError('candidate preprocessing implementations diverge')
            expected, _ = capture(ref, image, namespace)
            torch_values = dict(zip(names, model(torch.from_numpy(x), torch.from_numpy(hw))))
            actual = dict(zip(names, session.run(None, {'image': x, 'valid_hw': hw})))
            np.savez_compressed(args.output_dir / f'{case_name}-head-outputs.npz',
                                **{name: actual[name] for name in ('detections', 'prototypes', 'coefficients')})
            case = {'name': case_name, **image_identity(image), 'preprocessing': meta,
                    'input_fp32_sha256': hashlib.sha256(x.astype('<f4').tobytes()).hexdigest(),
                    'measurements': [], 'mixer_local_diagnostics': [], 'native_capture_status': native_cases[case_name]['status']}
            for comparison, reference_values in [('oasis_vs_onnx', expected), ('torch_candidate_vs_onnx', torch_values)]:
                for name in names:
                    metric = compare(reference_values[name].numpy(), actual[name])
                    case['measurements'].append({'comparison': comparison, 'tensor': name, **metric})
                    if metric['status'] == 'FAIL':
                        report['status'] = 'FAIL'
                        if report['first_divergent_operation'] is None:
                            report['first_divergent_operation'] = {'case': case_name, 'comparison': comparison, 'tensor': name}
            for name, value in expected.items():
                if name.endswith('_hconv_input'):
                    block = name.removesuffix('_hconv_input')
                    flat = value.reshape(-1, 512)
                    unfused = mixer_reference(flat, weights, block + '/hconv')['output']
                    folded = model.mixer(flat, block)
                    case['mixer_local_diagnostics'].append({'block': block, 'input_absolute_max': float(flat.abs().max()),
                        'input_rms': float(torch.sqrt((flat*flat).mean())),
                        'input_sha256': hashlib.sha256(flat.numpy().astype('<f4').tobytes()).hexdigest(),
                        **compare(unfused.numpy(), folded.numpy())})
                    if case['mixer_local_diagnostics'][-1]['status'] == 'FAIL':
                        report['status'] = 'FAIL'
                        if report['first_local_mixer_failure'] is None:
                            report['first_local_mixer_failure'] = {'case': case_name, 'block': block, 'tensor': 'isolated mixer output at independent candidate-reference input'}
            if native_cases[case_name]['status'] == 'OBSERVED':
                case['native_comparison'] = compare_native(decode(actual, meta), native_cases[case_name]['output'])
            else:
                case['native_comparison'] = {'status': 'BLOCKED_INVALID_NATIVE_OUTPUT', 'bbox_iou': None, 'mask_iou': None,
                                             'error': native_cases[case_name].get('error')}
            report['cases'].append(case)
            (args.output_dir / 'validation.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
            failures = sum(m['status']=='FAIL' for m in case['measurements'])
            print(f'{case_name}: compared={len(case["measurements"])} failed={failures} native={case["native_capture_status"]}', flush=True)
    report['completed'] = True
    report['status'] = 'FAIL' if report['status'] == 'FAIL' else 'PASS'
    (args.output_dir / 'validation.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    raise SystemExit(1 if report['status'] == 'FAIL' else 0)


if __name__ == '__main__':
    main()
