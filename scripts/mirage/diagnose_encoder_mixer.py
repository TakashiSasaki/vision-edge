"""Localize mixer rounding on actual candidate-encoder inputs, with an FP64 oracle.

This is an isolated diagnosis, not a full-encoder acceptance test. FP64 does not
validate FP32. The old thresholds and all observed FAIL results are retained.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from safetensors.torch import load_file
import torch

import hconvglu as hc
from encoder_fixtures import fixtures
from oasis_candidate_reference import load_reference, capture, OASIS_SHA256
from verify_checkpoint import ARTIFACTS, REVISION, identity, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'oasis-reference', 'photo', 'output-dir'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    checkpoint_id = verify(args.checkpoint, ARTIFACTS['mirage.safetensors'])
    weights = load_file(args.checkpoint)
    hc.require_weights(weights)
    torch.set_num_threads(2)
    ref, namespace = load_reference(args.oasis_reference, weights, 'symmetric')
    images = fixtures(args.photo)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
    sessions = {}
    report = {'scope': 'top_11 mixer only at actual candidate encoder inputs; full encoder not accepted',
              'checkpoint_revision': REVISION, 'checkpoint': checkpoint_id, 'reference_sha256': OASIS_SHA256,
              'source_identities': {name: identity(Path(__file__).parent/name) for name in
                  ('diagnose_encoder_mixer.py', 'hconvglu.py', 'encoder_fixtures.py', 'oasis_candidate_reference.py', 'fixtures/encoder-images.json')},
              'completed': False, 'status': 'RUNNING', 'block': 'top_11/hconv', 'graphs': [], 'cases': [],
              'availability_gate': 'MIRAGE_ONNX_NOT_VALIDATED'}
    for precision in ('float32', 'float64'):
        for folded in (False, True):
            name = precision + ('-folded' if folded else '-explicit')
            graph = hc.export(weights, 'top_11/hconv', folded, precision)
            path = args.output_dir / (name + '.onnx')
            path.write_bytes(graph.SerializeToString())
            session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
            if session.get_providers() != ['CPUExecutionProvider']:
                raise RuntimeError('unexpected provider')
            sessions[name] = session
            report['graphs'].append({'name': name, 'identity': identity(path), 'onnx_checker': 'PASS'})
    double_weights = {key: value.double() for key, value in weights.items() if '/hconv/' in key}
    with torch.inference_mode():
        for case_name in ('square_blank', 'photo_square_stretched'):
            expected, _ = capture(ref, images[case_name], namespace)
            x = expected['top_11_hconv_input'].reshape(-1, 512)
            np.save(args.output_dir / (case_name + '-input.npy'), x.numpy(), allow_pickle=False)
            fp32 = hc.reference(x, weights, 'top_11/hconv')
            oracle = hc.reference(x.double(), double_weights, 'top_11/hconv')
            case = {'name': case_name, 'input_shape': list(x.shape), 'input_sha256': hashlib.sha256(x.numpy().astype('<f4').tobytes()).hexdigest(),
                    'comparisons': {}, 'first_failed_stage': {}}
            for name, session in sessions.items():
                precision = name.split('-')[0]
                tensor = x if precision == 'float32' else x.double()
                reference = fp32 if precision == 'float32' else oracle
                tolerance = {} if precision == 'float32' else {'atol': 1e-8, 'rtol': 1e-9}
                result = []
                values = session.run(None, {'x': tensor.numpy()})
                for output, value in zip(session.get_outputs(), values):
                    result.append({'stage': output.name, **hc.compare(reference[output.name].numpy(), value, **tolerance)})
                case['comparisons'][name] = result
                case['first_failed_stage'][name] = next((m['stage'] for m in result if m['status']=='FAIL'), None)
            # Compare both FP32 implementations to the same higher-precision
            # equation, rounded to FP32 only at the comparison boundary.
            case['comparisons']['fp32_reference_vs_fp64_oracle'] = [
                {'stage': name, **hc.compare(oracle[name].numpy().astype(np.float32), value.numpy())} for name,value in fp32.items()]
            report['cases'].append(case)
            print(case_name, case['first_failed_stage'], flush=True)
    report['completed'] = True
    report['status'] = 'FAIL' if any(m['status']=='FAIL' for c in report['cases'] for rows in c['comparisons'].values() for m in rows) else 'PASS'
    (args.output_dir / 'diagnosis.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    raise SystemExit(1 if report['status']=='FAIL' else 0)


if __name__ == '__main__':
    main()
