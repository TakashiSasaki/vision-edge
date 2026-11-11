"""Export and measure all 28 checkpoint HConvGLUs. Deliberately no image-model parity claim."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import onnxruntime as ort
from safetensors.torch import load_file
import torch

import hconvglu as hc
import verify_checkpoint as checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path, help='must not exist')
    parser.add_argument('--precision', choices=['float32', 'float64'], default='float32')
    args = parser.parse_args()
    artifact = checkpoint.verify(args.checkpoint, checkpoint.ARTIFACTS['mirage.safetensors'])
    config_hash = checkpoint.verify(args.config, (1430, 'f7a3c37642086f954b1000fe80edb544a153c76fe68fb1b8173ddb97d465e915'))
    config = checkpoint.read_json(args.config.read_bytes())
    weights = load_file(args.checkpoint)
    required = hc.require_weights(weights)
    tensors = checkpoint.inventory(args.checkpoint)
    for row in tensors:
        used = row['name'] in required
        row.update(mapping_status='MAPPED_ISOLATED_HCONVGLU' if used else 'OUT_OF_SCOPE_UNMAPPED',
                   reference_operation=row['name'].rsplit('/', 1)[-1] if used else None,
                   scope_reason='Official HConvGLU specification' if used else 'Surrounding encoder, heads, language, reader or training tensors are not exported')
        if used:
            suffix = row['name'].rsplit('/', 1)[-1]
            row['reference_operation'] = {
                'gate_conv': 'a_j = sum_t w[j,t] * u[j+t-32] + bias_j',
                'gate_bias': 'Add bias_j to gate band output before tanh-approximation GELU',
                'value_conv': 'v_j = sum_t w[j,t] * u[j+t-32]',
                'out_conv': 'o_j = sum_t w[j,t] * (GELU(a)*v)[j+t-32]; y=Walsh(o)',
            }[suffix]
            if suffix != 'gate_bias':
                row.update(layout='[output_channel,tap]', active_taps=32224, zero_padded_tap_slots=1056,
                           boundary='out-of-range source channels multiply zero, never wrap')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    report = {'schema_version': 1, 'scope': '28 isolated HConvGLUs, not an image encoder/detector',
              'checkpoint_revision': checkpoint.REVISION, 'checkpoint': artifact,
              'config_identity': config_hash, 'config': config,
              'source_identities': {name: checkpoint.identity(Path(__file__).parent / name)
                                    for name in ('hconvglu.py', 'verify_hconvglu.py', 'verify_checkpoint.py', 'fixtures/hconvglu-inputs.json')},
              'reference': 'Official equations: independent unfused PyTorch butterfly + zero-padded tap windows',
              'specification': 'https://cactuscompute.com/blog/hconvglu',
              'runtime_versions': {k: importlib.metadata.version(k) for k in ('torch', 'numpy', 'onnx', 'onnxruntime', 'safetensors')},
              'python': platform.python_version(), 'provider': 'CPUExecutionProvider',
              'arithmetic_precision': args.precision,
              'precision_note': 'FP32 original weights; float64 arithmetic is diagnosis only and does not validate FP32/WebGPU',
              'required_tensors': len(required), 'total_tensors': len(tensors),
              'fixtures': {}, 'blocks': [], 'first_divergent_operation': None,
              'official_runtime_intermediate_comparison': 'NOT_AVAILABLE: public ABI exposes describe, not HConvGLU tensors',
              'official_observable_equivalence': 'NOT_RUN: this subset does not produce boxes, masks, names or text',
              'webgpu': 'NOT_RUN', 'availability_gate': 'MIRAGE_ONNX_NOT_VALIDATED'}
    inputs = hc.fixtures()
    pinned_fixtures = checkpoint.read_json((Path(__file__).parent / 'fixtures/hconvglu-inputs.json').read_bytes())
    if inputs.keys() != pinned_fixtures.keys():
        raise ValueError('fixture set differs from immutable manifest')
    for name, x in inputs.items():
        pinned = pinned_fixtures[name]
        if (list(x.shape) != pinned['shape'] or str(x.dtype) != pinned['dtype'] or
                hashlib.sha256(x.astype('<f4').tobytes()).hexdigest() != pinned['raw_little_endian_sha256']):
            raise ValueError(f'fixture identity mismatch: {name}')
    if args.precision == 'float64':
        inputs = {name: x.astype(np.float64) for name, x in inputs.items()}
    reference_weights = weights if args.precision == 'float32' else {k: v.double() for k, v in weights.items() if k in required}
    for name, x in inputs.items():
        report['fixtures'][name] = {'shape': list(x.shape), 'dtype': str(x.dtype),
                                    'raw_little_endian_sha256': hashlib.sha256(x.astype('<f4' if args.precision == 'float32' else '<f8').tobytes()).hexdigest(),
                                    'kind': 'synthetic feature vectors with original pretrained weights'}
        np.save(args.output_dir / f'input-{name}.npy', x, allow_pickle=False)
    with torch.inference_mode():
        for prefix in hc.PREFIXES:
            references = {name: {stage: value.numpy() for stage, value in hc.reference(torch.from_numpy(x), reference_weights, prefix).items()}
                          for name, x in inputs.items()}
            block = {'prefix': prefix, 'exports': []}
            for folded in (False, True):
                mode = 'folded' if folded else 'explicit'
                path = args.output_dir / (prefix.replace('/', '-') + '-' + mode + '.onnx')
                model = hc.export(weights, prefix, folded, args.precision)
                # Artifact metadata binds the constants to these exact source weights and converter.
                import onnx
                onnx.helper.set_model_props(model, {**{p.key: p.value for p in model.metadata_props},
                    'checkpoint_revision': checkpoint.REVISION, 'checkpoint_sha256': artifact['sha256'],
                    'converter_sha256': report['source_identities']['hconvglu.py']['sha256']})
                path.write_bytes(model.SerializeToString())
                session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
                if session.get_providers() != ['CPUExecutionProvider']:
                    raise RuntimeError(f'unexpected providers: {session.get_providers()}')
                result = {'mode': mode, 'onnx': path.name, 'identity': checkpoint.identity(path),
                          'onnx_checker': 'PASS', 'measurements': []}
                for name, x in inputs.items():
                    actuals = session.run(None, {'x': x})
                    for output, actual in zip(session.get_outputs(), actuals):
                        tolerance = {} if args.precision == 'float32' else {'atol': 1e-8, 'rtol': 1e-9}
                        metric = hc.compare(references[name][output.name], actual, **tolerance)
                        result['measurements'].append({'fixture': name, 'stage': output.name, **metric})
                        if metric['status'] == 'FAIL' and report['first_divergent_operation'] is None:
                            report['first_divergent_operation'] = {'block': prefix, 'mode': mode, 'fixture': name, 'stage': output.name}
                result['status'] = 'PASS' if all(m['status'] == 'PASS' for m in result['measurements']) else 'FAIL'
                block['exports'].append(result)
            report['blocks'].append(block)
            print(f'{prefix}: ' + ', '.join(f"{r['mode']}={r['status']}" for r in block['exports']), flush=True)
    report['status'] = 'PASS' if report['first_divergent_operation'] is None else 'FAIL'
    (args.output_dir / 'tensor-mapping.json').write_text(json.dumps({'tensors': tensors}, indent=2) + '\n')
    (args.output_dir / 'validation.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    raise SystemExit(0 if report['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
