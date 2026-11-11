"""Separate reference-order arithmetic and geometry-only preprocessing experiments.

No old criteria are relaxed. Geometry observations cannot establish score, OCR,
language or full native equivalence. Failed numerical runs exit nonzero.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from safetensors.torch import load_file
import torch

from acquire_reference import CONFIG_IDENTITY
from archive_encoder_weights import load_archive_weights
from encoder_candidate import output_names
from encoder_fixtures import image_identity
from encoder_reference_order import ReferenceOrderEncoder
from geometry_evidence import validate as validate_geometry
from hconvglu import compare
from oasis_candidate_reference import capture, load_reference, OASIS_SHA256
from perception_diagnostics import decode, compare_native
from refinement_inputs import images, preprocess, RULES
from refinement_baseline import validate as validate_baseline
from refinement_scope import numerical_names
from verify_checkpoint import ARTIFACTS, REVISION, identity, read_json, verify


def expected_at_input(reference, namespace, image, x, meta):
    original = reference.letterbox
    reference.letterbox = lambda ignored: (torch.from_numpy(x), meta)
    try:
        return capture(reference, image, namespace)[0]
    finally:
        reference.letterbox = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'config', 'cact', 'photo', 'oasis-reference', 'geometry-report', 'output-dir', 'baseline-onnx', 'baseline-validation'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--weights',choices=['fp32','cact-dequantized'])
    parser.add_argument('--rule',choices=RULES)
    parser.add_argument('--cases',nargs='+')
    phase=parser.add_mutually_exclusive_group()
    phase.add_argument('--numerical-only',action='store_true')
    phase.add_argument('--geometry-only',action='store_true')
    args = parser.parse_args()
    if args.geometry_only and args.weights!='cact-dequantized':
        parser.error('geometry-only requires --weights cact-dequantized')
    if args.weights=='fp32' and not args.numerical_only:
        parser.error('FP32-only selection requires --numerical-only')
    print('process_id',os.getpid(),flush=True)
    torch.set_num_threads(2)
    checkpoint_id = verify(args.checkpoint, ARTIFACTS['mirage.safetensors'])
    native_id = verify(args.cact, ARTIFACTS['mirage.cact'])
    config_id = verify(args.config, CONFIG_IDENTITY)
    inputs = images(args.photo)
    geometry = read_json(args.geometry_report.read_bytes())
    native_cases = validate_geometry(geometry, {name: image_identity(im) for name, im in inputs.items()})
    old_report = read_json(args.baseline_validation.read_bytes())
    validated_baseline=validate_baseline(old_report)
    baseline_id = verify(args.baseline_onnx, (validated_baseline['size'], validated_baseline['sha256']))
    if args.cases and (len(args.cases)!=len(set(args.cases)) or not set(args.cases)<=inputs.keys()):
        raise ValueError('unknown or duplicate requested case')
    requested=args.cases or [name for name in inputs if not args.numerical_only or not name.startswith('holdout_')]
    base_names=numerical_names(requested,geometry_only=args.geometry_only,numerical_only=args.numerical_only)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
    baseline=None
    if not args.numerical_only:
        baseline = ort.InferenceSession(str(args.baseline_onnx), sess_options=options, providers=['CPUExecutionProvider'])
        if baseline.get_providers() != ['CPUExecutionProvider']:
            raise ValueError('unexpected baseline provider')
    weights = load_file(args.checkpoint)
    archive_weights, _ = load_archive_weights(args.cact, weights)
    source_names = ('verify_refinement.py', 'encoder_reference_order.py', 'refinement_inputs.py', 'geometry_evidence.py',
        'encoder_candidate.py', 'encoder_fixtures.py', 'oasis_candidate_reference.py', 'hconvglu.py',
        'archive_encoder_weights.py', 'perception_diagnostics.py', 'fixtures/refinement-images.json', 'fixtures/encoder-images.json','refinement_baseline.py','refinement_scope.py')
    report = {'schema_version': 1, 'completed': False, 'status': 'RUNNING',
              'scope': 'reference-order candidate arithmetic and explicitly geometry-only preprocessing hypotheses',
              'checkpoint_revision': REVISION, 'checkpoint': checkpoint_id, 'cact': native_id, 'config': config_id,
              'source_identities': {name: identity(Path(__file__).parent/name) for name in source_names},
              'reference_sha256': OASIS_SHA256, 'native_geometry_report': identity(args.geometry_report),
              'baseline_onnx': baseline_id, 'baseline_validation': identity(args.baseline_validation),
              'baseline_status': old_report['status'], 'models': [], 'numerical_cases': [], 'geometry_cases': [],
              'provider':['CPUExecutionProvider'],'numeric_tolerance':{'atol':2e-4,'rtol':2e-4,'near_zero_denominator_floor':1e-6},
              'versions':{name:importlib.metadata.version(name) for name in ('torch','numpy','onnx','onnxruntime','safetensors','Pillow')},
              'optimization':str(options.graph_optimization_level),
              'process_id':os.getpid(),
              'requested_scope': {'phase':'numerical' if args.numerical_only else 'geometry' if args.geometry_only else 'both',
                                  'weights':args.weights or 'both','rule':args.rule or 'all','cases':requested},
              'full_native_equivalence': 'NOT_ESTABLISHED', 'score_equivalence': 'NOT_TESTED_GEOMETRY_ONLY_SCOPE',
              'availability_gate': 'MIRAGE_ONNX_NOT_VALIDATED', 'webgpu': 'NOT_RUN: CPU/native acceptance incomplete'}
    numerical_failed = False
    names = output_names()
    archive_session = None
    for mode, current_weights in [('fp32', weights), ('cact-dequantized', archive_weights)]:
        if args.weights and args.weights!=mode: continue
        model = ReferenceOrderEncoder(current_weights, 'symmetric').eval()
        reference, namespace = load_reference(args.oasis_reference, current_weights, 'symmetric')
        x, hw, _ = preprocess(inputs['square_blank'], 'pixel')
        path = args.output_dir/(mode+'-reference-order.onnx')
        torch.onnx.export(model, (torch.from_numpy(x), torch.from_numpy(hw)), str(path), input_names=['image','valid_hw'],
                          output_names=names, opset_version=17, dynamo=False)
        graph = onnx.load(str(path))
        onnx.helper.set_model_props(graph, {'scope': 'PARTIAL_CANDIDATE_ENCODER_AND_HEADS',
            'arithmetic': 'unfused-reference-order', 'weights_mode': mode, 'checkpoint_revision': REVISION,
            'weights_sha256': checkpoint_id['sha256'] if mode=='fp32' else native_id['sha256'],
            'converter_sha256': report['source_identities']['encoder_reference_order.py']['sha256'],
            'reference_sha256': OASIS_SHA256, 'availability_gate': 'MIRAGE_ONNX_NOT_VALIDATED'})
        onnx.checker.check_model(graph, full_check=True)
        onnx.save(graph, str(path)); del graph
        session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
        if session.get_providers()!=['CPUExecutionProvider']:
            raise ValueError('unexpected provider')
        report['models'].append({'weights': mode, 'onnx': identity(path), 'file': path.name, 'onnx_checker': 'PASS'})
        if mode=='cact-dequantized': archive_session=session
        with torch.inference_mode():
            for rule in (() if args.geometry_only else (args.rule,) if args.rule else ('pixel','nearest32')):
                for name in base_names:
                    image=inputs[name]; x,hw,meta=preprocess(image,rule)
                    expected=expected_at_input(reference,namespace,image,x,meta)
                    eager=dict(zip(names,model(torch.from_numpy(x),torch.from_numpy(hw))))
                    actual=dict(zip(names,session.run(None,{'image':x,'valid_hw':hw})))
                    case={'name':name,'rule':rule,'weights':mode,'input_sha256':hashlib.sha256(x.astype('<f4').tobytes()).hexdigest(),
                          'metadata':meta,'measurements':[],'first_failed_captured_checkpoint':{}}
                    for label,left,right in [('eager_vs_oasis',eager,expected),('onnx_vs_oasis',actual,expected),('onnx_vs_eager',actual,eager)]:
                        for tensor in names:
                            a=left[tensor].numpy() if torch.is_tensor(left[tensor]) else left[tensor]
                            b=right[tensor].numpy() if torch.is_tensor(right[tensor]) else right[tensor]
                            metric=compare(b,a)
                            case['measurements'].append({'comparison':label,'tensor':tensor,**metric})
                            if metric['status']=='FAIL':
                                numerical_failed=True
                                case['first_failed_captured_checkpoint'].setdefault(label,tensor)
                    report['numerical_cases'].append(case)
                    print(mode,rule,name,'numeric failures=',sum(m['status']=='FAIL' for m in case['measurements']),flush=True)
                    (args.output_dir/'validation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    # Freeze the numerical comparison before geometry-only analysis. None of
    # the native boxes/masks can fill any tensor or eager/ONNX comparison above.
    heads=['detections','prototypes','coefficients']
    for model_name,session in ([] if args.numerical_only else [('folded-baseline',baseline),('reference-order',archive_session)]):
        if session is None: raise ValueError('geometry model was not constructed')
        for rule in ((args.rule,) if args.rule else RULES):
            for name in requested:
                image=inputs[name]
                x,hw,meta=preprocess(image,rule)
                raw=dict(zip(heads,session.run(heads,{'image':x,'valid_hw':hw})))
                case={'name':name,'rule':rule,'model':model_name,'metadata':meta,**image_identity(image),
                      'input_sha256':hashlib.sha256(x.astype('<f4').tobytes()).hexdigest()}
                if native_cases[name]['status']=='OBSERVED':
                    case['comparison']=compare_native(decode(raw,meta),native_cases[name]['output'],compare_scores=False)
                else:
                    case['comparison']={'status':'BLOCKED_NATIVE_FAILURE','error':native_cases[name].get('error')}
                report['geometry_cases'].append(case)
            print('geometry',model_name,rule,'completed',len(requested),'images',flush=True)
            (args.output_dir/'validation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    report['completed']=True
    modes=1 if args.weights else 2
    num_rules=1 if args.rule else 2
    geo_rules=1 if args.rule else 3
    expected_num=0 if args.geometry_only else modes*num_rules*len(base_names)
    expected_geo=0 if args.numerical_only else 2*geo_rules*len(requested)
    if len(report['numerical_cases'])!=expected_num or len(report['geometry_cases'])!=expected_geo:
        raise ValueError('requested scope did not complete')
    report['status']='GEOMETRY_DIAGNOSTICS_ONLY' if args.geometry_only else 'FAIL' if numerical_failed else 'PASS_NUMERICAL_ONLY_NATIVE_PARITY_NOT_ESTABLISHED'
    (args.output_dir/'validation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    raise SystemExit(1 if numerical_failed else 0)


if __name__=='__main__':
    main()
