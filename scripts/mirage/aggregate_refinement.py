"""Audit exact case coverage and shared provenance before combining scoped runs."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from encoder_candidate import output_names
from verify_checkpoint import identity, read_json, ARTIFACTS, REVISION
from acquire_reference import CONFIG_IDENTITY

CONTRACT_KEYS=('checkpoint_revision','checkpoint','cact','config','source_identities','reference_sha256',
               'native_geometry_report','baseline_onnx','baseline_validation','baseline_status','provider',
               'numeric_tolerance','versions','optimization')
LABELS=('eager_vs_oasis','onnx_vs_oasis','onnx_vs_eager')


def exact_coverage(expected, actual):
    if not expected or len(actual)!=len(set(actual)) or set(actual)!=set(expected):
        raise ValueError('missing, duplicate, unexpected or zero-coverage records')


def same_contract(reference, candidate):
    if {k:reference[k] for k in CONTRACT_KEYS}!={k:candidate[k] for k in CONTRACT_KEYS}:
        raise ValueError('scoped experiment provenance/runtime/tolerance mismatch')


def geometry_groups(diagnostic):
    groups=diagnostic['groups']
    if set(groups)!={'objects','text'}:raise ValueError('missing or unexpected geometry groups')
    for group in groups.values():
        for key in ('candidate_count','native_count','matched_count','unmatched_candidate','unmatched_native'):
            if type(group[key]) is not int or group[key]<0:raise ValueError('invalid geometry count')
        if not isinstance(group['pairs'],list):raise ValueError('invalid geometry pair records')
    return groups.values()


def combine(paths):
    reports=[read_json(Path(p).read_bytes()) for p in paths]
    if not reports:raise ValueError('no reports')
    first=reports[0]
    if first['checkpoint_revision']!=REVISION or first['provider']!=['CPUExecutionProvider'] or first['numeric_tolerance']!={'atol':2e-4,'rtol':2e-4,'near_zero_denominator_floor':1e-6}:
        raise ValueError('unexpected experiment revision/provider/tolerance')
    for key,value in [('checkpoint',ARTIFACTS['mirage.safetensors']),('cact',ARTIFACTS['mirage.cact']),('config',CONFIG_IDENTITY)]:
        if first[key]!=dict(zip(('size','sha256'),value)):raise ValueError('experiment artifact pin mismatch')
    fixture_names=list(read_json((Path(__file__).parent/'fixtures/refinement-images.json').read_bytes()))
    base=[name for name in fixture_names if not name.startswith('holdout_')]
    expected_num={(w,r,n) for w in ('fp32','cact-dequantized') for r in ('pixel','nearest32') for n in base}
    expected_geo={(m,r,n) for m in ('folded-baseline','reference-order') for r in ('pixel','nearest32','ceil32') for n in fixture_names}
    numerical=[];geometry=[];models={};input_hashes={}
    for path,report in zip(paths,reports):
        if report.get('completed') is not True:raise ValueError('incomplete scoped experiment')
        same_contract(first,report)
        scope=report['requested_scope']
        nkeys=[(c['weights'],c['rule'],c['name']) for c in report['numerical_cases']]
        gkeys=[(c['model'],c['rule'],c['name']) for c in report['geometry_cases']]
        if scope['phase']=='numerical':
            if scope['weights'] not in ('fp32','cact-dequantized') or scope['rule'] not in ('pixel','nearest32'):
                raise ValueError('unexpected bounded numerical variant')
            exact_coverage({(scope['weights'],scope['rule'],n) for n in scope['cases']},nkeys)
            if gkeys:raise ValueError('unexpected geometry in numeric scope')
        elif scope['phase']=='geometry':
            if scope['weights']!='cact-dequantized' or scope['rule'] not in ('pixel','nearest32','ceil32'):
                raise ValueError('unexpected geometry variant')
            exact_coverage({(m,scope['rule'],n) for m in ('folded-baseline','reference-order') for n in scope['cases']},gkeys)
            if nkeys:raise ValueError('unexpected numeric data in geometry scope')
        else:raise ValueError('aggregation requires explicit bounded phases')
        wanted='GEOMETRY_DIAGNOSTICS_ONLY' if scope['phase']=='geometry' else 'FAIL' if any(m['status']=='FAIL' for c in report['numerical_cases'] for m in c['measurements']) else 'PASS_NUMERICAL_ONLY_NATIVE_PARITY_NOT_ESTABLISHED'
        if report['status']!=wanted:raise ValueError('scoped status contradicts measurements')
        if {m['weights'] for m in report['models']}!={scope['weights']}:raise ValueError('scoped model identity mismatch')
        for model in report['models']:
            file=Path(path).parent/model['file']
            if identity(file)!=model['onnx'] or model['onnx_checker']!='PASS':raise ValueError('ONNX artifact changed/unvalidated')
            old=models.setdefault(model['weights'],model['onnx'])
            if old!=model['onnx']:raise ValueError('different graph for same weight mode')
        for case in report['numerical_cases']:
            exact_coverage({(label,n) for label in LABELS for n in output_names()},
                           [(m['comparison'],m['tensor']) for m in case['measurements']])
            for metric in case['measurements']:
                if any(metric[k]!=first['numeric_tolerance'][k] for k in ('atol','rtol','near_zero_denominator_floor')):
                    raise ValueError('per-tensor tolerance mismatch')
                count=metric['failed_elements']
                if not 0<=count<=math.prod(metric['shape']) or metric['status']!=('FAIL' if count else 'PASS'):
                    raise ValueError('comparison aggregate contradicts elements')
                if not all(math.isfinite(metric[k]) and metric[k]>=0 for k in ('max_absolute_error','max_relative_error','rms_error')):
                    raise ValueError('invalid numerical metric')
            numerical.append(case)
        for case in report['geometry_cases']:
            diagnostic=case['comparison']
            if diagnostic['status']!='DIAGNOSTIC_NOT_PARITY_ACCEPTANCE' or diagnostic['score_scope']!='NOT_REQUESTED_GEOMETRY_ONLY':
                raise ValueError('unexpected native comparison scope')
            for group in geometry_groups(diagnostic):
                pairs=group['pairs'];a=[p['candidate'] for p in pairs];b=[p['native'] for p in pairs]
                if len(set(a))!=len(a) or len(set(b))!=len(b) or group['matched_count']!=len(pairs):raise ValueError('duplicate/inconsistent matches')
                if group['unmatched_candidate']!=group['candidate_count']-len(pairs) or group['unmatched_native']!=group['native_count']-len(pairs):
                    raise ValueError('unmatched counts inconsistent')
                for pair in pairs:
                    if not (0<=pair['candidate']<group['candidate_count'] and 0<=pair['native']<group['native_count'] and .5<=pair['bbox_iou']<=1):
                        raise ValueError('invalid matched index/IoU')
                    if pair['score_absolute_difference'] is not None:raise ValueError('fabricated geometry score comparison')
                    if pair['mask_iou'] is not None and not 0<=pair['mask_iou']<=1:raise ValueError('invalid mask IoU')
            geometry.append(case)
        for case in report['numerical_cases']+report['geometry_cases']:
            key=(case['rule'],case['name'])
            previous=input_hashes.setdefault(key,case['input_sha256'])
            if previous!=case['input_sha256']:raise ValueError('input tensor differs across weight/model variants')
    exact_coverage(expected_num,[(c['weights'],c['rule'],c['name']) for c in numerical])
    exact_coverage(expected_geo,[(c['model'],c['rule'],c['name']) for c in geometry])
    result={k:first[k] for k in CONTRACT_KEYS}
    counts={}
    for w in ('fp32','cact-dequantized'):
        for r in ('pixel','nearest32'):
            rows=[m for c in numerical if c['weights']==w and c['rule']==r for m in c['measurements']]
            counts[w+'/'+r]={label:dict(Counter(m['status'] for m in rows if m['comparison']==label)) for label in LABELS}
    result.update(schema_version=1,completed=True,status='FAIL' if any(m['status']=='FAIL' for c in numerical for m in c['measurements']) else 'NUMERICAL_ONLY_PASS',
                  scope='candidate arithmetic and exploratory geometry; no full native/score/OCR equivalence',
                  models=models,numerical_counts=counts,numerical_cases=numerical,geometry_cases=geometry,
                  reports=[{'path':str(p),'identity':identity(p)} for p in paths],
                  aggregator=identity(Path(__file__)),availability_gate='MIRAGE_ONNX_NOT_VALIDATED')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reports',nargs='+',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();result=combine(a.reports)
    with a.output.open('x') as f:json.dump(result,f,separators=(',',':'),allow_nan=False);f.write('\n')
    print(json.dumps(result['numerical_counts']))
    raise SystemExit(1 if result['status']=='FAIL' else 0)


if __name__=='__main__':main()
