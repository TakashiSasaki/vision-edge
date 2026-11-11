"""Trace early rounding and measure its candidate decoded-output effects.

These are diagnostics against unofficial Oasis, not native/model acceptance.
The existing failed aggregate, graph, weights and tolerances remain unchanged.
"""
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
from torch import nn
from torch.nn import functional as F

from acquire_reference import CONFIG_IDENTITY
from aggregate_refinement import exact_coverage
from archive_encoder_weights import load_archive_weights
from diagnose_reference_body import production_identity, PRODUCERS
from encoder_reference_order import ReferenceOrderEncoder
from encoder_fixtures import image_identity
from hconvglu import compare
from oasis_candidate_reference import load_reference, OASIS_SHA256
from perception_diagnostics import decode, box_iou
from refinement_inputs import images, preprocess
from verify_checkpoint import ARTIFACTS, REVISION, identity, read_json, verify
from verify_refinement import expected_at_input

HEADS=('detections','prototypes','coefficients')
STEM_CASES=('square_blank','small_objects','photo_square_stretched')
CONTRACT=('checkpoint_revision','checkpoint','cact','config','reference_sha256','sources',
          'aggregate','models','provider','versions','optimization','tolerance','decoder')


def proposal_map(rows, kind):
    result={}
    for row in rows:
        cell=row['cell'];box=row['bbox'];score=row['score']
        if len(cell)!=2 or any(type(v) is not int or not 0<=v<64 for v in cell):
            raise ValueError('invalid proposal cell')
        if tuple(cell) in result:raise ValueError('duplicate proposal cell')
        if len(box)!=4 or any(type(v) is not int or not 0<=v<=1000 for v in box) or box[0]>box[2] or box[1]>box[3]:
            raise ValueError('invalid decoded box')
        if not np.isfinite(score) or not 0<=score<=1:raise ValueError('invalid score')
        if kind=='objects' and (row['_mask'].shape!=(128,128) or row['_mask'].dtype!=np.bool_):
            raise ValueError('invalid decoded diagnostic mask')
        result[tuple(cell)]=row
    return result


def decoded_effects(reference, actual):
    if set(reference)!= {'objects','text'} or set(actual)!=set(reference):
        raise ValueError('both decoded groups are required')
    groups={};changed=False
    for kind in ('objects','text'):
        e,a=proposal_map(reference[kind],kind),proposal_map(actual[kind],kind)
        common=sorted(e.keys() & a.keys());pairs=[]
        order=[p['cell'] for p in reference[kind]]==[p['cell'] for p in actual[kind]]
        for cell in common:
            x,y=e[cell],a[cell];difference=abs(x['score']-y['score'])
            pair={'cell':list(cell),'reference_bbox':x['bbox'],'onnx_bbox':y['bbox'],
                  'bbox_equal':x['bbox']==y['bbox'],'bbox_iou':box_iou(x['bbox'],y['bbox']),
                  'reference_score':x['score'],'onnx_score':y['score'],'score_absolute_error':difference,
                  'score_tolerance_pass':difference<=2e-4+2e-4*abs(x['score']),
                  'mask_different_pixels':None,'mask_iou':None,'mask_status':'NOT_APPLICABLE'}
            # IoU has no informative union for identical zero-area boxes.
            if any(b[0]==b[2] or b[1]==b[3] for b in (x['bbox'],y['bbox'])):
                pair['bbox_iou']=None
            if kind=='objects':
                u=x['_mask']|y['_mask'];union=int(np.count_nonzero(u))
                pair.update(mask_different_pixels=int(np.count_nonzero(x['_mask']!=y['_mask'])),
                            mask_iou=float(np.count_nonzero(x['_mask']&y['_mask'])/union) if union else 1.,
                            mask_status='MEASURED' if union else 'EMPTY_BOTH')
            pairs.append(pair)
        missing=sorted(e.keys()-a.keys());extra=sorted(a.keys()-e.keys())
        differs=bool(missing or extra or not order or any(not p['bbox_equal'] or not p['score_tolerance_pass'] or p['mask_different_pixels'] not in (None,0) for p in pairs))
        changed|=differs
        groups[kind]={'reference_count':len(e),'onnx_count':len(a),'ordered_cells_equal':order,
                      'missing_cells':[list(c) for c in missing],'extra_cells':[list(c) for c in extra],
                      'reference_ordered_cells':[p['cell'] for p in reference[kind]],
                      'onnx_ordered_cells':[p['cell'] for p in actual[kind]],
                      'pairs':pairs,'decoded_difference':differs}
    return {'status':'DECODED_DIFFERENCES' if changed else 'SAME_DISCRETE_OUTPUTS_SCORES_WITHIN_TOLERANCE',
            'scope':'candidate decoder; integer boxes on0..1000 grid and128x128 masks; no names/OCR/native comparison',
            'groups':groups}


def stem_names():
    names=[]
    for i in range(5):
        names.extend([f'stem_{i}_conv'] if i==0 else [f'stem_{i}_silu',f'stem_{i}_depthwise',f'stem_{i}_pointwise'])
        names.extend(f'stem_{i}_{s}' for s in ('square','mean','epsilon','rsqrt','normalized','scaled'))
        names.append(f'stem_{i}')
    return names


class StemProbe(nn.Module):
    def __init__(self, weights):
        super().__init__();self.model=ReferenceOrderEncoder(weights,'symmetric')

    def forward(self,image,hw):
        m=self.model;result=[];x=image
        for i in range(5):
            if i:
                x=F.silu(x);result.append(x)
                x=m.conv(x,f'stem/dw_{i}',2,True);result.append(x)
                x=m.pointwise(x,f'stem/pw_{i}/kernel');result.append(x)
            else:x=m.conv(x,'stem/w0',2);result.append(x)
            square=x*x;mean=square.mean(1,keepdim=True);epsilon=mean+1e-6
            inv=torch.rsqrt(epsilon);normalized=x*inv
            scaled=normalized*(1+m.weight(f'stem/norm_{i}/scale')[None,:,None,None])
            x=scaled*m.mask(hw,2**(i+1))
            result.extend((square,mean,epsilon,inv,normalized,scaled,x))
        return tuple(result)


def measure(args):
    if not 1<=len(args.cases)<=5 or len(set(args.cases))!=len(args.cases):
        raise ValueError('one to five distinct fixtures required')
    torch.set_num_threads(2)
    pins={k:verify(p,v) for k,p,v in [('checkpoint',args.checkpoint,ARTIFACTS['mirage.safetensors']),('cact',args.cact,ARTIFACTS['mirage.cact']),('config',args.config,CONFIG_IDENTITY)]}
    baseline=read_json(args.aggregate.read_bytes());model_id=production_identity(baseline,args.weights)
    verify(args.production_onnx,(model_id['size'],model_id['sha256']))
    fixtures=images(args.photo)
    base_names={c['name'] for c in baseline['numerical_cases']}
    if not set(args.cases)<=base_names or (args.phase=='stem' and (args.rule!='pixel' or not set(args.cases)<=set(STEM_CASES))):
        raise ValueError('unknown or out-of-scope fixture/rule')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    source_names=(*PRODUCERS,'diagnose_reference_body.py','measure_numerical_effects.py')
    report={'schema_version':1,'phase':args.phase,'weights':args.weights,'rule':args.rule,'requested_cases':args.cases,
            'completed':False,'status':'RUNNING','checkpoint_revision':REVISION,**pins,'reference_sha256':OASIS_SHA256,
            'sources':{n:identity(Path(__file__).parent/n) for n in source_names},'aggregate':identity(args.aggregate),
            'models':baseline['models'],'provider':['CPUExecutionProvider'],
            'versions':{n:importlib.metadata.version(n) for n in ('torch','numpy','onnx','onnxruntime','safetensors','Pillow')},
            'optimization':str(options.graph_optimization_level),'tolerance':baseline['numeric_tolerance'],
            'decoder':'unchanged candidate0.3 local peaks; integer0..1000 boxes;128x128 diagnostic binary masks',
            'cases':[],'full_model_acceptance':'FAIL_RETAINED','native_equivalence':'NOT_ESTABLISHED',
            'availability_gate':'MIRAGE_ONNX_NOT_VALIDATED'}
    def save():
        (args.output_dir/'effects.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    weights=load_file(args.checkpoint)
    if args.weights=='cact-dequantized':weights,_=load_archive_weights(args.cact,weights)
    reference,namespace=load_reference(args.oasis_reference,weights,'symmetric')
    production=ort.InferenceSession(str(args.production_onnx),sess_options=options,providers=['CPUExecutionProvider'])
    probe=session=None
    if args.phase=='stem':
        probe=StemProbe(weights).eval();path=args.output_dir/'stem-stages.onnx'
        with torch.inference_mode():
            torch.onnx.export(probe,(torch.zeros(1,3,512,512),torch.tensor([512,512])),str(path),input_names=['image','valid_hw'],output_names=stem_names(),opset_version=17,dynamo=False)
        onnx.checker.check_model(onnx.load(str(path)),full_check=True)
        report['diagnostic_graph']={'identity':identity(path),'onnx_checker':'PASS'}
        session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
    if any(s.get_providers()!=['CPUExecutionProvider'] for s in (production,session) if s is not None):
        raise ValueError('unexpected provider')
    for name in args.cases:
        image=fixtures[name];x,hw,meta=preprocess(image,args.rule)
        expected=expected_at_input(reference,namespace,image,x,meta)
        old=next(c for c in baseline['numerical_cases'] if (c['weights'],c['rule'],c['name'])==(args.weights,args.rule,name))
        case={'name':name,'metadata':meta,**image_identity(image),'input_sha256':hashlib.sha256(x.tobytes()).hexdigest()}
        if case['input_sha256']!=old['input_sha256']:raise ValueError('input differs from failed baseline')
        if args.phase=='outputs':
            actual=dict(zip(HEADS,production.run(list(HEADS),{'image':x,'valid_hw':hw})))
            eager={n:expected[n].numpy() for n in HEADS}
            case['raw']=[{'tensor':n,**compare(eager[n],actual[n])} for n in HEADS]
            case['baseline_metrics_equal']=all(m=={k:v for k,v in next(p for p in old['measurements'] if p['comparison']=='onnx_vs_oasis' and p['tensor']==m['tensor']).items() if k!='comparison'} for m in case['raw'])
            case['decoded']=decoded_effects(decode(eager,meta),decode(actual,meta))
        else:
            with torch.inference_mode():eager=probe(torch.from_numpy(x),torch.from_numpy(hw))
            observed=session.run(None,{'image':x,'valid_hw':hw})
            case['stages']=[{'stage':n,**compare(e.numpy(),a),'different_elements':int(np.count_nonzero(e.numpy()!=a)),'reference_strides':list(e.stride())} for n,e,a in zip(stem_names(),eager,observed)]
            case['first_numerically_unequal_stage']=next((m['stage'] for m in case['stages'] if m['different_elements']),None)
            case['first_threshold_failure']=next((m['stage'] for m in case['stages'] if m['status']=='FAIL'),None)
            observed_final=production.run([f'stem_{i}' for i in range(5)],{'image':x,'valid_hw':hw})
            case['instrumentation_effect']=[]
            for i,a in enumerate(observed_final):
                j=stem_names().index(f'stem_{i}')
                if eager[j].numpy().tobytes()!=expected[f'stem_{i}'].numpy().tobytes():raise ValueError('stem probe disagrees with reference')
                case['instrumentation_effect'].append({'tensor':f'stem_{i}',**compare(a,observed[j])})
        report['cases'].append(case);save();print(args.phase,args.weights,args.rule,name,'measured',flush=True)
    if len(report['cases'])!=len(args.cases):raise ValueError('incomplete measurement')
    bad=any(c['decoded']['status']=='DECODED_DIFFERENCES' or not c['baseline_metrics_equal'] or any(m['status']=='FAIL' for m in c['raw']) for c in report['cases']) if args.phase=='outputs' else any(m['status']=='FAIL' for c in report['cases'] for m in c['stages']+c['instrumentation_effect'])
    report.update(completed=True,status='DIFFERENCES_RETAINED' if bad else 'MEASURED_NOT_FULL_MODEL_ACCEPTANCE');save()
    return 1 if bad else 0


def report_failed(report):
    if report['phase']=='outputs':
        return any(c['decoded']['status']=='DECODED_DIFFERENCES' or not c['baseline_metrics_equal'] or
                   any(m['status']=='FAIL' for m in c['raw']) for c in report['cases'])
    return any(m['status']=='FAIL' for c in report['cases'] for m in c['stages']+c['instrumentation_effect'])


def validate_metric(m, tolerance):
    import math
    if m['dtype']!='float32' or not m['shape'] or any(type(d) is not int or d<1 for d in m['shape']):
        raise ValueError('invalid metric shape/dtype')
    n=m['failed_elements']
    if type(n) is not int or not 0<=n<=math.prod(m['shape']) or m['status']!=('FAIL' if n else 'PASS'):
        raise ValueError('invalid metric count/status')
    if any(m[k]!=v for k,v in tolerance.items()):raise ValueError('metric tolerance changed')
    if any(not math.isfinite(m[k]) or m[k]<0 for k in ('max_absolute_error','max_relative_error','rms_error')):
        raise ValueError('invalid error values')


def validate_decoded(d):
    if set(d['groups'])!={'objects','text'}:raise ValueError('missing decoded group')
    any_difference=False
    for kind,g in d['groups'].items():
        e,a=g['reference_ordered_cells'],g['onnx_ordered_cells']
        for cells in (e,a):
            if any(len(c)!=2 or any(type(v) is not int or not 0<=v<64 for v in c) for c in cells):raise ValueError('invalid cell')
            if len(cells)!=len(set(map(tuple,cells))):raise ValueError('duplicate decoded cell')
        es,as_=set(map(tuple,e)),set(map(tuple,a))
        if type(g['reference_count']) is not int or type(g['onnx_count']) is not int:raise ValueError('invalid decoded count type')
        if g['reference_count']!=len(e) or g['onnx_count']!=len(a) or g['ordered_cells_equal']!=(e==a):raise ValueError('decoded count/order mismatch')
        if g['missing_cells']!=[list(c) for c in sorted(es-as_)] or g['extra_cells']!=[list(c) for c in sorted(as_-es)]:raise ValueError('missing/extra cells mismatch')
        exact_coverage(es&as_,[tuple(p['cell']) for p in g['pairs']]) if es&as_ else None
        if not es&as_ and g['pairs']:raise ValueError('unexpected pairs')
        different=bool(es!=as_ or e!=a)
        for pair in g['pairs']:
            x,y=pair['reference_bbox'],pair['onnx_bbox']
            for b in (x,y):
                if len(b)!=4 or any(type(v) is not int or not 0<=v<=1000 for v in b) or b[0]>b[2] or b[1]>b[3]:raise ValueError('invalid box')
            scores=[pair['reference_score'],pair['onnx_score']]
            if any(not np.isfinite(v) or not 0<=v<=1 for v in scores):raise ValueError('invalid decoded score')
            difference=abs(scores[0]-scores[1]);passed=difference<=2e-4+2e-4*abs(scores[0])
            if pair['score_absolute_error']!=difference or pair['score_tolerance_pass']!=passed or pair['bbox_equal']!=(x==y):raise ValueError('decoded pair mismatch')
            iou=pair['bbox_iou']
            expected_iou=None if any(b[0]==b[2] or b[1]==b[3] for b in (x,y)) else box_iou(x,y)
            if iou!=expected_iou:raise ValueError('invalid box IoU')
            pixels=pair['mask_different_pixels']
            if kind=='objects':
                if type(pixels) is not int or not 0<=pixels<=16384 or pair['mask_status'] not in ('MEASURED','EMPTY_BOTH') or not 0<=pair['mask_iou']<=1:raise ValueError('invalid mask metric')
                if (pixels==0)!=(pair['mask_iou']==1.) or pair['mask_status']=='EMPTY_BOTH' and pixels!=0:raise ValueError('inconsistent mask summary')
            elif pixels is not None or pair['mask_iou'] is not None or pair['mask_status']!='NOT_APPLICABLE':raise ValueError('unexpected text mask')
            different|=bool(x!=y or not passed or pixels not in (None,0))
        if g['decoded_difference']!=different:raise ValueError('decoded group status mismatch')
        any_difference|=different
    wanted='DECODED_DIFFERENCES' if any_difference else 'SAME_DISCRETE_OUTPUTS_SCORES_WITHIN_TOLERANCE'
    if d['status']!=wanted:raise ValueError('decoded aggregate mismatch')


def combine(paths):
    reports=[read_json(p.read_bytes()) for p in paths]
    if not reports:raise ValueError('no reports')
    baseline_path=Path(__file__).parents[2]/'docs/verification/mirage-refinement-results.json'
    baseline=read_json(baseline_path.read_bytes())
    for mode in ('fp32','cact-dequantized'):production_identity(baseline,mode)
    base=reports[0];numerical=[];stem=[]
    if base['aggregate']!=identity(baseline_path) or base['models']!=baseline['models'] or base['tolerance']!=baseline['numeric_tolerance'] or base['provider']!=['CPUExecutionProvider']:
        raise ValueError('baseline graph/tolerance/provider mismatch')
    for key in ('checkpoint_revision','checkpoint','cact','config','reference_sha256','versions'):
        if base[key]!=baseline[key]:raise ValueError('baseline provenance/runtime mismatch')
    names=(*PRODUCERS,'diagnose_reference_body.py','measure_numerical_effects.py')
    if base['sources']!={n:identity(Path(__file__).parent/n) for n in names}:raise ValueError('source mismatch')
    for r in reports:
        if r['completed'] is not True or {k:r[k] for k in CONTRACT}!={k:base[k] for k in CONTRACT}:raise ValueError('incomplete or mixed provenance')
        if r['phase'] not in ('outputs','stem') or r['weights'] not in ('fp32','cact-dequantized') or r['rule'] not in ('pixel','nearest32'):raise ValueError('unknown scope')
        exact_coverage(set(r['requested_cases']),[c['name'] for c in r['cases']])
        wanted='DIFFERENCES_RETAINED' if report_failed(r) else 'MEASURED_NOT_FULL_MODEL_ACCEPTANCE'
        if r['status']!=wanted:raise ValueError('scoped status contradicts measurements')
        for c in r['cases']:
            old=next(p for p in baseline['numerical_cases'] if (p['weights'],p['rule'],p['name'])==(r['weights'],r['rule'],c['name']))
            if c['input_sha256']!=old['input_sha256'] or c['metadata']!=old['metadata']:raise ValueError('baseline input mismatch')
            row={'weights':r['weights'],'rule':r['rule'],**c}
            metrics=c['raw'] if r['phase']=='outputs' else c['stages']+c['instrumentation_effect']
            for metric in metrics:validate_metric(metric,base['tolerance'])
            if r['phase']=='outputs':
                exact_coverage(set(HEADS),[m['tensor'] for m in c['raw']]);validate_decoded(c['decoded'])
                equal=all(m=={k:v for k,v in next(p for p in old['measurements'] if p['comparison']=='onnx_vs_oasis' and p['tensor']==m['tensor']).items() if k!='comparison'} for m in c['raw'])
                if c['baseline_metrics_equal']!=equal:raise ValueError('baseline replay flag mismatch')
                numerical.append(row)
            else:
                exact_coverage(set(stem_names()),[m['stage'] for m in c['stages']])
                if [m['stage'] for m in c['stages']]!=stem_names():raise ValueError('stem stage order mismatch')
                for metric in c['stages']:
                    count=metric['different_elements']
                    if type(count) is not int or not 0<=count<=int(np.prod(metric['shape'])) or (count==0)!=(metric['max_absolute_error']==0):raise ValueError('invalid unequal-element count')
                exact_coverage({f'stem_{i}' for i in range(5)},[m['tensor'] for m in c['instrumentation_effect']])
                first=next((m['stage'] for m in c['stages'] if m['different_elements']),None)
                first_fail=next((m['stage'] for m in c['stages'] if m['status']=='FAIL'),None)
                if c['first_numerically_unequal_stage']!=first or c['first_threshold_failure']!=first_fail:raise ValueError('first-stage summary mismatch')
                stem.append(row)
    exact_coverage({(c['weights'],c['rule'],c['name']) for c in baseline['numerical_cases']},[(c['weights'],c['rule'],c['name']) for c in numerical])
    exact_coverage({(w,'pixel',n) for w in ('fp32','cact-dequantized') for n in STEM_CASES},[(c['weights'],c['rule'],c['name']) for c in stem])
    result={k:base[k] for k in CONTRACT}
    result.update(completed=True,status='DIFFERENCES_RETAINED' if any(report_failed(r) for r in reports) else 'MEASURED_NOT_FULL_MODEL_ACCEPTANCE',
                  scope='candidate raw/decoded outputs and early rounding; global167FAIL retained; no native equivalence',
                  output_cases=numerical,stem_cases=stem,reports=[{'path':str(p),**identity(p)} for p in paths],
                  aggregator=identity(Path(__file__)),availability_gate='MIRAGE_ONNX_NOT_VALIDATED')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    m=sub.add_parser('measure')
    for n in ('checkpoint','config','cact','photo','oasis-reference','aggregate','production-onnx','output-dir'):
        m.add_argument('--'+n,type=Path,required=True)
    m.add_argument('--weights',choices=('fp32','cact-dequantized'),required=True)
    m.add_argument('--phase',choices=('outputs','stem'),required=True)
    m.add_argument('--rule',choices=('pixel','nearest32'),required=True)
    m.add_argument('--cases',nargs='+',required=True)
    c=sub.add_parser('combine');c.add_argument('--reports',type=Path,nargs='+',required=True);c.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.action=='measure':raise SystemExit(measure(a))
    result=combine(a.reports)
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print('completed',len(result['output_cases']),'output cases and',len(result['stem_cases']),'stem cases')
    # This is a measured diagnostic, never a full-model acceptance result.
    raise SystemExit(1 if result['status']=='DIFFERENCES_RETAINED' else 0)


if __name__=='__main__':main()
