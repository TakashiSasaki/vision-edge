"""Development-only policy selection and reserved official-native comparisons.

No native, numerical, names or OCR acceptance is implied by a completed study.
Native full-score failures are retained separately from geometry observations.
"""
import argparse
import ctypes
import datetime
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import onnxruntime as ort
from safetensors.torch import load_file
import torch

from aggregate_refinement import exact_coverage
from acquire_reference import CONFIG_IDENTITY
from archive_encoder_weights import load_archive_weights
from capture_native_reference import LIBRARY_SHA256, ENGINE_REVISION
from corpus_fixtures import cases, selected_cases, verify_image, MANIFEST, PROTOCOL
from corpus_postprocessing import decode_policy
from diagnose_reference_body import production_identity, PRODUCERS
from geometry_evidence import GEOMETRY_OPTIONS
from hconvglu import compare
from measure_numerical_effects import decoded_effects, validate_metric
from native_evidence_contract import OPTIONS
from oasis_candidate_reference import load_reference, OASIS_SHA256
from perception_diagnostics import compare_native
from refinement_inputs import preprocess, extent
from verify_checkpoint import ARTIFACTS, REVISION, identity, verify, read_json
from verify_refinement import expected_at_input

BASELINE=Path(__file__).parents[2]/'docs/verification/mirage-refinement-results.json'
HEADS=('detections','prototypes','coefficients')
REQUESTS={'full-score':OPTIONS,'geometry-only':GEOMETRY_OPTIONS}


def sources():
    names=(*PRODUCERS,'diagnose_reference_body.py','measure_numerical_effects.py',
           'native_corpus_study.py','corpus_fixtures.py','corpus_postprocessing.py',
           'fixtures/native-corpus.json','fixtures/native-protocol.json')
    return {n:identity(Path(__file__).parent/n) for n in names}


def header(kind):
    return {'schema_version':1,'kind':kind,'sources':sources(),'manifest':identity(MANIFEST),
            'protocol':identity(PROTOCOL),'checkpoint_revision':REVISION,'reference_sha256':OASIS_SHA256,
            'model':dict(zip(('size','sha256'),ARTIFACTS['mirage.cact'])),
            'library_sha256':LIBRARY_SHA256,'engine_revision':ENGINE_REVISION,
            'platform':{'system':platform.platform(),'machine':platform.machine(),'python':platform.python_version(),'torch_cpu':torch.backends.cpu.get_cpu_capability()},
            'versions':{n:importlib.metadata.version(n) for n in ('torch','numpy','onnx','onnxruntime','safetensors','Pillow')},
            'global_acceptance':'NOT_ESTABLISHED','availability_gate':'MIRAGE_ONNX_NOT_VALIDATED'}


def write(path,value,new=False):
    with path.open('x' if new else 'w') as f:json.dump(value,f,separators=(',',':'),allow_nan=False);f.write('\n')


def validate_header(report,kind):
    expected=header(kind)
    if any(report.get(k)!=v for k,v in expected.items()):raise ValueError('study provenance mismatch')
    if report.get('completed') is not True:raise ValueError('incomplete study scope')


def frozen_selection(path):
    if path is None:raise ValueError('reserved inference requires a frozen development-only selection')
    r=read_json(path.read_bytes());validate_header(r,'selection')
    expected=select_result([Path(p['path']) for p in r['candidate_reports']],Path(r['native_report']['path']))
    if any(r.get(k)!=v for k,v in expected.items()):raise ValueError('selection no longer matches development evidence')
    return r


def require_scope(args):
    rows=selected_cases(args.cases,args.split)
    selection=frozen_selection(args.selection) if args.split=='reserved_validation' else None
    if args.split=='development' and args.selection is not None:raise ValueError('development must not depend on reserved selection')
    return rows,selection


def native_requests_status(rows):
    return {mode:'FAIL' if any(c['requests'][mode]['status']=='FAIL' for c in rows) else 'OBSERVED' for mode in REQUESTS}


def parse_native(raw,mode):
    output=read_json(raw)
    if not isinstance(output,dict) or any(not isinstance(output.get(k),list) for k in ('objects','text')):
        raise ValueError('unexpected native structure')
    for group in ('objects','text'):
        for item in output[group]:
            if not isinstance(item,dict):raise ValueError('invalid native item')
            b=item.get('bbox')
            if not isinstance(b,list) or len(b)!=4 or any(type(v) not in (int,float) or not np.isfinite(v) for v in b):
                raise ValueError('invalid native box')
            if mode=='geometry-only' and 'score' in item:raise ValueError('unexpected geometry-only score')
            if mode=='full-score' and (type(item.get('score')) not in (int,float) or not np.isfinite(item['score']) or not 0<=item['score']<=1):
                raise ValueError('missing/nonfinite/out-of-range native score')
    return output


def capture(args):
    rows,selection=require_scope(args)
    model=verify(args.cact,ARTIFACTS['mirage.cact']);library=identity(args.library)
    if library['sha256']!=LIBRARY_SHA256:raise ValueError('unverified native library')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    lib=ctypes.CDLL(str(args.library.resolve()));text,integer=ctypes.c_char_p,ctypes.c_int
    bindings={'load':([text,ctypes.c_uint64],integer),'models':([],integer),'last_error':([],text),
              'set_threads':([integer],None),'describe':([text,integer,integer,text,text,text,text,integer],integer)}
    for name,(parameters,result) in bindings.items():
        method=getattr(lib,'cactus_'+name);method.argtypes=parameters;method.restype=result
    lib.cactus_set_threads(2);data=args.cact.read_bytes()
    if lib.cactus_load(data,len(data))<0 or not lib.cactus_models()&4:raise ValueError(lib.cactus_last_error().decode())
    report={**header('native'),'model':model,'library':library,'options':REQUESTS,'language':'en',
            'split':args.split,'selection':identity(args.selection) if selection else None,
            'requested_cases':args.cases,'cases':[],'completed':False,'status':'RUNNING','native_threads':2,
            'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    for c in rows:
        image=verify_image(args.images/c['file'],c);case={'name':c['name'],'rgb_sha256':c['rgb_sha256'],'shape':c['shape'],'requests':{}}
        for mode,options in REQUESTS.items():
            buffer=ctypes.create_string_buffer(16*1024*1024)
            code=lib.cactus_describe(image.tobytes(),image.width,image.height,json.dumps(options).encode(),b'en',None,buffer,len(buffer))
            path=args.output_dir/(c['name']+'-'+mode+'-raw.txt')
            with path.open('xb') as f:f.write(buffer.value)
            request={'native_code':code,'raw_path':str(path),'raw_identity':identity(path)}
            try:
                if code<0:raise ValueError(lib.cactus_last_error().decode())
                output=parse_native(buffer.value,mode)
                request.update(status='OBSERVED',output=output)
            except (ValueError,UnicodeError) as error:request.update(status='FAIL',error=f'{type(error).__name__}: {error}')
            case['requests'][mode]=request
        report['cases'].append(case);write(args.output_dir/'native.json',report)
        print(c['name'],[(k,v['status']) for k,v in case['requests'].items()],flush=True)
    report.update(completed=True,request_status=native_requests_status(report['cases']))
    report['status']='FAIL' if 'FAIL' in report['request_status'].values() else 'OBSERVED'
    write(args.output_dir/'native.json',report);return 1 if report['status']=='FAIL' else 0


def native_report(path,split,complete=True):
    r=read_json(path.read_bytes());validate_header(r,'native')
    if r['split']!=split or r['options']!=REQUESTS or r['language']!='en' or r['library']['sha256']!=LIBRARY_SHA256:raise ValueError('native scope mismatch')
    expected={n for n,c in cases().items() if c['split']==split} if complete else set(r['requested_cases'])
    exact_coverage(expected,[c['name'] for c in r['cases']])
    records=cases()
    for c in r['cases']:
        if c['rgb_sha256']!=records[c['name']]['rgb_sha256'] or c['shape']!=records[c['name']]['shape']:raise ValueError('native image mismatch')
        if set(c['requests'])!=set(REQUESTS):raise ValueError('missing native request scope')
        for mode,q in c['requests'].items():
            p=Path(q['raw_path'])
            if identity(p)!=q['raw_identity']:raise ValueError('native raw response changed')
            observed=None;failure=None
            if q['native_code']>=0:
                try:observed=parse_native(p.read_bytes(),mode)
                except (ValueError,UnicodeError) as error:failure=f'{type(error).__name__}: {error}'
            if q['status']=='OBSERVED':
                if q['native_code']<0 or observed is None or observed!=q['output']:raise ValueError('native parsed/raw mismatch')
            elif q['status']=='FAIL':
                if 'output' in q or q['native_code']>=0 and failure!=q['error']:raise ValueError('native failure/raw mismatch')
            else:raise ValueError('incomplete native request')
    status=native_requests_status(r['cases'])
    if r.get('request_status')!=status or r['status']!=('FAIL' if 'FAIL' in status.values() else 'OBSERVED'):raise ValueError('native status contradiction')
    return r


def join_native(paths,split):
    reports=[native_report(p,split,False) for p in paths]
    if not reports:raise ValueError('no native reports')
    first=reports[0];result={k:v for k,v in first.items() if k not in ('cases','requested_cases','started_utc')}
    rows=[c for r in reports for c in r['cases']]
    for r in reports:
        if r['selection']!=first['selection']:raise ValueError('mixed selection identities')
    expected=[n for n,c in cases().items() if c['split']==split];exact_coverage(set(expected),[c['name'] for c in rows])
    status=native_requests_status(rows)
    result.update(cases=rows,requested_cases=expected,request_status=status,status='FAIL' if 'FAIL' in status.values() else 'OBSERVED',
                  reports=[{'path':str(p),**identity(p)} for p in paths])
    return result


def decoded_summary(e,a):
    full=decoded_effects(e,a);result={'status':full['status'],'groups':{}}
    for kind,g in full['groups'].items():
        result['groups'][kind]={'reference_count':g['reference_count'],'onnx_count':g['onnx_count'],
            'ordered_cells_equal':g['ordered_cells_equal'],'missing_cells':g['missing_cells'],'extra_cells':g['extra_cells'],
            'bbox_differences':sum(not p['bbox_equal'] for p in g['pairs']),
            'mask_different_pixels':sum(p['mask_different_pixels'] or 0 for p in g['pairs']),
            'max_score_error':max((p['score_absolute_error'] for p in g['pairs']),default=0.),
            'score_tolerance_failures':sum(not p['score_tolerance_pass'] for p in g['pairs'])}
    return result


def measure(args):
    rows,selection=require_scope(args);torch.set_num_threads(2)
    baseline=read_json(BASELINE.read_bytes());graph_id=production_identity(baseline,args.weights)
    verify(args.checkpoint,ARTIFACTS['mirage.safetensors']);verify(args.config,CONFIG_IDENTITY);verify(args.cact,ARTIFACTS['mirage.cact'])
    verify(args.onnx,(graph_id['size'],graph_id['sha256']))
    native=native_report(args.native,args.split)
    if native['selection']!=(identity(args.selection) if selection else None):raise ValueError('native/reserved selection mismatch')
    ncases={c['name']:c for c in native['cases']};protocol=read_json(PROTOCOL.read_bytes())
    policies=protocol['candidate_grid'] if selection is None else [selection['selected_policy'],{'resize':'pixel','nms_iou':None}]
    policies=[p for i,p in enumerate(policies) if p['resize']==args.rule and p not in policies[:i]]
    if not policies:raise ValueError('rule is outside frozen reserved policies')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    weights=load_file(args.checkpoint)
    if args.weights=='cact-dequantized':weights,_=load_archive_weights(args.cact,weights)
    reference,namespace=load_reference(args.oasis_reference,weights,'symmetric')
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(args.onnx),sess_options=options,providers=['CPUExecutionProvider'])
    if session.get_providers()!=['CPUExecutionProvider']:raise ValueError('unexpected provider')
    report={**header('candidate'),'split':args.split,'weights':args.weights,'rule':args.rule,
            'baseline':identity(BASELINE),'onnx':graph_id,'provider':['CPUExecutionProvider'],
            'native_report':{'path':str(args.native),**identity(args.native)},'selection':identity(args.selection) if selection else None,
            'tolerance':baseline['numeric_tolerance'],'threads':{'torch':2,'ort_intra':2,'ort_inter':1},
            'optimization':str(options.graph_optimization_level),'checkpoint':identity(args.checkpoint),'config':identity(args.config),
            'requested_cases':args.cases,'completed':False,'cases':[]}
    for c in rows:
        image=verify_image(args.images/c['file'],c);x,hw,meta=preprocess(image,args.rule)
        expected=expected_at_input(reference,namespace,image,x,meta)
        actual=dict(zip(HEADS,session.run(list(HEADS),{'image':x,'valid_hw':hw})))
        eager={n:expected[n].numpy() for n in HEADS}
        case={'name':c['name'],'rgb_sha256':c['rgb_sha256'],'input_sha256':hashlib.sha256(x.tobytes()).hexdigest(),
              'metadata':meta,'raw':[{'tensor':n,**compare(eager[n],actual[n])} for n in HEADS],'policies':[]}
        geometry=ncases[c['name']]['requests']['geometry-only'];full=ncases[c['name']]['requests']['full-score']
        for policy in policies:
            p=decode_policy(actual,meta,policy['nms_iou']);e=decode_policy(eager,meta,policy['nms_iou'])
            row={'policy':policy,'pytorch_onnx_decoded':decoded_summary(e,p)}
            row['geometry']=compare_native(p,geometry['output'],compare_scores=False) if geometry['status']=='OBSERVED' else {'status':'BLOCKED_NATIVE_GEOMETRY_FAILURE'}
            if geometry['status']=='OBSERVED':
                pg=compare_native(e,geometry['output'],compare_scores=False)
                row['pytorch_native_geometry_equal']=pg==row['geometry']
                row['pytorch_native_geometry_if_different']=None if pg==row['geometry'] else pg
            row['scores']=compare_native(p,full['output']) if full['status']=='OBSERVED' else {'status':'BLOCKED_NATIVE_FULL_SCORE_FAILURE','error':full['error']}
            case['policies'].append(row)
        report['cases'].append(case);write(args.output_dir/'candidate.json',report);print(args.weights,args.rule,c['name'],'measured',flush=True)
    report.update(completed=True,numerical_status='FAIL' if any(m['status']=='FAIL' for c in report['cases'] for m in c['raw']) else 'PASS',
                  decoded_status='DIFFERENCES' if any(p['pytorch_onnx_decoded']['status']=='DECODED_DIFFERENCES' for c in report['cases'] for p in c['policies']) else 'SAME_DISCRETE_OUTPUTS',
                  status='MEASURED_NATIVE_PARITY_NOT_ESTABLISHED')
    write(args.output_dir/'candidate.json',report);return 1 if report['numerical_status']=='FAIL' or report['decoded_status']=='DIFFERENCES' else 0


def validate_comparison(d,scores):
    if d['status']!='DIAGNOSTIC_NOT_PARITY_ACCEPTANCE' or d['score_scope']!=('COMPARED' if scores else 'NOT_REQUESTED_GEOMETRY_ONLY') or set(d['groups'])!={'objects','text'}:
        raise ValueError('invalid comparison scope')
    for g in d['groups'].values():
        for k in ('candidate_count','native_count','matched_count','unmatched_candidate','unmatched_native'):
            if type(g[k]) is not int or g[k]<0:raise ValueError('invalid comparison count')
        pairs=g['pairs'];a=[p['candidate'] for p in pairs];b=[p['native'] for p in pairs]
        if len(set(a))!=len(a) or len(set(b))!=len(b) or len(pairs)!=g['matched_count'] or g['unmatched_candidate']!=g['candidate_count']-len(pairs) or g['unmatched_native']!=g['native_count']-len(pairs):
            raise ValueError('invalid one-to-one comparison')
        for p in pairs:
            if not 0<=p['candidate']<g['candidate_count'] or not 0<=p['native']<g['native_count'] or not .5<=p['bbox_iou']<=1:raise ValueError('invalid match index/IoU')
            if scores:
                if p['score_absolute_difference'] is None or not np.isfinite(p['score_absolute_difference']) or p['score_absolute_difference']<0:raise ValueError('invalid score metric')
            elif p['score_absolute_difference'] is not None:raise ValueError('fabricated geometry score')
            if p['mask_iou'] is not None and not 0<=p['mask_iou']<=1:raise ValueError('invalid mask IoU')


def validate_decoded_summary(d):
    if set(d['groups'])!={'objects','text'}:raise ValueError('missing decoded group')
    changed=False
    for kind,g in d['groups'].items():
        a,b=g['reference_count'],g['onnx_count']
        if type(a) is not int or type(b) is not int or min(a,b)<0:raise ValueError('invalid decoded count')
        n=a-len(g['missing_cells'])
        if n<0 or n!=b-len(g['extra_cells']):raise ValueError('decoded unmatched count mismatch')
        for field,maximum in [('bbox_differences',n),('score_tolerance_failures',n),('mask_different_pixels',16384*n if kind=='objects' else 0)]:
            if type(g[field]) is not int or not 0<=g[field]<=maximum:raise ValueError('invalid decoded metric')
        if not np.isfinite(g['max_score_error']) or g['max_score_error']<0:raise ValueError('invalid decoded score')
        if g['ordered_cells_equal'] and (a!=b or g['missing_cells'] or g['extra_cells']):raise ValueError('decoded order mismatch')
        changed|=bool(g['missing_cells'] or g['extra_cells'] or not g['ordered_cells_equal'] or g['bbox_differences'] or g['mask_different_pixels'] or g['score_tolerance_failures'])
    wanted='DECODED_DIFFERENCES' if changed else 'SAME_DISCRETE_OUTPUTS_SCORES_WITHIN_TOLERANCE'
    if d['status']!=wanted:raise ValueError('decoded status mismatch')


def validate_anchors(comparison,native,decoded,candidate_field):
    """Bind internally consistent aggregates to their actual inference records."""
    for kind in ('objects','text'):
        group=comparison['groups'][kind]
        if group['native_count']!=len(native[kind]) or group['candidate_count']!=decoded['groups'][kind][candidate_field]:
            raise ValueError('comparison count contradicts native/decoded evidence')


def candidate_reports(paths,split,native_path,selection=None):
    baseline=read_json(BASELINE.read_bytes());reports=[]
    observed=native_report(native_path,split);native_cases={c['name']:c for c in observed['cases']}
    protocol=read_json(PROTOCOL.read_bytes())
    reserved=frozen_selection(selection) if split=='reserved_validation' else None
    for path in paths:
        r=read_json(path.read_bytes());validate_header(r,'candidate')
        if r['weights'] not in ('fp32','cact-dequantized') or r['rule'] not in ('pixel','nearest32','ceil32'):raise ValueError('unknown candidate scope')
        if r['threads']!={'torch':2,'ort_intra':2,'ort_inter':1} or r['optimization']!=baseline['optimization'] or r['checkpoint']!=baseline['checkpoint'] or r['config']!=baseline['config']:raise ValueError('candidate runtime/artifact mismatch')
        if r['split']!=split or r['provider']!=['CPUExecutionProvider'] or r['baseline']!=identity(BASELINE) or r['onnx']!=baseline['models'][r['weights']] or r['native_report']!={'path':str(native_path),**identity(native_path)} or r['tolerance']!=baseline['numeric_tolerance']:
            raise ValueError('candidate inference contract mismatch')
        if r['selection']!=(identity(selection) if selection else None):raise ValueError('candidate selection mismatch')
        exact_coverage(set(r['requested_cases']),[c['name'] for c in r['cases']])
        selected_cases(r['requested_cases'],split)
        for c in r['cases']:
            if cases()[c['name']]['split']!=split or c['rgb_sha256']!=cases()[c['name']]['rgb_sha256']:raise ValueError('candidate input mismatch')
            h,w,_=cases()[c['name']]['shape'];rw,rh=extent(w,h,r['rule'])
            expected_meta={'W':w,'H':h,'w':rw,'h':rh,'r':512/max(w,h),'coordinate_scale':[512/rw,512/rh,512/rw,512/rh],'resize_rule':r['rule']}
            if c['metadata']!=expected_meta:raise ValueError('candidate resize metadata mismatch')
            exact_coverage(set(HEADS),[m['tensor'] for m in c['raw']])
            shapes=dict(zip(HEADS,([1,64,64,10],[1,32,128,128],[1,32,64,64])))
            for m in c['raw']:
                validate_metric(m,r['tolerance'])
                if m['shape']!=shapes[m['tensor']]:raise ValueError('unexpected raw head shape')
            policies=protocol['candidate_grid'] if reserved is None else [reserved['selected_policy'],{'resize':'pixel','nms_iou':None}]
            allowed={(p['resize'],p['nms_iou']) for p in policies if p['resize']==r['rule']}
            exact_coverage(allowed,[(p['policy']['resize'],p['policy']['nms_iou']) for p in c['policies']])
            for policy in c['policies']:
                validate_decoded_summary(policy['pytorch_onnx_decoded'])
                for field,mode,scores in [('geometry','geometry-only',False),('scores','full-score',True)]:
                    if native_cases[c['name']]['requests'][mode]['status']=='OBSERVED':
                        validate_comparison(policy[field],scores)
                        validate_anchors(policy[field],native_cases[c['name']]['requests'][mode]['output'],policy['pytorch_onnx_decoded'],'onnx_count')
                    elif policy[field]['status']!=('BLOCKED_NATIVE_FULL_SCORE_FAILURE' if scores else 'BLOCKED_NATIVE_GEOMETRY_FAILURE'):raise ValueError('native failure was hidden')
                if native_cases[c['name']]['requests']['geometry-only']['status']=='OBSERVED':
                    other=policy['pytorch_native_geometry_if_different']
                    if policy['pytorch_native_geometry_equal']:
                        if other is not None:raise ValueError('contradictory PyTorch comparison')
                        other=policy['geometry']
                    else:
                        validate_comparison(other,False)
                        if other==policy['geometry']:raise ValueError('contradictory PyTorch equality')
                    validate_anchors(other,native_cases[c['name']]['requests']['geometry-only']['output'],policy['pytorch_onnx_decoded'],'reference_count')
        wanted='FAIL' if any(m['status']=='FAIL' for c in r['cases'] for m in c['raw']) else 'PASS'
        decoded='DIFFERENCES' if any(p['pytorch_onnx_decoded']['status']=='DECODED_DIFFERENCES' for c in r['cases'] for p in c['policies']) else 'SAME_DISCRETE_OUTPUTS'
        if r['decoded_status']!=decoded:raise ValueError('decoded status contradiction')
        if r['numerical_status']!=wanted or r['status']!='MEASURED_NATIVE_PARITY_NOT_ESTABLISHED':raise ValueError('candidate status mismatch')
        reports.append(r)
    return reports


def geometry_totals(rows):
    totals={}
    if not rows:raise ValueError('zero geometry coverage')
    for row in rows:validate_comparison(row['geometry'],False)
    for kind in ('objects','text'):
        groups=[r['geometry']['groups'][kind] for r in rows];pairs=[p for g in groups for p in g['pairs']]
        a=sum(g['candidate_count'] for g in groups);b=sum(g['native_count'] for g in groups);tp=sum(g['matched_count'] for g in groups)
        totals[kind]={'candidate_count':a,'native_count':b,'matched':tp,'unmatched_candidate':a-tp,'unmatched_native':b-tp,
            'micro_f1':2*tp/(a+b) if a+b else 1.,'mean_matched_box_iou':sum(p['bbox_iou'] for p in pairs)/len(pairs) if pairs else None,
            'mean_valid_matched_mask_iou':float(np.mean([p['mask_iou'] for p in pairs if p['mask_iou'] is not None])) if any(p['mask_iou'] is not None for p in pairs) else None,
            'invalid_native_masks':sum(p['mask_status']=='INVALID_NATIVE_POLYGON' for p in pairs)}
    a=sum(g['candidate_count'] for g in totals.values());b=sum(g['native_count'] for g in totals.values());tp=sum(g['matched'] for g in totals.values())
    weighted=sum(g['mean_matched_box_iou']*g['matched'] for g in totals.values() if g['matched'])
    return {'groups':totals,'candidate_count':a,'native_count':b,'matched':tp,'unmatched_candidate':a-tp,'unmatched_native':b-tp,
            'micro_f1':2*tp/(a+b) if a+b else 1.,'mean_matched_box_iou':weighted/tp if tp else None}


def select_result(paths,native_path):
    native_report(native_path,'development');reports=candidate_reports(paths,'development',native_path)
    names=[n for n,c in cases().items() if c['split']=='development'];grid=read_json(PROTOCOL.read_bytes())['candidate_grid']
    exact_coverage({(w,r,n) for w in ('fp32','cact-dequantized') for r in ('pixel','nearest32','ceil32') for n in names},
                   [(r['weights'],r['rule'],c['name']) for r in reports for c in r['cases']])
    table=[]
    for policy in grid:
        selected=[]
        for report in reports:
            for c in report['cases']:
                exact_coverage({(p['resize'],p['nms_iou']) for p in grid if p['resize']==report['rule']},[(p['policy']['resize'],p['policy']['nms_iou']) for p in c['policies']])
                if report['weights']=='cact-dequantized':selected.extend(p for p in c['policies'] if p['policy']==policy)
        exact_coverage(set(names),[c['name'] for r in reports if r['weights']=='cact-dequantized' and r['rule']==policy['resize'] for c in r['cases']])
        table.append({'policy':policy,**geometry_totals(selected)})
    best=max(range(len(table)),key=lambda i:(table[i]['micro_f1'],table[i]['mean_matched_box_iou'] or 0,-i))
    return {**header('selection'),'completed':True,'status':'DEVELOPMENT_ONLY_FROZEN',
            'selection_weight_mode':'cact-dequantized','selection_native_scope':'geometry-only','selected_policy':table[best]['policy'],
            'policy_table':table,'development_cases':names,'candidate_reports':[{'path':str(p),**identity(p)} for p in paths],
            'native_report':{'path':str(native_path),**identity(native_path)}}


def summarize(selection_path,native_path,paths):
    selection=frozen_selection(selection_path);native=native_report(native_path,'reserved_validation')
    if native['selection']!=identity(selection_path):raise ValueError('native validation did not use frozen selection')
    reports=candidate_reports(paths,'reserved_validation',native_path,selection_path)
    names=[n for n,c in cases().items() if c['split']=='reserved_validation'];policies=[selection['selected_policy'],{'resize':'pixel','nms_iou':None}]
    policies=[p for i,p in enumerate(policies) if p not in policies[:i]];rules={p['resize'] for p in policies}
    exact_coverage({(w,r,n) for w in ('fp32','cact-dequantized') for r in rules for n in names},[(r['weights'],r['rule'],c['name']) for r in reports for c in r['cases']])
    table=[]
    for weight in ('fp32','cact-dequantized'):
        for policy in policies:
            rows=[p for r in reports if r['weights']==weight for c in r['cases'] for p in c['policies'] if p['policy']==policy]
            if len(rows)!=len(names):raise ValueError('missing/duplicate reserved policy')
            table.append({'weights':weight,'policy':policy,**geometry_totals(rows)})
    return {**header('summary'),'completed':True,'status':'DIAGNOSTIC_NATIVE_PARITY_NOT_ESTABLISHED',
            'selection':{'path':str(selection_path),**identity(selection_path)},'development':selection,
            'native_validation':{'path':str(native_path),**identity(native_path)},'reserved_cases':names,
            'reserved_table':table,'candidate_reports':[{'path':str(p),**identity(p)} for p in paths],
            'reserved_measurements':reports}


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for command in ('capture','measure'):
        q=sub.add_parser(command)
        for name in ('images','cact','output-dir'):q.add_argument('--'+name,type=Path,required=True)
        q.add_argument('--split',choices=('development','reserved_validation'),required=True);q.add_argument('--cases',nargs='+',required=True);q.add_argument('--selection',type=Path)
        if command=='capture':q.add_argument('--library',type=Path,required=True)
        else:
            for name in ('checkpoint','config','oasis-reference','onnx','native'):q.add_argument('--'+name,type=Path,required=True)
            q.add_argument('--weights',choices=('fp32','cact-dequantized'),required=True);q.add_argument('--rule',choices=('pixel','nearest32','ceil32'),required=True)
    q=sub.add_parser('join-native');q.add_argument('--reports',nargs='+',type=Path,required=True);q.add_argument('--split',choices=('development','reserved_validation'),required=True);q.add_argument('--output',type=Path,required=True)
    q=sub.add_parser('select');q.add_argument('--reports',nargs='+',type=Path,required=True);q.add_argument('--native',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    q=sub.add_parser('summarize');q.add_argument('--selection',type=Path,required=True);q.add_argument('--native',type=Path,required=True);q.add_argument('--reports',nargs='+',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.command=='capture':raise SystemExit(capture(a))
    if a.command=='measure':raise SystemExit(measure(a))
    if a.command=='join-native':r=join_native(a.reports,a.split)
    elif a.command=='select':r=select_result(a.reports,a.native);r['frozen_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    else:r=summarize(a.selection,a.native,a.reports)
    write(a.output,r,True)
    print(a.command,r['status'],r.get('selected_policy',''),flush=True)
    raise SystemExit(1 if r['status']=='FAIL' else 0)


if __name__=='__main__':main()
