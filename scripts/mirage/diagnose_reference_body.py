"""Distinguish local FP32 arithmetic from upstream drift at a candidate body block.

This is diagnostic evidence, not native equivalence or model acceptance. Exposing
intermediates can change ONNX optimization; its effect is measured separately.
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
from archive_encoder_weights import load_archive_weights
from encoder_reference_order import ReferenceOrderEncoder
from hconvglu import compare
from oasis_candidate_reference import load_reference, OASIS_SHA256
from refinement_inputs import images, preprocess
from verify_checkpoint import ARTIFACTS, REVISION, identity, read_json, verify
from verify_refinement import expected_at_input

STAGES = ('mix_norm','mix_value','mix_gate','mix_gate_silu','mix_product','mix_down',
          'residual_pre','hconv_input','walsh','gate_linear','gate_pre','gate','value',
          'product','out_band','hconv_output','grn_r','grn_mean','grn','body_output')
PRODUCERS = ('verify_refinement.py', 'encoder_reference_order.py', 'refinement_inputs.py', 'geometry_evidence.py', 'encoder_candidate.py', 'encoder_fixtures.py', 'oasis_candidate_reference.py', 'hconvglu.py', 'archive_encoder_weights.py', 'perception_diagnostics.py', 'fixtures/refinement-images.json', 'fixtures/encoder-images.json', 'refinement_baseline.py', 'refinement_scope.py')


def production_identity(report, mode):
    if report.get('completed') is not True or report.get('status') != 'FAIL':
        raise ValueError('a completed, retained failed refinement aggregate is required')
    if report['checkpoint_revision'] != REVISION or report['reference_sha256'] != OASIS_SHA256 or report['provider'] != ['CPUExecutionProvider']:
        raise ValueError('production provenance mismatch')
    for key, expected in [('checkpoint',ARTIFACTS['mirage.safetensors']),('cact',ARTIFACTS['mirage.cact']),('config',CONFIG_IDENTITY)]:
        if report[key] != dict(zip(('size','sha256'),expected)):
            raise ValueError('production artifact pin mismatch')
    if report['numeric_tolerance'] != {'atol':2e-4,'rtol':2e-4,'near_zero_denominator_floor':1e-6}:
        raise ValueError('production tolerance mismatch')
    if set(report['source_identities']) != set(PRODUCERS):
        raise ValueError('unexpected production source scope')
    for name in PRODUCERS:
        if report['source_identities'][name] != identity(Path(__file__).parent/name):
            raise ValueError('production source mismatch: '+name)
    if mode not in ('fp32','cact-dequantized'):
        raise ValueError('unknown weight mode')
    return report['models'][mode]


def failure_mask(expected, actual):
    if expected.shape != actual.shape or not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError('matching finite arrays required')
    a,b=expected.astype(np.float64),actual.astype(np.float64)
    return np.abs(a-b)>2e-4+2e-4*np.abs(a)


def failure_examples(expected, actual, limit=8):
    indices=np.argwhere(failure_mask(expected,actual))
    a,b=expected.astype(np.float64),actual.astype(np.float64)
    tolerance=2e-4+2e-4*np.abs(a)
    return [{'index':index.tolist(),'reference':float(a[tuple(index)]),'actual':float(b[tuple(index)]),
             'absolute_error':float(abs(a[tuple(index)]-b[tuple(index)])),
             'allowed_error':float(tolerance[tuple(index)])} for index in indices[:limit]]


class BodyProbe(nn.Module):
    def __init__(self, weights, block):
        super().__init__()
        if type(block) is not int or not 0<=block<16:
            raise ValueError('body block must be0..15')
        self.model=ReferenceOrderEncoder(weights,'symmetric')
        self.prefix=f'body_{block}'

    def forward(self,x,mask):
        m,p=self.model,self.prefix
        z=m.norm(x,p+'/mix_norm/scale',1)
        v=m.conv(z,p+'/mix/dw_value',depthwise=True)+m.weight(p+'/mix/b_value')[None,:,None,None]
        g=m.conv(z,p+'/mix/dw_gate',depthwise=True)+m.weight(p+'/mix/b_gate')[None,:,None,None]
        a=F.silu(g);product=v*a;down=m.conv(product,p+'/mix/dw_down',depthwise=True)
        t=(x+mask*down).permute(0,2,3,1)
        hinput=m.norm(t,p+'/hconv_norm/scale')
        u=hinput@m.explicit_walsh
        linear=m.band(u,m.weight(p+'/hconv/gate_conv'))
        pre=linear+m.weight(p+'/hconv/gate_bias')
        gate=F.gelu(pre,approximate='tanh')
        value=m.band(u,m.weight(p+'/hconv/value_conv'))
        hproduct=gate*value
        band=m.band(hproduct,m.weight(p+'/hconv/out_conv'))
        h=band@m.explicit_walsh
        mm=mask.permute(0,2,3,1)
        r=torch.sqrt(((mm*h)**2).sum((1,2),keepdim=True)+1e-6)
        mean=r.mean(-1,keepdim=True)
        grn=h+m.weight(p+'/grn/gamma')*h*r/(mean+1e-6)+m.weight(p+'/grn/beta')
        return (z,v,g,a,product,down,t,hinput,u,linear,pre,gate,value,hproduct,band,h,r,mean,grn,(t+mm*grn).permute(0,3,1,2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','config','cact','photo','oasis-reference','aggregate','production-onnx','output-dir'):
        parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--weights',choices=('fp32','cact-dequantized'),default='cact-dequantized')
    parser.add_argument('--block',type=int,choices=range(1,16),default=3)
    parser.add_argument('--cases',nargs='+',default=['square_blank','small_objects','photo_square_stretched'])
    args=parser.parse_args()
    if not 1<=len(args.cases)<=5 or len(set(args.cases))!=len(args.cases):
        parser.error('request one to five distinct fixture names per bounded run')
    torch.set_num_threads(2)
    pins={k:verify(path,expected) for k,path,expected in [('checkpoint',args.checkpoint,ARTIFACTS['mirage.safetensors']),('cact',args.cact,ARTIFACTS['mirage.cact']),('config',args.config,CONFIG_IDENTITY)]}
    aggregate=read_json(args.aggregate.read_bytes())
    graph_id=production_identity(aggregate,args.weights)
    verify(args.production_onnx,(graph_id['size'],graph_id['sha256']))
    inputs=images(args.photo)
    if not set(args.cases)<=inputs.keys():raise ValueError('unknown fixture')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    source_names=(*PRODUCERS,'diagnose_reference_body.py')
    report={'scope':'isolated body arithmetic, upstream drift and instrumentation effect; no official native acceptance',
        'completed':False,'status':'RUNNING','checkpoint_revision':REVISION,**pins,'reference_sha256':OASIS_SHA256,
        'source_identities':{n:identity(Path(__file__).parent/n) for n in source_names},
        'aggregate':identity(args.aggregate),'production_onnx':graph_id,'weights':args.weights,'block':args.block,
        'rule':'pixel','requested_cases':args.cases,'cases':[],
        'provider':['CPUExecutionProvider'],'threads':{'torch':2,'ort_intra':2,'ort_inter':1},
        'versions':{n:importlib.metadata.version(n) for n in ('torch','numpy','onnx','onnxruntime','safetensors','Pillow')},
        'numeric_tolerance':aggregate['numeric_tolerance'],'availability_gate':'MIRAGE_ONNX_NOT_VALIDATED',
        'limitations':['Oasis is an unofficial candidate; no native intermediates are used.','Outputs expose20stages and can affect ONNX fusion/layout; replay measures rather than assumes instrumentation identity.','A local stage PASS does not accept accumulated full-model error.']}
    def save():
        (args.output_dir/'diagnosis.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    weights=load_file(args.checkpoint)
    if args.weights=='cact-dequantized':weights,_=load_archive_weights(args.cact,weights)
    reference,namespace=load_reference(args.oasis_reference,weights,'symmetric')
    probe=BodyProbe(weights,args.block).eval()
    path=args.output_dir/'body-stages.onnx'
    with torch.inference_mode():
        torch.onnx.export(probe,(torch.zeros(1,512,16,16),torch.ones(1,1,16,16)),str(path),input_names=['x','mask'],output_names=list(STAGES),opset_version=17,dynamo=False)
    onnx.checker.check_model(onnx.load(str(path)),full_check=True)
    report['diagnostic_graph']={'identity':identity(path),'onnx_checker':'PASS'}
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    report['optimization']=str(options.graph_optimization_level)
    diagnostic=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
    production=ort.InferenceSession(str(args.production_onnx),sess_options=options,providers=['CPUExecutionProvider'])
    if any(s.get_providers()!=['CPUExecutionProvider'] for s in (diagnostic,production)):
        raise ValueError('unexpected provider')
    prefix=f'body_{args.block}'
    previous=f'body_{args.block-1}'
    with torch.inference_mode():
        for name in args.cases:
            x,hw,meta=preprocess(inputs[name],'pixel')
            expected=expected_at_input(reference,namespace,inputs[name],x,meta)
            input_tensor=expected[previous]
            mask=probe.model.mask(torch.from_numpy(hw),32)
            eager=probe(input_tensor,mask)
            if not torch.equal(eager[-1],expected[prefix]):raise ValueError('diagnostic body no longer exactly reproduces candidate reference')
            actual=diagnostic.run(None,{'x':input_tensor.numpy(),'mask':mask.numpy()})
            observed=production.run([previous,prefix],{'image':x,'valid_hw':hw})
            production_input=observed[0]
            replay=diagnostic.run(['body_output'],{'x':production_input,'mask':mask.numpy()})[0]
            # Preserve the reference tensor's strides while changing only its values.
            replay_tensor=torch.empty_strided(input_tensor.shape,input_tensor.stride()).copy_(torch.from_numpy(production_input))
            torch_replay=probe(replay_tensor,mask)[-1].numpy()
            target=expected[prefix].numpy()
            case={'name':name,'metadata':meta,'input_shape':list(input_tensor.shape),'input_strides':list(input_tensor.stride()),
                'reference_input_sha256':hashlib.sha256(input_tensor.numpy().tobytes()).hexdigest(),
                'production_input_sha256':hashlib.sha256(production_input.tobytes()).hexdigest(),
                'local':[{'stage':n,**compare(e.numpy(),a)} for n,e,a in zip(STAGES,eager,actual)],
                'upstream_drift':compare(input_tensor.numpy(),production_input),
                'production':compare(target,observed[1]),'onnx_replay':compare(target,replay),
                'torch_replay':compare(target,torch_replay),
                'instrumentation_effect':compare(observed[1],replay),
                'production_failure_examples':failure_examples(target,observed[1]),
                'onnx_replay_failure_examples':failure_examples(target,replay),
                'torch_replay_failure_examples':failure_examples(target,torch_replay),
                'failure_overlap':{}}
            production_failed=failure_mask(target,observed[1])
            for label,values in [('onnx_replay',replay),('torch_replay',torch_replay)]:
                failed=failure_mask(target,values)
                case['failure_overlap'][label]={'same_failure_mask':bool(np.array_equal(production_failed,failed)),
                    'intersection_elements':int(np.count_nonzero(production_failed & failed)),
                    'production_failed_elements':int(np.count_nonzero(production_failed)),
                    'replay_failed_elements':int(np.count_nonzero(failed))}
            report['cases'].append(case);save()
            print(name,'local_failed_stages',sum(m['status']=='FAIL' for m in case['local']),
                  'production_failed_elements',case['production']['failed_elements'],
                  'torch_replay_failed_elements',case['torch_replay']['failed_elements'],flush=True)
    if len(report['cases'])!=len(args.cases):raise ValueError('incomplete diagnosis')
    report['completed']=True
    # Every actual numerical failure is retained, including production and replay.
    rows=[m for c in report['cases'] for m in c['local']+[c[k] for k in ('upstream_drift','production','onnx_replay','torch_replay','instrumentation_effect')]]
    report['status']='FAIL' if any(m['status']=='FAIL' for m in rows) else 'LOCAL_DIAGNOSTIC_PASS_FULL_MODEL_NOT_ACCEPTED'
    save();raise SystemExit(1 if report['status']=='FAIL' else 0)


if __name__=='__main__':main()
