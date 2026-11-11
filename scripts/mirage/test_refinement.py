"""Model-free controls for explicit refinement hypotheses and geometry scope."""
import copy
import unittest

import numpy as np
from PIL import Image
import torch
from torch import nn

from capture_native_reference import LIBRARY_SHA256
from encoder_reference_order import ReferenceOrderEncoder
from geometry_evidence import GEOMETRY_OPTIONS, validate
from hconvglu import band_reference
from perception_diagnostics import compare_native, decode
from refinement_inputs import extent, preprocess
from verify_checkpoint import ARTIFACTS, REVISION
from acquire_reference import CONFIG_IDENTITY
from encoder_candidate import required_shapes
from refinement_baseline import validate as validate_baseline
from refinement_scope import numerical_names
from aggregate_refinement import exact_coverage, same_contract, geometry_groups, CONTRACT_KEYS


class RefinementTests(unittest.TestCase):
    def test_empty_or_malformed_geometry_groups_are_not_coverage(self):
        g={'candidate_count':0,'native_count':0,'matched_count':0,'unmatched_candidate':0,'unmatched_native':0,'pairs':[]}
        original={'groups':{'objects':copy.deepcopy(g),'text':copy.deepcopy(g)}}
        self.assertEqual(len(list(geometry_groups(original))),2)
        for mutation in ('empty','missing','negative','boolean'):
            d=copy.deepcopy(original)
            if mutation=='empty':d['groups']={}
            elif mutation=='missing':del d['groups']['text']
            elif mutation=='negative':d['groups']['text']['native_count']=-1
            else:d['groups']['objects']['candidate_count']=False
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):geometry_groups(d)

    def test_coverage_rejects_duplicates_missing_and_empty(self):
        expected={('model','rule','a'),('model','rule','b')}
        exact_coverage(expected,list(expected))
        for actual in ([],[('model','rule','a')],[('model','rule','a')]*2,[('model','rule','a'),('model','rule','other')]):
            with self.assertRaises(ValueError):exact_coverage(expected,actual)
        with self.assertRaises(ValueError):exact_coverage(set(),[])

    def test_aggregate_contract_rejects_mixed_sources_runtime_and_tolerance(self):
        original={key:{'value':key} for key in CONTRACT_KEYS}
        same_contract(original,copy.deepcopy(original))
        for key in ('checkpoint','source_identities','provider','numeric_tolerance','versions','native_geometry_report','baseline_onnx'):
            changed=copy.deepcopy(original);changed[key]={'value':'changed'}
            with self.subTest(key=key),self.assertRaises(ValueError):same_contract(original,changed)

    def test_empty_numerical_scope_never_reports_green(self):
        with self.assertRaises(ValueError): numerical_names(['holdout_264_portrait'])
        with self.assertRaises(ValueError): numerical_names([])
        self.assertEqual(numerical_names(['holdout_264_portrait'],geometry_only=True),[])
        with self.assertRaises(ValueError):numerical_names(['square_blank','holdout_264_portrait'],numerical_only=True)
        self.assertEqual(numerical_names(['square_blank'],numerical_only=True),['square_blank'])

    def test_constructor_registers_rope_without_unused_folded_matrices(self):
        weights={name:torch.zeros(shape) for name,shape in required_shapes().items()}
        m=ReferenceOrderEncoder(weights,'symmetric')
        self.assertEqual(len(m.keys),452)
        self.assertFalse(hasattr(m,'body_0_gate_map'))
        x=torch.ones(1,4,260,32)
        y=m.rope(x)
        self.assertEqual(tuple(y.shape),tuple(x.shape))
        torch.testing.assert_close(y[:,:,256:],x[:,:,256:],rtol=0,atol=0)

    def test_baseline_requires_current_pins_and_cpu_but_retains_fail(self):
        report={'completed':True,'status':'FAIL','checkpoint_revision':REVISION,'weights_mode':'cact-dequantized',
                'padding':'symmetric','provider':['CPUExecutionProvider'],'onnx':{'size':1,'sha256':'graph'}}
        for k,value in [('checkpoint',ARTIFACTS['mirage.safetensors']),('cact',ARTIFACTS['mirage.cact']),('config',CONFIG_IDENTITY)]:
            report[k]=dict(zip(('size','sha256'),value))
        self.assertEqual(validate_baseline(report),report['onnx'])
        self.assertEqual(report['status'],'FAIL')
        for key,value in [('checkpoint_revision','main'),('checkpoint',{}),('cact',{}),('config',{}),('provider',['CUDAExecutionProvider']),('completed',False)]:
            bad=copy.deepcopy(report);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_baseline(bad)

    def test_half_up_alignment_is_not_bankers_rounding_or_ceil(self):
        self.assertEqual(extent(264,512,'nearest32'),(256,512))
        self.assertEqual(extent(272,512,'nearest32'),(288,512))
        self.assertEqual(extent(321,512,'nearest32'),(320,512))
        self.assertEqual(extent(264,512,'ceil32'),(288,512))
        self.assertEqual(extent(512,264,'nearest32'),(512,256))
        self.assertEqual(extent(1,4096,'nearest32'),(32,512))
        for w,h,rule in [(0,1,'pixel'),(1,-1,'pixel'),(1,1,'guess')]:
            with self.assertRaises(ValueError): extent(w,h,rule)

    def test_inverse_uses_actual_two_axis_resize_extents(self):
        x,hw,meta=preprocess(Image.new('RGB',(342,512)), 'nearest32')
        np.testing.assert_array_equal(hw,[512,352])
        self.assertEqual(meta['W'],342)
        self.assertAlmostEqual(meta['coordinate_scale'][0],512/352)
        self.assertEqual(meta['coordinate_scale'][1],1)
        self.assertEqual(tuple(x.shape),(1,3,512,512))
        np.testing.assert_array_equal(x[0,:,0,400],[0,0,0])

    def test_gather_band_matches_independent_windows_and_zero_edges(self):
        m=ReferenceOrderEncoder.__new__(ReferenceOrderEncoder);nn.Module.__init__(m)
        m.register_buffer('band_gather_indices',(torch.arange(512)[:,None]+torch.arange(65)[None,:]).reshape(-1))
        x=torch.eye(512); w=(torch.arange(512*65).reshape(512,65)%97).float()/128
        torch.testing.assert_close(m.band(x,w),band_reference(x,w),atol=0,rtol=0)
        self.assertEqual(float(m.band(x,w)[0,-1]),0)
        self.assertEqual(float(m.band(x,w)[-1,0]),0)

    def test_geometry_scope_has_no_fabricated_score(self):
        proposal={'objects':[{'bbox':[0,0,100,100],'score':.7}], 'text':[]}
        native={'objects':[{'bbox':[0,0,100,100]}], 'text':[]}
        result=compare_native(proposal,native,compare_scores=False)
        self.assertEqual(result['score_scope'],'NOT_REQUESTED_GEOMETRY_ONLY')
        self.assertIsNone(result['groups']['objects']['pairs'][0]['score_absolute_difference'])
        with self.assertRaises(KeyError): compare_native(proposal,native)

    def valid_evidence(self):
        inputs={'one':{'shape':[1,1,3],'rgb_sha256':'fixed'}}
        data={'completed':True,'scope':'GEOMETRY_ONLY_NO_SCORE_PARITY','options':copy.deepcopy(GEOMETRY_OPTIONS),'language':'en',
              'checkpoint_revision':REVISION,'model':dict(zip(('size','sha256'),ARTIFACTS['mirage.cact'])),
              'library':{'sha256':LIBRARY_SHA256},'status':'OBSERVED',
              'cases':[{'name':'one',**inputs['one'],'status':'OBSERVED','output':{'objects':[],'text':[]}}]}
        return data,inputs

    def test_geometry_capture_rejects_different_scope_scores_and_duplicates(self):
        original,inputs=self.valid_evidence();self.assertEqual(len(validate(original,inputs)),1)
        for mutation in ('scope','options','score','duplicate','input','aggregate','partial'):
            d=copy.deepcopy(original)
            if mutation=='scope':d['scope']='FULL'
            elif mutation=='options':d['options']['fields'].append('score')
            elif mutation=='score':d['cases'][0]['output']['objects']=[{'score':.7}]
            elif mutation=='duplicate':d['cases'].append(d['cases'][0])
            elif mutation=='input':d['cases'][0]['rgb_sha256']='changed'
            elif mutation=='aggregate':d['status']='FAIL'
            else:d['completed']=False
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):validate(d,inputs)

    def test_decode_honors_explicit_coordinate_inverse(self):
        raw=np.zeros((1,64,64,10),np.float32);raw[...,0:2]=-20
        raw[0,8,8,0]=10;raw[0,8,8,2:4]=[4,4]
        outputs={'detections':raw,'prototypes':np.zeros((1,32,128,128),np.float32),'coefficients':np.zeros((1,32,64,64),np.float32)}
        meta={'W':342,'H':512,'w':352,'h':512,'r':1,'coordinate_scale':[512/352,1,512/352,1]}
        got=decode(outputs,meta)['objects'][0]['bbox']
        self.assertEqual(got,[136,94,227,156])
        meta['coordinate_scale']=[1,2]
        with self.assertRaises(ValueError):decode(outputs,meta)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
