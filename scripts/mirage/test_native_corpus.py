"""Model-free controls for NMS, native failures, corpus identity and blind split guards."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import urllib.error

import numpy as np
from PIL import Image

from corpus_fixtures import cases, selected_cases, verify_image, acquire
from corpus_postprocessing import decode_policy
from native_corpus_study import require_scope, parse_native, validate_comparison, validate_anchors, geometry_totals
from perception_diagnostics import compare_native


def raw_outputs():
    raw=np.zeros((1,64,64,10),np.float32);raw[...,:2]=-20
    return {'detections':raw,'prototypes':np.zeros((1,32,128,128),np.float32),'coefficients':np.zeros((1,32,64,64),np.float32)}

META={'W':512,'H':512,'w':512,'h':512,'r':1.,'coordinate_scale':[1.,1.,1.,1.]}


class CorpusTests(unittest.TestCase):
    def test_nms_before_cap_keeps_lower_ranked_distinct_proposal(self):
        out=raw_outputs();raw=out['detections']
        for row in range(8):
            for col in range(8):
                i,j=2+row*3,2+col*3;raw[0,i,j,0]=4;raw[0,i,j,2:4]=4;raw[0,i,j,6:8]=[10-j,10-i]
        raw[0,59,59,0]=3;raw[0,59,59,2:4]=4
        self.assertEqual(len(decode_policy(out,META,None)['objects']),64)
        selected=decode_policy(out,META,.5)['objects']
        self.assertEqual([p['cell'] for p in selected],[[2,2],[59,59]])

    def test_exact_half_iou_is_not_suppressed(self):
        out=raw_outputs()
        for j,score in [(24,4),(32,3)]:
            out['detections'][0,24,j,0]=score;out['detections'][0,24,j,2:4]=24
        self.assertEqual(len(decode_policy(out,META,.5)['objects']),2)

    def test_unknown_policy_and_invalid_axis_scale_rejected(self):
        with self.assertRaises(ValueError):decode_policy(raw_outputs(),META,.4)
        bad={**META,'coordinate_scale':[1.,float('nan'),1.,1.]}
        with self.assertRaises(ValueError):decode_policy(raw_outputs(),bad,.5)

    def test_exact_corpus_splits_and_bounded_cases(self):
        records=cases();dev=[n for n,c in records.items() if c['split']=='development'];reserved=[n for n,c in records.items() if c['split']=='reserved_validation']
        self.assertEqual((len(dev),len(reserved)),(8,4))
        self.assertEqual(len(selected_cases(dev[:4],'development')),4)
        for names in ([],dev[:5],[dev[0],dev[0]],[reserved[0]],['unknown']):
            with self.subTest(names=names),self.assertRaises(ValueError):selected_cases(names,'development')

    def test_reserved_inference_requires_frozen_development_selection(self):
        reserved=next(n for n,c in cases().items() if c['split']=='reserved_validation')
        with self.assertRaises(ValueError):require_scope(SimpleNamespace(cases=[reserved],split='reserved_validation',selection=None))

    def test_image_byte_and_decoded_identity_rejected_when_corrupted(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'image.png';Image.new('RGB',(3,2),(15,20,30)).save(path);raw=path.read_bytes();image=Image.open(path).convert('RGB')
            record={'size':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'shape':[2,3,3],
                    'rgb_sha256':hashlib.sha256(image.tobytes()).hexdigest(),'exif_orientation':None}
            self.assertEqual(verify_image(path,record).size,(3,2))
            bad={**record,'rgb_sha256':'0'*64}
            with self.assertRaises(ValueError):verify_image(path,bad)
            path.write_bytes(raw[:-1])
            with self.assertRaises(ValueError):verify_image(path,record)

    def test_denied_image_download_is_recorded_and_not_retried(self):
        with tempfile.TemporaryDirectory() as d:
            with patch('corpus_fixtures.urllib.request.urlopen',side_effect=urllib.error.HTTPError('https://images.cocodataset.org/',403,'Forbidden',None,None)) as fetch:
                first=acquire(Path(d),False);second=acquire(Path(d),False)
            self.assertEqual(fetch.call_count,1)
            self.assertEqual((first['status'],second['status']),('FAIL','FAIL'))
            self.assertFalse(first['completed']);self.assertFalse(second['completed'])

    def test_native_nonfinite_and_scope_failures_are_not_sanitized(self):
        geometry=b'{"objects":[{"bbox":[0,0,1000,1000]}],"text":[]}'
        self.assertEqual(len(parse_native(geometry,'geometry-only')['objects']),1)
        for raw,scope in [(b'{"objects":[{"bbox":[0,0,1,1],"score":-nan}],"text":[]}','full-score'),
                          (b'{"objects":[{"bbox":[0,0,1,1],"score":NaN}],"text":[]}','full-score'),
                          (geometry,'full-score'),(b'{"objects":[{"bbox":[0,0,1,1],"score":0.5}],"text":[]}','geometry-only')]:
            with self.subTest(scope=scope),self.assertRaises(ValueError):parse_native(raw,scope)

    def test_matching_counts_and_geometry_score_omission_are_enforced(self):
        proposals={'objects':[{'bbox':[0,0,1000,1000],'score':.7}],'text':[]}
        native={'objects':[{'bbox':[0,0,1000,1000]}],'text':[]}
        value=compare_native(proposals,native,compare_scores=False);validate_comparison(value,False)
        self.assertEqual(geometry_totals([{'geometry':value}])['micro_f1'],1.)
        for mutation in ('score','count','index'):
            bad=copy.deepcopy(value)
            if mutation=='score':bad['groups']['objects']['pairs'][0]['score_absolute_difference']=0.
            elif mutation=='count':bad['groups']['objects']['unmatched_candidate']=1
            else:bad['groups']['objects']['pairs'][0]['candidate']=10
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):validate_comparison(bad,False)
        with self.assertRaises(ValueError):geometry_totals([])

    def test_internally_consistent_counts_must_match_raw_evidence(self):
        native={'objects':[{'bbox':[0,0,1000,1000]}],'text':[]}
        proposals={'objects':[{'bbox':[0,0,1000,1000],'score':.7}],'text':[]}
        value=compare_native(proposals,native,compare_scores=False)
        decoded={'groups':{'objects':{'onnx_count':1,'reference_count':1},'text':{'onnx_count':0,'reference_count':0}}}
        validate_anchors(value,native,decoded,'onnx_count')
        for count,unmatched in [('native_count','unmatched_native'),('candidate_count','unmatched_candidate')]:
            bad=copy.deepcopy(value);bad['groups']['objects'][count]+=1;bad['groups']['objects'][unmatched]+=1
            validate_comparison(bad,False)
            for field in ('onnx_count','reference_count'):
                with self.subTest(count=count,field=field),self.assertRaises(ValueError):validate_anchors(bad,native,decoded,field)


if __name__=='__main__':unittest.main()
