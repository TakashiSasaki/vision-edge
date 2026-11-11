"""Model-free negative controls for decoded comparisons and aggregate integrity."""
import copy
import unittest
import numpy as np
from measure_numerical_effects import decoded_effects, validate_decoded, validate_metric, combine, stem_names


def proposal(cell=(4,5),score=.7):
    return {'cell':list(cell),'bbox':[100,200,300,400],'score':score,'_mask':np.zeros((128,128),bool)}


class NumericalEffectsTests(unittest.TestCase):
    def test_score_rounding_does_not_hide_discrete_changes(self):
        ref={'objects':[proposal()],'text':[]};actual=copy.deepcopy(ref)
        actual['objects'][0]['score']+=1e-6
        result=decoded_effects(ref,actual);validate_decoded(result)
        self.assertEqual(result['status'],'SAME_DISCRETE_OUTPUTS_SCORES_WITHIN_TOLERANCE')
        actual['objects'][0]['_mask'][0,0]=True
        result=decoded_effects(ref,actual);validate_decoded(result)
        self.assertEqual(result['status'],'DECODED_DIFFERENCES')
        self.assertEqual(result['groups']['objects']['pairs'][0]['mask_different_pixels'],1)
        bad=copy.deepcopy(result);bad['status']='SAME_DISCRETE_OUTPUTS_SCORES_WITHIN_TOLERANCE'
        with self.assertRaises(ValueError):validate_decoded(bad)

    def test_empty_groups_and_missing_ranked_cells_are_explicit(self):
        empty={'objects':[],'text':[]};result=decoded_effects(empty,empty);validate_decoded(result)
        ref={'objects':[proposal(),proposal((6,7))],'text':[]}
        for actual in ({'objects':[proposal((6,7))],'text':[]},{'objects':list(reversed(ref['objects'])),'text':[]}):
            result=decoded_effects(ref,actual);validate_decoded(result)
            self.assertEqual(result['status'],'DECODED_DIFFERENCES')
        with self.assertRaises(ValueError):decoded_effects(ref,{'objects':[proposal(),proposal()],'text':[]})
        with self.assertRaises(ValueError):decoded_effects(ref,{'objects':[]})

    def test_malformed_masks_boxes_scores_rejected(self):
        ref={'objects':[proposal()],'text':[]}
        for field,value in [('_mask',np.zeros((2,2),bool)),('bbox',[300,200,100,400]),('score',float('nan')),('cell',[64,0])]:
            bad=copy.deepcopy(ref);bad['objects'][0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):decoded_effects(ref,bad)

    def test_metric_and_zero_coverage_rejections(self):
        tolerance={'atol':2e-4,'rtol':2e-4,'near_zero_denominator_floor':1e-6}
        metric={**tolerance,'shape':[2],'dtype':'float32','failed_elements':0,'status':'PASS',
                'max_absolute_error':0.,'max_relative_error':0.,'rms_error':0.}
        validate_metric(metric,tolerance)
        for field,value in [('failed_elements',3),('status','FAIL'),('max_absolute_error',float('nan')),('atol',.1),('shape',[])]:
            bad=copy.deepcopy(metric);bad[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):validate_metric(bad,tolerance)
        with self.assertRaises(ValueError):combine([])
        self.assertEqual(len(stem_names()),48)
        self.assertEqual(len(set(stem_names())),48)


if __name__=='__main__':unittest.main()
