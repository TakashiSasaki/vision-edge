"""Model-free controls for diagnostic provenance and actual failure examples."""
import copy
import unittest
from pathlib import Path

import numpy as np
from diagnose_reference_body import production_identity, failure_examples, PRODUCERS
from acquire_reference import CONFIG_IDENTITY
from oasis_candidate_reference import OASIS_SHA256
from verify_checkpoint import ARTIFACTS, REVISION, identity


class DiagnosisControls(unittest.TestCase):
    def test_production_provenance_rejects_unverified_sources(self):
        report={'completed':True,'status':'FAIL','checkpoint_revision':REVISION,
                'reference_sha256':OASIS_SHA256,'provider':['CPUExecutionProvider'],
                'numeric_tolerance':{'atol':2e-4,'rtol':2e-4,'near_zero_denominator_floor':1e-6},
                'source_identities':{n:identity(Path(__file__).parent/n) for n in PRODUCERS},
                'models':{'cact-dequantized':{'size':1,'sha256':'test identity only'}}}
        for key,value in [('checkpoint',ARTIFACTS['mirage.safetensors']),('cact',ARTIFACTS['mirage.cact']),('config',CONFIG_IDENTITY)]:
            report[key]=dict(zip(('size','sha256'),value))
        self.assertEqual(production_identity(report,'cact-dequantized'),report['models']['cact-dequantized'])
        for key,value in [('completed',False),('status','PASS'),('checkpoint_revision','wrong'),
                          ('reference_sha256','wrong'),('provider',[]),('checkpoint',{}),('cact',{}),('config',{}),
                          ('numeric_tolerance',{}),('source_identities',{n:{} for n in PRODUCERS})]:
            invalid=copy.deepcopy(report);invalid[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):production_identity(invalid,'cact-dequantized')
        for name in PRODUCERS:
            invalid=copy.deepcopy(report);invalid['source_identities'][name]={'size':0,'sha256':'changed'}
            with self.subTest(source=name),self.assertRaises(ValueError):production_identity(invalid,'cact-dequantized')
        invalid=copy.deepcopy(report);invalid['source_identities']['unexpected.py']={}
        with self.assertRaises(ValueError):production_identity(invalid,'cact-dequantized')
        with self.assertRaises(ValueError):production_identity(report,'unknown')

    def test_failure_examples_use_elementwise_threshold_and_reject_invalid_values(self):
        reference=np.array([[0.,100.,-2.]],np.float32)
        actual=np.array([[.0003,100.001,-2.01]],np.float32)
        rows=failure_examples(reference,actual)
        self.assertEqual([r['index'] for r in rows],[[0,0],[0,2]])
        self.assertTrue(all(r['absolute_error']>r['allowed_error'] for r in rows))
        self.assertEqual(len(failure_examples(reference,actual,limit=1)),1)
        with self.assertRaises(ValueError):failure_examples(reference,actual[:,:2])
        with self.assertRaises(ValueError):failure_examples(reference,np.full_like(reference,np.nan))


if __name__=='__main__':unittest.main()
