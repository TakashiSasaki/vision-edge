"""Model-free controls for candidate encoder math, integrity and diagnostics."""
import tempfile
from pathlib import Path
import unittest
import copy

import numpy as np
from PIL import Image
import torch
from torch import nn

from archive_encoder_weights import decode_cq
from encoder_candidate import EncoderCandidate, required_shapes, validate_weights
from encoder_fixtures import preprocess, fixtures
from oasis_candidate_reference import load_reference
from perception_diagnostics import box_iou, compare_native
from native_evidence_contract import validate_native_evidence, OPTIONS
from verify_checkpoint import ARTIFACTS, REVISION
from capture_native_reference import LIBRARY_SHA256


class EncoderTests(unittest.TestCase):
    def test_native_options_duplicates_and_incomplete_evidence_rejected(self):
        inputs = {'fixture': {'shape': [1, 1, 3], 'rgb_sha256': 'test'}}
        report = {'completed': True, 'model': dict(zip(('size', 'sha256'), ARTIFACTS['mirage.cact'])),
                  'checkpoint_revision': REVISION, 'library': {'sha256': LIBRARY_SHA256},
                  'options': copy.deepcopy(OPTIONS), 'language': 'en', 'status': 'OBSERVED',
                  'cases': [{'name': 'fixture', **inputs['fixture'], 'status': 'OBSERVED', 'output': {'objects': [], 'text': []}}]}
        self.assertEqual(len(validate_native_evidence(report, inputs)), 1)
        for option, value in [('conf', 0.5), ('max_objects', 20), ('mask_format', 'bitmap')]:
            bad = copy.deepcopy(report); bad['options'][option] = value
            with self.subTest(option=option), self.assertRaises(ValueError): validate_native_evidence(bad, inputs)
        for mutation in ('duplicate', 'language', 'incomplete', 'aggregate'):
            bad = copy.deepcopy(report)
            if mutation == 'duplicate': bad['cases'].append(bad['cases'][0])
            elif mutation == 'language': bad['language'] = 'de'
            elif mutation == 'aggregate': bad['status'] = 'FAIL'
            else: bad['completed'] = False
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): validate_native_evidence(bad, inputs)

    def model(self, tensor, padding='symmetric'):
        model = EncoderCandidate.__new__(EncoderCandidate)
        nn.Module.__init__(model)
        model.padding = padding
        model.keys = {'test': 'test_weight'}
        model.register_buffer('test_weight', tensor)
        return model

    def test_padding_impulse_distinguishes_sample_alignment(self):
        kernel = torch.zeros(3, 3, 1, 1)
        kernel[1, 1] = 1
        image = torch.arange(16).float().reshape(1, 1, 4, 4)
        symmetric = self.model(kernel).conv(image, 'test', 2)
        same = self.model(kernel, 'same').conv(image, 'test', 2)
        torch.testing.assert_close(symmetric[0, 0], torch.tensor([[0., 2.], [8., 10.]]))
        torch.testing.assert_close(same[0, 0], torch.tensor([[5., 7.], [13., 15.]]))

    def test_rms_uses_one_plus_scale_without_mean_subtraction(self):
        model = self.model(torch.tensor([0.5, 1.0]))
        x = torch.tensor([[3., 4.]])
        expected = x * torch.tensor([1.5, 2.0]) / np.sqrt(12.5+1e-6)
        torch.testing.assert_close(model.norm(x, 'test'), expected)

    def test_valid_mask_includes_partial_stride_cell(self):
        mask = EncoderCandidate.mask(torch.tensor([33, 1]), 32)
        self.assertEqual(int(mask.sum()), 2)
        self.assertEqual(tuple(mask.shape), (1, 1, 16, 16))
        self.assertEqual(float(mask[0, 0, 1, 0]), 1)
        self.assertEqual(float(mask[0, 0, 0, 1]), 0)

    def test_cq_lsb_first_codes_and_natural_walsh_golden(self):
        # Repeated [-3,-1,1,3] * .5 has only natural-Hadamard bins 1 and 2.
        raw = bytes([0b11100100]) * 32 + np.array([0.5], dtype='<f2').tobytes()
        actual = decode_cq(raw, (1, 128), 128, 2, np.array([-3., -1., 1., 3.]))
        expected = np.zeros((1, 128), np.float32)
        expected[0, 1], expected[0, 2] = -64/np.sqrt(128), -128/np.sqrt(128)
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0)

    def test_bad_cq_encoding_is_rejected(self):
        for raw, group, bits in [(b'bad', 128, 2), (b'', 64, 2), (b'', 128, 3)]:
            with self.subTest(group=group, bits=bits), self.assertRaises(ValueError):
                decode_cq(raw, (1, 128), group, bits, np.zeros(4))

    def test_required_scope_never_fills_missing_extra_or_invalid_tensors(self):
        weights = {name: torch.zeros(shape) for name, shape in required_shapes().items()}
        self.assertEqual(len(validate_weights(weights)), 452)
        name = 'top_0/self_attn/q_proj/kernel'
        original = weights.pop(name)
        with self.assertRaises(ValueError): validate_weights(weights)
        for bad in (torch.zeros(1), original.double(), torch.full_like(original, float('nan'))):
            weights[name] = bad
            with self.assertRaises(ValueError): validate_weights(weights)
        weights[name] = original
        weights['body_16/invented'] = torch.zeros(1)
        with self.assertRaises(ValueError): validate_weights(weights)

    def test_candidate_preprocessing_zero_padding_and_rgb_scaling(self):
        image = Image.new('RGB', (2, 1), (255, 0, 127))
        pixels, hw, meta = preprocess(image)
        np.testing.assert_array_equal(hw, [256, 512])
        self.assertEqual(meta['r'], 256)
        np.testing.assert_array_equal(pixels[0, :, 300, 0], [0, 0, 0])
        np.testing.assert_allclose(pixels[0, :, 0, 0], [1, -1, 127/127.5-1], atol=1e-7)

    def test_unpinned_external_source_and_photo_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.py'
            path.write_text('raise RuntimeError("must not execute")')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                load_reference(path, {}, 'symmetric')
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                fixtures(path)

    def test_matching_is_one_to_one_and_reports_unmatched_predictions(self):
        predicted = {'objects': [{'bbox': [0, 0, 100, 100], 'score': 0.5}, {'bbox': [0, 0, 100, 100], 'score': 0.6}], 'text': []}
        native = {'objects': [{'bbox': [0, 0, 100, 100], 'score': 0.5}], 'text': []}
        result = compare_native(predicted, native)
        group = result['groups']['objects']
        self.assertEqual(group['matched_count'], 1)
        self.assertEqual(group['unmatched_candidate'], 1)
        self.assertIsNone(result['ocr_text_agreement'])
        self.assertEqual(result['status'], 'DIAGNOSTIC_NOT_PARITY_ACCEPTANCE')
        self.assertEqual(box_iou([0, 0, 0, 0], [0, 0, 0, 0]), 0)

    def test_invalid_native_polygon_is_preserved_without_fabricated_iou(self):
        proposals = {'objects': [{'bbox': [0, 0, 100, 100], 'score': 0.5, '_mask': np.ones((128, 128), bool)}], 'text': []}
        native = {'objects': [{'bbox': [0, 0, 100, 100], 'score': 0.5, 'mask': [[0, 0, 100, 100]]}], 'text': []}
        pair = compare_native(proposals, native)['groups']['objects']['pairs'][0]
        self.assertEqual(pair['mask_status'], 'INVALID_NATIVE_POLYGON')
        self.assertIsNone(pair['mask_iou'])
        self.assertEqual(pair['native_mask'], [[0, 0, 100, 100]])


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
