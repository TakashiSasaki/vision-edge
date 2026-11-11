"""Synthetic operator controls; run in the isolated research Python environment."""
import unittest

import numpy as np
import torch

import hconvglu as hc


class OperatorTests(unittest.TestCase):
    def test_diverse_fixture_has_distinct_rows(self):
        x = hc.fixtures()['diverse_integer_mix']
        self.assertEqual(len(np.unique(x, axis=0)), 33)

    def test_sequency_is_not_just_an_involution(self):
        x = torch.eye(512)
        w = hc.walsh_matrix()
        np.testing.assert_allclose(hc.butterfly(x).numpy(), w, atol=1e-7, rtol=0)
        np.testing.assert_allclose(w @ w.T, np.eye(512), atol=1e-12, rtol=0)
        signs = np.sign(w)
        np.testing.assert_array_equal(np.count_nonzero(signs[:, 1:] != signs[:, :-1], axis=1), np.arange(512))
        for n in (0, 3):
            with self.assertRaises(ValueError): hc.walsh_matrix(n)

    def test_asymmetric_output_indexed_bands_and_edges(self):
        # Asymmetry detects transposition, tap reversal, sharing and circular padding.
        taps = np.arange(24, dtype=np.float32).reshape(8, 3) / 8
        x = np.eye(8, dtype=np.float32)
        expected = np.zeros_like(x)
        for source in range(8):
            for output in range(8):
                tap = source - output + 1
                if 0 <= tap < 3:
                    expected[source, output] = taps[output, tap]
        np.testing.assert_array_equal(x @ hc.band_matrix(taps).T, expected)
        np.testing.assert_array_equal(hc.band_reference(torch.from_numpy(x), torch.from_numpy(taps)).numpy(), expected)
        self.assertEqual(expected[0, -1], 0)
        self.assertEqual(expected[-1, 0], 0)

    def test_identity_bands_recover_tanh_gelu_in_walsh_basis(self):
        weights = {}
        prefix = 'test'
        for name in ('gate_conv', 'value_conv', 'out_conv'):
            weights[prefix + '/' + name] = torch.ones((512, 1))
        weights[prefix + '/gate_bias'] = torch.zeros(512)
        x = torch.from_numpy(hc.fixtures()['bounded_integer_pattern'])
        w = torch.from_numpy(hc.walsh_matrix().astype(np.float32))
        u = x @ w.T
        expected = (torch.nn.functional.gelu(u, approximate='tanh') * u) @ w
        np.testing.assert_allclose(hc.reference(x, weights, prefix)['output'].numpy(), expected.numpy(), atol=2e-5, rtol=2e-5)

    def test_missing_unexpected_shape_dtype_and_nonfinite_weights_rejected(self):
        weights = {f'{p}/{s}': torch.zeros((512,) if s == 'gate_bias' else (512, 65))
                   for p in hc.PREFIXES for s in hc.SUFFIXES}
        self.assertEqual(len(hc.require_weights(weights)), 112)
        key = next(iter(weights))
        original = weights.pop(key)
        with self.assertRaises(ValueError): hc.require_weights(weights)
        weights[key] = original
        for bad in (original.double(), torch.zeros(1), torch.full_like(original, float('nan'))):
            weights[key] = bad
            with self.assertRaises(ValueError): hc.require_weights(weights)
        weights[key] = original
        weights['body_16/hconv/gate_bias'] = torch.zeros(512)
        with self.assertRaises(ValueError): hc.require_weights(weights)

    def test_comparison_reports_real_failure_and_near_zero_policy(self):
        a = np.array([0, 1], np.float32)
        self.assertEqual(hc.compare(a, a)['status'], 'PASS')
        result = hc.compare(a, np.array([1e-3, 1], np.float32))
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(result['failed_elements'], 1)
        self.assertAlmostEqual(result['max_relative_error'], 1000, places=3)
        with self.assertRaises(ValueError): hc.compare(a, np.array([np.nan, 1], np.float32))
        with self.assertRaises(ValueError): hc.compare(a, a.astype(np.float64))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
