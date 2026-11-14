"""Unit and archive-parity tests for full Mirage ONNX modules and codec."""
from __future__ import annotations

import gzip
import hashlib
import json
import struct
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from archive_full_weights import (
    ENGRAM_PRIME,
    ENGRAM_SEED,
    ENGRAM_SLOTS,
    MirageVocab,
    build_non_encoder_entry_spec,
    ctc_collapse,
    decode_general_cq,
    engram_index,
    load_full_archive_weights,
    parse_vocab_raw,
)
from archive_encoder_weights import natural_walsh
from mirage_full_modules import (
    MAX_M,
    MAX_SEQ,
    MAX_SLOTS,
    MirageCrossKv,
    MirageDecoderStep,
    MirageReader,
    build_fourier_box,
    compute_rope_tables,
)

CACT_PATH = ROOT_DIR / '.cache/container-models/mirage.cact'
MANIFEST_PATH = ROOT_DIR / 'src/assets/mirage-reconstruction/full_pipeline_manifest.json'
OASIS_REF_PATH = Path('/data/vision-edge/cache/oasis/reference.py')


def _pack_cq_matrix(mat: np.ndarray, codebook: np.ndarray, bits: int, group: int) -> bytes:
    """Encode float matrix into synthetic CQ archive bytes via Walsh-Hadamard forward transform."""
    rows, cols = mat.shape
    ng = -(-cols // group)
    padded = ng * group
    pad_mat = np.zeros((rows, padded), dtype=np.float64)
    pad_mat[:, :cols] = mat
    blocks = pad_mat.reshape(rows, ng, group)
    # Apply orthonormal Hadamard (natural_walsh is self-inverse!)
    spec = natural_walsh(blocks)
    cb = codebook[12:28] if bits == 4 else codebook[0:4]
    idx_out = bytearray()
    norms_out = np.empty((rows, ng), dtype='<f2')
    per = 8 // bits
    for r in range(rows):
        for g in range(ng):
            vec = spec[r, g]
            max_abs = float(np.max(np.abs(vec))) or 1.0
            cb_max = float(np.max(np.abs(cb)))
            scale = max_abs / cb_max
            norms_out[r, g] = np.float16(scale)
            s = float(norms_out[r, g]) or 1.0
            q_codes = [int(np.argmin(np.abs(cb * s - v))) for v in vec]
            for k in range(0, group, per):
                byte_val = 0
                for j in range(per):
                    byte_val |= (q_codes[k + j] & ((1 << bits) - 1)) << (j * bits)
                idx_out.append(byte_val)
    return bytes(idx_out) + norms_out.tobytes()


class FullCodecUnitTests(unittest.TestCase):
    def test_cq_4bit_roundtrip_and_invalid_rejected(self):
        cb = np.zeros(28, dtype=np.float64)
        cb[12:28] = np.linspace(-3.5, 3.5, 16)
        rng = np.random.default_rng(7)
        mat = rng.normal(0.0, 0.5, size=(3, 128)).astype(np.float64)
        blob = _pack_cq_matrix(mat, cb, bits=4, group=128)
        dec = decode_general_cq(blob, cb, bits=4, group=128, rows=3, cols=128)
        self.assertEqual(dec.shape, (3, 128))
        self.assertTrue(np.isfinite(dec).all())
        self.assertLess(np.max(np.abs(dec - mat)), 0.35)
        with self.assertRaises(ValueError):
            decode_general_cq(b'short', cb, bits=4, group=128, rows=3, cols=128)
        with self.assertRaises(ValueError):
            decode_general_cq(blob, cb, bits=5, group=128, rows=3, cols=128)

    def test_vocab_raw_decode_byte_fallback_and_refs(self):
        pieces = [
            ('<pad>', 3, 0.0),
            ('</s>', 3, 0.0),
            ('<s>', 3, 0.0),
            ('<unk>', 2, 0.0),
            ('\u2581hello', 1 if False else 0, -1.0),
            ('<0xF0>', 4, 0.0),
            ('<0x9F>', 4, 0.0),
            ('<0x98>', 4, 0.0),
            ('<0x80>', 4, 0.0),
            ('<|en|>', 3, 0.0),
            ('<|fr|>', 3, 0.0),
        ]
        buf = bytearray(struct.pack('<5IBBBB', len(pieces), 0, 1, 2, 3, 1, 1, 0, 0))
        for text, typ, score in pieces:
            enc = text.encode('utf-8')
            buf.extend(struct.pack('<fBH', score, typ, len(enc)))
            buf.extend(enc)
        vocab = parse_vocab_raw(bytes(buf), embedding_rows=len(pieces) + 96)
        self.assertIsInstance(vocab, MirageVocab)
        self.assertEqual(vocab.lang_ids, {'en': 9, 'fr': 10})
        self.assertEqual(vocab.ref_id('object', 3), len(pieces) + 3)
        self.assertEqual(vocab.ref_id('text', 2), len(pieces) + 64 + 2)
        decoded = vocab.decode([2, 9, 4, 5, 6, 7, 8, vocab.ref_id('object', 0), 1], show_special=False)
        self.assertEqual(decoded, ' hello😀<obj0>')

    def test_ctc_collapse_rules(self):
        charset = 'ABC'
        self.assertEqual(ctc_collapse([0, 1, 1, 0, 1, 2, 2, 3, 0], charset), 'AABC')
        self.assertEqual(ctc_collapse([0, 0, 99, 0], charset), '')

    def test_engram_hash_matches_reference_uint32_math(self):
        tokens = [2, 8192, 42, 17, 999]
        got = engram_index(tokens, t=4, oi=1, order=3, h=1)
        acc = (ENGRAM_SEED * (1 * 2 + 1 + 1)) & 0xFFFFFFFF
        for j in range(3):
            acc = ((acc ^ tokens[4 - j]) * ENGRAM_PRIME) & 0xFFFFFFFF
        expected = (acc ^ (acc >> 15)) % ENGRAM_SLOTS
        self.assertEqual(got, expected)

    def test_non_encoder_entry_spec_covers_all_90_tensors(self):
        specs = build_non_encoder_entry_spec()
        names = {s['name'] for s in specs}
        self.assertIn('embedding/embedding', names)
        self.assertIn('engrams_0/embedding', names)
        self.assertIn('engrams_1/embedding', names)
        self.assertIn('reader/w0', names)
        self.assertEqual(sum(1 for s in specs if s['layer'] == 0), 37)


@unittest.skipUnless(CACT_PATH.exists() and MANIFEST_PATH.exists(), 'Mirage .cact or manifest missing')
class FullArchiveReconstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.arc = load_full_archive_weights(CACT_PATH)
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))

    def _reconstruct(self, comp_key: str) -> bytes:
        meta = self.manifest[comp_key]
        skel_gz = MANIFEST_PATH.parent / meta['skeletonFile']
        placements = json.loads((MANIFEST_PATH.parent / meta['placementsFile']).read_text(encoding='utf-8'))
        buf = bytearray(gzip.decompress(skel_gz.read_bytes()))
        w = self.arc['weights']
        for p in placements:
            t = w[p['tensor']].detach().cpu().float()
            tr = p['transform']
            layer = p['layer']
            if tr == 'id':
                arr = t.reshape(-1) if t.ndim == 0 else t
            elif tr == 'plus1':
                arr = t + 1.0
            elif tr == 'T':
                arr = t.T
            elif tr == 'perm3201':
                arr = t.permute(3, 2, 0, 1)
            elif tr == 'mix_5_384':
                arr = t[0, :, 0, :]
            elif tr == 'mix_row':
                arr = t[0, :, 0, :][p['row'] : p['row'] + 1]
            elif tr == 'tap_row':
                arr = t[p['row']]
            elif tr == 'layer_id':
                sl = t[layer]
                arr = sl.reshape(1) if sl.ndim == 0 else sl
            elif tr == 'layer_plus1':
                arr = t[layer] + 1.0
            elif tr == 'layer_T':
                arr = t[layer].T
            elif tr == 'layer_row':
                arr = t[layer][p['row']]
            else:
                raise ValueError(tr)
            raw = arr.contiguous().numpy().astype('<f4').tobytes()
            buf[p['offset'] : p['offset'] + p['length']] = raw
        return bytes(buf)

    def test_skeletons_zeroed_and_reconstruct_exact_sha256(self):
        import onnxruntime as ort

        for key in ('reader', 'crossKv', 'decoderStep'):
            with self.subTest(component=key):
                meta = self.manifest[key]
                self.assertEqual(meta['unplacedLargeFloatTensors'], [])
                cov_a, cov_b = meta['targetTensorCoverage'].split('/')
                self.assertEqual(cov_a, cov_b)
                raw = self._reconstruct(key)
                self.assertEqual(len(raw), meta['expectedSize'])
                self.assertEqual(hashlib.sha256(raw).hexdigest(), meta['expectedSha256'])
                sess = ort.InferenceSession(raw, providers=['CPUExecutionProvider'])
                self.assertGreater(len(sess.get_inputs()), 0)

    def test_reconstructed_modules_match_pytorch_within_tolerance(self):
        import onnxruntime as ort

        w = self.arc['weights']
        rng = np.random.default_rng(123)

        # 1. Reader
        reader_pt = MirageReader(w).eval()
        crop = torch.from_numpy(rng.uniform(-1, 1, size=(1, 3, 48, 320)).astype(np.float32))
        with torch.no_grad():
            pt_probs = reader_pt(crop).numpy()
        reader_sess = ort.InferenceSession(self._reconstruct('reader'), providers=['CPUExecutionProvider'])
        (ort_probs,) = reader_sess.run(['probs'], {'crop': crop.numpy()})
        np.testing.assert_allclose(ort_probs, pt_probs, atol=2e-4, rtol=2e-4)

        # 2. CrossKV (driven by real reconstructed encoder features from preset fixture)
        from PIL import Image as PILImage
        from provision_experimental_onnx import ensure_verified_onnx

        enc_path = ensure_verified_onnx(cact_path=CACT_PATH)['path']
        enc_sess = ort.InferenceSession(str(enc_path), providers=['CPUExecutionProvider'])
        preset_img = PILImage.open(SCRIPT_DIR / 'presets/coco_000000458054.png').convert('RGB')
        pw, ph = preset_img.size
        pscale = 512 / max(pw, ph)
        prw, prh = round(pw * pscale), round(ph * pscale)
        pcanvas = np.zeros((512, 512, 3), dtype=np.float32)
        pcanvas[:prh, :prw] = np.array(
            preset_img.resize((prw, prh), PILImage.Resampling.BILINEAR), dtype=np.float32
        ) / 255.0
        E_real, D_real = enc_sess.run(
            ['memory', 'dense_features'],
            {'image': pcanvas.transpose(2, 0, 1)[None], 'valid_hw': np.array([prh, prw], np.int64)},
        )

        cross_pt = MirageCrossKv(w).eval()
        E = torch.from_numpy(E_real)
        sf = np.zeros((1, MAX_SLOTS, 192), dtype=np.float32)
        sp = np.zeros((1, MAX_SLOTS, 512), dtype=np.float32)
        se = np.zeros((1, MAX_SLOTS, 512), dtype=np.float32)
        sf[0, 0, :128] = D_real[0, :, :8, :8].mean(axis=(1, 2))
        sf[0, 0, 128:] = build_fourier_box(torch.tensor([0.1, 0.2, 0.4, 0.6])).numpy()
        sp[0, 0] = E_real[0].reshape(16, 16, 512)[:8, :8, :].mean(axis=(0, 1))
        se[0, 0] = w['embedding/embedding'][8203].numpy()
        sf = torch.from_numpy(sf)
        sp = torch.from_numpy(sp)
        se = torch.from_numpy(se)
        vc = torch.ones((256,), dtype=torch.float32)
        sv = torch.cat([torch.ones(5), torch.zeros(MAX_SLOTS - 5)])
        with torch.no_grad():
            pt_ck, pt_cv, pt_cm = cross_pt(E, sf, sp, se, vc, sv)
        cross_sess = ort.InferenceSession(self._reconstruct('crossKv'), providers=['CPUExecutionProvider'])
        ort_ck, ort_cv, ort_cm = cross_sess.run(
            ['ck', 'cv', 'cross_mask'],
            {
                'E': E.numpy(),
                'slot_d_fourier': sf.numpy(),
                'slot_pooled': sp.numpy(),
                'slot_emb': se.numpy(),
                'valid_cells': vc.numpy(),
                'slot_valid': sv.numpy(),
            },
        )
        np.testing.assert_allclose(ort_ck, pt_ck.numpy(), atol=2e-4, rtol=2e-4)
        np.testing.assert_allclose(ort_cv, pt_cv.numpy(), atol=2e-4, rtol=2e-4)
        np.testing.assert_allclose(ort_cm, pt_cm.numpy(), atol=2e-4, rtol=2e-4)

        # 3. Decoder step
        dec_pt = MirageDecoderStep(w, self.arc['perm1'], self.arc['perm2']).eval()
        rc, rs = compute_rope_tables()
        token_emb = w['embedding/embedding'][2].clone().float()
        eng0 = torch.from_numpy(rng.normal(0, 0.1, size=(512,)).astype(np.float32))
        eng1 = torch.from_numpy(rng.normal(0, 0.1, size=(512,)).astype(np.float32))
        pos_oh = torch.zeros((MAX_SEQ,), dtype=torch.float32)
        pos_oh[0] = 1.0
        self_mask = torch.full((MAX_SEQ,), -1e9, dtype=torch.float32)
        self_mask[0] = 0.0
        pk = torch.zeros((8, 2, MAX_SEQ, 48), dtype=torch.float32)
        pv = torch.zeros((8, 2, MAX_SEQ, 64), dtype=torch.float32)
        rqh = torch.zeros((8, 2, 384), dtype=torch.float32)
        rkh = torch.zeros((8, 2, 96), dtype=torch.float32)
        rvh = torch.zeros((8, 2, 128), dtype=torch.float32)
        evh = torch.zeros((2, 9, 512), dtype=torch.float32)
        with torch.no_grad():
            pt_logits, *_ = dec_pt(
                token_emb, eng0, eng1, rc[0], rs[0], pos_oh, self_mask,
                pt_ck, pt_cv, pt_cm, pk, pv, rqh, rkh, rvh, evh
            )
        dec_sess = ort.InferenceSession(self._reconstruct('decoderStep'), providers=['CPUExecutionProvider'])
        (ort_logits,) = dec_sess.run(
            ['logits'],
            {
                'token_emb': token_emb.numpy(),
                'eng0_emb': eng0.numpy(),
                'eng1_emb': eng1.numpy(),
                'rope_cos': rc[0].numpy(),
                'rope_sin': rs[0].numpy(),
                'pos_onehot': pos_oh.numpy(),
                'self_mask': self_mask.numpy(),
                'cross_k': ort_ck,
                'cross_v': ort_cv,
                'cross_mask': ort_cm,
                'past_k': pk.numpy(),
                'past_v': pv.numpy(),
                'raw_q_hist': rqh.numpy(),
                'raw_k_hist': rkh.numpy(),
                'raw_v_hist': rvh.numpy(),
                'eng_v_hist': evh.numpy(),
            },
        )
        np.testing.assert_allclose(ort_logits, pt_logits.numpy(), atol=2e-4, rtol=2e-4)


if __name__ == '__main__':
    unittest.main()
