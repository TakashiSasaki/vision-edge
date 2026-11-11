"""Audit-limited loader for the pinned, external, third-party candidate reference.

No downloaded module imports or top-level program are executed. Only listed
forward definitions are compiled after a complete SHA-256 match. This reference
is not official and cannot establish original runtime parity by itself.
"""
import ast
import hashlib
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

OASIS_REVISION = '75bf8f3047e88a21c10d1f4175450656704a3c1e'
OASIS_SHA256 = 'ce84a5da7016d141269ecce6bcb4eabcfcaada74a2ff189856a4e94c3913ff62'
FUNCTIONS = {'zc_norm', 'rms_unit', '_pad', 'conv3', 'dwconv', 'pw', 'cnorm', 'walsh_sequency', 'band', 'rope_half'}
METHODS = {'letterbox', 'valid_mask', 'hconvglu', 'encode', 'body_block', 'top_block', 'res_block', 'neck', 'detect', 'head', 'masks', 'to_1000'}


def load_reference(path, weights, padding):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != OASIS_SHA256:
        raise ValueError('external candidate source hash mismatch')
    if padding not in ('symmetric', 'same'):
        raise ValueError('unknown padding')
    tree = ast.parse(raw)
    definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in FUNCTIONS]
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Mirage']
    if len(definitions) != len(FUNCTIONS) or len(classes) != 1:
        raise ValueError('pinned source definitions absent')
    cls = classes[0]
    cls.body = [m for m in cls.body if isinstance(m, ast.FunctionDef) and m.name in METHODS]
    if {m.name for m in cls.body} != METHODS:
        raise ValueError('pinned reference methods absent')
    definitions.append(cls)
    options = {'pad_mode': 'sym' if padding == 'symmetric' else 'same', 'canvas_fill': 0.0,
               'rope2d_rows_first': True, 'rope2d_layout': 'pairs16', 'walsh': 'sequency',
               'level_gate': 'raw', 'resample': Image.Resampling.BILINEAR}
    namespace = {'np': np, 'torch': torch, 'F': F, 'Image': Image, 'math': math,
                 'EPS': 1e-6, 'D_MODEL': 512, 'CANVAS': 512, 'N_OBJ': 64, 'N_TXT': 32, 'OPTS': options}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), 'exec'), namespace)
    reference = namespace['Mirage'].__new__(namespace['Mirage'])
    reference.p = weights
    reference.W = namespace['walsh_sequency'](512)
    return reference, namespace


def capture(reference, image, namespace):
    """Observe candidate intermediates without replacing its neural computations."""
    outputs = {}
    original_cnorm = namespace['cnorm']
    original_body = reference.body_block
    original_top = reference.top_block
    original_mixer = reference.hconvglu
    stem_scales = {id(reference.p[f'stem/norm_{i}/scale']): i for i in range(5)}
    def cnorm(x, scale):
        result = original_cnorm(x, scale)
        if id(scale) in stem_scales:
            outputs[f'stem_{stem_scales[id(scale)]}'] = result
        return result
    def body(x, i, mask):
        result = original_body(x, i, mask)
        outputs[f'body_{i}'] = result
        return result
    def top(z, i, *args):
        result = original_top(z, i, *args)
        outputs[f'top_{i}'] = result
        return result
    def mixer(x, prefix):
        outputs[prefix.split('/')[0] + '_hconv_input'] = x
        return original_mixer(x, prefix)
    namespace['cnorm'] = cnorm
    reference.body_block, reference.top_block, reference.hconvglu = body, top, mixer
    try:
        with torch.inference_mode():
            enc = reference.encode(image)
            outputs['memory'], outputs['dense_features'] = enc['E'], enc['D']
            raw = torch.einsum('bchw,cd->bhwd', enc['D'], reference.p['dense/out/kernel']) + reference.p['dense/bias']
            outputs['detections'] = raw
            up = F.interpolate(enc['D'], scale_factor=2, mode='nearest')
            edge = namespace['pw'](enc['S4'], reference.p['dense/edge/kernel'])
            outputs['prototypes'] = reference.head(enc['ms'][4] * (up + edge), 'proto', enc['ms'][4])
            outputs['coefficients'] = reference.head(enc['D'], 'coef', enc['ms'][8])
            # Stem output in encode is masked after cnorm; capture that same checkpoint.
            for i in range(5):
                outputs[f'stem_{i}'] = outputs[f'stem_{i}'] * enc['ms'][2 ** (i + 1)]
        return outputs, enc
    finally:
        namespace['cnorm'] = original_cnorm
        reference.body_block, reference.top_block, reference.hconvglu = original_body, original_top, original_mixer
