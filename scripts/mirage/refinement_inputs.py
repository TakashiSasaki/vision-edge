"""Pinned aspect-ratio probes and explicit preprocessing hypotheses."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from encoder_fixtures import fixtures, image_identity

RULES = ('pixel', 'nearest32', 'ceil32')


def images(photo):
    result = fixtures(photo)
    original = result['photo_original_portrait']
    for short in (264, 272, 280, 321, 327, 400):
        portrait = original.resize((short, 512), Image.Resampling.BILINEAR)
        result[f'holdout_{short}_portrait'] = portrait
        result[f'holdout_{short}_landscape'] = portrait.transpose(Image.Transpose.ROTATE_90)
    manifest = json.loads((Path(__file__).parent / 'fixtures/refinement-images.json').read_text())
    if {name: image_identity(image) for name, image in result.items()} != manifest:
        raise ValueError('refinement fixture bytes differ from manifest')
    return result


def extent(width, height, rule):
    if width < 1 or height < 1 or rule not in RULES:
        raise ValueError('positive dimensions and explicit known rule required')
    scale = 512 / max(width, height)
    dims = [width * scale, height * scale]
    if rule == 'pixel':
        return tuple(max(1, round(d)) for d in dims)
    if rule == 'nearest32':
        return tuple(min(512, max(32, math.floor(d / 32 + 0.5) * 32)) for d in dims)
    return tuple(min(512, max(32, math.ceil(d / 32) * 32)) for d in dims)


def preprocess(image, rule):
    image = image.convert('RGB')
    width, height = image.size
    w, h = extent(width, height, rule)
    x = np.zeros((512, 512, 3), np.float32)
    x[:h, :w] = np.asarray(image.resize((w, h), Image.Resampling.BILINEAR), np.float32) / 127.5 - 1
    meta = {'W': width, 'H': height, 'w': w, 'h': h, 'r': 512/max(width, height),
            'coordinate_scale': [512/w, 512/h, 512/w, 512/h], 'resize_rule': rule}
    return x.transpose(2, 0, 1)[None].copy(), np.array([h, w], np.int64), meta
