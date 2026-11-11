"""Deterministic image inputs for candidate conversion research, not accuracy labels."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from capture_native_reference import image_fixtures

PHOTO_SHA256 = '327de5cf31c479b7f4a5ca73b8b757f534f2e31ff6c452348f72aee86fb5b49b'
PHOTO_URL = 'https://cactuscompute.com/mirage/coco-231049.jpg'


def fixtures(photo_path):
    if hashlib.sha256(Path(photo_path).read_bytes()).hexdigest() != PHOTO_SHA256:
        raise ValueError('photo fixture identity mismatch')
    cases = {name: Image.fromarray(pixels) for name, pixels in image_fixtures().items()}
    with Image.open(photo_path) as source:
        photo = source.convert('RGB')
    cases['photo_original_portrait'] = photo
    cases['photo_square_stretched'] = photo.resize((512, 512), Image.Resampling.BILINEAR)
    cases['photo_rotated_landscape'] = photo.transpose(Image.Transpose.ROTATE_90)
    manifest = json.loads((Path(__file__).parent / 'fixtures/encoder-images.json').read_text())
    if {name: image_identity(image) for name, image in cases.items()} != manifest:
        raise ValueError('immutable fixture set/bytes changed')
    return cases


def preprocess(image):
    """Candidate choice: Pillow bilinear, nearest-integer size, top-left, zero normalized fill."""
    image = image.convert('RGB')
    width, height = image.size
    ratio = 512 / max(width, height)
    w, h = max(1, round(width * ratio)), max(1, round(height * ratio))
    x = np.zeros((512, 512, 3), np.float32)
    x[:h, :w] = np.asarray(image.resize((w, h), Image.Resampling.BILINEAR), dtype=np.float32) / 127.5 - 1
    return x.transpose(2, 0, 1)[None].copy(), np.array([h, w], np.int64), {'W': width, 'H': height, 'w': w, 'h': h, 'r': ratio}


def image_identity(image):
    pixels = np.asarray(image.convert('RGB'))
    return {'shape': list(pixels.shape), 'rgb_sha256': hashlib.sha256(pixels.tobytes()).hexdigest()}
