"""Diagnostic reconstruction of the pinned archive's encoder/head weights.

Float entries are matched byte-for-byte to FP16/FP32 casts of the original tensors.
The 60 CQ entry positions are the pinned Oasis candidate mapping, independently
shape/format checked here, not an official native-forward implementation.
"""
import hashlib
import math
import struct

import numpy as np
import torch

from encoder_candidate import required_shapes, validate_weights
from verify_checkpoint import ARTIFACTS, verify


def natural_walsh(x):
    n = x.shape[-1]
    if n < 1 or n & (n - 1):
        raise ValueError('power of two required')
    width = 1
    while width < n:
        b = x.reshape(*x.shape[:-1], n // (2 * width), 2, width)
        a, c = b[..., 0, :], b[..., 1, :]
        x = np.stack((a + c, a - c), axis=-2).reshape(x.shape)
        width *= 2
    return x / math.sqrt(n)


def decode_cq(raw, shape, group, bits, codebook):
    rows, cols = math.prod(shape[:-1]), shape[-1]
    if group != 128 or bits != 2 or cols % group:
        raise ValueError('unsupported candidate encoder CQ format')
    count = rows * cols
    index_bytes = count * bits // 8
    norm_count = count // group
    if len(raw) != index_bytes + 2 * norm_count:
        raise ValueError('CQ byte count mismatch')
    packed = np.frombuffer(raw[:index_bytes], np.uint8)
    codes = ((packed[:, None] >> (np.arange(4) * 2)) & 3).reshape(rows, cols // group, group)
    norms = np.frombuffer(raw[index_bytes:], '<f2').astype(np.float64).reshape(rows, cols // group, 1)
    result = natural_walsh(codebook[:4][codes] * norms).reshape(shape)
    if not np.isfinite(result).all():
        raise ValueError('nonfinite CQ result')
    return result.astype(np.float32)


def load_archive_weights(path, source_weights):
    validate_weights(source_weights)
    verify(path, ARTIFACTS['mirage.cact'])
    with open(path, 'rb') as f:
        data = f.read()
    fields = struct.unpack_from('<48I', data)
    if fields[0] != 0x05E12A84 or fields[1] != 806 or fields[2] != 28:
        raise ValueError('unexpected pinned archive header')
    codebook = np.frombuffer(data, '<f4', count=28, offset=196).astype(np.float64)
    offset = 196 + 4 * fields[2]
    entries, hashes = [], {}
    for i in range(fields[1]):
        dtype, ndim = data[offset], data[offset + 1]
        if dtype not in (1, 2, 3, 4) or not 0 <= ndim <= 4:
            raise ValueError('invalid archive dimensionality')
        shape = struct.unpack_from('<4I', data, offset + 4)[:ndim]
        start, length = struct.unpack_from('<QQ', data, offset + 20)
        group, bits = struct.unpack_from('<II', data, offset + 36)
        if start < 196 + 112 + 44 * 806 or start + length > len(data):
            raise ValueError('archive entry out of bounds')
        entry = {'index': i, 'dtype': dtype, 'shape': list(shape), 'offset': start,
                 'bytes': length, 'group': group, 'bits': bits}
        entries.append(entry)
        if dtype in (1, 2):
            if length != (2 if dtype == 1 else 4) * math.prod(shape):
                raise ValueError('floating-point byte count mismatch')
            digest = hashlib.sha256(data[start:start+length]).hexdigest()
            hashes.setdefault((dtype, digest), []).append(entry)
        offset += 44
    weights, mapping = {}, []
    quantized = {f'top_{i}/self_attn/{part}_proj/kernel': 542 + 15 * i + j
                 for i in range(12) for j, part in enumerate(('q', 'k', 'v', 'gate', 'out'))}
    for name, shape in required_shapes().items():
        if name in quantized:
            entry = entries[quantized[name]]
            if entry['dtype'] != 3 or entry['shape'] != list(shape[::-1]):
                raise ValueError(f'CQ entry shape/type mismatch: {name}')
            raw = data[entry['offset']:entry['offset']+entry['bytes']]
            array = decode_cq(raw, tuple(entry['shape']), entry['group'], entry['bits'], codebook).T.copy()
            row = {'name': name, 'entry': entry, 'mapping_basis': 'Oasis candidate CQ index, checked shape and encoding'}
        else:
            dtype = 2 if name in ('dense/out/kernel', 'dense/bias', 'dense/protos/kernel', 'dense/coef/kernel') else 1
            numpy_dtype = '<f4' if dtype == 2 else '<f2'
            source_bytes = source_weights[name].numpy().astype(numpy_dtype).tobytes()
            matches = hashes.get((dtype, hashlib.sha256(source_bytes).hexdigest()), [])
            # One scalar coincides with a decoder gate. Its encoder position is
            # fixed by the 15-entry attention record and verified against bytes.
            if name == 'top_11/attn_gate':
                matches = [e for e in matches if e['index'] == 550 + 15 * 11]
            if len(matches) != 1:
                raise ValueError(f'nonunique or absent original floating-point byte match: {name}')
            entry = matches[0]
            raw = data[entry['offset']:entry['offset']+entry['bytes']]
            if raw != source_bytes:
                raise ValueError('FP16 bytes mismatch')
            array = np.frombuffer(raw, numpy_dtype).astype(np.float32).reshape(shape)
            row = {'name': name, 'entry': entry, 'mapping_basis': 'exact original floating-point byte identity, row-major reshape'}
        weights[name] = torch.from_numpy(array)
        mapping.append(row)
    validate_weights(weights)
    return weights, mapping
