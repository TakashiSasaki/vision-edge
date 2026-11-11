"""Unfused candidate preserving the pinned PyTorch reference's FP32 operation order.

This remains a third-party-reference candidate, not official native parity.
No band/Walsh folding, synthetic weights, or relaxed numerical criteria are used.
"""
import torch
from torch import nn
from torch.nn import functional as F

from encoder_candidate import EncoderCandidate, validate_weights
from hconvglu import walsh_matrix


class ReferenceOrderEncoder(EncoderCandidate):
    def __init__(self, weights, padding):
        # No folded matrices are used by this graph. Avoid computing or storing
        # them while retaining the same strict required-tensor validation.
        nn.Module.__init__(self)
        if padding not in ('symmetric', 'same'):
            raise ValueError('explicit known padding required')
        self.padding = padding
        self.keys = {}
        for i, name in enumerate(validate_weights(weights)):
            key=f'weight_{i}'
            self.keys[name]=key
            self.register_buffer(key,weights[name].clone())
        self.register_buffer('explicit_walsh', torch.tensor(walsh_matrix(), dtype=torch.float32))
        indices = torch.arange(512)[:, None] + torch.arange(65)[None, :]
        self.register_buffer('band_gather_indices', indices.reshape(-1))
        # Reciprocal-after-positive-power matches the candidate reference's
        # rounding. Negative powers are mathematically equal but not FP32 equal.
        rows, cols = torch.arange(256) // 16, torch.arange(256) % 16
        frequencies = 1 / (10000.0 ** (torch.arange(0, 16, 2).float() / 16))
        angle = torch.cat((rows[:, None] * frequencies, cols[:, None] * frequencies), -1)
        self.register_buffer('rope_cos',angle.cos())
        self.register_buffer('rope_sin',angle.sin())

    def band(self, x, weights):
        # Gather is a compact ONNX alternative to hundreds of per-channel
        # Slice nodes. Padding and tap orientation are unchanged from unfold.
        padded = F.pad(x, (32, 32))
        windows = padded.index_select(-1, self.band_gather_indices).reshape(*x.shape[:-1], 512, 65)
        return (windows * weights).sum(-1)

    def mixer(self, x, block):
        u = x @ self.explicit_walsh
        g = F.gelu(self.band(u, self.weight(block + '/hconv/gate_conv')) +
                   self.weight(block + '/hconv/gate_bias'), approximate='tanh')
        v = self.band(u, self.weight(block + '/hconv/value_conv'))
        return self.band(g * v, self.weight(block + '/hconv/out_conv')) @ self.explicit_walsh

    def pointwise(self, x, name):
        return (x.permute(0, 2, 3, 1) @ self.weight(name)).permute(0, 3, 1, 2)
