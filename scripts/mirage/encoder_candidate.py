"""Candidate Mirage encoder/dense heads from official equations; native parity unproven.

The padding mode is explicit. Image resizing and RoPE layout remain candidate
choices, never implicit production defaults. No language/reader output is invented.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from hconvglu import band_matrix, walsh_matrix

EPS = 1e-6


def required_shapes():
    shapes = {'stem/w0': (3, 3, 3, 32)}
    widths = (32, 64, 128, 256, 512)
    for i, width in enumerate(widths):
        shapes[f'stem/norm_{i}/scale'] = (width,)
        if i:
            shapes[f'stem/dw_{i}'] = (3, 3, 1, widths[i-1])
            shapes[f'stem/pw_{i}/kernel'] = (widths[i-1], width)
    for kind, count in [('body', 16), ('top', 12)]:
        for i in range(count):
            p = f'{kind}_{i}'
            for part in ('gate_conv', 'value_conv', 'out_conv'):
                shapes[f'{p}/hconv/{part}'] = (512, 65)
            shapes[f'{p}/hconv/gate_bias'] = (512,)
            shapes[f'{p}/hconv_norm/scale'] = (512,)
            if kind == 'body':
                for part in ('dw_value', 'dw_gate', 'dw_down'):
                    shapes[f'{p}/mix/{part}'] = (3, 3, 1, 512)
                for part in ('b_value', 'b_gate'):
                    shapes[f'{p}/mix/{part}'] = (512,)
                for part in ('gamma', 'beta'):
                    shapes[f'{p}/grn/{part}'] = (512,)
                shapes[f'{p}/mix_norm/scale'] = (512,)
            else:
                for part in ('attn_norm', 'post_attn_norm'):
                    shapes[f'{p}/{part}/scale'] = (512,)
                shapes[f'{p}/attn_gate'] = (1,)
                for part in ('q_norm', 'k_norm'):
                    shapes[f'{p}/self_attn/{part}/scale'] = (32,)
                for part, shape in [('q', (512, 128)), ('k', (512, 32)), ('v', (512, 64)),
                                    ('gate', (512, 256)), ('out', (256, 512))]:
                    shapes[f'{p}/self_attn/{part}_proj/kernel'] = shape
    shapes.update({'registers': (4, 512), 'enc_norm/scale': (512,), 'level_gates': (12, 512)})
    for i in range(12):
        shapes[f'level_norm_{i}/scale'] = (512,)
    for part, shape in [('coarse', (512, 128)), ('mid', (256, 128)), ('fine', (128, 128)),
                        ('edge', (64, 128)), ('out', (128, 10)), ('protos', (128, 32)), ('coef', (128, 32))]:
        shapes[f'dense/{part}/kernel'] = shape
    shapes['dense/bias'] = (10,)
    for norm, dw, pw in [('neck_norm', 'neck_dw', 'neck_pw')] + [
            (f'block_norm_{i}', f'dw_{i}', f'pw_{i}') for i in range(3)] + [
            (f'{part}_norm_{i}', f'{part}_dw_{i}', f'{part}_pw_{i}') for part in ('proto', 'coef') for i in range(2)]:
        shapes[f'dense/{norm}/scale'] = (128,)
        shapes[f'dense/{dw}'] = (3, 3, 1, 128)
        shapes[f'dense/{pw}/kernel'] = (128, 128)
    for part in ('final_norm', 'proto_final_norm', 'coef_final_norm'):
        shapes[f'dense/{part}/scale'] = (128,)
    return shapes


def validate_weights(weights):
    required = required_shapes()
    for name, shape in required.items():
        if name not in weights:
            raise ValueError(f'missing learned tensor: {name}')
        x = weights[name]
        if tuple(x.shape) != shape or x.dtype != torch.float32 or not torch.isfinite(x).all():
            raise ValueError(f'invalid learned tensor: {name}, expected finite FP32 {shape}')
    scope = {name for name in weights if name.startswith(('stem/', 'body_', 'top_', 'dense/', 'level_norm_'))
             or name in ('registers', 'enc_norm/scale', 'level_gates')}
    if scope != required.keys():
        raise ValueError(f'unexpected scope tensors: {sorted(scope-required.keys())}')
    return required


class EncoderCandidate(nn.Module):
    """512 canvas -> raw detection/prototype/coefficient maps plus diagnostics."""
    def __init__(self, weights, padding):
        super().__init__()
        if padding not in ('symmetric', 'same'):
            raise ValueError('explicit padding must be symmetric or same')
        self.padding = padding
        self.keys = {}
        for i, name in enumerate(validate_weights(weights)):
            key = f'weight_{i}'
            self.keys[name] = key
            self.register_buffer(key, weights[name].clone())
        w = walsh_matrix()
        for kind, count in [('body', 16), ('top', 12)]:
            for i in range(count):
                prefix = f'{kind}_{i}/hconv'
                for part in ('gate', 'value', 'out'):
                    band = band_matrix(weights[f'{prefix}/{part}_conv'].numpy())
                    matrix = band.T @ w if part == 'out' else w.T @ band.T
                    self.register_buffer(f'{kind}_{i}_{part}_map', torch.from_numpy(matrix.astype(np.float32)))
        rows, cols = torch.arange(256) // 16, torch.arange(256) % 16
        freq = 10000.0 ** (-torch.arange(0, 16, 2).float() / 16)
        angle = torch.cat((rows[:, None] * freq, cols[:, None] * freq), -1)
        self.register_buffer('rope_cos', angle.cos())
        self.register_buffer('rope_sin', angle.sin())

    def weight(self, name):
        return getattr(self, self.keys[name])

    def norm(self, x, name, dim=-1):
        shape = [1] * x.dim()
        shape[dim] = -1
        return x * torch.rsqrt((x * x).mean(dim, keepdim=True) + EPS) * (1 + self.weight(name).reshape(shape))

    def conv(self, x, name, stride=1, depthwise=False):
        pads = (0, 1, 0, 1) if stride == 2 and self.padding == 'same' else (1, 1, 1, 1)
        kernel = self.weight(name).permute(3, 2, 0, 1)
        return F.conv2d(F.pad(x, pads), kernel, stride=stride, groups=x.shape[1] if depthwise else 1)

    def pointwise(self, x, name):
        # Convolution in this candidate vs NHWC matmul in the independent reference.
        kernel = self.weight(name).T[:, :, None, None]
        return F.conv2d(x, kernel)

    def mixer(self, x, block):
        gate = x @ getattr(self, f'{block}_gate_map') + self.weight(f'{block}/hconv/gate_bias')
        # Explicit tanh-GELU formula is also the ONNX graph, not a sigmoid substitute.
        gelu = 0.5 * gate * (1 + torch.tanh(math.sqrt(2 / math.pi) * (gate + 0.044715 * gate ** 3)))
        return (gelu * (x @ getattr(self, f'{block}_value_map'))) @ getattr(self, f'{block}_out_map')

    @staticmethod
    def mask(valid_hw, stride):
        positions = torch.arange(512 // stride) * stride
        return ((positions[:, None] < valid_hw[0]) & (positions[None, :] < valid_hw[1])).float()[None, None]

    def rope(self, x):
        a, b = x[:, :, :256, :16], x[:, :, :256, 16:]
        rotated = torch.cat((a * self.rope_cos - b * self.rope_sin,
                             b * self.rope_cos + a * self.rope_sin), -1)
        return torch.cat((rotated, x[:, :, 256:]), 2)

    def residual(self, x, norm, dw, pw, mask):
        y = F.silu(self.norm(x, 'dense/' + norm + '/scale', 1))
        return mask * (x + self.pointwise(self.conv(y, 'dense/' + dw, depthwise=True), 'dense/' + pw + '/kernel'))

    def head(self, x, part, mask):
        for i in range(2):
            x = self.residual(x, f'{part}_norm_{i}', f'{part}_dw_{i}', f'{part}_pw_{i}', mask)
        x = F.silu(self.norm(x, f'dense/{part}_final_norm/scale', 1))
        return self.pointwise(x, 'dense/' + ('protos' if part == 'proto' else 'coef') + '/kernel')

    def forward(self, image, valid_hw):
        outputs = {}
        masks = {s: self.mask(valid_hw, s) for s in (2, 4, 8, 16, 32)}
        x = masks[2] * self.norm(self.conv(image, 'stem/w0', 2), 'stem/norm_0/scale', 1)
        features = {2: x}
        outputs['stem_0'] = x
        for i in range(1, 5):
            x = self.conv(F.silu(x), f'stem/dw_{i}', 2, True)
            x = masks[2 ** (i + 1)] * self.norm(self.pointwise(x, f'stem/pw_{i}/kernel'), f'stem/norm_{i}/scale', 1)
            features[2 ** (i + 1)] = x
            outputs[f'stem_{i}'] = x
        s4, s8, s16 = [F.silu(features[s]) for s in (4, 8, 16)]
        x = F.silu(x)
        m = masks[32]
        for i in range(16):
            block = f'body_{i}'
            z = self.norm(x, block + '/mix_norm/scale', 1)
            v = self.conv(z, block + '/mix/dw_value', depthwise=True) + self.weight(block + '/mix/b_value')[None, :, None, None]
            g = self.conv(z, block + '/mix/dw_gate', depthwise=True) + self.weight(block + '/mix/b_gate')[None, :, None, None]
            t = (x + m * self.conv(v * F.silu(g), block + '/mix/dw_down', depthwise=True)).permute(0, 2, 3, 1)
            hinput = self.norm(t, block + '/hconv_norm/scale')
            outputs[block + '_hconv_input'] = hinput
            h = self.mixer(hinput, block)
            mm = m.permute(0, 2, 3, 1)
            r = torch.sqrt(((mm * h) ** 2).sum((1, 2), keepdim=True) + EPS)
            grn = h + self.weight(block + '/grn/gamma') * h * r / (r.mean(-1, keepdim=True) + EPS) + self.weight(block + '/grn/beta')
            x = (t + mm * grn).permute(0, 3, 1, 2)
            outputs[block] = x
        z = torch.cat((x.flatten(2).transpose(1, 2), self.weight('registers')[None]), 1)
        valid = torch.cat((m.flatten().bool(), torch.ones(4, dtype=torch.bool)))
        mv = valid[None, :, None].float()
        levels = [z]
        for i in range(12):
            block = f'top_{i}'
            n = self.norm(z, block + '/attn_norm/scale')
            pre = block + '/self_attn/'
            q = (n @ self.weight(pre + 'q_proj/kernel')).reshape(1, 260, 4, 32).transpose(1, 2)
            k = (n @ self.weight(pre + 'k_proj/kernel')).reshape(1, 260, 1, 32).transpose(1, 2)
            v = (n @ self.weight(pre + 'v_proj/kernel')).reshape(1, 260, 1, 64).transpose(1, 2)
            q, k = self.rope(self.norm(q, pre + 'q_norm/scale')), self.rope(self.norm(k, pre + 'k_norm/scale'))
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(32)
            scores = scores.masked_fill(~valid[None, None, None, :], float('-inf'))
            a = (scores.softmax(-1) @ v).transpose(1, 2).reshape(1, 260, 256)
            a = (a * torch.sigmoid(n @ self.weight(pre + 'gate_proj/kernel'))) @ self.weight(pre + 'out_proj/kernel')
            t = z + mv * torch.sigmoid(self.weight(block + '/attn_gate')) * self.norm(a, block + '/post_attn_norm/scale')
            hinput = self.norm(t, block + '/hconv_norm/scale')
            outputs[block + '_hconv_input'] = hinput
            z = t + mv * self.mixer(hinput, block)
            levels.append(z)
            outputs[block] = z
        e = self.norm(z[:, :256], 'enc_norm/scale')
        for i in range(12):
            e = e + self.weight('level_gates')[i] * self.norm(levels[i][:, :256], f'level_norm_{i}/scale')
        e = m.flatten()[None, :, None] * e
        outputs['memory'] = e
        up = lambda x: F.interpolate(x, scale_factor=2, mode='nearest')
        t = masks[16] * (up(self.pointwise(e.transpose(1, 2).reshape(1, 512, 16, 16), 'dense/coarse/kernel'))
                         + self.pointwise(s16, 'dense/mid/kernel'))
        t = self.residual(t, 'neck_norm', 'neck_dw', 'neck_pw', masks[16])
        d = masks[8] * (up(t) + self.pointwise(s8, 'dense/fine/kernel'))
        for i in range(3):
            d = self.residual(d, f'block_norm_{i}', f'dw_{i}', f'pw_{i}', masks[8])
        d = masks[8] * F.silu(self.norm(d, 'dense/final_norm/scale', 1))
        outputs['dense_features'] = d
        outputs['detections'] = self.pointwise(d, 'dense/out/kernel').permute(0, 2, 3, 1) + self.weight('dense/bias')
        outputs['prototypes'] = self.head(masks[4] * (up(d) + self.pointwise(s4, 'dense/edge/kernel')), 'proto', masks[4])
        outputs['coefficients'] = self.head(d, 'coef', masks[8])
        return tuple(outputs.values())


def output_names():
    return ([f'stem_{i}' for i in range(5)] +
            [name for kind, count in [('body', 16), ('top', 12)] for i in range(count)
             for name in (f'{kind}_{i}_hconv_input', f'{kind}_{i}')] +
            ['memory', 'dense_features', 'detections', 'prototypes', 'coefficients'])
