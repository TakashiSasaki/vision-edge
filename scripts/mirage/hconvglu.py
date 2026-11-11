"""Research-only Mirage HConvGLU exporter. No image encoder or detector is exported.

Specification: https://cactuscompute.com/blog/hconvglu
Row-vector convention: u=x W.T, gate=u B_g.T+b, y=h B_o.T W.
The reference uses a butterfly and sliding windows; ONNX uses explicit matrices.
"""
import math

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import torch
import torch.nn.functional as F

DIM, TAPS = 512, 65
PREFIXES = tuple(f'{kind}_{i}/hconv' for kind, count in [('body', 16), ('top', 12)] for i in range(count))
SUFFIXES = ('gate_conv', 'gate_bias', 'value_conv', 'out_conv')
STAGES = ('walsh', 'gate_pre', 'gate', 'value', 'product', 'out_band', 'output')


def require_weights(weights):
    """Validate the entire declared subset before any export; never fill missing tensors."""
    expected = {f'{p}/{s}' for p in PREFIXES for s in SUFFIXES}
    observed = {k for k in weights if '/hconv/' in k}
    if observed != expected:
        raise ValueError(f'HConvGLU tensor mismatch: missing={sorted(expected-observed)}, unexpected={sorted(observed-expected)}')
    for name in sorted(expected):
        w = weights[name]
        shape = (DIM,) if name.endswith('/gate_bias') else (DIM, TAPS)
        if tuple(w.shape) != shape or w.dtype != torch.float32 or not torch.isfinite(w).all():
            raise ValueError(f'invalid FP32 HConvGLU tensor: {name}')
    return expected


def butterfly(x):
    """Official sequency butterfly, independently implemented from the matrix formula."""
    n = x.shape[-1]
    if n < 1 or n & (n - 1):
        raise ValueError('Walsh width must be a power of two')
    width = 1
    while width < n:
        blocks = x.reshape(*x.shape[:-1], n // (2 * width), 2, width)
        signs = 1 - 2 * (torch.arange(width) % 2)
        left, right = blocks[..., 0, :], blocks[..., 1, :] * signs
        x = torch.stack((left + right, left - right), dim=-1).reshape(x.shape)
        width *= 2
    return x / math.sqrt(n)


def band_reference(x, w):
    radius = w.shape[-1] // 2
    return (F.pad(x, (radius, radius)).unfold(-1, w.shape[-1], 1) * w).sum(-1)


def reference(x, weights, prefix):
    """Unfused FP32 PyTorch checkpoints, with zero-padded, output-indexed bands."""
    u = butterfly(x)
    a = band_reference(u, weights[prefix + '/gate_conv']) + weights[prefix + '/gate_bias']
    g = F.gelu(a, approximate='tanh')
    v = band_reference(u, weights[prefix + '/value_conv'])
    h = g * v
    o = band_reference(h, weights[prefix + '/out_conv'])
    return dict(zip(STAGES, (u, a, g, v, h, o, butterfly(o))))


def walsh_matrix(n=DIM):
    if n < 1 or n & (n - 1):
        raise ValueError('Walsh width must be a power of two')
    bits = n.bit_length() - 1
    rows = [int(f'{j ^ (j >> 1):0{bits}b}'[::-1], 2) for j in range(n)]
    return np.array([[(-1.0 if (j & k).bit_count() % 2 else 1.0) / math.sqrt(n)
                      for k in range(n)] for j in rows], dtype=np.float64)


def band_matrix(weights):
    n, taps = weights.shape
    if taps % 2 != 1:
        raise ValueError('band width must be odd')
    matrix = np.zeros((n, n), dtype=np.float64)
    for output in range(n):
        for tap in range(taps):
            source = output + tap - taps // 2
            if 0 <= source < n:
                matrix[output, source] = weights[output, tap]
    return matrix


def export(weights, prefix, folded=False, precision='float32'):
    """Build opset-17 ONNX. Float64 is a separate rounding-diagnosis experiment."""
    require_weights(weights)
    if prefix not in PREFIXES:
        raise ValueError('unknown block')
    if precision not in ('float32', 'float64'):
        raise ValueError('unsupported arithmetic precision')
    dtype = np.float32 if precision == 'float32' else np.float64
    proto = TensorProto.FLOAT if precision == 'float32' else TensorProto.DOUBLE
    w = walsh_matrix()
    bg, bv, bo = [band_matrix(weights[prefix + '/' + s].numpy())
                  for s in ('gate_conv', 'value_conv', 'out_conv')]
    arrays = {'bias': weights[prefix + '/gate_bias'].numpy(),
              'half': np.array(0.5), 'one': np.array(1.0),
              'cubic': np.array(0.044715), 'root': np.array(math.sqrt(2 / math.pi))}
    nodes = []
    def node(op, inputs, output):
        nodes.append(helper.make_node(op, inputs, [output], name=output))
    if folded:
        # Fold in float64, then round each final constant once to FP32.
        arrays.update(gate_map=w.T @ bg.T, value_map=w.T @ bv.T, out_map=bo.T @ w)
        source = 'x'
    else:
        arrays.update(W=w.T, gate_map=bg.T, value_map=bv.T, out_map=bo.T)
        node('MatMul', ['x', 'W'], 'walsh')
        source = 'walsh'
    node('MatMul', [source, 'gate_map'], 'gate_linear')
    node('Add', ['gate_linear', 'bias'], 'gate_pre')
    node('Mul', ['gate_pre', 'gate_pre'], 'square')
    node('Mul', ['square', 'gate_pre'], 'cube')
    node('Mul', ['cube', 'cubic'], 'scaled_cube')
    node('Add', ['gate_pre', 'scaled_cube'], 'gelu_inner')
    node('Mul', ['gelu_inner', 'root'], 'gelu_arg')
    node('Tanh', ['gelu_arg'], 'tanh')
    node('Add', ['one', 'tanh'], 'gelu_plus')
    node('Mul', ['gate_pre', 'half'], 'half_gate')
    node('Mul', ['half_gate', 'gelu_plus'], 'gate')
    node('MatMul', [source, 'value_map'], 'value')
    node('Mul', ['gate', 'value'], 'product')
    if folded:
        node('MatMul', ['product', 'out_map'], 'output')
        outputs = ('gate_pre', 'gate', 'value', 'product', 'output')
    else:
        node('MatMul', ['product', 'out_map'], 'out_band')
        node('MatMul', ['out_band', 'W'], 'output')
        outputs = STAGES
    info = lambda name: helper.make_tensor_value_info(name, proto, ['vectors', DIM])
    graph = helper.make_graph(nodes, 'partial-mirage-hconvglu', [info('x')], [info(n) for n in outputs],
                              [numpy_helper.from_array(np.asarray(v, dtype=dtype), k) for k, v in arrays.items()])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=9,
                              producer_name='vision-edge research HConvGLU exporter')
    helper.set_model_props(model, {'scope': 'isolated HConvGLU only; not an encoder or detector',
                                   'block': prefix, 'folded': str(folded),
                                   'arithmetic_precision': precision,
                                   'availability_gate': 'MIRAGE_ONNX_NOT_VALIDATED'})
    onnx.checker.check_model(model, full_check=True)
    return model


def fixtures():
    """Immutable, exact binary-fraction vectors; no random generator or image inference."""
    i = np.arange(17 * DIM, dtype=np.int64).reshape(17, DIM)
    j = np.arange(33 * DIM, dtype=np.int64).reshape(33, DIM)
    return {'zero': np.zeros((1, DIM), np.float32),
            'all_channel_impulses': np.eye(DIM, dtype=np.float32),
            'signed_ramp': ((np.arange(DIM, dtype=np.int64)[None] - 256) / 256).astype(np.float32),
            'bounded_integer_pattern': (((i * 73 + 19) % 1024 - 512) / 512).astype(np.float32),
            'diverse_integer_mix': ((((j * 73) ^ (j >> 3) ^ ((j >> 7) * 193)) % 2048 - 1024) / 1024).astype(np.float32),
            'stress_integer_pattern': (((i * 137 + 31) % 1024 - 512) / 64).astype(np.float32)}


def compare(expected, actual, atol=2e-4, rtol=2e-4):
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        raise ValueError('shape/dtype mismatch')
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError('nonfinite comparison')
    error = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
    limit = atol + rtol * np.abs(expected.astype(np.float64))
    return {'shape': list(expected.shape), 'dtype': str(actual.dtype),
            'max_absolute_error': float(error.max()),
            'max_relative_error': float((error / np.maximum(np.abs(expected), 1e-6)).max()),
            'rms_error': float(np.sqrt(np.mean(error ** 2))), 'atol': atol, 'rtol': rtol,
            'near_zero_denominator_floor': 1e-6,
            'failed_elements': int(np.count_nonzero(error > limit)),
            'status': 'PASS' if np.all(error <= limit) else 'FAIL'}
