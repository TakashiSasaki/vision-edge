"""Fail-closed Mirage checkpoint acquisition and Safetensors inventory (stdlib only)."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import tempfile
import urllib.request

REVISION = '76c1e9e53381aca8fc3843c636216c96ff220122'
ARTIFACTS = {
    'mirage.safetensors': (216370504, '6ba97c5c1bef661c9b0c80782135682b38b4312fb33e0b645c93c10604bb16cd'),
    'mirage.cact': (21611043, '4249a0f7515272ff76b9414bec17d7e7daf84b15b66c7fb4f17fce1e66f9f237'),
}
WIDTHS = {'BOOL': 1, 'U8': 1, 'I8': 1, 'I16': 2, 'U16': 2, 'F16': 2,
          'BF16': 2, 'I32': 4, 'U32': 4, 'F32': 4, 'I64': 8, 'U64': 8, 'F64': 8}


def unique_object(pairs):
    result = {}
    for k, v in pairs:
        if k in result:
            raise ValueError(f'duplicate JSON key: {k}')
        result[k] = v
    return result


def read_json(data):
    def invalid(value):
        raise ValueError(f'nonfinite JSON number: {value}')
    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f'nonfinite JSON number: {value}')
        return result
    return json.loads(data, object_pairs_hook=unique_object, parse_constant=invalid, parse_float=finite_float)


def identity(path):
    digest = hashlib.sha256()
    size = 0
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            size += len(chunk)
            digest.update(chunk)
    return {'size': size, 'sha256': digest.hexdigest()}


def verify(path, expected):
    actual = identity(path)
    if (actual['size'], actual['sha256']) != expected:
        raise ValueError(f'artifact identity mismatch: {Path(path).name}: {actual}')
    return actual


def inventory(path):
    """Reject malformed headers, unsupported dtypes, gaps, overlaps and trailing data."""
    path = Path(path)
    size = path.stat().st_size
    with path.open('rb') as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise ValueError('truncated header length')
        length = struct.unpack('<Q', prefix)[0]
        if not 2 <= length <= min(16 * 1024 * 1024, size - 8) or length % 8:
            raise ValueError('invalid or excessive header length')
        raw = f.read(length)
    if not raw.startswith(b'{'):
        raise ValueError('header must begin with an object')
    header = read_json(raw)
    if not isinstance(header, dict):
        raise ValueError('header must be an object')
    rows = []
    for name, entry in header.items():
        if name == '__metadata__':
            if not isinstance(entry, dict) or not all(isinstance(v, str) for v in entry.values()):
                raise ValueError('metadata values must be strings')
            continue
        if not isinstance(entry, dict) or set(entry) != {'dtype', 'shape', 'data_offsets'}:
            raise ValueError(f'invalid tensor entry: {name}')
        dtype, shape, offsets = entry['dtype'], entry['shape'], entry['data_offsets']
        if not isinstance(dtype, str) or dtype not in WIDTHS:
            raise ValueError(f'unsupported dtype: {name}')
        if not isinstance(shape, list) or not all(type(x) is int and x >= 0 for x in shape):
            raise ValueError(f'invalid shape: {name}')
        if not isinstance(offsets, list) or len(offsets) != 2 or not all(type(x) is int for x in offsets):
            raise ValueError(f'invalid offsets: {name}')
        start, end = offsets
        if not 0 <= start <= end <= size - 8 - length:
            raise ValueError(f'out-of-bounds offsets: {name}')
        if end - start != math.prod(shape) * WIDTHS[dtype]:
            raise ValueError(f'byte length does not match shape/dtype: {name}')
        rows.append({'name': name, 'dtype': dtype, 'shape': shape, 'data_offsets': offsets,
                     'file_offsets': [8 + length + start, 8 + length + end],
                     'mapping_status': 'UNMAPPED', 'reference_operation': None})
    cursor = 0
    for row in sorted(rows, key=lambda r: (*r['data_offsets'], r['name'])):
        start, end = row['data_offsets']
        if start != cursor:
            raise ValueError(f'gap or overlap: {row["name"]}')
        cursor = end
    if not rows or cursor != size - 8 - length:
        raise ValueError('empty inventory or unindexed data')
    return sorted(rows, key=lambda r: r['name'])


def acquire(directory, name):
    """Explicit network action only; no fallback host and no retry on denied acquisition."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / name
    expected = ARTIFACTS[name]
    if destination.exists():
        verify(destination, expected)
        return destination
    url = f'https://huggingface.co/Cactus-Compute/mirage/resolve/{REVISION}/{name}'
    fd, temp = tempfile.mkstemp(prefix=f'.{name}.', dir=directory)
    try:
        with os.fdopen(fd, 'wb') as out, urllib.request.urlopen(url, timeout=60) as response:
            if response.status != 200:
                raise ValueError(f'expected complete HTTP 200, got {response.status}')
            size = 0
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > expected[0]:
                    raise ValueError('download exceeds expected size')
                out.write(chunk)
        verify(temp, expected)
        # An existing destination is never overwritten, including concurrent acquisition.
        os.link(temp, destination)
    finally:
        Path(temp).unlink(missing_ok=True)
    return destination


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--cact', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--download-to', type=Path, help='explicitly acquire pinned Safetensors; never bypass a recorded denial')
    args = p.parse_args()
    if bool(args.checkpoint) == bool(args.download_to):
        p.error('provide exactly one of --checkpoint or --download-to')
    checkpoint = args.checkpoint or acquire(args.download_to, 'mirage.safetensors')
    # Verify all data before emitting an inventory. No model conversion or availability change.
    artifact = verify(checkpoint, ARTIFACTS['mirage.safetensors'])
    config = read_json(args.config.read_bytes())
    if not isinstance(config, dict):
        raise ValueError('config must be an object')
    report = {'schema_version': 1, 'revision': REVISION, 'artifact': artifact,
              'identity_source': 'task-pinned; live upstream metadata not independently verified',
              'config_identity': identity(args.config), 'config': config,
              'config_checkpoint_compatibility': 'NOT_VERIFIED',
              'tensors': inventory(checkpoint), 'conversion_status': 'NOT_IMPLEMENTED',
              'numerical_equivalence': 'NOT_RUN', 'availability_gate': 'MIRAGE_ONNX_NOT_VALIDATED'}
    if args.cact:
        report['native_artifact'] = verify(args.cact, ARTIFACTS['mirage.cact'])
    # Serialize before publication; errors never leave a partial success report.
    payload = json.dumps(report, indent=2, allow_nan=False) + '\n'
    fd, temp = tempfile.mkstemp(prefix='.inventory.', dir=args.output.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as out:
            out.write(payload)
        os.link(temp, args.output)  # exclusive, atomic; never overwrite input or old reports
    finally:
        Path(temp).unlink(missing_ok=True)


if __name__ == '__main__':
    main()
