"""Explicit one-shot official Mirage acquisition, with live metadata and full integrity checks.

No retries, alternative hosts, runtime installation, or application-startup hooks.
Run only when model downloads are authorized in the current environment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

import verify_checkpoint as cp

CONFIG_IDENTITY = (1430, 'f7a3c37642086f954b1000fe80edb544a153c76fe68fb1b8173ddb97d465e915')
BASE = 'https://huggingface.co/Cactus-Compute/mirage'


def validate_metadata(metadata):
    if metadata.get('sha') != cp.REVISION or metadata.get('id') != 'Cactus-Compute/mirage':
        raise ValueError('unexpected upstream model/revision')
    entries = metadata['siblings']
    names = [s['rfilename'] for s in entries]
    if len(names) != len(set(names)):
        raise ValueError('duplicate artifact metadata')
    by_name = {s['rfilename']: s for s in entries}
    for name, (size, digest) in cp.ARTIFACTS.items():
        entry = by_name[name]
        if entry['size'] != size or entry['lfs']['size'] != size or entry['lfs']['sha256'] != digest:
            raise ValueError(f'upstream identity mismatch: {name}')
    if by_name['config.json']['size'] != CONFIG_IDENTITY[0]:
        raise ValueError('unexpected config size')
    return by_name


def fetch_small(url, limit):
    with urllib.request.urlopen(url, timeout=60) as response:
        if response.status != 200:
            raise ValueError(f'expected HTTP 200: {response.status}')
        data = response.read(limit + 1)
        if len(data) > limit:
            raise ValueError('response exceeds limit')
        return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True, type=Path, help='existing ignored cache or new directory')
    parser.add_argument('--evidence-dir', required=True, type=Path, help='new exclusive evidence directory')
    args = parser.parse_args()
    args.evidence_dir.mkdir(parents=True, exist_ok=False)
    metadata = fetch_small(f'https://huggingface.co/api/models/Cactus-Compute/mirage/revision/{cp.REVISION}?blobs=true', 1024 * 1024)
    (args.evidence_dir / 'hub-metadata.json').write_bytes(metadata)
    entries = validate_metadata(cp.read_json(metadata))
    args.directory.mkdir(parents=True, exist_ok=True)
    config = args.directory / 'config.json'
    if not config.exists():
        payload = fetch_small(f'{BASE}/resolve/{cp.REVISION}/config.json', CONFIG_IDENTITY[0])
        digest = hashlib.sha256(payload).hexdigest()
        git_blob = hashlib.sha1(f'blob {len(payload)}\0'.encode() + payload).hexdigest()
        if (len(payload), digest) != CONFIG_IDENTITY or git_blob != entries['config.json']['blobId']:
            raise ValueError('config identity mismatch')
        with config.open('xb') as out:
            out.write(payload)
    cp.verify(config, CONFIG_IDENTITY)
    artifacts = {name: cp.identity(cp.acquire(args.directory, name)) for name in cp.ARTIFACTS}
    result = {'revision': cp.REVISION, 'identity_source': 'live official Hub metadata and local full-file hashes',
              'artifacts': artifacts, 'config': cp.identity(config), 'status': 'PASS'}
    (args.evidence_dir / 'acquisition.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
