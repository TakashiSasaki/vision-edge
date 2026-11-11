"""Synthetic format/integrity controls; these do not establish model equivalence."""
import hashlib
import io
import sys
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import verify_checkpoint as verifier


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'fixture.safetensors'

    def write(self, header, data=b'\0' * 8):
        raw = header.encode() if isinstance(header, str) else json.dumps(header).encode()
        raw += b' ' * (-len(raw) % 8)
        self.path.write_bytes(struct.pack('<Q', len(raw)) + raw + data)

    def entry(self, **changes):
        return {'dtype': 'F32', 'shape': [2], 'data_offsets': [0, 8], **changes}

    def test_inventory(self):
        self.write({'weight': self.entry(), '__metadata__': {'source': 'synthetic'}})
        row, = verifier.inventory(self.path)
        self.assertEqual(row['shape'], [2])
        self.assertEqual(row['mapping_status'], 'UNMAPPED')
        self.assertEqual(row['file_offsets'][1], self.path.stat().st_size)

    def test_scalar_and_empty_tensor(self):
        self.write({'empty': self.entry(shape=[0], data_offsets=[0, 0]),
                    'scalar': self.entry(shape=[], data_offsets=[0, 4])}, b'\0' * 4)
        self.assertEqual(len(verifier.inventory(self.path)), 2)

    def test_corruption_controls(self):
        cases = [self.entry(dtype='UNKNOWN'), self.entry(shape=[True]),
                 self.entry(shape=[-1]), self.entry(shape=[3]),
                 self.entry(data_offsets=[True, 8]), self.entry(data_offsets=[-1, 7]),
                 self.entry(data_offsets=[0, 12]), self.entry(extra=0)]
        for entry in cases:
            with self.subTest(entry=entry):
                self.write({'weight': entry})
                with self.assertRaises(ValueError): verifier.inventory(self.path)

    def test_overlap_and_gap(self):
        for offsets in [[0, 8], [12, 20]]:
            self.write({'a': self.entry(), 'b': self.entry(data_offsets=offsets)}, b'\0' * 20)
            with self.assertRaises(ValueError): verifier.inventory(self.path)

    def test_trailing_and_truncated_data(self):
        for length in [7, 9]:
            self.write({'a': self.entry()}, b'\0' * length)
            with self.assertRaises(ValueError): verifier.inventory(self.path)

    def test_duplicate_json_keys(self):
        self.write('{"a":{},"a":{}}')
        with self.assertRaisesRegex(ValueError, 'duplicate'): verifier.inventory(self.path)

    def test_oversized_header(self):
        self.path.write_bytes(struct.pack('<Q', 2**63))
        with self.assertRaises(ValueError): verifier.inventory(self.path)

    def test_identity_rejects_corruption(self):
        self.path.write_bytes(b'abc')
        expected = (3, hashlib.sha256(b'abc').hexdigest())
        verifier.verify(self.path, expected)
        self.path.write_bytes(b'abd')
        with self.assertRaises(ValueError): verifier.verify(self.path, expected)

    def test_nonfinite_json(self):
        for value in ['NaN', 'Infinity', '-Infinity', '1e400', '-1e400']:
            with self.subTest(value=value):
                with self.assertRaises(ValueError): verifier.read_json('{"x":' + value + '}')

    def test_existing_invalid_download_never_fetches(self):
        path = Path(self.temp.name) / 'mirage.safetensors'
        path.write_bytes(b'invalid')
        with patch.object(verifier.urllib.request, 'urlopen') as fetch:
            with self.assertRaises(ValueError): verifier.acquire(path.parent, path.name)
            fetch.assert_not_called()
        self.assertEqual(path.read_bytes(), b'invalid')

    def test_download_rejection_removes_temporary_file(self):
        with patch.object(verifier.urllib.request, 'urlopen', side_effect=OSError('denied')):
            with self.assertRaises(OSError): verifier.acquire(self.path.parent, 'mirage.safetensors')
        self.assertEqual(list(self.path.parent.iterdir()), [])

    def test_download_success_and_corruption(self):
        class Response(io.BytesIO):
            status = 200
        name = 'mirage.safetensors'
        expected = (3, hashlib.sha256(b'abc').hexdigest())
        for data in [b'ab', b'abd', b'abcd', b'abc']:
            with self.subTest(data=data), patch.dict(verifier.ARTIFACTS, {name: expected}):
                with patch.object(verifier.urllib.request, 'urlopen', return_value=Response(data)):
                    if data == b'abc':
                        target = verifier.acquire(self.path.parent, name)
                        self.assertEqual(target.read_bytes(), data)
                    else:
                        with self.assertRaises(ValueError): verifier.acquire(self.path.parent, name)
                        self.assertEqual(list(self.path.parent.iterdir()), [])

    def test_cli_output_is_fail_closed_and_exclusive(self):
        self.write({'weight': self.entry()})
        ident = verifier.identity(self.path)
        config = self.path.parent / 'config.json'
        output = self.path.parent / 'inventory.json'
        argv = ['verify', '--checkpoint', str(self.path), '--config', str(config), '--output', str(output)]
        with patch.dict(verifier.ARTIFACTS, {'mirage.safetensors': (ident['size'], ident['sha256'])}), patch.object(sys, 'argv', argv):
            config.write_text('{"x":1e400}')
            with self.assertRaises(ValueError): verifier.main()
            self.assertFalse(output.exists())
            config.write_text('{"synthetic":true}')
            verifier.main()
            original = output.read_bytes()
            report = json.loads(original)
            self.assertEqual(report['numerical_equivalence'], 'NOT_RUN')
            self.assertEqual(len(report['tensors']), 1)
            with self.assertRaises(FileExistsError): verifier.main()
            self.assertEqual(output.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
