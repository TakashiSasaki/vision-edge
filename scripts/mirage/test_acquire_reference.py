"""Offline metadata controls; never download real weights during ordinary CI."""
import copy
import unittest

import acquire_reference as acquire
import verify_checkpoint as cp


class MetadataTests(unittest.TestCase):
    def metadata(self):
        return {'id': 'Cactus-Compute/mirage', 'sha': cp.REVISION, 'siblings': [
            {'rfilename': name, 'size': size, 'lfs': {'size': size, 'sha256': digest}}
            for name, (size, digest) in cp.ARTIFACTS.items()
        ] + [{'rfilename': 'config.json', 'size': acquire.CONFIG_IDENTITY[0]}]}

    def test_live_identity_validation(self):
        self.assertEqual(len(acquire.validate_metadata(self.metadata())), 3)

    def test_reject_changed_revision_hash_size_and_duplicates(self):
        for mutation in ('revision', 'hash', 'size', 'duplicate', 'missing'):
            data = copy.deepcopy(self.metadata())
            if mutation == 'revision': data['sha'] = 'main'
            elif mutation == 'hash': data['siblings'][0]['lfs']['sha256'] = '0' * 64
            elif mutation == 'size': data['siblings'][0]['size'] -= 1
            elif mutation == 'duplicate': data['siblings'].append(data['siblings'][0])
            else: data['siblings'].pop(0)
            with self.subTest(mutation=mutation), self.assertRaises((ValueError, KeyError)):
                acquire.validate_metadata(data)


if __name__ == '__main__':
    unittest.main()
