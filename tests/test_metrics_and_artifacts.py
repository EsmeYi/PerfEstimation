"""Regression checks for statistical conventions and safe artifact installation."""
import gzip
import io
import math
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import download_artifacts as artifacts
from metrics import aggregate, ranks, score

class MetricTests(unittest.TestCase):
    def test_tied_ranks(self):
        self.assertEqual(ranks([4, 1, 4, 2]), [2.5, 0, 2.5, 1])

    def test_perfect_prediction_and_monotone_ranking(self):
        rows = [{'ipc_llvm_mca': x, 'probe_log1p': math.log1p(x)} for x in [1, 2, 2, 8]]
        result = score(rows)
        self.assertAlmostEqual(result['r2_log1p'], 1)
        self.assertAlmostEqual(result['mae_ipc'], 0)
        self.assertAlmostEqual(result['spearman'], 1)
        for row in rows:
            row['probe_log1p'] += 1
        self.assertAlmostEqual(score(rows)['spearman'], 1)
        self.assertLess(score(rows)['r2_log1p'], 1)

    def test_seed_sd_is_sample_sd(self):
        self.assertEqual(aggregate([{'x': 1}, {'x': 2}, {'x': 3}])['x'], {'mean': 2, 'sample_sd': 1})

    def test_invalid_and_constant_predictions(self):
        with self.assertRaises(ValueError):
            score([{'ipc_llvm_mca': 1, 'probe_log1p': float('nan')}] * 2)
        result = score([{'ipc_llvm_mca': 1, 'probe_log1p': 0}] * 2)
        self.assertIsNone(result['r2_log1p'])
        self.assertIsNone(result['spearman'])

class ArtifactTests(unittest.TestCase):
    def test_reject_archive_traversal_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset = root / 'bad.tar.gz'
            with tarfile.open(asset, 'w:gz') as archive:
                info = tarfile.TarInfo('../escape')
                info.size = 4
                archive.addfile(info, io.BytesIO(b'evil'))
            with patch.object(artifacts, 'P', root), self.assertRaises(ValueError):
                artifacts.install({'group': 'data'}, asset)
            self.assertFalse((root.parent / 'escape').exists())

    def test_valid_checkpoint_and_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset = root / 'model.gz'
            with gzip.open(asset, 'wb') as stream:
                stream.write(b'valid saved tensor bytes')
            import hashlib
            item = {'group': 'compact', 'path': 'checkpoints/model.pt', 'bytes': 24,
                    'sha256': hashlib.sha256(b'valid saved tensor bytes').hexdigest()}
            with patch.object(artifacts, 'P', root):
                artifacts.install(item, asset)
                self.assertEqual((root / item['path']).read_bytes(), b'valid saved tensor bytes')
                item['path'] = 'checkpoints/corrupt.pt'
                item['sha256'] = '0' * 64
                with self.assertRaises(ValueError):
                    artifacts.install(item, asset)
                self.assertFalse((root / item['path']).exists())

    def test_reject_checkpoint_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(artifacts, 'P', root), self.assertRaises(ValueError):
                artifacts.install({'group': 'compact', 'path': '../escape'}, root / 'unused')

if __name__ == '__main__':
    unittest.main()
