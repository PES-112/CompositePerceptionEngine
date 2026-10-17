import gzip
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.perception_stack.depth_loader import decode_sanpo_depth, load_depth_map


def sanpo_file(depth: np.ndarray, header: bool = True) -> bytes:
    h, w = depth.shape
    values = np.concatenate([[h, w], depth.ravel()]) if header else depth.ravel()
    return gzip.compress(values.astype(np.float16).tobytes())


class SanpoDepthTests(unittest.TestCase):
    def test_header_gives_the_shape(self):
        """SANPO-Real depth_maps/: [720, 1280] then the image. Reading it as 960x960 scrambles it."""
        depth = np.tile(np.linspace(1.0, 30.0, 720, dtype=np.float32)[:, None], (1, 1280))
        depth[300:500, 100:300] = 1.5                           # a person close on the left
        raw = np.frombuffer(gzip.decompress(sanpo_file(depth)), dtype=np.float16)
        self.assertEqual(raw.size, 921_602)                     # = 960 * 960 + 2, the old trap
        out = decode_sanpo_depth(raw)
        self.assertEqual(out.shape, (720, 1280))
        self.assertAlmostEqual(float(np.median(out[300:500, 100:300])), 1.5, delta=0.01)
        self.assertAlmostEqual(float(out[0, 0]), 1.0, delta=0.01)   # the header isn't a pixel

    def test_load_depth_map_reads_the_header(self):
        depth = np.full((720, 1280), 4.0, dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "000000.float16.gz"
            path.write_bytes(sanpo_file(depth))
            out = load_depth_map(path)
        self.assertEqual(out.shape, (720, 1280))
        self.assertTrue(np.allclose(out, 4.0))

    def test_headerless_known_shape_still_loads(self):
        depth = np.full((720, 1280), 2.0, dtype=np.float32)
        raw = np.frombuffer(gzip.decompress(sanpo_file(depth, header=False)), dtype=np.float16)
        self.assertEqual(decode_sanpo_depth(raw).shape, (720, 1280))

    def test_unknown_size_raises(self):
        with self.assertRaises(ValueError):
            decode_sanpo_depth(np.zeros(1000, dtype=np.float16))


if __name__ == "__main__":
    unittest.main()
