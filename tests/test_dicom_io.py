"""dicom/io.py and dicom/preprocess.py on synthetic inputs."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rsna_lumbar.dicom.io import get_image_position_x, read_dicom
from rsna_lumbar.dicom.preprocess import resize_and_pad
from tests.synthetic_dicom import write_synthetic_dicom


class TestReadDicom(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_minmax_to_uint8(self):
        p = write_synthetic_dicom(self.tmp / "a.dcm", np.arange(64, dtype=np.uint16).reshape(8, 8) * 10)
        img = read_dicom(p)
        self.assertEqual(img.dtype, np.uint8)
        self.assertEqual(int(img.min()), 0)
        self.assertGreaterEqual(int(img.max()), 254)

    def test_monochrome1_inversion(self):
        px = np.arange(64, dtype=np.uint16).reshape(8, 8)
        p = write_synthetic_dicom(self.tmp / "m1.dcm", px, photometric="MONOCHROME1")
        img = read_dicom(p)
        self.assertEqual(int(img[0, 0]), int(img.max()))
        self.assertEqual(int(img[-1, -1]), 0)
        img_nofix = read_dicom(p, fix_monochrome=False)
        self.assertEqual(int(img_nofix[0, 0]), 0)

    def test_constant_image_does_not_divide_by_zero(self):
        p = write_synthetic_dicom(self.tmp / "c.dcm", np.full((4, 4), 7, dtype=np.uint16))
        img = read_dicom(p)
        self.assertTrue(np.all(img == 0))

    def test_image_position_x(self):
        p = write_synthetic_dicom(self.tmp / "ipp.dcm", np.zeros((2, 2), dtype=np.uint16), image_position=(-12.5, 3.0, 4.0))
        self.assertEqual(get_image_position_x(p), -12.5)

    def test_image_position_missing_returns_none(self):
        p = write_synthetic_dicom(self.tmp / "noipp.dcm", np.zeros((2, 2), dtype=np.uint16))
        self.assertIsNone(get_image_position_x(p))
        self.assertIsNone(get_image_position_x(self.tmp / "does_not_exist.dcm"))


class TestResizeAndPad(unittest.TestCase):
    def test_portrait_image_pads_width(self):
        img = np.full((100, 50), 200, dtype=np.uint8)
        out, scale, left, top = resize_and_pad(img, (64, 64))
        self.assertEqual(out.shape, (64, 64))
        self.assertAlmostEqual(scale, 0.64)
        self.assertEqual((left, top), (16, 0))
        self.assertTrue(np.all(out[:, :16] == 0))
        self.assertTrue(np.all(out[:, 48:] == 0))
        self.assertTrue(np.all(out[:, 16:48] == 200))

    def test_landscape_image_pads_height(self):
        img = np.full((30, 90), 100, dtype=np.uint8)
        out, scale, left, top = resize_and_pad(img, (90, 90))
        self.assertEqual(out.shape, (90, 90))
        self.assertAlmostEqual(scale, 1.0)
        self.assertEqual((left, top), (0, 30))

    def test_coordinate_back_projection_is_consistent(self):
        # a single bright pixel in the source lands where scale/pad predict
        img = np.zeros((200, 100), dtype=np.uint8)
        img[150, 25] = 255
        out, scale, left, top = resize_and_pad(img, (128, 128))
        expected_y = int(150 * scale) + top
        expected_x = int(25 * scale) + left
        py, px = np.unravel_index(np.argmax(out), out.shape)
        self.assertLessEqual(abs(int(py) - expected_y), 1)
        self.assertLessEqual(abs(int(px) - expected_x), 1)
        # and the inverse mapping recovers the original within one pixel
        self.assertLessEqual(abs((px - left) / scale - 25), 1.5)
        self.assertLessEqual(abs((py - top) / scale - 150), 1.5)

    def test_three_channel_input_is_supported(self):
        img = np.zeros((40, 20, 3), dtype=np.uint8)
        out, _, _, _ = resize_and_pad(img, (32, 32))
        self.assertEqual(out.shape, (32, 32, 3))


if __name__ == "__main__":
    unittest.main()
