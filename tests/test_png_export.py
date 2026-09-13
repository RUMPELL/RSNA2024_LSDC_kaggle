"""DICOM conversion, cropping rules, and 2.5D slice stacking (preprocess/png_export.py).

Uses synthetic pixel arrays only. Requires numpy, pydicom, and cv2 (for the PNG round trip).
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rsna_lumbar.preprocess.png_export import (
    CropRuleConfig,
    DicomToImageConfig,
    compute_crop_params,
    condition_to_dir,
    crop_image,
    dicom_array_to_image,
    export_crop_3ch_png,
    level_to_disc_index,
)
from tests.synthetic_dicom import read_dataset, write_synthetic_dicom


def _dataset_with(pixels, photometric="MONOCHROME2"):
    with tempfile.TemporaryDirectory() as d:
        p = write_synthetic_dicom(Path(d) / "1.dcm", pixels, photometric=photometric)
        return read_dataset(p)


class TestDicomArrayToImage(unittest.TestCase):
    def test_output_is_uint8_with_same_shape(self):
        ds = _dataset_with(np.arange(64, dtype=np.uint16).reshape(8, 8))
        out = dicom_array_to_image(ds, ds.pixel_array, DicomToImageConfig())
        self.assertEqual(out.shape, (8, 8))
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(int(out.min()), 0)
        self.assertLessEqual(int(out.max()), 255)

    def test_monochrome1_is_inverted(self):
        pixels = np.arange(64, dtype=np.uint16).reshape(8, 8)
        ds2 = _dataset_with(pixels, "MONOCHROME2")
        ds1 = _dataset_with(pixels, "MONOCHROME1")
        out2 = dicom_array_to_image(ds2, ds2.pixel_array, DicomToImageConfig())
        out1 = dicom_array_to_image(ds1, ds1.pixel_array, DicomToImageConfig())
        # brightest input pixel becomes darkest output under MONOCHROME1
        self.assertEqual(int(out2[-1, -1]), int(out2.max()))
        self.assertEqual(int(out1[-1, -1]), 0)
        self.assertEqual(int(out1[0, 0]), int(out1.max()))

    def test_monochrome1_fix_can_be_disabled(self):
        pixels = np.arange(64, dtype=np.uint16).reshape(8, 8)
        ds1 = _dataset_with(pixels, "MONOCHROME1")
        out = dicom_array_to_image(ds1, ds1.pixel_array, DicomToImageConfig(fix_monochrome=False))
        self.assertEqual(int(out[0, 0]), 0)

    def test_percentile_clipping_limits_outlier_influence(self):
        # 400 pixels: a ramp 0..399 plus one extreme outlier
        pixels = np.arange(400, dtype=np.uint16).reshape(20, 20)
        pixels[0, 0] = 60000
        ds = _dataset_with(pixels)
        clipped = dicom_array_to_image(ds, ds.pixel_array, DicomToImageConfig(clip_percentiles=(1.0, 99.0)))
        unclipped = dicom_array_to_image(ds, ds.pixel_array, DicomToImageConfig(clip_percentiles=(0.0, 100.0)))
        # without clipping the ramp is squashed into a few grey levels; with clipping it spans the range
        self.assertGreater(len(np.unique(clipped)), len(np.unique(unclipped)))
        self.assertGreaterEqual(int(clipped[19, 19]), 250)

    def test_lut_path_is_identity_when_no_lut_tags(self):
        pixels = np.arange(64, dtype=np.uint16).reshape(8, 8)
        ds = _dataset_with(pixels)
        with_lut = dicom_array_to_image(ds, ds.pixel_array, DicomToImageConfig(try_lut=True))
        without = dicom_array_to_image(ds, ds.pixel_array, DicomToImageConfig(try_lut=False))
        np.testing.assert_array_equal(with_lut, without)


class TestCropImage(unittest.TestCase):
    def setUp(self):
        self.img = np.arange(100 * 120, dtype=np.uint8).reshape(100, 120)

    def test_interior_crop_has_requested_size(self):
        c = crop_image(self.img, x=60, y=50, crop_size=20)
        self.assertEqual(c.shape, (20, 20))
        np.testing.assert_array_equal(c, self.img[40:60, 50:70])

    def test_boundary_is_clamped_not_padded(self):
        # documented notebook behaviour: crops at the edge shrink rather than pad
        c = crop_image(self.img, x=0, y=0, crop_size=20)
        self.assertEqual(c.shape, (10, 10))
        # far edge keeps half+1 pixels because the upper bound is inclusive of x
        c = crop_image(self.img, x=119, y=99, crop_size=20)
        self.assertEqual(c.shape, (11, 11))

    def test_offsets_shift_the_window(self):
        base = crop_image(self.img, x=60, y=50, crop_size=20)
        shifted = crop_image(self.img, x=60, y=50, crop_size=20, offset_x=5, offset_y=-3)
        np.testing.assert_array_equal(shifted, self.img[37:57, 55:75])
        self.assertFalse(np.array_equal(base, shifted))

    def test_odd_crop_size_uses_floor_half(self):
        c = crop_image(self.img, x=60, y=50, crop_size=21)
        self.assertEqual(c.shape, (20, 20))


class TestLevelAndConditionMapping(unittest.TestCase):
    def test_level_variants(self):
        for level, expected in [("l1_l2", 1), ("L5_S1", 5), ("disc3", 3), ("4", 4), (2, 2), ("L3/L4", 3)]:
            self.assertEqual(level_to_disc_index(level), expected, level)

    def test_unmappable_level_raises(self):
        with self.assertRaises(ValueError):
            level_to_disc_index("cervical")

    def test_condition_title_and_snake_case(self):
        self.assertEqual(condition_to_dir("Spinal Canal Stenosis"), "spinal_canal_stenosis")
        self.assertEqual(condition_to_dir("left_subarticular_stenosis"), "left_subarticular_stenosis")
        self.assertEqual(condition_to_dir("Right Neural Foraminal Narrowing"), "right_neural_foraminal_narrowing")

    def test_unknown_condition_raises(self):
        with self.assertRaises(ValueError):
            condition_to_dir("disc bulge")


class TestComputeCropParams(unittest.TestCase):
    def setUp(self):
        self.rules = CropRuleConfig()

    def test_axial_uses_axial_scale_and_minimums(self):
        size, ox, oy = compute_crop_params(512, 512, "Axial T2", 3, self.rules)
        # int(512*0.13)=66 >= 64 -> max(66, 112)
        self.assertEqual((size, ox, oy), (112, 0, 0))
        size, _, _ = compute_crop_params(400, 400, "Axial T2", 3, self.rules)
        # int(400*0.13)=52 < 64 -> max(52, 64)
        self.assertEqual(size, 64)

    def test_sagittal_t2_offsets(self):
        size, ox, oy = compute_crop_params(500, 400, "Sagittal T2/STIR", 3, self.rules)
        self.assertEqual(size, int(400 * 0.20))
        self.assertEqual(ox, int(400 * -0.05))
        self.assertEqual(oy, 0)
        _, _, oy5 = compute_crop_params(500, 400, "Sagittal T2/STIR", 5, self.rules)
        self.assertEqual(oy5, int(500 * 0.05))

    def test_sagittal_t1_has_no_offsets(self):
        size, ox, oy = compute_crop_params(500, 400, "Sagittal T1", 5, self.rules)
        self.assertEqual((ox, oy), (0, 0))
        self.assertEqual(size, 80)


class TestExport3chBoundaries(unittest.TestCase):
    """[prev, center, next] selection, including the first/last-slice fallbacks.

    Slice k has its top k rows bright, so after normalisation each channel's
    mean identifies which instance it came from.
    """

    N = 5  # instances 1..5

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.images = root / "train_images"
        self.out = root / "png"
        for k in range(1, self.N + 1):
            px = np.zeros((32, 32), dtype=np.uint16)
            px[:k, :] = 1000
            write_synthetic_dicom(self.images / "7" / "70" / f"{k}.dcm", px)
        self.addCleanup(self._tmp.cleanup)

    def _channel_slices(self, instance):
        import cv2

        # non_axial_scale=1.0 makes the crop cover the whole 32x32 slice
        p = export_crop_3ch_png(
            self.images, self.out, study_id=7, series_id=70, instance_number=instance,
            level="l3_l4", series_description="Sagittal T1", condition="Spinal Canal Stenosis",
            x=16, y=16, crop_rules=CropRuleConfig(non_axial_scale=1.0),
        )
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        self.assertEqual(img.shape, (32, 32, 3))
        # bright rows per channel == originating instance number
        return tuple(int(np.count_nonzero(img[:, 0, c] > 128)) for c in range(3))

    def test_interior_slice_stacks_prev_center_next(self):
        self.assertEqual(self._channel_slices(3), (2, 3, 4))

    def test_first_slice_is_promoted_to_instance_two(self):
        self.assertEqual(self._channel_slices(1), (1, 2, 3))

    def test_last_slice_falls_back_by_one(self):
        # instance 6 does not exist -> notebook fallback: center=inst-2, prev=inst-3, next=inst-1
        self.assertEqual(self._channel_slices(self.N), (2, 3, 4))

    def test_output_path_encodes_condition_and_disc(self):
        p = export_crop_3ch_png(
            self.images, self.out, 7, 70, 3, "l3_l4", "Sagittal T1", "Spinal Canal Stenosis", 16, 16,
        )
        self.assertEqual(p, self.out / "spinal_canal_stenosis" / "7_disc3.png")
        self.assertTrue(p.is_file())


if __name__ == "__main__":
    unittest.main()
