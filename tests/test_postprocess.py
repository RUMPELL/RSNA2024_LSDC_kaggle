"""Detection post-processing (postprocess/filtering.py, postprocess/lr_split.py)."""
import unittest

import pandas as pd

from rsna_lumbar.postprocess.filtering import fill_missing_classes, filter_top_confidence_per_class
from rsna_lumbar.postprocess.lr_split import add_lr_direction, infer_lr_direction


class TestTopConfidencePerClass(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "class_id": [0, 0, 1, 1, 1, 2],
                "confidence": [0.2, 0.9, 0.5, 0.7, 0.6, 0.3],
                "dicom_path": list("abcdef"),
            }
        )

    def test_keeps_single_best_per_class(self):
        out = filter_top_confidence_per_class(self.df, by=("class_id",), n=1)
        self.assertEqual(len(out), 3)
        best = out.set_index("class_id")["dicom_path"].to_dict()
        self.assertEqual(best, {0: "b", 1: "d", 2: "f"})

    def test_top_n_greater_than_one(self):
        out = filter_top_confidence_per_class(self.df, n=2)
        self.assertEqual(out[out["class_id"] == 1]["confidence"].tolist(), [0.7, 0.6])
        self.assertEqual(len(out[out["class_id"] == 2]), 1)

    def test_empty_input_returns_empty(self):
        empty = self.df.iloc[:0]
        out = filter_top_confidence_per_class(empty)
        self.assertTrue(out.empty)


class TestFillMissingClasses(unittest.TestCase):
    def test_inserts_rows_for_missing_levels(self):
        df = pd.DataFrame({"class_id": [0, 2], "confidence": [0.9, 0.8]})
        out = fill_missing_classes(df, required_classes=range(5), fallback_conf=0.0)
        self.assertEqual(sorted(out["class_id"].tolist()), [0, 1, 2, 3, 4])
        inserted = out[out["class_id"].isin([1, 3, 4])]
        self.assertTrue((inserted["confidence"] == 0.0).all())

    def test_complete_input_is_unchanged(self):
        df = pd.DataFrame({"class_id": [0, 1], "confidence": [0.5, 0.6]})
        out = fill_missing_classes(df, required_classes=[0, 1])
        pd.testing.assert_frame_equal(out, df)

    def test_empty_input_gets_all_classes(self):
        df = pd.DataFrame(columns=["class_id", "confidence"])
        out = fill_missing_classes(df, required_classes=[0, 1, 2], fallback_conf=0.1)
        self.assertEqual(len(out), 3)
        self.assertTrue((out["confidence"] == 0.1).all())


class TestLeftRight(unittest.TestCase):
    def test_sign_heuristic(self):
        self.assertEqual(infer_lr_direction(12.0), "Right")
        self.assertEqual(infer_lr_direction(-3.5), "Left")
        self.assertEqual(infer_lr_direction(0.0), "Left")  # documented: > 0 is Right
        self.assertIsNone(infer_lr_direction(None))

    def test_add_lr_direction_column(self):
        df = pd.DataFrame({"image_position_x": [5.0, -5.0]})
        out = add_lr_direction(df)
        self.assertEqual(out["LR_Direction"].tolist(), ["Right", "Left"])
        self.assertNotIn("LR_Direction", df.columns)

    def test_missing_position_in_a_float_column_currently_maps_to_left(self):
        # Documents present behaviour: pandas stores None as NaN in a float
        # column, and NaN > 0 is False, so a missing position is labelled
        # "Left" rather than None. Callers must filter missing positions first.
        df = pd.DataFrame({"image_position_x": [5.0, None]})
        out = add_lr_direction(df)
        self.assertEqual(out["LR_Direction"].tolist(), ["Right", "Left"])


if __name__ == "__main__":
    unittest.main()
