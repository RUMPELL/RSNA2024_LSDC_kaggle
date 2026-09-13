"""Submission utilities (pipeline/submit.py).

pipeline/submit.py imports torch at module level, so these tests are skipped
when torch is not installed (as in the dependency-light CI). They exercise
only pure NumPy/pandas helpers; no model is built.
"""
import unittest

import numpy as np

try:
    import torch  # noqa: F401

    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestSubmissionHelpers(unittest.TestCase):
    def test_template_has_25_rows_and_competition_columns(self):
        from rsna_lumbar.pipeline.submit import SEVERITY_CLASSES, build_submission_template

        df = build_submission_template(4003253)
        self.assertEqual(len(df), 25)
        self.assertEqual(list(df.columns), ["row_id"] + SEVERITY_CLASSES)
        self.assertIn("4003253_spinal_canal_stenosis_l1", df["row_id"].tolist())
        self.assertIn("4003253_right_subarticular_stenosis_l5", df["row_id"].tolist())
        self.assertEqual(df["row_id"].nunique(), 25)

    def test_fallback_probabilities_are_a_distribution(self):
        from rsna_lumbar.pipeline.submit import fallback_probs

        p = fallback_probs()
        self.assertEqual(p.shape, (3,))
        self.assertAlmostEqual(float(p.sum()), 1.0, places=6)
        np.testing.assert_array_equal(p, fallback_probs())

    def test_softmax_rows_sum_to_one(self):
        from rsna_lumbar.pipeline.submit import softmax_np

        x = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
        s = softmax_np(x)
        np.testing.assert_allclose(s.sum(axis=1), [1.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(s[1], [1 / 3] * 3, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
