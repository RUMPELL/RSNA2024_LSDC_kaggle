"""Study-level GroupKFold: no study may appear in both train and validation."""
import unittest

import numpy as np
import pandas as pd

from rsna_lumbar.data.make_splits import add_group_kfold


def synthetic_frame(n_studies=40, rows_per_study=(3, 5, 7)):
    rows = []
    rng = np.random.RandomState(0)
    for s in range(n_studies):
        for _ in range(int(rng.choice(rows_per_study))):
            rows.append({"study_id": 1000 + s, "level": int(rng.randint(1, 6)), "severity": int(rng.randint(0, 3))})
    return pd.DataFrame(rows)


class TestGroupKFoldLeakageGuard(unittest.TestCase):
    def setUp(self):
        self.df = synthetic_frame()

    def test_no_study_is_split_across_train_and_validation(self):
        out = add_group_kfold(self.df, n_splits=5, group_col="study_id")
        for fold in range(5):
            val_studies = set(out.loc[out["fold"] == fold, "study_id"])
            train_studies = set(out.loc[out["fold"] != fold, "study_id"])
            self.assertEqual(val_studies & train_studies, set(), f"fold {fold} leaks studies")

    def test_every_row_is_assigned_exactly_one_fold(self):
        out = add_group_kfold(self.df, n_splits=5)
        self.assertFalse((out["fold"] == -1).any())
        self.assertEqual(sorted(out["fold"].unique().tolist()), [0, 1, 2, 3, 4])
        self.assertEqual(len(out), len(self.df))

    def test_all_rows_of_a_study_share_a_fold(self):
        out = add_group_kfold(self.df, n_splits=4)
        per_study = out.groupby("study_id")["fold"].nunique()
        self.assertTrue((per_study == 1).all())

    def test_deterministic(self):
        a = add_group_kfold(self.df, n_splits=5)["fold"].tolist()
        b = add_group_kfold(self.df, n_splits=5)["fold"].tolist()
        self.assertEqual(a, b)

    def test_input_frame_is_not_mutated(self):
        before = self.df.copy()
        add_group_kfold(self.df, n_splits=3)
        pd.testing.assert_frame_equal(self.df, before)
        self.assertNotIn("fold", self.df.columns)


if __name__ == "__main__":
    unittest.main()
