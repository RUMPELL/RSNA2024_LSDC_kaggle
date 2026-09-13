"""utils/config.py and utils/seed.py."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from rsna_lumbar.utils.config import dump_yaml, load_yaml
from rsna_lumbar.utils.seed import set_seed


class TestYamlConfig(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.yaml"
            dump_yaml({"a": 1, "b": [1, 2], "c": {"d": "x"}}, str(p))
            self.assertEqual(load_yaml(str(p)), {"a": 1, "b": [1, 2], "c": {"d": "x"}})

    def test_environment_variables_are_expanded_recursively(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"RSNA_ROOT": "/data/rsna"}):
            p = Path(d) / "c.yaml"
            p.write_text("root: $RSNA_ROOT/train\nnested:\n  - ${RSNA_ROOT}/a\n  - plain\n", encoding="utf-8")
            cfg = load_yaml(str(p))
        self.assertEqual(cfg["root"], "/data/rsna/train")
        self.assertEqual(cfg["nested"], ["/data/rsna/a", "plain"])

    def test_example_paths_config_has_every_key_the_scripts_read(self):
        repo = Path(__file__).resolve().parents[1]
        cfg = load_yaml(str(repo / "configs" / "paths.example.yaml"))
        for key in ("train_csv", "train_label_coordinates_csv", "test_series_descriptions_csv",
                    "train_images_dir", "png_out_dir", "png_root", "models_root"):
            self.assertIn(key, cfg)


class TestSeed(unittest.TestCase):
    def test_numpy_stream_is_reproducible(self):
        set_seed(123)
        a = np.random.rand(5)
        set_seed(123)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(os.environ.get("PYTHONHASHSEED"), "123")


if __name__ == "__main__":
    unittest.main()
