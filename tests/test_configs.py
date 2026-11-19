import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import unittest
import yaml

class TestConfigs(unittest.TestCase):
    def test_all_configs_valid_yaml(self):
        """Verifies every YAML configuration in configs/ parses without error."""
        config_dir = "configs"
        self.assertTrue(os.path.exists(config_dir), "configs directory must exist!")

        expected_configs = [
            "model_a.yaml",
            "model_b.yaml",
            "model_c.yaml",
            "model_d.yaml",
            "model_e.yaml",
            "datasets.yaml",
            "training.yaml",
            "evaluation.yaml"
        ]

        for fname in expected_configs:
            fpath = os.path.join(config_dir, fname)
            self.assertTrue(os.path.exists(fpath), f"Missing config file: {fpath}")
            with open(fpath, "r") as f:
                data = yaml.safe_load(f)
                self.assertIsInstance(data, dict, f"Config {fname} did not parse into a dictionary!")

if __name__ == "__main__":
    unittest.main()

