"""Small CPU checks of the mathematical and data contracts."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from palr_df.data import ClockData
from palr_df.evaluate import window_metrics
from palr_df.io import sha256, verify_manifest, safe_path
from palr_df.model import PaLRDF, from_config
from palr_df.results import checked_records, comparison_summary
from palr_df.spectra import compact_singular_values, energy_rank, loss_gram

torch.set_num_threads(2)


class SpectralTests(unittest.TestCase):
    def test_qr_matches_dense_svd(self):
        rng = np.random.default_rng(4)
        a, b = rng.normal(size=(19, 4)), rng.normal(size=(4, 7))
        compact = compact_singular_values(a, b)
        dense = np.linalg.svd(a @ b, compute_uv=False)
        np.testing.assert_allclose(compact, dense[:4], rtol=1e-12, atol=1e-12)

    def test_loss_metric_matches_explicit_operator(self):
        rng = np.random.default_rng(5)
        basis = rng.normal(size=(3, 11))
        horizon = basis.shape[1]
        c = np.tril(np.ones((horizon, horizon)))
        g = np.eye(horizon) / horizon + c.T @ c / horizon + .1 * np.ones((horizon, horizon))
        np.testing.assert_allclose(loss_gram(basis), basis @ g @ basis.T, atol=1e-12)

    def test_zero_and_energy_threshold(self):
        self.assertEqual(energy_rank(np.zeros(3)), 0)
        self.assertEqual(energy_rank([3, 1], .9), 1)
        self.assertEqual(energy_rank([3, 1], .95), 2)


class ModelTests(unittest.TestCase):
    def test_released_comparison_recipes(self):
        repo = Path(__file__).resolve().parents[1]
        expected = {"cnn": 415904, "lstm": 177600, "lstm_attention": 117792, "informer": 219984, "ssa_tcn": 973632, "mgamnet": 511361}
        for name, parameters in expected.items():
            config = json.loads((repo / f"configs/baselines/{name}_12h.json").read_text(encoding="utf-8"))
            model = from_config(config).eval()
            self.assertEqual(sum(p.numel() for p in model.parameters()), parameters)
            with torch.inference_mode():
                result = model(torch.full((1, 5760), .125))
            torch.testing.assert_close(result, torch.full((1, 1440), .125), atol=1e-7, rtol=0)

    def test_parameter_counts_and_constant_preservation(self):
        for horizon, count in ((1440, 33019), (2880, 37339)):
            model = PaLRDF(5760, horizon).eval()
            self.assertEqual(sum(p.numel() for p in model.parameters()), count)
            with torch.inference_mode():
                y = model(torch.full((2, 5760), .125))
            self.assertEqual(tuple(y.shape), (2, horizon))
            torch.testing.assert_close(y, torch.full_like(y, .125), atol=1e-7, rtol=0)

    def test_shape_validation(self):
        model = PaLRDF(5760, 1440)
        with self.assertRaises(ValueError):
            model(torch.zeros(2, 120))

    def test_trainable_decoder(self):
        model = PaLRDF(5760, 1440).train()
        prediction = model(torch.randn(2, 5760) * .01)
        loss = (prediction - .02).square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(model.coefficient_head.weight.grad).all())
        self.assertGreater(float(model.coefficient_head.weight.grad.abs().sum()), 0)


class DataTests(unittest.TestCase):
    def data(self, folder, omit_index=None):
        epochs = np.datetime64("2024-01-01", "ns") + np.arange(6 * 2880) * np.timedelta64(30, "s")
        biases = (np.arange(len(epochs), dtype=np.float64) ** 2)[:, None] * 1e-5
        if omit_index is not None:
            epochs, biases = np.delete(epochs, omit_index), np.delete(biases, omit_index, axis=0)
        path = folder / "data.npz"
        np.savez(path, epochs=epochs, satellites=np.asarray(["C29"]), bias_ns=biases, time_system=np.asarray("GPS"), sampling_seconds=np.asarray(30))
        return ClockData(path), biases

    def config(self):
        return {"satellites": ["C29"], "sampling_seconds": 30, "input_length": 5760, "prediction_length": 1440, "training_stride": 240, "validation_stride": 240, "train_target_intervals": [["2024-01-04", "2024-01-06"]]}

    def test_minimal_interval_origin_and_clock_anchor(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, biases = self.data(Path(tmp))
            samples = data.samples(self.config(), "C29", "train")
            # Minimal raw range starts Jan 2, so first target is Jan 4 00:00:30.
            self.assertEqual(samples.target_start[0], np.datetime64("2024-01-04T00:00:30"))
            np.testing.assert_array_equal(samples.x_difference_ns[0], np.diff(biases[2880:8641, 0]).astype(np.float32))
            self.assertEqual(samples.anchor_bias_ns[0], biases[8640, 0])
            np.testing.assert_array_equal(samples.target_bias_ns[0], biases[8641:10081, 0])

    def test_gap_and_overlapping_intervals_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, _ = self.data(Path(tmp), 3000)
            with self.assertRaises(ValueError):
                data.samples(self.config(), "C29", "train")
        with tempfile.TemporaryDirectory() as tmp:
            data, _ = self.data(Path(tmp))
            config = self.config()
            config["train_target_intervals"] *= 2
            with self.assertRaises(ValueError):
                data.samples(config, "C29", "train")


class EvaluationTests(unittest.TestCase):
    def test_window_average_is_not_pooled_rmse(self):
        prediction = np.asarray([[0, 0], [3, 3]])
        metrics = window_metrics(prediction, np.zeros_like(prediction))
        self.assertEqual(float(metrics["rmse_ns"].mean()), 1.5)
        self.assertEqual(metrics["eae_ns"].tolist(), [0, 3])
        self.assertEqual(metrics["re_ns"].tolist(), [0, 0])

    def test_macro_then_seed_sd_and_missing_satellite(self):
        frame = pd.DataFrame({"horizon": ["12h"] * 4, "model": ["PaLR-DF"] * 4, "satellite": ["A", "B", "A", "B"], "seed": [1, 1, 2, 2], "rmse_ns": [0., 2., 2., 4.]})
        checked_records(frame, ["horizon", "model"], ["A", "B"], [1, 2])
        _, summary = comparison_summary(frame)
        self.assertAlmostEqual(summary.iloc[0].macro_rmse_ns, 2)
        self.assertAlmostEqual(summary.iloc[0].sample_sd_ns, np.sqrt(2))
        with self.assertRaises(ValueError):
            checked_records(frame.iloc[:3], ["horizon", "model"], ["A", "B"], [1, 2])


class IntegrityTests(unittest.TestCase):
    def test_parent_segments_in_asset_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "nested").mkdir()
            asset = folder / "value.bin"
            asset.write_bytes(b"ok")
            self.assertEqual(safe_path(folder / "nested" / "..", "value.bin"), asset.resolve())
            with self.assertRaises(ValueError):
                safe_path(folder / "nested" / "..", "../outside.bin")

    def test_tampered_asset_and_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            asset = folder / "value.bin"
            asset.write_bytes(b"ok")
            manifest = folder / "manifest.json"
            manifest.write_text(json.dumps({"files": [{"path": "value.bin", "bytes": 2, "sha256": sha256(asset)}]}))
            self.assertEqual(verify_manifest(folder, manifest)["verified_files"], 1)
            asset.write_bytes(b"no")
            with self.assertRaises(ValueError):
                verify_manifest(folder, manifest)
            with self.assertRaises(ValueError):
                safe_path(folder, "../outside")


if __name__ == "__main__":
    unittest.main()
