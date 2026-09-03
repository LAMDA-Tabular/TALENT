import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from TALENT.api import build_args
from TALENT.model.method_registry import get_method_spec
from TALENT.model.models.trc import TRC


class _TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 6)
        self.head = nn.Linear(6, 2)

    def encode(self, x_num, x_cat=None):
        return torch.relu(self.encoder(x_num))

    def forward(self, x_num, x_cat=None):
        return self.head(self.encode(x_num, x_cat))


class TRCModelTest(unittest.TestCase):
    def test_forward_freezes_backbone_and_preserves_shape(self):
        model = TRC(
            backbone=_TinyBackbone(),
            hidden_dim=6,
            d_out=2,
            embedding_num=3,
        )
        model.train()
        output = model(torch.randn(5, 3))

        self.assertEqual(output.shape, (5, 2))
        self.assertFalse(model.backbone.training)
        self.assertTrue(all(not p.requires_grad for p in model.backbone.parameters()))
        self.assertGreaterEqual(model.orthogonality_loss().item(), 0.0)

    def test_ablation_without_space_mapping(self):
        model = TRC(
            backbone=_TinyBackbone(),
            hidden_dim=6,
            d_out=1,
            shift_estimator=False,
            space_mapping=False,
            loss_orth=False,
        )
        self.assertEqual(model(torch.randn(4, 3)).shape, (4,))
        self.assertEqual(model.orthogonality_loss().item(), 0.0)


class TRCIntegrationTest(unittest.TestCase):
    def test_registry_and_packaged_config(self):
        spec = get_method_spec("trc")
        self.assertEqual(spec.cat_policy, ("indices",))
        config_path = Path(__file__).parents[1] / "TALENT" / "configs" / "default" / "trc.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertIn("trc", config)
        self.assertEqual(config["trc"]["model"]["embedding_num"], 10)

    def test_tiny_cpu_fit_and_predict(self):
        rng = np.random.default_rng(7)
        sizes = {"train": 12, "val": 6, "test": 6}
        numerical = {
            split: rng.normal(size=(size, 2)).astype("float32")
            for split, size in sizes.items()
        }
        categorical = {
            split: np.asarray(
                [["a" if index % 2 else "b"] for index in range(size)],
                dtype=object,
            )
            for split, size in sizes.items()
        }
        labels = {
            split: np.asarray([index % 2 for index in range(size)], dtype="int64")
            for split, size in sizes.items()
        }
        train_val = (
            {key: numerical[key] for key in ("train", "val")},
            {key: categorical[key] for key in ("train", "val")},
            {key: labels[key] for key in ("train", "val")},
        )
        test = (
            {"test": numerical["test"]},
            {"test": categorical["test"]},
            {"test": labels["test"]},
        )
        info = {"task_type": "binclass", "n_num_features": 2, "n_cat_features": 1}

        config = {
            "model": {
                "embedding_num": 3,
                "tau": 0.2,
                "optimal_loss_fraction": 0.5,
                "perturb_times": 1,
                "backbone": {
                    "token_bias": True,
                    "n_layers": 1,
                    "d_token": 8,
                    "n_heads": 2,
                    "d_ffn_factor": 1.0,
                    "attention_dropout": 0.0,
                    "ffn_dropout": 0.0,
                    "residual_dropout": 0.0,
                    "activation": "reglu",
                    "prenormalization": False,
                    "initialization": "kaiming",
                    "kv_compression": None,
                    "kv_compression_sharing": None,
                },
            },
            "training": {
                "lr": 1e-3,
                "weight_decay": 0.0,
                "backbone_lr": 1e-3,
                "backbone_weight_decay": 0.0,
                "backbone_patience": 1,
                "trc_patience": 1,
            },
            "general": {},
        }
        with tempfile.TemporaryDirectory() as save_path:
            args = build_args(
                "trc",
                save_path=save_path,
                config=copy.deepcopy(config),
                max_epoch=1,
                batch_size=4,
                normalization="none",
                num_policy="none",
                use_float=True,
            )
            method = get_method_spec("trc").get_class()(args, is_regression=False)
            method.fit(train_val, info)
            loss, metrics, names, predictions = method.predict(test, info, "best-val")

        self.assertTrue(np.isfinite(loss))
        self.assertEqual(predictions.shape, (sizes["test"], 2))
        self.assertIn("Accuracy", names)
        self.assertTrue(np.isfinite(metrics[0]))


if __name__ == "__main__":
    unittest.main()
