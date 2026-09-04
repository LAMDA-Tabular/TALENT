"""TALENT integration for Deep Tabular Representation Corrector (TRC)."""

from __future__ import annotations

import copy
import os
import os.path as osp
import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from TALENT.model.lib.data import Dataset
from TALENT.model.lib.tuning_metric import select_objective
from TALENT.model.methods.base import Method
from TALENT.model.utils import Averager


class TRCMethod(Method):
    """Train an FT-Transformer backbone, freeze it, then fit TRC.

    TRC is a post-hoc representation method rather than a conventional
    end-to-end architecture.  Consequently ``fit`` implements the two stages
    from Algorithm 1 instead of using :class:`Method`'s single training loop.
    """

    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)
        assert args.cat_policy == "indices"

    @staticmethod
    def _model_defaults(config):
        config = copy.deepcopy(config)
        config.setdefault("embedding_num", 10)
        config.setdefault("shift_estimator", True)
        config.setdefault("space_mapping", True)
        config.setdefault("loss_orth", True)
        config.setdefault("reg_weight", 0.1)
        config.setdefault("tau", 0.01)
        config.setdefault("optimal_loss_fraction", 0.2)
        config.setdefault("perturb_times", 3)
        config.setdefault("mask_keep_min", 0.1)
        config.setdefault("mask_keep_max", 0.3)
        backbone_defaults = {
            "token_bias": True,
            "n_layers": 3,
            "d_token": 192,
            "n_heads": 8,
            "d_ffn_factor": 4.0 / 3.0,
            "attention_dropout": 0.2,
            "ffn_dropout": 0.1,
            "residual_dropout": 0.0,
            "activation": "reglu",
            "prenormalization": False,
            "initialization": "kaiming",
            "kv_compression": None,
            "kv_compression_sharing": None,
        }
        backbone_config = config.setdefault("backbone", {})
        for key, value in backbone_defaults.items():
            backbone_config.setdefault(key, value)
        return config

    @staticmethod
    def _training_defaults(config):
        config = copy.deepcopy(config)
        config.setdefault("lr", 1e-4)
        config.setdefault("weight_decay", 1e-5)
        config.setdefault("backbone_lr", 1e-4)
        config.setdefault("backbone_weight_decay", 1e-5)
        config.setdefault("backbone_patience", 10)
        config.setdefault("trc_patience", 10)
        return config

    def construct_model(self, model_config=None):
        from TALENT.model.models.ftt import Transformer
        from TALENT.model.models.trc import TRC

        if model_config is None:
            model_config = self.args.config["model"]
        model_config = self._model_defaults(model_config)
        self.trc_config = model_config

        backbone_config = model_config["backbone"]
        backbone = Transformer(
            d_numerical=self.d_in,
            categories=self.categories,
            d_out=self.d_out,
            **backbone_config,
        )
        self.model = TRC(
            backbone=backbone,
            hidden_dim=backbone_config["d_token"],
            d_out=self.d_out,
            embedding_num=model_config["embedding_num"],
            shift_estimator=model_config["shift_estimator"],
            space_mapping=model_config["space_mapping"],
            loss_orth=model_config["loss_orth"],
        ).to(self.args.device)
        if self.args.use_float:
            self.model.float()
        else:
            self.model.double()

    def _split_features(
        self, X
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.N is not None and self.C is not None:
            return X[0], X[1]
        if self.C is not None:
            return None, X
        return X, None

    def _run_backbone_epoch(self, loader, optimizer=None):
        backbone = self.model.backbone
        backbone.train(optimizer is not None)
        total = Averager()
        context = torch.enable_grad() if optimizer is not None else torch.no_grad()
        with context:
            for X, y in loader:
                X_num, X_cat = self._split_features(X)
                loss = self.criterion(backbone(X_num, X_cat), y)
                if optimizer is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                total.add(loss.item())
        return total.item()

    def _fit_backbone(self, training_config):
        backbone = self.model.backbone
        for parameter in backbone.parameters():
            parameter.requires_grad_(True)
        optimizer = torch.optim.AdamW(
            backbone.parameters(),
            lr=training_config["backbone_lr"],
            weight_decay=training_config["backbone_weight_decay"],
        )

        best_loss = float("inf")
        best_state = None
        stale_epochs = 0
        patience = int(training_config["backbone_patience"])
        for epoch in range(max(1, int(self.args.max_epoch))):
            train_loss = self._run_backbone_epoch(self.train_loader, optimizer)
            val_loss = self._run_backbone_epoch(self.val_loader)
            print(
                f"Backbone epoch {epoch}: train loss={train_loss:.4f}, "
                f"val loss={val_loss:.4f}"
            )
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = copy.deepcopy(backbone.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break

        if best_state is None:
            raise RuntimeError("The TRC backbone did not complete an epoch")
        backbone.load_state_dict(best_state)
        self.model.freeze_backbone()

    def _per_sample_losses(self):
        losses = []
        backbone = self.model.backbone
        backbone.eval()
        with torch.no_grad():
            for X, y in self.val_loader:
                X_num, X_cat = self._split_features(X)
                prediction = backbone(X_num, X_cat)
                if self.is_regression:
                    batch_losses = (prediction.reshape(-1) - y.reshape(-1)).square()
                else:
                    batch_losses = F.cross_entropy(prediction, y, reduction="none")
                losses.append(batch_losses.detach())
        if not losses:
            raise ValueError("TRC requires a non-empty validation split")
        return torch.cat(losses)

    def _gradient_norm(self, dataset_index: int) -> float:
        X, y = self.val_loader.dataset[dataset_index]
        X_num, X_cat = self._split_features(X)
        X_num = None if X_num is None else X_num.unsqueeze(0)
        X_cat = None if X_cat is None else X_cat.unsqueeze(0)
        y = y.unsqueeze(0)

        backbone = self.model.backbone
        parameters = [parameter for parameter in backbone.parameters()]
        prediction = backbone(X_num, X_cat)
        loss = self.criterion(prediction, y)
        gradients = torch.autograd.grad(
            loss,
            parameters,
            allow_unused=True,
            retain_graph=False,
            create_graph=False,
        )
        nonempty = [
            gradient.detach().abs().reshape(-1)
            for gradient in gradients
            if gradient is not None
        ]
        if not nonempty:
            return float("inf")
        return torch.cat(nonempty).mean().item()

    def _find_optimal_subset(self):
        """Search validation samples with low loss, then low gradient norm."""
        losses = self._per_sample_losses()
        n_validation = len(losses)
        fraction = float(self.trc_config["optimal_loss_fraction"])
        if not 0.0 < fraction <= 1.0:
            raise ValueError("optimal_loss_fraction must be in (0, 1]")
        tau = float(self.trc_config["tau"])
        if not 0.0 < tau <= 1.0:
            raise ValueError("tau must be in (0, 1]")

        candidate_count = max(1, int(n_validation * fraction))
        candidates = torch.argsort(losses)[:candidate_count].cpu().tolist()

        backbone = self.model.backbone
        for parameter in backbone.parameters():
            parameter.requires_grad_(True)
        backbone.eval()
        gradient_norms = [self._gradient_norm(index) for index in candidates]
        self.model.freeze_backbone()

        selected_count = min(candidate_count, max(1, int(n_validation * tau)))
        ordering = np.argsort(np.asarray(gradient_norms))[:selected_count]
        selected_indices = [candidates[index] for index in ordering]
        subset = Subset(self.val_loader.dataset, selected_indices)
        return DataLoader(
            subset,
            batch_size=min(self.args.batch_size, len(subset)),
            shuffle=True,
            num_workers=0,
        )

    @staticmethod
    def _perturb_tensor(values, distribution, keep_probability):
        if values is None:
            return None
        batch_size, n_features = values.shape
        rows = torch.randint(
            distribution.shape[0],
            (batch_size, n_features),
            device=values.device,
        )
        columns = torch.arange(n_features, device=values.device).expand(batch_size, -1)
        noise = distribution[rows, columns]
        mask = (
            torch.rand(batch_size, n_features, device=values.device)
            < keep_probability
        )
        return torch.where(mask, values, noise)

    def _perturb(self, X_num, X_cat):
        keep_min = float(self.trc_config["mask_keep_min"])
        keep_max = float(self.trc_config["mask_keep_max"])
        if not 0.0 <= keep_min <= keep_max <= 1.0:
            raise ValueError(
                "mask keep probabilities must satisfy 0 <= min <= max <= 1"
            )
        batch_size = X_num.shape[0] if X_num is not None else X_cat.shape[0]
        keep_probability = torch.empty(
            batch_size, 1, device=self.args.device
        ).uniform_(keep_min, keep_max)
        train_dataset = self.train_loader.dataset
        return (
            self._perturb_tensor(X_num, train_dataset.X_num, keep_probability),
            self._perturb_tensor(X_cat, train_dataset.X_cat, keep_probability),
        )

    def _train_shift_estimator(self, optimal_loader, optimizer):
        if self.model.shift_estimator is None:
            return 0.0
        self.model.train()
        total = Averager()
        perturb_times = int(self.trc_config["perturb_times"])
        if perturb_times <= 0:
            raise ValueError("perturb_times must be positive")

        for X, _ in optimal_loader:
            X_num, X_cat = self._split_features(X)
            clean = self.model.encode_backbone(X_num, X_cat)
            loss = F.mse_loss(self.model.shift_estimator(clean), torch.zeros_like(clean))
            for _ in range(perturb_times):
                corrupted_num, corrupted_cat = self._perturb(X_num, X_cat)
                corrupted = self.model.encode_backbone(corrupted_num, corrupted_cat)
                target_shift = corrupted - clean
                loss = loss + F.mse_loss(
                    self.model.shift_estimator(corrupted), target_shift
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total.add(loss.item())
        return total.item()

    def _run_trc_epoch(self, loader, optimizer=None):
        self.model.train(optimizer is not None)
        total = Averager()
        context = torch.enable_grad() if optimizer is not None else torch.no_grad()
        with context:
            for X, y in loader:
                X_num, X_cat = self._split_features(X)
                loss = self.criterion(self.model(X_num, X_cat), y)
                if self.model.use_orthogonal_loss:
                    loss = (
                        loss
                        + float(self.trc_config["reg_weight"])
                        * self.model.orthogonality_loss()
                    )
                if optimizer is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                total.add(loss.item())
        return total.item()

    def _validation_metrics(self):
        logits, labels = [], []
        self.model.eval()
        with torch.no_grad():
            for X, y in self.val_loader:
                X_num, X_cat = self._split_features(X)
                logits.append(self.model(X_num, X_cat))
                labels.append(y)
        logits = torch.cat(logits)
        labels = torch.cat(labels)
        val_loss = self.criterion(logits, labels).item()
        metrics, names = self.metric(logits, labels, self.y_info)
        score, _ = select_objective(metrics, names, self.args, self.is_regression)
        return val_loss, score

    def _fit_trc(self, training_config):
        optimal_loader = (
            self._find_optimal_subset()
            if self.model.shift_estimator is not None
            else None
        )
        parameters = list(self.model.correction_parameters())
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=training_config["lr"],
            weight_decay=training_config["weight_decay"],
        )

        best_loss = float("inf")
        best_state = None
        stale_epochs = 0
        patience = int(training_config["trc_patience"])
        for epoch in range(max(1, int(self.args.max_epoch))):
            shift_loss = (
                self._train_shift_estimator(optimal_loader, self.optimizer)
                if optimal_loader is not None
                else 0.0
            )
            train_loss = self._run_trc_epoch(self.train_loader, self.optimizer)
            val_loss, score = self._validation_metrics()
            self.trlog["train_loss"].append(train_loss)
            print(
                f"TRC epoch {epoch}: shift loss={shift_loss:.4f}, "
                f"train loss={train_loss:.4f}, val loss={val_loss:.4f}"
            )

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                self.trlog["best_epoch"] = epoch
                self.trlog["best_res"] = score
                torch.save(
                    {"params": best_state},
                    osp.join(self.args.save_path, f"best-val-{self.args.seed}.pth"),
                )
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break

        if best_state is None:
            raise RuntimeError("TRC did not complete an epoch")
        self.model.load_state_dict(best_state)

    def fit(self, data, info, train=True, config=None):
        start = time.time()
        N, C, y = data
        self.D = Dataset(N, C, y, info)
        self.N, self.C, self.y = self.D.N, self.D.C, self.D.y
        self.is_binclass = self.D.is_binclass
        self.is_multiclass = self.D.is_multiclass
        self.is_regression = self.D.is_regression
        self.n_num_features = self.D.n_num_features
        self.n_cat_features = self.D.n_cat_features
        if config is not None:
            self.reset_stats_withconfig(config)

        self.data_format(is_train=True)
        self.construct_model()
        if not train:
            return

        os.makedirs(self.args.save_path, exist_ok=True)
        training_config = self._training_defaults(self.args.config["training"])
        self._fit_backbone(training_config)
        self._fit_trc(training_config)
        torch.save(
            {"params": self.model.state_dict()},
            osp.join(self.args.save_path, f"epoch-last-{self.args.seed}.pth"),
        )
        torch.save(self.trlog, osp.join(self.args.save_path, "trlog"))
        self.fit_time = time.time() - start
