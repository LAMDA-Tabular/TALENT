"""Tabular Representation Corrector (TRC).

The implementation follows Algorithm 1 of "Deep Tabular Representation
Corrector".  TRC is deliberately kept independent of the TALENT training
loop: the method wrapper owns the two-stage optimisation, while this module
contains the reusable frozen-backbone correction network.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class TRC(nn.Module):
    """Correct representations produced by an already-trained backbone.

    The backbone must expose ``encode(x_num, x_cat)`` and return a 2-D tensor
    of shape ``(batch, hidden_dim)``.  Its parameters are frozen permanently;
    only the shift estimator, coordinate estimator, embedding vectors and the
    re-initialised prediction head are optimised during the TRC stage.
    """

    def __init__(
        self,
        *,
        backbone: nn.Module,
        hidden_dim: int,
        d_out: int,
        embedding_num: int = 10,
        shift_estimator: bool = True,
        space_mapping: bool = True,
        loss_orth: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if embedding_num <= 0:
            raise ValueError("embedding_num must be positive")
        self.backbone = backbone
        self.hidden_dim = hidden_dim
        self.d_out = d_out
        self.embedding_num = embedding_num
        self.use_shift_estimator = shift_estimator
        self.use_space_mapping = space_mapping
        self.use_orthogonal_loss = loss_orth and space_mapping

        if self.use_shift_estimator:
            self.shift_estimator = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.shift_estimator = None

        if self.use_space_mapping:
            self.coordinate_estimator = nn.Linear(hidden_dim, embedding_num)
            self.embedding_vectors = nn.Parameter(
                torch.empty(embedding_num, hidden_dim)
            )
            nn.init.normal_(self.embedding_vectors)
        else:
            self.coordinate_estimator = None
            self.register_parameter("embedding_vectors", None)

        self.head = nn.Linear(hidden_dim, d_out)
        self.freeze_backbone()

    def freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True):
        """Keep dropout/normalisation in the frozen backbone deterministic."""
        super().train(mode)
        self.backbone.eval()
        return self

    def encode_backbone(self, x_num, x_cat) -> torch.Tensor:
        with torch.no_grad():
            representation = self.backbone.encode(x_num, x_cat)
        if representation.ndim != 2 or representation.shape[-1] != self.hidden_dim:
            raise RuntimeError(
                "backbone.encode() must return (batch, hidden_dim); got "
                f"{tuple(representation.shape)}"
            )
        return representation.detach()

    def reestimate(self, representation: torch.Tensor) -> torch.Tensor:
        if self.shift_estimator is None:
            return representation
        return representation - self.shift_estimator(representation)

    def map_space(self, representation: torch.Tensor) -> torch.Tensor:
        if self.coordinate_estimator is None:
            return representation
        coordinates = F.softmax(self.coordinate_estimator(representation), dim=-1)
        return coordinates @ self.embedding_vectors

    def forward_from_representation(self, representation: torch.Tensor) -> torch.Tensor:
        corrected = self.reestimate(representation)
        corrected = self.map_space(corrected)
        output = self.head(corrected)
        return output.squeeze(-1) if self.d_out == 1 else output

    def forward(self, x_num, x_cat=None) -> torch.Tensor:
        representation = self.encode_backbone(x_num, x_cat)
        return self.forward_from_representation(representation)

    def correction_parameters(self) -> Iterable[nn.Parameter]:
        """Yield all and only parameters trained in the TRC stage."""
        if self.shift_estimator is not None:
            yield from self.shift_estimator.parameters()
        if self.coordinate_estimator is not None:
            yield from self.coordinate_estimator.parameters()
            yield self.embedding_vectors
        yield from self.head.parameters()

    def orthogonality_loss(self) -> torch.Tensor:
        """Diversify light-space vectors as in the authors' implementation."""
        if self.embedding_vectors is None:
            return self.head.weight.new_zeros(())
        normalized = F.normalize(self.embedding_vectors, p=2, dim=1)
        similarity = (normalized @ normalized.T).abs().clamp(0.0, 1.0)
        l1 = similarity.sum()
        l2_squared = similarity.square().sum().clamp_min(
            torch.finfo(similarity.dtype).eps
        )
        sparse_term = l1 / l2_squared
        constraint_term = (l1 - self.embedding_num).abs()
        return sparse_term + 0.5 * constraint_term
