"""Mitra-v2 method.

Mitra-v2 (AutoGluon / Amazon, September 2026, arXiv 2609.04540) is the second
generation of Mitra. Both released checkpoints use the same Tab2D architecture
as v1, so they run zero-shot through TALENT's bundled Mitra code:

* ``autogluon/mitra-classifier-2`` -- same configuration as v1 (10 classes);
* ``autogluon/mitra-regressor-2`` -- a 1000-bin distributional regression head.
  As in AutoGluon's cross-entropy regression path, targets are min-max scaled
  over the support set and passed as bin ids; predictions are the
  softmax-weighted mean of the bin centres (the decode the released regressor
  was evaluated with in ``mitra-finetune``).

This wrapper is zero-shot (in-context only); the fine-tuned + bagged recipe of
the Mitra-v2 report requires the ``mitra-finetune`` package.

The weights are Apache-2.0 licensed and downloaded automatically from the
Hugging Face Hub unless bundled under ``TALENT/model/models/models_mitra_v2/{cls,reg}/``.
"""

from TALENT.model.methods.mitra import MitraMethod


class MitraV2Method(MitraMethod):
    BUNDLED_DIR = "models_mitra_v2"
    HF_REPOS = {"cls": "autogluon/mitra-classifier-2", "reg": "autogluon/mitra-regressor-2"}
