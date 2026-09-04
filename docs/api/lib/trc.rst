Deep Tabular Representation Corrector (TRC)
===========================================

TRC is a post-hoc representation method. TALENT first trains an
FT-Transformer backbone, freezes all of its parameters, searches the
validation split for approximated optimal representations, and then learns:

* a two-layer shift estimator for representation re-estimation;
* a coordinate estimator and compact set of embedding vectors for light-space
  mapping;
* a re-initialized prediction head.

The implementation follows Algorithm 1 from `Deep Tabular Representation
Corrector <https://arxiv.org/abs/2603.16569>`_ and supports regression, binary
classification, and multiclass classification.

Usage
-----

.. code-block:: bash

   python train_model_deep.py --model_type trc --cat_policy indices

The main parameters are defined in ``TALENT/configs/default/trc.json``.
``tau`` controls the ratio of approximated optimal validation representations,
``perturb_times`` controls the number of simulated shifts, ``embedding_num``
sets the number of light-space vectors, and ``reg_weight`` weights the
orthogonality objective.

API
---

.. automodule:: TALENT.model.models.trc
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: TALENT.model.methods.trc
   :members:
   :undoc-members:
   :show-inheritance:
