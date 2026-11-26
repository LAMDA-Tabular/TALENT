**TabCaps**
===========

A capsule network that encapsulates all feature values of a record into vectorial features.


Functions
~~~~~~~~~

.. code-block:: python

    class TabCapsModel(BaseEstimator)

TabCaps model for tabular data classification using capsule networks.

**Parameters:**

* **decode** *(bool, optional, Default is False)* - Whether to use reconstruction.
* **mean** *(int, optional)* - Mean value for normalization.
* **std** *(int, optional)* - Standard deviation for normalization.
* **sub_class** *(int, optional, Default is 1)* - Number of sub-classes.
* **init_dim** *(int, optional)* - Initial dimension.
* **primary_capsule_size** *(int, optional, Default is 16)* - Primary capsule size.
* **digit_capsule_size** *(int, optional, Default is 16)* - Digit capsule size.
* **leaves** *(int, optional, Default is 32)* - Number of leaves.
* **seed** *(int, optional, Default is 0)* - Random seed.
* **verbose** *(int, optional, Default is 1)* - Verbosity level.
* **optimizer_fn** *(Any, optional)* - Optimizer function.
* **optimizer_params** *(Dict, optional)* - Optimizer parameters.
* **scheduler_fn** *(Any, optional)* - Scheduler function.
* **scheduler_params** *(Dict, optional)* - Scheduler parameters.
* **input_dim** *(int, optional)* - Input dimension.
* **output_dim** *(int, optional)* - Output dimension.
* **device_name** *(str, optional, Default is "auto")* - Device name.

**Methods:**

* **fit(self, X_train, y_train, eval_set=None, eval_name=None, eval_metric=None, max_epochs=100, patience=10, batch_size=1024, virtual_batch_size=256, callbacks=None, logname=None, resume_dir=None, device_id=None, cfg=None)** - Train the model.
* **predict(self, X, y, decode=False)** - Make predictions.
* **save_check(self, path, seed)** - Save model checkpoint.
* **load_model(self, filepath, input_dim, output_dim)** - Load saved model.


.. code-block:: python

    class CapsuleClassifier(nn.Module)

Capsule network classifier for tabular data.

**Parameters:**

* **input_dim** *(int)* - Input dimension.
* **output_dim** *(int)* - Output dimension.
* **out_capsule_num** *(int)* - Number of output capsules.
* **init_dim** *(int)* - Initial dimension.
* **primary_capsule_dim** *(int)* - Primary capsule dimension.
* **digit_capsule_dim** *(int)* - Digit capsule dimension.
* **n_leaves** *(int)* - Number of leaves.

**Input:**

* **x** *(Tensor)* - Input tensor.

**Output:**

* **Tensor** - Classification output.


.. code-block:: python

    class ReconstructCapsNet(nn.Module)

Capsule network with reconstruction capabilities.

**Parameters:**

* **input_dim** *(int)* - Input dimension.
* **output_dim** *(int)* - Output dimension.
* **out_capsule_num** *(int)* - Number of output capsules.
* **init_dim** *(int)* - Initial dimension.
* **primary_capsule_dim** *(int)* - Primary capsule dimension.
* **digit_capsule_dim** *(int)* - Digit capsule dimension.
* **n_leaves** *(int)* - Number of leaves.

**Input:**

* **x** *(Tensor)* - Input tensor.
* **y_one_hot** *(Tensor)* - One-hot encoded labels.

**Output:**

* **tuple** - (classification_output, reconstruction_output).


.. code-block:: python

    class MarginLoss(nn.Module)

Margin loss for capsule networks.

**Parameters:**

* **m_plus** *(float, optional, Default is 0.9)* - Positive margin.
* **m_minus** *(float, optional, Default is 0.1)* - Negative margin.
* **lambda_val** *(float, optional, Default is 0.5)* - Lambda value.

**Input:**

* **y_pred** *(Tensor)* - Predicted outputs.
* **y_true** *(Tensor)* - True labels.

**Output:**

* **Tensor** - Loss value.


.. code-block:: python

    class AbstractLayer(nn.Module)

Abstract layer for capsule networks.

**Parameters:**

* **base_input_dim** *(int)* - Base input dimension.
* **base_output_dim** *(int)* - Base output dimension.
* **k** *(int)* - Number of branches.
* **virtual_batch_size** *(int)* - Virtual batch size.
* **bias** *(bool, optional, Default is False)* - Whether to use bias.

**Input:**

* **x** *(Tensor)* - Input tensor.

**Output:**

* **Tensor** - Layer output.

**References:**

Jintai Chen, Kuanlun Liao, Yanwen Fang, Danny Z. Chen, Jian Wu. **TABCAPS: A CAPSULE NEURAL NETWORK FOR TABULAR DATA CLASSIFICATION WITH BOW ROUTING**. In *Proceedings of the 11th International Conference on Learning Representations*, 2023. `<https://openreview.net/pdf?id=G32oY4Vnm8>`_