**HyperFast**
=============

A meta-trained hypernetwork that generates task-specific neural networks for instant classification of tabular data.


Functions
~~~~~~~~~

.. code-block:: python

    class HyperFast(nn.Module)

HyperFast model using hypernetworks for dynamic weight generation.

**Parameters:**

* **cfg** - Configuration object containing:
  - **n_dims** *(int)* - Number of dimensions.
  - **max_categories** *(int)* - Maximum number of categories.
  - **rf_size** *(int)* - Random feature size.
  - **torch_pca** *(bool)* - Whether to use torch PCA.
  - **clip_data_value** *(float)* - Data clipping value.
  - **hn_n_layers** *(int)* - Number of hypernetwork layers.
  - **hn_hidden_size** *(int)* - Hypernetwork hidden size.
  - **main_n_layers** *(int)* - Number of main network layers.

**Input:**

* **X** *(Tensor)* - Input features.
* **y** *(Tensor)* - Target labels.
* **n_classes** *(int)* - Number of classes.

**Output:**

* **Tensor** - Model predictions.


.. code-block:: python

    class TorchPCA(nn.Module)

PyTorch implementation of PCA for dimensionality reduction.

**Parameters:**

* **n_components** *(int)* - Number of components to keep.

**Methods:**

* **fit_transform(self, X)** - Fit PCA and transform data.
* **transform(self, X)** - Transform data using fitted PCA.


.. code-block:: python

    def create_random_features(X, rf_size, device)

Creates random features for input data.

**Parameters:**

* **X** *(Tensor)* - Input data.
* **rf_size** *(int)* - Random feature size.
* **device** *(torch.device)* - Target device.

**Returns:**

* **Tensor** - Random features.


.. code-block:: python

    def compute_class_means(X, y, n_classes)

Computes per-class mean features.

**Parameters:**

* **X** *(Tensor)* - Input features.
* **y** *(Tensor)* - Target labels.
* **n_classes** *(int)* - Number of classes.

**Returns:**

* **Tensor** - Per-class mean features. 


**References:**

David Bonet, Daniel Mas Montserrat, Xavier Giró-i-Nieto, Alexander G. Ioannidis. **HyperFast: Instant Classification for Tabular Data**. arXiv:2402.14335 [cs.LG], 2024. `<https://arxiv.org/abs/2402.14335>`_