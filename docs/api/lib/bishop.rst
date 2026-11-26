**BiSHop**
==========

An end-to-end framework for deep tabular learning which leverages a sparse Hopfield model with adaptable sparsity, enhanced by column-wise and row-wise modules.


Functions
~~~~~~~~~

.. code-block:: python

    class NumEmb(torch.nn.Module)

Numerical embedding layer for tabular data using quantile-based binning.

**Parameters:**

* **n_num** *(int)* - Number of numerical features.
* **emb_dim** *(int)* - Embedding dimension (number of bins).

**Methods:**

* **get_bins(self, data, identifier='num')** - Compute quantile bins from data.
* **forward(self, x)** - Forward pass through numerical embedding.
* **_to(self, device)** - Move model to specified device.

**Input Shape:**

`(batch_size, n_num)`

**Output Shape:**

`(batch_size, n_num, emb_dim)`


.. code-block:: python

    class FullEmbDropout(torch.nn.Module)

Full embedding dropout layer.

**Parameters:**

* **dropout** *(float, optional, Default is 0.1)* - Dropout rate.

**Input:**

* **X** *(torch.Tensor)* - Input tensor.

**Output:**

* **torch.Tensor** - Tensor with dropout applied.


.. code-block:: python

    def _trunc_normal_(x, mean=0., std=1.)

Truncated normal initialization approximation.

**Parameters:**

* **x** *(Tensor)* - Tensor to initialize.
* **mean** *(float, optional, Default is 0.)* - Mean of normal distribution.
* **std** *(float, optional, Default is 1.)* - Standard deviation.

**Returns:**

* **Tensor** - Initialized tensor.


.. code-block:: python

    class _Embedding(torch.nn.Embedding)

Embedding layer with truncated normal initialization.

**Parameters:**

* **ni** *(int)* - Number of input features.
* **nf** *(int)* - Number of output features.
* **std** *(float, optional, Default is 0.01)* - Standard deviation for initialization.


.. code-block:: python

    class CatEmb(torch.nn.Module)

Categorical embedding layer with sharing options.

**Parameters:**

* **n_cat** *(int)* - Number of categorical features.
* **emb_dim** *(int)* - Embedding dimension.
* **n_class** *(list)* - List of number of classes for each categorical feature.
* **share** *(bool, optional, Default is True)* - Whether to share embeddings.
* **share_add** *(bool, optional, Default is False)* - Whether to add shared embeddings.
* **share_div** *(int, optional, Default is 8)* - Division factor for shared embeddings.
* **full_dropout** *(bool, optional, Default is False)* - Whether to use full dropout.
* **emb_dropout** *(float, optional, Default is 0.1)* - Embedding dropout rate.

**Input:**

* **x** *(torch.Tensor)* - Categorical feature tensor.

**Output:**

* **torch.Tensor** - Embedded categorical features.


.. code-block:: python

    class PatchEmb(torch.nn.Module)

Patch embedding layer for transformer architectures.

**Parameters:**

* **patch_dim** *(int)* - Patch dimension.
* **d_model** *(int)* - Model dimension.

**Input:**

* **x** *(torch.Tensor)* - Input tensor.

**Output:**

* **torch.Tensor** - Patch embedded tensor. 


**Referencses:**

Xu, C., Huang, Y.-C., Hu, J. Y.-C., Li, W., Gilani, A., Goan, H.-S., & Liu, H. (2024). BiSHop: Bi-Directional Cellular Learning for Tabular Data with Generalized Sparse Modern Hopfield Model. In Proceedings of the 41st International Conference on Machine Learning (ICML). `<https://arxiv.org/abs/2404.03830>`_