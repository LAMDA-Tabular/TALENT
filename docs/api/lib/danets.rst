**DANet**
=========

A neural network designed to enhance tabular data processing by grouping correlated features and reducing computational complexity.


Functions
~~~~~~~~~

.. code-block:: python

    class AcceleratedCreator(object)

Creates accelerated versions of neural networks by extracting and compressing layers.

**Parameters:**

* **input_dim** *(int)* - Input dimension.
* **base_out_dim** *(int)* - Base output dimension.
* **k** *(int)* - Number of branches.

**Methods:**

* **__call__(self, network)** - Accelerate a network by modifying its layers.
* **extract_module(self, basicblock, base_input_dim, fix_input_dim)** - Extract and compress a module.


.. code-block:: python

    class Extractor(object)

Extracts parameters from abstract layers and computes compressed weights.

**Parameters:**

* **k** *(int)* - Number of branches.

**Methods:**

* **get_parameter(self, abs_layer)** - Extract parameters from an abstract layer.
* **compute_weights(self, a, b, eps, mu, var, sw, pw, pb, base_input_dim, base_output_dim, k)** - Compute compressed weights.
* **__call__(self, abslayer, input_dim, base_out_dim)** - Extract and compress a layer.


.. code-block:: python

    class CompressAbstractLayer(nn.Module)

Compressed abstract layer with attention and feature weights.

**Parameters:**

* **att_w** *(Tensor)* - Attention weights.
* **f_w** *(Tensor)* - Feature weights.
* **att_b** *(Tensor)* - Attention bias.
* **f_b** *(Tensor)* - Feature bias.

**Input:**

* **x** *(Tensor)* - Input tensor.

**Output:**

* **Tensor** - Compressed output with attention mechanism.


.. code-block:: python

    def get_parameter(abs_layer)

Extract parameters from an abstract layer.

**Parameters:**

* **abs_layer** - Abstract layer to extract parameters from.

**Returns:**

* **tuple** - Extracted parameters (alpha, beta, eps, mu, var, sparse_weight, process_weight, process_bias).


.. code-block:: python

    def compute_weights(a, b, eps, mu, var, sw, pw, pb, base_input_dim, base_output_dim, k)

Compute compressed weights from extracted parameters.

**Parameters:**

* **a** *(Tensor)* - Alpha parameter.
* **b** *(Tensor)* - Beta parameter.
* **eps** *(float)* - Epsilon value.
* **mu** *(Tensor)* - Mean parameter.
* **var** *(Tensor)* - Variance parameter.
* **sw** *(Tensor)* - Sparse weights.
* **pw** *(Tensor)* - Process weights.
* **pb** *(Tensor)* - Process bias.
* **base_input_dim** *(int)* - Base input dimension.
* **base_output_dim** *(int)* - Base output dimension.
* **k** *(int)* - Number of branches.

**Returns:**

* **tuple** - Computed weights (W_att, W_fc, B_att, B_fc). 


**Referencses:**

Chen, J., Liao, K., Wan, Y., Chen, D. Z., & Wu, J. (2022). DANets: Deep Abstract Networks for Tabular Data Classification and Regression. arXiv:2112.02962 [cs.LG]. `<https://arxiv.org/abs/2112.02962>`_