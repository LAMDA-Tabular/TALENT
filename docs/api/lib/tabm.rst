**TabM**
========

A model based on MLP and variations of BatchEnsemble.


Functions
~~~~~~~~~

.. code-block:: python

    def _init_scaling_by_sections(weight: Tensor, distribution: Literal['normal', 'random-signs'], init_sections: list[int]) -> None

Initializes scaling weights by sections for efficient ensemble members.

**Parameters:**

* **weight** *(Tensor)* - Weight tensor to initialize.
* **distribution** *(str)* - Initialization distribution ('normal' or 'random-signs').
* **init_sections** *(list[int])* - List of section sizes.


.. code-block:: python

    def init_rsqrt_uniform_(x: Tensor, d: int) -> Tensor

Initializes tensor with uniform distribution scaled by reciprocal square root.

**Parameters:**

* **x** *(Tensor)* - Tensor to initialize.
* **d** *(int)* - Dimension for scaling.

**Returns:**

* **Tensor** - Initialized tensor.


.. code-block:: python

    def init_random_signs_(x: Tensor) -> Tensor

Initializes tensor with random signs (-1 or 1).

**Parameters:**

* **x** *(Tensor)* - Tensor to initialize.

**Returns:**

* **Tensor** - Tensor with random signs.


.. code-block:: python

    class Identity(nn.Module)

Identity module that returns input unchanged.

**Input:**

* **x** *(Tensor)* - Input tensor.

**Output:**

* **Tensor** - Same as input.


.. code-block:: python

    class Mean(nn.Module)

Computes mean along specified dimension.

**Parameters:**

* **dim** *(int)* - Dimension to compute mean along.

**Input:**

* **x** *(Tensor)* - Input tensor.

**Output:**

* **Tensor** - Mean along specified dimension.


.. code-block:: python

    class ScaleEnsemble(nn.Module)

Scales ensemble members with learnable weights.

**Parameters:**

* **k** *(int)* - Number of ensemble members.
* **d** *(int)* - Feature dimension.
* **init** *(str)* - Weight initialization ('ones', 'normal', 'random-signs').

**Input:**

* **x** *(Tensor)* - Input tensor of shape (B, K, D).

**Output:**

* **Tensor** - Scaled tensor.


.. code-block:: python

    class ElementwiseAffineEnsemble(nn.Module)

Element-wise affine transformation for ensemble members.

**Parameters:**

* **k** *(int)* - Number of ensemble members.
* **d** *(int)* - Feature dimension.
* **bias** *(bool)* - Whether to use bias.
* **weight_init** *(str)* - Weight initialization method.

**Input:**

* **x** *(Tensor)* - Input tensor of shape (B, K, D).

**Output:**

* **Tensor** - Transformed tensor.


.. code-block:: python

    class LinearEfficientEnsemble(nn.Module)

Efficient ensemble linear layer with configurable scaling.

**Parameters:**

* **in_features** *(int)* - Input feature dimension.
* **out_features** *(int)* - Output feature dimension.
* **bias** *(bool, optional, Default is True)* - Whether to use bias.
* **k** *(int)* - Number of ensemble members.
* **ensemble_scaling_in** *(bool)* - Whether to ensemble input scaling.
* **ensemble_scaling_out** *(bool)* - Whether to ensemble output scaling.
* **ensemble_bias** *(bool)* - Whether to ensemble bias.
* **scaling_init** *(str)* - Scaling initialization method.

**Input:**

* **x** *(Tensor)* - Input tensor of shape (B, K, D).

**Output:**

* **Tensor** - Linear transformation result.


.. code-block:: python

    def make_efficient_ensemble(module: nn.Module, **kwargs) -> None

Converts a module to use efficient ensemble methods.

**Parameters:**

* **module** *(nn.Module)* - Module to convert.
* **kwargs** - Ensemble configuration parameters.


.. code-block:: python

    class OneHotEncoding0d(nn.Module)

One-hot encoding for categorical features.

**Parameters:**

* **cardinalities** *(list[int])* - List of category counts for each feature.

**Input:**

* **x** *(Tensor)* - Categorical feature tensor.

**Output:**

* **Tensor** - One-hot encoded tensor. 

**References:**

Yury Gorishniy, Akim Kotelnikov, and Artem Babenko. **TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling**. arXiv:2410.24210 [cs.LG], 2025. `<https://arxiv.org/abs/2410.24210>`_