**ExcelFormer**
===============

A deep learning model for tabular data prediction, featuring a semi-permeable attention module to address rotational invariance, tailored data augmentation, and an attentive feedforward network, making it a reliable solution across diverse datasets.


Functions
~~~~~~~~~

.. code-block:: python

    class Lambda(nn.Module)

Lambda layer that applies a custom function.

**Parameters:**

* **f** *(callable)* - Function to apply.

**Input:**

* **x** - Input tensor.

**Output:**

* **Tensor** - Result of applying function f to input.


.. code-block:: python

    class RMSNorm(nn.Module)

Root Mean Square Layer Normalization.

**Parameters:**

* **d** *(int)* - Model size.
* **p** *(float, optional, Default is -1.0)* - Partial RMSNorm parameter.
* **eps** *(float, optional, Default is 1e-5)* - Epsilon value.
* **bias** *(bool, optional, Default is False)* - Whether to use bias term.

**Input:**

* **x** *(Tensor)* - Input tensor.

**Output:**

* **Tensor** - Normalized tensor.


.. code-block:: python

    class ScaleNorm(nn.Module)

Scale normalization layer.

**Parameters:**

* **d** *(int)* - Model dimension.
* **eps** *(float, optional, Default is 1e-5)* - Epsilon value.
* **clamp** *(bool, optional, Default is False)* - Whether to clamp norms.

**Input:**

* **x** *(Tensor)* - Input tensor.

**Output:**

* **Tensor** - Scaled and normalized tensor.


.. code-block:: python

    def reglu(x: Tensor) -> Tensor

ReGLU activation function.

**Parameters:**

* **x** *(Tensor)* - Input tensor.

**Returns:**

* **Tensor** - ReGLU output.


.. code-block:: python

    def geglu(x: Tensor) -> Tensor

GEGLU activation function.

**Parameters:**

* **x** *(Tensor)* - Input tensor.

**Returns:**

* **Tensor** - GEGLU output.


.. code-block:: python

    def tanglu(x: Tensor) -> Tensor

TanGLU activation function.

**Parameters:**

* **x** *(Tensor)* - Input tensor.

**Returns:**

* **Tensor** - TanGLU output.


.. code-block:: python

    class ReGLU(nn.Module)

ReGLU activation module.

**Input:**

* **x** *(Tensor)* - Input tensor.

**Output:**

* **Tensor** - ReGLU output.


.. code-block:: python

    class GEGLU(nn.Module)

GEGLU activation module.

**Input:**

* **x** *(Tensor)* - Input tensor.

**Output:**

* **Tensor** - GEGLU output.


.. code-block:: python

    def make_optimizer(optimizer: str, parameter_groups, lr: float, weight_decay: float) -> optim.Optimizer

Creates an optimizer with specified parameters.

**Parameters:**

* **optimizer** *(str)* - Optimizer type.
* **parameter_groups** - Parameter groups.
* **lr** *(float)* - Learning rate.
* **weight_decay** *(float)* - Weight decay.

**Returns:**

* **optim.Optimizer** - Configured optimizer.


.. code-block:: python

    class RAdam(optim.Optimizer)

Rectified Adam optimizer.

**Parameters:**

* **params** - Model parameters.
* **lr** *(float, optional, Default is 1e-3)* - Learning rate.
* **betas** *(tuple, optional, Default is (0.9, 0.999))* - Beta parameters.
* **eps** *(float, optional, Default is 1e-8)* - Epsilon value.
* **weight_decay** *(float, optional, Default is 0)* - Weight decay.
* **degenerated_to_sgd** *(bool, optional, Default is True)* - Whether to degenerate to SGD.


.. code-block:: python

    class AdaBelief(optim.Optimizer)

AdaBelief optimizer.

**Parameters:**

* **params** - Model parameters.
* **lr** *(float, optional, Default is 1e-3)* - Learning rate.
* **betas** *(tuple, optional, Default is (0.9, 0.999))* - Beta parameters.
* **eps** *(float, optional, Default is 1e-16)* - Epsilon value.
* **weight_decay** *(float, optional, Default is 0)* - Weight decay.
* **amsgrad** *(bool, optional, Default is False)* - Whether to use AMSGrad.
* **weight_decouple** *(bool, optional, Default is True)* - Whether to decouple weight decay.
* **fixed_decay** *(bool, optional, Default is False)* - Whether to use fixed decay.
* **rectify** *(bool, optional, Default is True)* - Whether to use rectification.
* **degenerated_to_sgd** *(bool, optional, Default is True)* - Whether to degenerate to SGD. 


**Referencses:**

ExcelFormer: A Neural Network Surpassing GBDTs on Tabular Data
Jintai Chen, Jiahuan Yan, Qiyuan Chen, Danny Ziyi Chen, Jian Wu, Jimeng Sun. **ExcelFormer: A Neural Network Surpassing GBDTs on Tabular Data**. arXiv:2301.02819 [cs.LG], 2024. `<https://arxiv.org/abs/2301.02819>`_