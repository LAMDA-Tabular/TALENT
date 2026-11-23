**TabR**
========

A deep learning model that integrates a KNN component to enhance tabular data predictions through an efficient attention-like mechanism.


Functions
~~~~~~~~~

.. code-block:: python

    def _initialize_embeddings(weight: Tensor, d: Optional[int]) -> None

Initializes embedding weights with uniform distribution.

**Parameters:**

* **weight** *(Tensor)* - Weight tensor to initialize.
* **d** *(Optional[int])* - Embedding dimension.


.. code-block:: python

    def make_trainable_vector(d: int) -> Parameter

Creates a trainable parameter vector with proper initialization.

**Parameters:**

* **d** *(int)* - Vector dimension.

**Returns:**

* **Parameter** - Trainable parameter vector.


.. code-block:: python

    class CLSEmbedding(nn.Module)

Adds a CLS token to the beginning of input sequences.

**Parameters:**

* **d_embedding** *(int)* - Embedding dimension.

**Input Shape:**

`(*, seq_len, d_embedding)`

**Output Shape:**

`(*, seq_len + 1, d_embedding)`


.. code-block:: python

    class ResNet(nn.Module)

Residual network with customizable blocks and normalization.

**Parameters:**

* **d_in** *(Optional[int])* - Input dimension.
* **d_out** *(Optional[int])* - Output dimension.
* **n_blocks** *(int)* - Number of residual blocks.
* **d_block** *(int)* - Block dimension.
* **dropout** *(float)* - Dropout rate.
* **d_hidden_multiplier** *(float)* - Hidden dimension multiplier.
* **n_linear_layers_per_block** *(int, optional, Default is 2)* - Number of linear layers per block.
* **activation** *(str, optional, Default is 'ReLU')* - Activation function.
* **normalization** *(str)* - Normalization type.
* **first_normalization** *(bool)* - Whether to apply normalization first.


.. code-block:: python

    class LinearEmbeddings(nn.Module)

Linear embeddings for continuous features.

**Parameters:**

* **n_features** *(int)* - Number of features.
* **d_embedding** *(int)* - Embedding dimension.
* **bias** *(bool, optional, Default is True)* - Whether to use bias.


.. code-block:: python

    class PeriodicEmbeddings(nn.Module)

Periodic embeddings using frequency-based encoding.

**Parameters:**

* **n_features** *(int)* - Number of features.
* **n_frequencies** *(int)* - Number of frequencies.
* **frequency_scale** *(float)* - Frequency scaling factor.


.. code-block:: python

    class NLinear(nn.Module)

Feature-wise linear layer.

**Parameters:**

* **n_features** *(int)* - Number of features.
* **d_in** *(int)* - Input dimension.
* **d_out** *(int)* - Output dimension.
* **bias** *(bool, optional, Default is True)* - Whether to use bias.


.. code-block:: python

    class LREmbeddings(nn.Sequential)

Linear + ReLU embeddings.

**Parameters:**

* **n_features** *(int)* - Number of features.
* **d_embedding** *(int)* - Embedding dimension.


.. code-block:: python

    class PLREmbeddings(nn.Sequential)

Periodic + Linear + ReLU embeddings.

**Parameters:**

* **n_features** *(int)* - Number of features.
* **n_frequencies** *(int)* - Number of frequencies.
* **frequency_scale** *(float)* - Frequency scaling factor.
* **d_embedding** *(int)* - Embedding dimension.
* **lite** *(bool)* - Whether to use lite version.


.. code-block:: python

    class PBLDEmbeddings(nn.Module)

Periodic + BatchNorm + Linear + Dropout embeddings.

**Parameters:**

* **n_features** *(int)* - Number of features.
* **n_frequencies** *(int)* - Number of frequencies.
* **frequency_scale** *(float)* - Frequency scaling factor.
* **d_embedding** *(int)* - Embedding dimension.
* **lite** *(bool)* - Whether to use lite version.
* **plr_act_name** *(str, optional, Default is 'relu')* - Activation function name.
* **plr_use_densenet** *(bool, optional, Default is True)* - Whether to use dense connections.


.. code-block:: python

    class MLP(nn.Module)

Multi-layer perceptron with SELU activation.

**Parameters:**

* **d_in** *(Optional[int])* - Input dimension.
* **d_out** *(Optional[int])* - Output dimension.
* **n_blocks** *(int)* - Number of blocks.
* **d_block** *(int)* - Block dimension.
* **dropout** *(float)* - Dropout rate.
* **activation** *(str, optional, Default is 'SELU')* - Activation function.


.. code-block:: python

    def make_module(spec, *args, **kwargs) -> nn.Module

Creates a module based on specification.

**Parameters:**

* **spec** - Module specification.
* **args** - Positional arguments.
* **kwargs** - Keyword arguments.

**Returns:**

* **nn.Module** - Created module.


.. code-block:: python

    def make_module1(type: str, *args, **kwargs) -> nn.Module

Creates a module of specified type.

**Parameters:**

* **type** *(str)* - Module type.
* **args** - Positional arguments.
* **kwargs** - Keyword arguments.

**Returns:**

* **nn.Module** - Created module. 

**References:**

Yury Gorishniy, Ivan Rubachev, Nikolay Kartashev, Daniil Shlenskii, Akim Kotelnikov, and Artem Babenko. **TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023**. arXiv:2307.14338 [cs.LG], 2023. `<https://arxiv.org/abs/2307.14338>`_