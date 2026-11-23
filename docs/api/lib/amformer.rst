**AMFormer**
============

A token-based method which improves the transformer architecture for tabular data by incorporating parallel addition and multiplication attention mechanisms, utilizing prompt tokens to constrain feature interactions.


Functions
~~~~~~~~~

.. code-block:: python

    class GEGLU(nn.Module)

GEGLU (Gated Exponential Linear Unit) activation function.

**Input:**

* **x** *(Tensor)* - Input tensor.

**Output:**

* **Tensor** - GEGLU output.


.. code-block:: python

    def FeedForward(dim, mult=4, dropout=0.)

Creates a feedforward network with GEGLU activation.

**Parameters:**

* **dim** *(int)* - Input/output dimension.
* **mult** *(int, optional, Default is 4)* - Multiplier for hidden dimension.
* **dropout** *(float, optional, Default is 0.)* - Dropout rate.

**Returns:**

* **nn.Sequential** - Feedforward network.


.. code-block:: python

    class Attention(nn.Module)

Multi-head attention mechanism.

**Parameters:**

* **heads** *(int, optional, Default is 8)* - Number of attention heads.
* **dim** *(int, optional, Default is 64)* - Input dimension.
* **dropout** *(float, optional, Default is 0.)* - Dropout rate.
* **inner_dim** *(int, optional, Default is 0)* - Inner dimension (0 for same as dim).

**Input:**

* **x** *(Tensor)* - Input tensor.
* **attn_out** *(bool, optional, Default is False)* - Whether to return attention weights.

**Output:**

* **Tensor** - Attention output, or tuple (output, attention_weights) if attn_out=True.


.. code-block:: python

    class MemoryBlock(nn.Module)

Memory block with grouped attention mechanism.

**Parameters:**

* **token_num** *(int)* - Number of tokens.
* **heads** *(int)* - Number of attention heads.
* **dim** *(int)* - Input dimension.
* **attn_dropout** *(float)* - Attention dropout rate.
* **cluster** *(bool)* - Whether to use clustering.
* **target_mode** *(str)* - Target mode for attention.
* **groups** *(int)* - Number of groups.
* **num_per_group** *(int)* - Number of tokens per group.
* **use_cls_token** *(bool)* - Whether to use CLS token.
* **sum_or_prod** *(str, optional)* - Sum or product operation.
* **qk_relu** *(bool, optional, Default is False)* - Whether to use ReLU in QK computation.

**Input:**

* **x** *(Tensor)* - Input tensor.

**Output:**

* **Tensor** - Memory block output.


.. code-block:: python

    class Transformer(nn.Module)

Transformer model with memory blocks.

**Parameters:**

* **dim** *(int)* - Input dimension.
* **depth** *(int)* - Number of transformer layers.
* **heads** *(int)* - Number of attention heads.
* **attn_dropout** *(float)* - Attention dropout rate.
* **ff_dropout** *(float)* - Feedforward dropout rate.
* **use_cls_token** *(bool)* - Whether to use CLS token.
* **groups** *(int)* - Number of groups.
* **sum_num_per_group** *(int)* - Number per group for sum operation.
* **prod_num_per_group** *(int)* - Number per group for product operation.
* **cluster** *(bool)* - Whether to use clustering.
* **target_mode** *(str)* - Target mode.
* **token_num** *(int)* - Number of tokens.
* **token_descent** *(bool, optional, Default is False)* - Whether to use token descent.
* **use_prod** *(bool, optional, Default is True)* - Whether to use product operation.
* **qk_relu** *(bool, optional, Default is False)* - Whether to use ReLU in QK.

**Input:**

* **x** *(Tensor)* - Input tensor.

**Output:**

* **Tensor** - Transformer output.


.. code-block:: python

    class NumericalEmbedder(nn.Module)

Numerical feature embedder.

**Parameters:**

* **dim** *(int)* - Embedding dimension.
* **num_numerical_types** *(int)* - Number of numerical feature types.

**Input:**

* **x** *(Tensor)* - Numerical feature tensor.

**Output:**

* **Tensor** - Embedded numerical features. 



**Referencses:**

Cheng, Y., Hu, R., Ying, H., Shi, X., Wu, J., & Lin, W. (2024). Arithmetic Feature Interaction Is Necessary for Deep Tabular Learning. arXiv:2402.02334. `<https://arxiv.org/abs/2402.02334>`_
