**NODE**
========

A tree-mimic method that generalizes oblivious decision trees, combining gradient-based optimization with hierarchical representation learning.


Functions
~~~~~~~~~

.. code-block:: python

    class ODST(ModuleWithInit)

Oblivious Differentiable Sparsemax Trees - a differentiable decision tree implementation.

**Parameters:**

* **in_features** *(int)* - Number of input features.
* **num_trees** *(int)* - Number of trees in the ensemble.
* **depth** *(int, optional, Default is 6)* - Depth of each tree.
* **tree_dim** *(int, optional, Default is 1)* - Number of response channels per tree.
* **flatten_output** *(bool, optional, Default is True)* - Whether to flatten output.
* **choice_function** *(callable, optional, Default is sparsemax)* - Feature selection function.
* **bin_function** *(callable, optional, Default is sparsemoid)* - Binary decision function.
* **initialize_response_** *(callable, optional, Default is nn.init.normal_)* - Response initialization.
* **initialize_selection_logits_** *(callable, optional, Default is nn.init.uniform_)* - Selection logits initialization.
* **threshold_init_beta** *(float, optional, Default is 1.0)* - Threshold initialization beta.
* **threshold_init_cutoff** *(float, optional, Default is 1.0)* - Threshold initialization cutoff.

**Input Shape:**

`(batch_size, in_features)`

**Output Shape:**

`(batch_size, num_trees * tree_dim)` if flatten_output=True, else `(batch_size, num_trees, tree_dim)`

**Methods:**

* **initialize(self, input, eps=1e-6)** - Data-aware initialization of thresholds and temperatures.
* **forward(self, input)** - Forward pass through the oblivious trees.


.. code-block:: python

    def sparsemax(x, dim=-1)

Sparsemax activation function.

**Parameters:**

* **x** *(Tensor)* - Input tensor.
* **dim** *(int, optional, Default is -1)* - Dimension to apply sparsemax.

**Returns:**

* **Tensor** - Sparsemax output.


.. code-block:: python

    def sparsemoid(x)

Sparsemoid activation function.

**Parameters:**

* **x** *(Tensor)* - Input tensor.

**Returns:**

* **Tensor** - Sparsemoid output.


.. code-block:: python

    class ModuleWithInit(nn.Module)

Base class for modules with custom initialization.

**Methods:**

* **initialize(self, input, eps=1e-6)** - Initialize module parameters based on input data. 

**References:**

Sergei Popov, Stanislav Morozov, Artem Babenko. **Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data**. arXiv:1909.06312 [cs.LG], 2019. `<https://arxiv.org/abs/1909.06312>`_
