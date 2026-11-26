**Periodic Tab DL**
===================

Periodic Tab DL provides Fourier-based feature encoding and periodic neural networks for tabular data processing.


Functions
~~~~~~~~~

.. code-block:: python

    class FourierEncoder(nn.Module)

Fourier Feature Encoder Layer with optional Random Fourier Features (RFF).

**Parameters:**

* **input_size** *(int)* - Number of input features.
* **num_features_per_input** *(int)* - Number of Fourier features per input feature.
* **kernel_size** *(int, optional, Default is 1)* - Size of the kernel for local feature extraction.
* **scale_input** *(bool, optional, Default is True)* - Whether to scale the input with learnable parameters.
* **init_frequency_range** *(tuple, optional, Default is (0.0, 5.0))* - Range for frequency initialization.
* **use_feature_scaling** *(bool, optional, Default is True)* - Whether to add learnable scaling for Fourier features.
* **use_convolution** *(bool, optional, Default is True)* - Whether to apply convolution over input features.
* **activation** *(str, optional, Default is 'sin_cos')* - Activation function ('sin_cos', 'sin', 'cos', 'tanh').
* **frequency_init** *(str, optional, Default is 'log')* - Method for initializing frequencies.
* **frequency_scale** *(float, optional, Default is None)* - Scaling factor for frequencies.
* **use_phase_shift** *(bool, optional, Default is True)* - Whether to use learnable phase shifts.
* **use_learnable_amplitude** *(bool, optional, Default is True)* - Whether to use learnable amplitude scaling.
* **use_rff** *(bool, optional, Default is True)* - Whether to use Random Fourier Features.
* **rff_sigma** *(float, optional, Default is 1.0)* - Bandwidth parameter for RFF.

**Input:**

* **x** *(Tensor)* - Input tensor of shape (batch_size, input_size).

**Output:**

* **Tensor** - Fourier encoded features.


.. code-block:: python

    class FourierBlock(nn.Module)

Fourier block for periodic neural networks.

**Parameters:**

* **input_dim** *(int)* - Input dimension.
* **hidden_dim** *(int)* - Hidden dimension.
* **output_dim** *(int)* - Output dimension.
* **activation** *(str, optional, Default is 'relu')* - Activation function.

**Input:**

* **x** *(Tensor)* - Input tensor.

**Output:**

* **Tensor** - Block output.


.. code-block:: python

    class FourierNet(nn.Module)

Fourier-based neural network for tabular data.

**Parameters:**

* **input_dim** *(int)* - Input dimension.
* **hidden_dims** *(List[int])* - List of hidden dimensions.
* **output_dim** *(int)* - Output dimension.
* **activation** *(str, optional, Default is 'relu')* - Activation function.

**Input:**

* **x** *(Tensor)* - Input tensor.

**Output:**

* **Tensor** - Network output.


.. code-block:: python

    class TabFourierNet(nn.Module)

Tabular Fourier network with feature encoding.

**Parameters:**

* **input_dim** *(int)* - Input dimension.
* **hidden_dims** *(List[int])* - List of hidden dimensions.
* **output_dim** *(int)* - Output dimension.
* **fourier_params** *(Dict)* - Fourier encoder parameters.

**Input:**

* **x** *(Tensor)* - Input tensor.

**Output:**

* **Tensor** - Network output.


.. code-block:: python

    class OrthogonalPolynomial

Orthogonal polynomial basis functions.

**Parameters:**

* **degree** *(int)* - Polynomial degree.
* **basis_type** *(str)* - Type of orthogonal basis.

**Methods:**

* **evaluate(self, x)** - Evaluate polynomial at given points.


.. code-block:: python

    class PnPBlock(nn.Module)

Plug-and-Play neural network block.

**Parameters:**

* **input_dim** *(int)* - Input dimension.
* **output_dim** *(int)* - Output dimension.
* **activation** *(str)* - Activation function.

**Input:**

* **x** *(Tensor)* - Input tensor.

**Output:**

* **Tensor** - Block output.


.. code-block:: python

    class AutoPnP(nn.Module)

Automatic Plug-and-Play network architecture.

**Parameters:**

* **input_dim** *(int)* - Input dimension.
* **hidden_dims** *(List[int])* - List of hidden dimensions.
* **output_dim** *(int)* - Output dimension.

**Input:**

* **x** *(Tensor)* - Input tensor.

**Output:**

* **Tensor** - Network output.


