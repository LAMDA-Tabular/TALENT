**DNNR**
========

Enhances KNN by using local gradients and Taylor approximations for more accurate and interpretable predictions.


Functions
~~~~~~~~~

.. code-block:: python

    class DNNR(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin)

DNNR model for regression using nearest neighbors and derivative approximation.

**Parameters:**

* **n_neighbors** *(int, optional, Default is 3)* - Number of nearest neighbors to use.
* **n_derivative_neighbors** *(int, optional, Default is -1)* - Number of neighbors for derivative approximation.
* **order** *(str, optional, Default is "1")* - Taylor approximation order ('1', '2', '2diag', '3diag').
* **fit_intercept** *(bool, optional, Default is False)* - Whether to fit intercept.
* **solver** *(Union[str, Solver], optional, Default is "linear_regression")* - Equation solver.
* **index** *(Union[str, nn_index.BaseIndex], optional, Default is "annoy")* - Nearest neighbor index.
* **index_kwargs** *(dict, optional, Default is {})* - Index constructor arguments.
* **scaling** *(Union[None, str, InputScaling], optional, Default is "learned")* - Input scaling method.
* **scaling_kwargs** *(dict, optional, Default is {})* - Scaling method arguments.
* **precompute_derivatives** *(bool, optional, Default is False)* - Whether to precompute derivatives.
* **clip** *(bool, optional, Default is False)* - Whether to clip predictions.

**Methods:**

* **fit(self, X_train, y_train)** - Fit the DNNR model.
* **predict(self, X_test)** - Make predictions.
* **point_analysis(self, X_test, y_test=None)** - Analyze predictions for each point.


.. code-block:: python

    class InputScaling(sklearn.base.BaseEstimator, metaclass=abc.ABCMeta)

Abstract base class for input scaling methods.

**Methods:**

* **fit(self, X_train, y_train, X_test=None, y_test=None)** - Fit the scaling method.
* **transform(self, X)** - Transform input data.
* **fit_transform(self, X, y)** - Fit and transform data.


.. code-block:: python

    class NoScaling(InputScaling)

No scaling implementation.

**Methods:**

* **fit(self, X_train, y_train, X_test=None, y_test=None)** - No-op fit method.
* **transform(self, X)** - Returns input unchanged.


.. code-block:: python

    class LearnedScaling(InputScaling)

Learned input scaling using optimization.

**Parameters:**

* **n_epochs** *(int, optional, Default is 1)* - Number of training epochs.
* **optimizer** *(Union[str, Type[_Optimizer]], optional, Default is SGD)* - Optimizer type.
* **optimizer_params** *(dict, optional, Default is {})* - Optimizer parameters.
* **shuffle** *(bool, optional, Default is True)* - Whether to shuffle data.
* **epsilon** *(float, optional, Default is 1e-6)* - Gradient computation epsilon.
* **show_progress** *(bool, optional, Default is False)* - Whether to show progress bar.
* **fail_on_nan** *(bool, optional, Default is False)* - Whether to fail on NaN values.
* **index** *(Union[str, Type[nn_index.BaseIndex]], optional, Default is 'annoy')* - Nearest neighbor index.
* **index_kwargs** *(dict, optional, Default is {})* - Index constructor arguments.

**Methods:**

* **fit(self, X_train, y_train, X_val=None, y_val=None, val_size=None)** - Fit the scaling method.
* **transform(self, X)** - Transform input data.


.. code-block:: python

    class SGD(_Optimizer)

Stochastic gradient descent optimizer.

**Parameters:**

* **parameters** *(List[np.ndarray])* - Parameters to optimize.
* **lr** *(float, optional, Default is 0.01)* - Learning rate.

**Methods:**

* **step(self, gradients)** - Update parameters using gradients.


.. code-block:: python

    class RMSPROP(_Optimizer)

RMSPROP optimizer.

**Parameters:**

* **parameters** *(List[np.ndarray])* - Parameters to optimize.
* **lr** *(float, optional, Default is 1e-4)* - Learning rate.
* **γ** *(float, optional, Default is 0.99)* - Decay rate.
* **eps** *(float, optional, Default is 1e-08)* - Epsilon for numerical stability.

**Methods:**

* **step(self, gradients)** - Update parameters using RMSPROP algorithm.


.. code-block:: python

    class NeighborPrediction

Data class for neighbor prediction results.

**Fields:**

* **neighbor_x** *(np.ndarray)* - Neighbor feature values.
* **neighbor_y** *(np.ndarray)* - Neighbor target values.
* **neighbors_xs** *(np.ndarray)* - All neighbor features.
* **neighbors_ys** *(np.ndarray)* - All neighbor targets.
* **query** *(np.ndarray)* - Query point.
* **local_prediction** *(np.ndarray)* - Local prediction.
* **derivative** *(np.ndarray)* - Estimated derivative.
* **prediction_fn** *(Callable)* - Prediction function.
* **intercept** *(Optional[np.ndarray])* - Intercept term.


.. code-block:: python

    class DNNRPrediction

Data class for DNNR prediction results.

**Fields:**

* **query** *(np.ndarray)* - Query point.
* **y_pred** *(np.ndarray)* - Predicted value.
* **neighbor_predictions** *(list[NeighborPrediction])* - Individual neighbor predictions.
* **y_true** *(Optional[np.ndarray])* - True target value. 

**Referencses:**

Youssef Nader, Leon Sixt, Tim Landgraf. **DNNR: Differential Nearest Neighbors Regression**. In *Proceedings of the 39th International Conference on Machine Learning*, 2022. `<https://proceedings.mlr.press/v162/nader22a/nader22a.pdf>`_
