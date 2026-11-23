Method Base
===========

.. automodule:: TALENT.model.methods.base
   :members:
   :undoc-members:
   :show-inheritance:

Utility Functions
-----------------

.. autofunction:: TALENT.model.utils.check_softmax
   :noindex:

Core Method Class
-----------------

.. class:: Method
   :noindex:

   Base class for TALENT's deep method adapters.
   
   This class provides a unified interface for training, validation, and prediction
   for deep method adapters. Classical methods have their own base class.
   
   **Attributes:**
   
   * **args** (*argparse.Namespace*) -- Command line arguments and configuration
   * **is_regression** (*bool*) -- Whether the task is regression
   * **D** (*Dataset*) -- Dataset object containing features and labels
   * **train_step** (*int*) -- Current training step counter
   * **val_count** (*int*) -- Counter for validation without improvement
   * **continue_training** (*bool*) -- Whether to continue training
   * **timer** (*Timer*) -- Timer for tracking training time
   * **trlog** (*dict*) -- Training log containing loss, best results, etc.
   * **model** (*torch.nn.Module*) -- The neural network model (to be implemented by subclasses)
   * **optimizer** (*torch.optim.Optimizer*) -- Optimizer for training
   * **criterion** (*callable*) -- Loss function
   
   **Methods:**
   
   .. method:: __init__(args, is_regression)
      :noindex:
      
      Initialize the method with arguments and task type.
      
      **Parameters:**
      
      * **args** (*argparse.Namespace*) -- Command line arguments and configuration
      * **is_regression** (*bool*) -- Whether the task is regression
      
      **Initialization:**
      
      * Sets up training statistics and logging
      * Initializes device (CPU/GPU)
      * Sets up training log with appropriate best result tracking
   
   .. method:: reset_stats_withconfig(config)
      :noindex:
      
      Reset training statistics with a new configuration.
      
      **Parameters:**
      
      * **config** (*dict*) -- New configuration dictionary
      
      **Actions:**
      
      * Resets random seeds for reproducibility
      * Clears training step counter and validation counter
      * Resets training log with new configuration
      * Reinitializes timer
   
   .. method:: data_format(is_train=True, N=None, C=None, y=None)
      :noindex:
      
      Format and preprocess data for training or testing.
      
      **Parameters:**
      
      * **is_train** (*bool, optional*) -- Whether data is for training. Defaults to True.
      * **N** (*dict, optional*) -- Numerical features dictionary. Defaults to None.
      * **C** (*dict, optional*) -- Categorical features dictionary. Defaults to None.
      * **y** (*dict, optional*) -- Target labels dictionary. Defaults to None.
      
      **Processing Pipeline:**
      
      * **Training Mode:**
        * Handle missing values (NaN processing)
        * Process labels (standardization for regression, encoding for classification)
        * Apply numerical feature encoding (PLE, Unary, etc.)
        * Apply categorical feature encoding (ordinal, one-hot, etc.)
        * Apply normalization to numerical features
        * Create DataLoaders for training and validation
        * Set up loss function
      
      * **Testing Mode:**
        * Apply same preprocessing using fitted encoders and normalizers
        * Create DataLoader for testing
        * Prepare test data tensors
   
   .. method:: fit(data, info, train=True, config=None)
      :noindex:
      
      Fit the method to the training data.
      
      **Parameters:**
      
      * **data** (*tuple*) -- Tuple of (N, C, y) where N=numerical, C=categorical, y=labels
      * **info** (*dict*) -- Dataset information including task type and feature counts
      * **train** (*bool, optional*) -- Whether to train the model. Defaults to True.
      * **config** (*dict, optional*) -- Configuration dictionary. Defaults to None.
      
      **Returns:**
      
      * **float** -- Total training time in seconds
      
      **Training Process:**
      
      * Initialize dataset and extract features
      * Format data for training
      * Construct model (implemented by subclasses)
      * Set up optimizer (AdamW)
      * Train for specified number of epochs
      * Save best model and training log
   
   .. method:: predict(data, info, model_name)
      :noindex:
      
      Make predictions on test data.
      
      **Parameters:**
      
      * **data** (*tuple*) -- Tuple of (N, C, y) test data
      * **info** (*dict*) -- Dataset information
      * **model_name** (*str*) -- Name of the saved model file
      
      **Returns:**
      
      * **tuple** -- (loss, metrics, metric_names, predictions) where:
        * loss: Test loss value
        * metrics: List of evaluation metrics
        * metric_names: Names of the metrics
        * predictions: Model predictions
      
      **Prediction Process:**
      
      * Load trained model weights
      * Format test data using fitted preprocessors
      * Run inference on test set
      * Compute evaluation metrics
      * Return results
   
   .. method:: train_epoch(epoch)
      :noindex:
      
      Train the model for one epoch.
      
      **Parameters:**
      
      * **epoch** (*int*) -- Current epoch number
      
      **Training Loop:**
      
      * Set model to training mode
      * Iterate through training batches
      * Forward pass and compute loss
      * Backward pass and update weights
      * Log training progress
      * Update training statistics
   
   .. method:: validate(epoch)
      :noindex:
      
      Validate the model on validation set.
      
      **Parameters:**
      
      * **epoch** (*int*) -- Current epoch number
      
      **Validation Process:**
      
      * Set model to evaluation mode
      * Run inference on validation set
      * Compute validation metrics
      * Check for improvement
      * Save best model if improved
      * Implement early stopping (20 epochs without improvement)
      * Save training log
   
   .. method:: metric(predictions, labels, y_info)
      :noindex:
      
      Compute evaluation metrics based on task type.
      
      **Parameters:**
      
      * **predictions** (*np.ndarray*) -- Model predictions
      * **labels** (*np.ndarray*) -- Ground truth labels
      * **y_info** (*dict*) -- Label information including processing policy
      
      **Returns:**
      
      * **tuple** -- (metrics, metric_names) where:
        * metrics: List of computed metric values
        * metric_names: Names of the metrics
      
      **Metrics by Task Type:**
      
      * **Regression:**
        * MAE (Mean Absolute Error)
        * R² (Coefficient of determination)
        * RMSE (Root Mean Squared Error)
      
      * **Binary Classification:**
        * Accuracy
        * Balanced Recall
        * Macro Precision
        * F1 Score
        * Log Loss
        * AUC (Area Under ROC Curve)
      
      * **Multi-class Classification:**
        * Accuracy
        * Balanced Recall
        * Macro Precision
        * Macro F1 Score
        * Log Loss
        * Macro AUC (One-vs-Rest)

Abstract Methods
----------------

The following methods must be implemented by subclasses:

.. method:: construct_model()
   :noindex:
   
   Construct the neural network model architecture.
   
   **Implementation Required:**
   
   Subclasses must implement this method to create their specific model architecture.
   The model should be assigned to `self.model` and should accept numerical and categorical
   features as separate inputs.
   
   **Expected Model Interface:**
   
   .. code-block:: python
      
      def forward(self, X_num, X_cat):
          # X_num: numerical features tensor or None
          # X_cat: categorical features tensor or None
          # Return: predictions tensor
          pass

Usage Example
-------------

.. code-block:: python
   
   from TALENT.model.methods.base import Method
   import torch.nn as nn
   
   class MyModel(Method):
       def construct_model(self):
           # Define your model architecture
           self.model = nn.Sequential(
               nn.Linear(self.d_in, 128),
               nn.ReLU(),
               nn.Linear(128, self.d_out)
           )
   
   # Usage
   method = MyModel(args, is_regression=True)
   time_cost = method.fit(train_data, info)
   loss, metrics, metric_names, predictions = method.predict(test_data, info, 'best-val')
