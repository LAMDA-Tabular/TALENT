====================================
Support Vector Machine
====================================

Support Vector Machine classical method implementation.

This section contains the Support Vector Machine (SVM) implementation for classification and regression.

.. automodule:: TALENT.model.classical_methods.svm
   :members:
   :undoc-members:
   :show-inheritance:

.. class:: SvmMethod
   :noindex:

   Linear support vector classifier or regressor.
   
   **Key Features:**
   
   - Uses LinearSVC with three-fold sigmoid calibration for classification and LinearSVR for regression
   - Finds optimal hyperplane for class separation
   - Supports both binary and multiclass classification
   - Automatically handles data preprocessing including normalization and encoding
   - Saves trained model to pickle file for later use
   - Provides probability predictions
   
   **Algorithm:**
   
   SVM is a supervised learning algorithm that finds a hyperplane to separate data points of different classes with maximum margin. The TALENT adapter uses a linear kernel.

   .. method:: __init__(args, is_regression)
      :noindex:
      
      Initialize the SVM method.
      
      **Parameters:**
      
      * **args** (*object*) -- Configuration arguments containing model settings
      * **is_regression** (*bool*) -- Whether the task is regression (True) or classification (False)

   .. method:: construct_model(model_config=None)
      :noindex:
      
      Construct the SVM model instance.
      
      **Parameters:**
      
      * **model_config** (*dict, optional*) -- Model configuration parameters for SVM
      
      **Model Creation:**
      
      - Creates a calibrated LinearSVC or a LinearSVR
      - Passes the configured parameters to the underlying linear estimator.

   .. method:: fit(data, info, train=True, config=None)
      :noindex:
      
      Train the SVM model on the provided data.
      
      **Parameters:**
      
      * **data** (*tuple*) -- Tuple containing (N, C, y) where N is numerical features, C is categorical features, y is labels
      * **info** (*dict*) -- Dataset information
      * **train** (*bool, default=True*) -- Whether to train the model or just load from checkpoint
      * **config** (*dict, optional*) -- Additional configuration parameters
      
      **Returns:**
      
      * **time_cost** (*float*) -- Training time in seconds
      
      **Training Process:**
      
      1. **Data Preprocessing:** Handles missing values, categorical encoding, normalization
      2. **Model Training:** Fits the SVM model with optimal hyperplane
      3. **Model Saving:** Saves the trained model to disk for later use

   .. method:: predict(data, info, model_name)
      :noindex:
      
      Make predictions using the trained SVM model.
      
      **Parameters:**
      
      * **data** (*tuple*) -- Tuple containing (N, C, y) where N is numerical features, C is categorical features, y is labels
      * **info** (*dict*) -- Dataset information
      * **model_name** (*str*) -- Name of the model for saving/loading
      
      **Returns:**
      
      A tuple ``(metric_values, metric_names, predictions)``. The predictions contain class probabilities or regression values in the original target units.
      
      **Prediction Process:**
      
      1. **Data Preprocessing:** Applies same preprocessing as training data
      2. **Model Loading:** Loads the trained SVM model
      3. **Prediction:** Generates probability predictions
      4. **Output:** Returns class probabilities or predicted target values

**Evaluation Metrics:**

- **For regression:** returns MAE, R2 and RMSE
- **For classification:** returns Accuracy, Avg_Recall, Avg_Precision, F1, LogLoss, AUC, Brier and ECE

**References:**

``[1] Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine learning, 20(3), 273-297.`` 