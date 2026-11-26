====================================
Dummy Baseline
====================================

Dummy Classifier classical method implementation.

The dummy baseline uses class frequencies for classification and the training mean for regression by default.

.. automodule:: TALENT.model.classical_methods.dummy
   :members:
   :undoc-members:
   :show-inheritance:

.. class:: DummyMethod
   :noindex:

   Baseline method for classification and regression.
   
   **Key Features:**
   
   - Uses scikit-learn's DummyClassifier and DummyRegressor
   - Provides baseline predictions for comparison
   - Supports both binary and multiclass classification
   - Automatically handles data preprocessing including normalization and encoding
   - Saves trained model to pickle file for later use
   - Simple baseline classifier
   
   **Algorithm:**
   
   Dummy Classifier is a classifier that makes predictions using simple rules. It is useful as a baseline for comparison with more sophisticated classifiers. It can use various strategies like most frequent, stratified, uniform, constant, etc.

   .. method:: __init__(args, is_regression)
      :noindex:
      
      Initialize the Dummy Classifier method.
      
      **Parameters:**
      
      * **args** (*object*) -- Configuration arguments containing model settings
      * **is_regression** (*bool*) -- Whether the task is regression (True) or classification (False)

   .. method:: construct_model(model_config=None)
      :noindex:
      
      Construct the Dummy Classifier model instance.
      
      **Parameters:**
      
      * **model_config** (*dict, optional*) -- Model configuration parameters for Dummy Classifier
      
      **Model Creation:**
      
      - Creates a classifier or regressor for the dataset task
      - Configures parameters like strategy, random_state, etc.

   .. method:: fit(data, info, train=True, config=None)
      :noindex:
      
      Train the Dummy Classifier model on the provided data.
      
      **Parameters:**
      
      * **data** (*tuple*) -- Tuple containing (N, C, y) where N is numerical features, C is categorical features, y is labels
      * **info** (*dict*) -- Dataset information
      * **train** (*bool, default=True*) -- Whether to train the model or just load from checkpoint
      * **config** (*dict, optional*) -- Additional configuration parameters
      
      **Returns:**
      
      * **time_cost** (*float*) -- Training time in seconds
      
      **Training Process:**
      
      1. **Data Preprocessing:** Handles missing values, categorical encoding, normalization
      2. **Model Training:** Fits the Dummy Classifier model
      3. **Model Saving:** Saves the trained model to disk for later use

   .. method:: predict(data, info, model_name)
      :noindex:
      
      Make predictions using the trained Dummy Classifier model.
      
      **Parameters:**
      
      * **data** (*tuple*) -- Tuple containing (N, C, y) where N is numerical features, C is categorical features, y is labels
      * **info** (*dict*) -- Dataset information
      * **model_name** (*str*) -- Name of the model for saving/loading
      
      **Returns:**
      
      A tuple ``(metric_values, metric_names, predictions)``. The predictions contain class probabilities or regression values in the original target units.
      
      **Prediction Process:**
      
      1. **Data Preprocessing:** Applies same preprocessing as training data
      2. **Model Loading:** Loads the trained Dummy Classifier model
      3. **Prediction:** Generates baseline predictions using simple rules
      4. **Output:** Returns class probabilities or predicted target values

**Evaluation Metrics:**

- **For regression:** returns MAE, R2 and RMSE
- **For classification:** returns Accuracy, Avg_Recall, Avg_Precision, F1, LogLoss, AUC, Brier and ECE

**References:**

``[1] scikit-learn developers. (2023). DummyClassifier. scikit-learn documentation.`` 