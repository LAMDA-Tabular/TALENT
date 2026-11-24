====================================
Random Forest
====================================

Random Forest classical method implementation.

This section contains the Random Forest implementation for classification and regression tasks using ensemble of decision trees. Random Forest is an ensemble learning method that operates by constructing multiple decision trees and outputting the class that is the mode of the classes predicted by individual trees.

.. automodule:: TALENT.model.classical_methods.randomforest
   :members:
   :undoc-members:
   :show-inheritance:

.. class:: RandomForestMethod
   :noindex:

   Random Forest method for classification and regression tasks using ensemble of decision trees.
   
   **Key Features:**
   
   - Uses sklearn's RandomForestClassifier for classification and RandomForestRegressor for regression
   - Inherits from KnnMethod class for common functionality
   - Supports both binary and multiclass classification
   - Automatically handles data preprocessing including normalization and encoding
   - Saves trained model to pickle file for later use
   - Ensemble method that combines multiple decision trees
   
   **Algorithm:**
   
   Random Forest builds multiple decision trees during training and outputs the class that is the mode of the classes predicted by individual trees for classification, or the mean prediction for regression.

   .. method:: __init__(args, is_regression)
      :noindex:
      
      Initialize the Random Forest method.
      
      **Parameters:**
      
      * **args** (*object*) -- Configuration arguments containing model settings
      * **is_regression** (*bool*) -- Whether the task is regression (True) or classification (False)

   .. method:: construct_model(model_config=None)
      :noindex:
      
      Construct the Random Forest model instance.
      
      **Parameters:**
      
      * **model_config** (*dict, optional*) -- Model configuration parameters for Random Forest
      
      **Model Creation:**
      
      - For classification: creates `RandomForestClassifier`
      - For regression: creates `RandomForestRegressor`
      - Configures ensemble parameters like number of trees, max depth, etc.

   .. method:: fit(data, info, train=True, config=None)
      :noindex:
      
      Train the Random Forest model on the provided data.
      
      **Parameters:**
      
      * **data** (*tuple*) -- Tuple containing (N, C, y) where N is numerical features, C is categorical features, y is labels
      * **info** (*dict*) -- Dataset information
      * **train** (*bool, default=True*) -- Whether to train the model or just load from checkpoint
      * **config** (*dict, optional*) -- Additional configuration parameters
      
      **Returns:**
      
      * **time_cost** (*float*) -- Training time in seconds
      
      **Training Process:**
      
      1. **Data Preprocessing:** Handles missing values, categorical encoding, normalization
      2. **Model Training:** Fits the Random Forest ensemble to the training data
      3. **Model Saving:** Saves the trained model to disk for later use

   .. method:: predict(data, info, model_name)
      :noindex:
      
      Make predictions using the trained Random Forest model.
      
      **Parameters:**
      
      * **data** (*tuple*) -- Tuple containing (N, C, y) where N is numerical features, C is categorical features, y is labels
      * **info** (*dict*) -- Dataset information
      * **model_name** (*str*) -- Name of the model for saving/loading
      
      **Returns:**
      
      A tuple ``(metric_values, metric_names, predictions)``. The predictions contain class probabilities or regression values in the original target units.
      
      **Prediction Process:**
      
      1. **Data Preprocessing:** Applies same preprocessing as training data
      2. **Model Loading:** Loads the trained Random Forest model
      3. **Prediction:** Generates predictions using the ensemble
      4. **Output:** Returns probabilities for classification or values for regression

**Evaluation Metrics:**

- **For regression:** returns MAE, R2, RMSE metrics
- **For classification:** returns Accuracy, Avg_Recall, Avg_Precision, F1, LogLoss, AUC, Brier and ECE

**References:**

``[1] Breiman, L. (2001). Random forests. Machine learning, 45(1), 5-32.`` 