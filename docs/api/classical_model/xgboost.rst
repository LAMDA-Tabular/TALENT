====================================
XGBoost
====================================

XGBoost classical method implementation.

This section contains the XGBoost implementation for classification and regression tasks. XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable.

.. automodule:: TALENT.model.classical_methods.xgboost
   :members:
   :undoc-members:
   :show-inheritance:

.. class:: XGBoostMethod
   :noindex:

   XGBoost method for classification and regression tasks using gradient boosting.
   
   **Key Features:**
   
   - Uses XGBoost library for gradient boosting implementation
   - Supports both classification and regression tasks
   - Handles missing values automatically
   - Provides feature importance analysis
   - Supports early stopping to prevent overfitting
   - Efficient implementation with parallel processing
   
   **Algorithm:**
   
   XGBoost is an implementation of gradient boosted decision trees designed for speed and performance. It uses a more regularized model formalization to control overfitting.

   .. method:: __init__(args, is_regression)
      :noindex:
      
      Initialize the XGBoost method.
      
      **Parameters:**
      
      * **args** (*object*) -- Configuration arguments containing model settings
      * **is_regression** (*bool*) -- Whether the task is regression (True) or classification (False)

   .. method:: construct_model(model_config=None)
      :noindex:
      
      Construct the XGBoost model instance.
      
      **Parameters:**
      
      * **model_config** (*dict, optional*) -- Model configuration parameters for XGBoost
      
      **Model Creation:**
      
      - For classification: creates `XGBClassifier`
      - For regression: creates `XGBRegressor`
      - Configures boosting parameters like learning rate, max depth, etc.

   .. method:: fit(data, info, train=True, config=None)
      :noindex:
      
      Train the XGBoost model on the provided data.
      
      **Parameters:**
      
      * **data** (*tuple*) -- Tuple containing (N, C, y) where N is numerical features, C is categorical features, y is labels
      * **info** (*dict*) -- Dataset information
      * **train** (*bool, default=True*) -- Whether to train the model or just load from checkpoint
      * **config** (*dict, optional*) -- Additional configuration parameters
      
      **Returns:**
      
      * **time_cost** (*float*) -- Training time in seconds
      
      **Training Process:**
      
      1. **Data Preprocessing:** Handles missing values, categorical encoding, normalization
      2. **Model Training:** Fits the XGBoost model with gradient boosting
      3. **Model Saving:** Saves the trained model to disk for later use

   .. method:: predict(data, info, model_name)
      :noindex:
      
      Make predictions using the trained XGBoost model.
      
      **Parameters:**
      
      * **data** (*tuple*) -- Tuple containing (N, C, y) where N is numerical features, C is categorical features, y is labels
      * **info** (*dict*) -- Dataset information
      * **model_name** (*str*) -- Name of the model for saving/loading
      
      **Returns:**
      
      A tuple ``(metric_values, metric_names, predictions)``. The predictions contain class probabilities or regression values in the original target units.
      
      **Prediction Process:**
      
      1. **Data Preprocessing:** Applies same preprocessing as training data
      2. **Model Loading:** Loads the trained XGBoost model
      3. **Prediction:** Generates predictions using the gradient boosting model
      4. **Output:** Returns probabilities for classification or values for regression

**Evaluation Metrics:**

- **For regression:** returns MAE, R2, RMSE metrics
- **For classification:** returns Accuracy, Avg_Recall, Avg_Precision, F1, LogLoss, AUC, Brier and ECE

**References:**

``[1] Chen, T., & Guestrin, C. (2016). Xgboost: A scalable tree boosting system. In Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining (pp. 785-794).`` 