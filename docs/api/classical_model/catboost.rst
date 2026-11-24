====================================
CatBoost
====================================

CatBoost classical method implementation.

This section contains the CatBoost implementation for classification and regression tasks. CatBoost is a gradient boosting algorithm that handles categorical features automatically and provides high performance with minimal parameter tuning.

.. automodule:: TALENT.model.classical_methods.catboost
   :members:
   :undoc-members:
   :show-inheritance:

.. class:: CatBoostMethod
   :noindex:

   CatBoost method for classification and regression tasks using gradient boosting with categorical features support.
   
   **Key Features:**
   
   - Uses CatBoost library for gradient boosting implementation
   - Automatically handles categorical features without preprocessing
   - Supports both classification and regression tasks
   - Provides feature importance analysis
   - Robust to overfitting with built-in regularization
   - Efficient implementation with GPU support
   
   **Algorithm:**
   
   CatBoost is a gradient boosting algorithm that uses ordered boosting and innovative algorithms for processing categorical features, which helps reduce overfitting and improve prediction quality.

   .. method:: __init__(args, is_regression)
      :noindex:
      
      Initialize the CatBoost method.
      
      **Parameters:**
      
      * **args** (*object*) -- Configuration arguments containing model settings
      * **is_regression** (*bool*) -- Whether the task is regression (True) or classification (False)

   .. method:: construct_model(model_config=None)
      :noindex:
      
      Construct the CatBoost model instance.
      
      **Parameters:**
      
      * **model_config** (*dict, optional*) -- Model configuration parameters for CatBoost
      
      **Model Creation:**
      
      - For classification: creates `CatBoostClassifier`
      - For regression: creates `CatBoostRegressor`
      - Configures boosting parameters like learning rate, depth, etc.

   .. method:: fit(data, info, train=True, config=None)
      :noindex:
      
      Train the CatBoost model on the provided data.
      
      **Parameters:**
      
      * **data** (*tuple*) -- Tuple containing (N, C, y) where N is numerical features, C is categorical features, y is labels
      * **info** (*dict*) -- Dataset information
      * **train** (*bool, default=True*) -- Whether to train the model or just load from checkpoint
      * **config** (*dict, optional*) -- Additional configuration parameters
      
      **Returns:**
      
      * **time_cost** (*float*) -- Training time in seconds
      
      **Training Process:**
      
      1. **Data Preprocessing:** Handles missing values, categorical encoding, normalization
      2. **Model Training:** Fits the CatBoost model with gradient boosting
      3. **Model Saving:** Saves the trained model to disk for later use

   .. method:: predict(data, info, model_name)
      :noindex:
      
      Make predictions using the trained CatBoost model.
      
      **Parameters:**
      
      * **data** (*tuple*) -- Tuple containing (N, C, y) where N is numerical features, C is categorical features, y is labels
      * **info** (*dict*) -- Dataset information
      * **model_name** (*str*) -- Name of the model for saving/loading
      
      **Returns:**
      
      A tuple ``(metric_values, metric_names, predictions)``. The predictions contain class probabilities or regression values in the original target units.
      
      **Prediction Process:**
      
      1. **Data Preprocessing:** Applies same preprocessing as training data
      2. **Model Loading:** Loads the trained CatBoost model
      3. **Prediction:** Generates predictions using the gradient boosting model
      4. **Output:** Returns probabilities for classification or values for regression

**Evaluation Metrics:**

- **For regression:** returns MAE, R2, RMSE metrics
- **For classification:** returns Accuracy, Avg_Recall, Avg_Precision, F1, LogLoss, AUC, Brier and ECE

**References:**

``[1] Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: unbiased boosting with categorical features. Advances in neural information processing systems, 31.`` 