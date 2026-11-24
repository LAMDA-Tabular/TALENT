====================================
K-Nearest Neighbors
====================================

K-Nearest Neighbors classical method implementation.

This section contains the K-Nearest Neighbors (KNN) implementation for classification and regression tasks. KNN is a non-parametric method used for classification and regression that makes predictions based on the similarity of input to the training data.

.. automodule:: TALENT.model.classical_methods.knn
   :members:
   :undoc-members:
   :show-inheritance:

.. class:: KnnMethod
   :noindex:

   K-Nearest Neighbors method for classification and regression tasks.
   
   **Key Features:**
   
   - Uses sklearn's KNeighborsClassifier for classification and KNeighborsRegressor for regression
   - Non-parametric method that makes predictions based on similarity
   - Supports both classification and regression tasks
   - Automatically handles data preprocessing including normalization and encoding
   - Saves trained model to pickle file for later use
   - Instance-based learning approach
   
   **Algorithm:**
   
   KNN is a non-parametric method used for classification and regression. The input consists of the k closest training examples in the feature space. The output depends on whether KNN is used for classification or regression.

   .. method:: __init__(args, is_regression)
      :noindex:
      
      Initialize the KNN method.
      
      **Parameters:**
      
      * **args** (*object*) -- Configuration arguments containing model settings
      * **is_regression** (*bool*) -- Whether the task is regression (True) or classification (False)

   .. method:: construct_model(model_config=None)
      :noindex:
      
      Construct the KNN model instance.
      
      **Parameters:**
      
      * **model_config** (*dict, optional*) -- Model configuration parameters for KNN
      
      **Model Creation:**
      
      - For classification: creates `KNeighborsClassifier`
      - For regression: creates `KNeighborsRegressor`
      - Configures parameters like number of neighbors, distance metric, etc.

   .. method:: fit(data, info, train=True, config=None)
      :noindex:
      
      Train the KNN model on the provided data.
      
      **Parameters:**
      
      * **data** (*tuple*) -- Tuple containing (N, C, y) where N is numerical features, C is categorical features, y is labels
      * **info** (*dict*) -- Dataset information
      * **train** (*bool, default=True*) -- Whether to train the model or just load from checkpoint
      * **config** (*dict, optional*) -- Additional configuration parameters
      
      **Returns:**
      
      * **time_cost** (*float*) -- Training time in seconds
      
      **Training Process:**
      
      1. **Data Preprocessing:** Handles missing values, categorical encoding, normalization
      2. **Model Training:** Stores training data for nearest neighbor search
      3. **Model Saving:** Saves the trained model to disk for later use

   .. method:: predict(data, info, model_name)
      :noindex:
      
      Make predictions using the trained KNN model.
      
      **Parameters:**
      
      * **data** (*tuple*) -- Tuple containing (N, C, y) where N is numerical features, C is categorical features, y is labels
      * **info** (*dict*) -- Dataset information
      * **model_name** (*str*) -- Name of the model for saving/loading
      
      **Returns:**
      
      A tuple ``(metric_values, metric_names, predictions)``. The predictions contain class probabilities or regression values in the original target units.
      
      **Prediction Process:**
      
      1. **Data Preprocessing:** Applies same preprocessing as training data
      2. **Model Loading:** Loads the trained KNN model
      3. **Prediction:** Finds k nearest neighbors and makes prediction
      4. **Output:** Returns probabilities for classification or values for regression

**Evaluation Metrics:**

- **For regression:** returns MAE, R2, RMSE metrics
- **For classification:** returns Accuracy, Avg_Recall, Avg_Precision, F1, LogLoss, AUC, Brier and ECE

**References:**

``[1] Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. IEEE transactions on information theory, 13(1), 21-27.`` 