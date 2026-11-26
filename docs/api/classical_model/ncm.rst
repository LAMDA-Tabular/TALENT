====================================
Nearest Centroid Method
====================================

Nearest Centroid Method classical method implementation.

This section contains the Nearest Centroid Method (NCM) implementation for classification tasks. NCM is a classification algorithm that assigns to samples the label of the class of training samples whose mean (centroid) is closest to the sample.

.. automodule:: TALENT.model.classical_methods.ncm
   :members:
   :undoc-members:
   :show-inheritance:

.. class:: NCMMethod
   :noindex:

   Nearest Centroid Method for classification tasks.
   
   **Key Features:**
   
   - Uses sklearn's NearestCentroid for classification
   - Assigns labels based on closest class centroid
   - Supports both binary and multiclass classification
   - Automatically handles data preprocessing including normalization and encoding
   - Saves trained model to pickle file for later use
   - Simple and interpretable classification method
   
   **Algorithm:**
   
   The Nearest Centroid Method assigns to samples the label of the class of training samples whose mean (centroid) is closest to the sample. It is a simple classification algorithm that works well when classes are well-separated.

   .. method:: __init__(args, is_regression)
      :noindex:
      
      Initialize the NCM method.
      
      **Parameters:**
      
      * **args** (*object*) -- Configuration arguments containing model settings
      * **is_regression** (*bool*) -- Whether the task is regression (True) or classification (False)

   .. method:: construct_model(model_config=None)
      :noindex:
      
      Construct the NCM model instance.
      
      **Parameters:**
      
      * **model_config** (*dict, optional*) -- Model configuration parameters for NCM
      
      **Model Creation:**
      
      - Creates `NearestCentroid` classifier
      - Uses the default Euclidean metric.

   .. method:: fit(data, info, train=True, config=None)
      :noindex:
      
      Train the NCM model on the provided data.
      
      **Parameters:**
      
      * **data** (*tuple*) -- Tuple containing (N, C, y) where N is numerical features, C is categorical features, y is labels
      * **info** (*dict*) -- Dataset information
      * **train** (*bool, default=True*) -- Whether to train the model or just load from checkpoint
      * **config** (*dict, optional*) -- Additional configuration parameters
      
      **Returns:**
      
      * **time_cost** (*float*) -- Training time in seconds
      
      **Training Process:**
      
      1. **Data Preprocessing:** Handles missing values, categorical encoding, normalization
      2. **Model Training:** Computes class centroids from training data
      3. **Model Saving:** Saves the trained model to disk for later use

   .. method:: predict(data, info, model_name)
      :noindex:
      
      Make predictions using the trained NCM model.
      
      **Parameters:**
      
      * **data** (*tuple*) -- Tuple containing (N, C, y) where N is numerical features, C is categorical features, y is labels
      * **info** (*dict*) -- Dataset information
      * **model_name** (*str*) -- Name of the model for saving/loading
      
      **Returns:**
      
      A tuple ``(metric_values, metric_names, predictions)``. The predictions contain class probabilities.
      
      **Prediction Process:**
      
      1. **Data Preprocessing:** Applies same preprocessing as training data
      2. **Model Loading:** Loads the trained NCM model
      3. **Prediction:** Finds closest centroid for each sample
      4. **Output:** Returns class probabilities computed from negative centroid distances

**Evaluation Metrics:**

- **For classification:** returns Accuracy, Avg_Recall, Avg_Precision, F1, LogLoss, AUC, Brier and ECE

**References:**

``[1] Hastie, T., Tibshirani, R., & Friedman, J. (2009). The elements of statistical learning: data mining, inference, and prediction. Springer Science & Business Media.`` 