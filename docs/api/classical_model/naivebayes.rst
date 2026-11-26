====================================
Naive Bayes
====================================

Naive Bayes classical method implementation.

This section contains the Naive Bayes implementation for classification tasks. Naive Bayes is a probabilistic classifier based on applying Bayes' theorem with the "naive" assumption of conditional independence between every pair of features.

.. automodule:: TALENT.model.classical_methods.naivebayes
   :members:
   :undoc-members:
   :show-inheritance:

.. class:: NaiveBayesMethod
   :noindex:

   Naive Bayes method for classification tasks.
   
   **Key Features:**
   
   - Uses sklearn's GaussianNB for classification
   - Probabilistic classifier based on Bayes' theorem
   - Supports both binary and multiclass classification
   - Automatically handles data preprocessing including normalization and encoding
   - Saves trained model to pickle file for later use
   - Provides probability predictions
   
   **Algorithm:**
   
   Naive Bayes is a probabilistic classifier based on applying Bayes' theorem with the "naive" assumption of conditional independence between every pair of features given the value of the class variable.

   .. method:: __init__(args, is_regression)
      :noindex:
      
      Initialize the Naive Bayes method.
      
      **Parameters:**
      
      * **args** (*object*) -- Configuration arguments containing model settings
      * **is_regression** (*bool*) -- Whether the task is regression (True) or classification (False)

   .. method:: construct_model(model_config=None)
      :noindex:
      
      Construct the Naive Bayes model instance.
      
      **Parameters:**
      
      * **model_config** (*dict, optional*) -- Model configuration parameters for Naive Bayes
      
      **Model Creation:**
      
      - Creates `GaussianNB` classifier
      - Uses the GaussianNB estimator defaults.

   .. method:: fit(data, info, train=True, config=None)
      :noindex:
      
      Train the Naive Bayes model on the provided data.
      
      **Parameters:**
      
      * **data** (*tuple*) -- Tuple containing (N, C, y) where N is numerical features, C is categorical features, y is labels
      * **info** (*dict*) -- Dataset information
      * **train** (*bool, default=True*) -- Whether to train the model or just load from checkpoint
      * **config** (*dict, optional*) -- Additional configuration parameters
      
      **Returns:**
      
      * **time_cost** (*float*) -- Training time in seconds
      
      **Training Process:**
      
      1. **Data Preprocessing:** Handles missing values, categorical encoding, normalization
      2. **Model Training:** Fits the Naive Bayes model
      3. **Model Saving:** Saves the trained model to disk for later use

   .. method:: predict(data, info, model_name)
      :noindex:
      
      Make predictions using the trained Naive Bayes model.
      
      **Parameters:**
      
      * **data** (*tuple*) -- Tuple containing (N, C, y) where N is numerical features, C is categorical features, y is labels
      * **info** (*dict*) -- Dataset information
      * **model_name** (*str*) -- Name of the model for saving/loading
      
      **Returns:**
      
      A tuple ``(metric_values, metric_names, predictions)``. The predictions contain class probabilities.
      
      **Prediction Process:**
      
      1. **Data Preprocessing:** Applies same preprocessing as training data
      2. **Model Loading:** Loads the trained Naive Bayes model
      3. **Prediction:** Generates probability predictions using Bayes' theorem
      4. **Output:** Returns probabilities for classification

**Evaluation Metrics:**

- **For classification:** returns Accuracy, Avg_Recall, Avg_Precision, F1, LogLoss, AUC, Brier and ECE

**References:**

``[1] Rish, I. (2001). An empirical study of the naive Bayes classifier. In IJCAI 2001 workshop on empirical methods in artificial intelligence (Vol. 3, No. 22, pp. 41-46).`` 