====================================
Logistic Regression
====================================

Logistic Regression classical method implementation.

This section contains the Logistic Regression implementation for classification tasks. Logistic Regression is a linear model for classification that uses a logistic function to model the probability of a certain class or event.

.. automodule:: TALENT.model.classical_methods.logreg
   :members:
   :undoc-members:
   :show-inheritance:

.. class:: LogRegMethod
   :noindex:

   Logistic Regression method for classification tasks.
   
   **Key Features:**
   
   - Uses sklearn's LogisticRegression for classification
   - Linear model with logistic function for probability estimation
   - Supports both binary and multiclass classification
   - Automatically handles data preprocessing including normalization and encoding
   - Saves trained model to pickle file for later use
   - Provides probability predictions
   
   **Algorithm:**
   
   Logistic Regression is a linear model for classification that uses a logistic function to model the probability of a certain class or event. It is a special case of linear regression where the dependent variable is categorical.

   .. method:: __init__(args, is_regression)
      :noindex:
      
      Initialize the Logistic Regression method.
      
      **Parameters:**
      
      * **args** (*object*) -- Configuration arguments containing model settings
      * **is_regression** (*bool*) -- Whether the task is regression (True) or classification (False)

   .. method:: construct_model(model_config=None)
      :noindex:
      
      Construct the Logistic Regression model instance.
      
      **Parameters:**
      
      * **model_config** (*dict, optional*) -- Model configuration parameters for Logistic Regression
      
      **Model Creation:**
      
      - Creates `LogisticRegression` classifier
      - Configures parameters like regularization, solver, etc.

   .. method:: fit(data, info, train=True, config=None)
      :noindex:
      
      Train the Logistic Regression model on the provided data.
      
      **Parameters:**
      
      * **data** (*tuple*) -- Tuple containing (N, C, y) where N is numerical features, C is categorical features, y is labels
      * **info** (*dict*) -- Dataset information
      * **train** (*bool, default=True*) -- Whether to train the model or just load from checkpoint
      * **config** (*dict, optional*) -- Additional configuration parameters
      
      **Returns:**
      
      * **time_cost** (*float*) -- Training time in seconds
      
      **Training Process:**
      
      1. **Data Preprocessing:** Handles missing values, categorical encoding, normalization
      2. **Model Training:** Fits the Logistic Regression model
      3. **Model Saving:** Saves the trained model to disk for later use

   .. method:: predict(data, info, model_name)
      :noindex:
      
      Make predictions using the trained Logistic Regression model.
      
      **Parameters:**
      
      * **data** (*tuple*) -- Tuple containing (N, C, y) where N is numerical features, C is categorical features, y is labels
      * **info** (*dict*) -- Dataset information
      * **model_name** (*str*) -- Name of the model for saving/loading
      
      **Returns:**
      
      A tuple ``(metric_values, metric_names, predictions)``. The predictions contain class probabilities.
      
      **Prediction Process:**
      
      1. **Data Preprocessing:** Applies same preprocessing as training data
      2. **Model Loading:** Loads the trained Logistic Regression model
      3. **Prediction:** Generates probability predictions
      4. **Output:** Returns probabilities for classification

**Evaluation Metrics:**

- **For classification:** returns Accuracy, Avg_Recall, Avg_Precision, F1, LogLoss, AUC, Brier and ECE

**References:**

``[1] Hosmer Jr, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). Applied logistic regression (Vol. 398). John Wiley & Sons.`` 