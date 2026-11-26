====================================
Linear Regression
====================================

Linear Regression classical method implementation.

This section contains the Linear Regression implementation for regression tasks. Linear Regression is a linear approach to modeling the relationship between a scalar response and one or more explanatory variables.

.. automodule:: TALENT.model.classical_methods.lr
   :members:
   :undoc-members:
   :show-inheritance:

.. class:: LinearRegressionMethod
   :noindex:

   Linear Regression method for regression tasks.
   
   **Key Features:**
   
   - Uses sklearn's LinearRegression for regression
   - Linear model for continuous target prediction
   - Supports multiple explanatory variables
   - Automatically handles data preprocessing including normalization and encoding
   - Saves trained model to pickle file for later use
   - Provides coefficient interpretation
   
   **Algorithm:**
   
   Linear Regression is a linear approach to modeling the relationship between a scalar response and one or more explanatory variables. It assumes a linear relationship between the input variables and the single output variable.

   .. method:: __init__(args, is_regression)
      :noindex:
      
      Initialize the Linear Regression method.
      
      **Parameters:**
      
      * **args** (*object*) -- Configuration arguments containing model settings
      * **is_regression** (*bool*) -- Whether the task is regression (True) or classification (False)

   .. method:: construct_model(model_config=None)
      :noindex:
      
      Construct the Linear Regression model instance.
      
      **Parameters:**
      
      * **model_config** (*dict, optional*) -- Model configuration parameters for Linear Regression
      
      **Model Creation:**
      
      - Creates `LinearRegression` regressor
      - Configures parameters such as fit_intercept.

   .. method:: fit(data, info, train=True, config=None)
      :noindex:
      
      Train the Linear Regression model on the provided data.
      
      **Parameters:**
      
      * **data** (*tuple*) -- Tuple containing (N, C, y) where N is numerical features, C is categorical features, y is labels
      * **info** (*dict*) -- Dataset information
      * **train** (*bool, default=True*) -- Whether to train the model or just load from checkpoint
      * **config** (*dict, optional*) -- Additional configuration parameters
      
      **Returns:**
      
      * **time_cost** (*float*) -- Training time in seconds
      
      **Training Process:**
      
      1. **Data Preprocessing:** Handles missing values, categorical encoding, normalization
      2. **Model Training:** Fits the Linear Regression model
      3. **Model Saving:** Saves the trained model to disk for later use

   .. method:: predict(data, info, model_name)
      :noindex:
      
      Make predictions using the trained Linear Regression model.
      
      **Parameters:**
      
      * **data** (*tuple*) -- Tuple containing (N, C, y) where N is numerical features, C is categorical features, y is labels
      * **info** (*dict*) -- Dataset information
      * **model_name** (*str*) -- Name of the model for saving/loading
      
      **Returns:**
      
      A tuple ``(metric_values, metric_names, predictions)``. The predictions contain values in the original target units.
      
      **Prediction Process:**
      
      1. **Data Preprocessing:** Applies same preprocessing as training data
      2. **Model Loading:** Loads the trained Linear Regression model
      3. **Prediction:** Generates continuous value predictions
      4. **Output:** Returns predicted values for regression

**Evaluation Metrics:**

- **For regression:** returns MAE, R2, RMSE metrics

**References:**

``[1] Montgomery, D. C., Peck, E. A., & Vining, G. G. (2021). Introduction to linear regression analysis. John Wiley & Sons.`` 