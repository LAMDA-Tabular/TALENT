Classical Methods
=================

Overview
--------

The classical methods module provides implementations of traditional machine learning algorithms for tabular data analysis. All methods inherit from the base `classical_methods` class and provide a unified interface for training, prediction, and evaluation.

Available Methods
-----------------

**Classification and Regression Methods:**
- Support Vector Machine (SVM)
- XGBoost
- K-Nearest Neighbors (KNN)
- Random Forest
- LightGBM
- CatBoost

**Classification Only Methods:**
- Logistic Regression
- Naive Bayes
- Nearest Centroid Method (NCM)

**Regression Only Methods:**
- Linear Regression

**Baseline Methods:**
- Dummy Classifier/Regressor

Common Features
---------------

All classical methods in TALENT share the following features:

- Automatic data preprocessing (missing value handling, encoding, normalization)
- Unified evaluation metrics
- Model persistence (save/load functionality)
- Support for both numerical and categorical features
- Configurable hyperparameters
- Training time measurement

Usage Example
-------------

.. code-block:: python

    from TALENT.model.classical_methods import SvmMethod
    
    # Initialize the model
    svm = SvmMethod(args, is_regression=False)
    
    # Train the model
    time_cost = svm.fit(data, info, train=True)
    
    # Make predictions
    vres, metric_name, predictions = svm.predict(test_data, info, model_name) 
    
This section contains documentation for all classical machine learning methods implemented in TALENT.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   base
   svm
   xgboost
   knn
   logreg
   catboost
   randomforest
   lightgbm
   naivebayes
   ncm
   dummy
   lr

