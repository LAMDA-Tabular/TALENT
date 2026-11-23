====================================
Classical Methods
====================================

Overview
--------

The classical methods module provides implementations of traditional machine learning algorithms for tabular data analysis. All methods inherit from the base `classical_methods` class and provide a unified interface for training, prediction, and evaluation.


Available Methods
-----------------

**Base Components:**

- :doc:`Base <classical_model/base>`: Base class for all classical machine learning methods, providing common interface and utilities

**Tree-Based Methods:**

- :doc:`Random Forest <classical_model/randomforest>`: Ensemble learning method using multiple decision trees
- :doc:`XGBoost <classical_model/xgboost>`: Gradient boosting framework with optimized implementation
- :doc:`LightGBM <classical_model/lightgbm>`: Light gradient boosting machine with high efficiency
- :doc:`CatBoost <classical_model/catboost>`: Gradient boosting with categorical features support

**Linear Methods:**

- :doc:`Logistic Regression <classical_model/logreg>`: Linear model for classification tasks
- :doc:`Linear Regression <classical_model/lr>`: Linear model for regression tasks
- :doc:`Support Vector Machine <classical_model/svm>`: SVM classifier with kernel methods

**Distance-Based Methods:**

- :doc:`K-Nearest Neighbors <classical_model/knn>`: Instance-based learning using nearest neighbors
- :doc:`Nearest Centroid Method <classical_model/ncm>`: Classification based on centroid distances

**Probabilistic Methods:**

- :doc:`Naive Bayes <classical_model/naivebayes>`: Probabilistic classifier based on Bayes theorem

**Utility Methods:**

- :doc:`Dummy Classifier <classical_model/dummy>`: Baseline classifier for comparison and testing

.. toctree::
   :maxdepth: 2
   :caption: Classical Methods:

   classical_model/index


