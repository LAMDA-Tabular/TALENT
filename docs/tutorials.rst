Using TALENT
============

Install from a source checkout to run the bundled examples.

.. code-block:: bash

   git clone https://github.com/LAMDA-Tabular/TALENT.git
   cd TALENT
   python -m pip install -e .

Python API
----------

From the checkout root, run logistic regression on the bundled bank dataset:

.. code-block:: python

   import TALENT

   result = TALENT.run_from_dataset("LogReg", "bank", "./example_datasets")
   print(dict(zip(result.metric_names, result.metrics)))
   print(result.predict_proba.shape)

For data already in memory, pass the training and validation splits, test
split and dataset metadata to ``TALENT.run(method, train_val, test, info)``.
Both entry points return a ``RunResult``. For classifiers with probabilistic
outputs, ``predict_proba`` has one column per class. Binary threshold tuning
uses the validation split. Regressors return predictions in target units.

Use ``TALENT.get_method_spec(name)`` to check supported tasks, preprocessing,
hyperparameter tuning and weights. Calling ``spec.preflight()`` reads local
package metadata. See :doc:`dependencies` for optional backends and
:doc:`methods` for model descriptions.

Command-line experiments
------------------------

From the same checkout root:

.. code-block:: bash

   python test/train_model_classical.py --model_type LogReg --dataset bank --dataset_path example_datasets --seed_num 1
   python test/train_model_deep.py --model_type mlp --dataset bank --dataset_path example_datasets --seed_num 1 --max_epoch 1

The launchers print metrics and write run artifacts below ``--model_path``.
Pass ``--help`` to the relevant launcher for its options.

Adding a method
---------------

Implement the adapter under ``TALENT/model/methods/`` or
``TALENT/model/classical_methods/`` and register it in
``TALENT/model/method_registry.py``. Add matching JSON files under
``TALENT/configs/default/`` and ``TALENT/configs/opt_space/``. The Python
runner and both CLI launchers use the registry for method selection.
