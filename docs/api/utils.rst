Utils
=====

.. automodule:: TALENT.model.utils
   :members:
   :undoc-members:
   :show-inheritance:

File and Path Utilities
-----------------------

.. function:: mkdir(path)
   :noindex:

   Create a directory if it does not exist.

   **Parameters:**
   
   * **path** (*str*) -- Path to the directory to create
   
   **Raises:**
   
   * **OSError** -- If directory creation fails for reasons other than already existing

.. function:: set_gpu(x)
   :noindex:

   Set environment variable CUDA_VISIBLE_DEVICES to specify which GPU to use.
   
   **Parameters:**
   
   * **x** (*str*) -- GPU ID to use (e.g., "0", "1", "0,1")
   
   **Example:**
   
   .. code-block:: python
      
      set_gpu("0")  # Use GPU 0
      set_gpu("0,1")  # Use GPUs 0 and 1

.. function:: ensure_path(path, remove=True)
   :noindex:

   Ensure a path exists, optionally removing existing directory.
   
   **Parameters:**
   
   * **path** (*str*) -- Path to the directory
   * **remove** (*bool, optional*) -- Whether to remove the directory if it exists. Defaults to True.
   
   **Note:**
   
   If the path exists and remove=True, will prompt user for confirmation before removing.

Random Seed and Device Management
---------------------------------

.. function:: set_seeds(base_seed, one_cuda_seed=False)
   :noindex:

   Set random seeds for reproducibility across all random number generators.
   
   **Parameters:**
   
   * **base_seed** (*int*) -- Base seed value (must be 0 <= base_seed < 2^32 - 10000)
   * **one_cuda_seed** (*bool, optional*) -- Whether to set one seed for all GPUs. Defaults to False.
   
   **Note:**
   
   Sets seeds for Python random, NumPy, PyTorch CPU, and PyTorch CUDA generators.
   Each generator gets a different seed derived from the base_seed.

.. function:: get_device(device=None)
   :noindex:

   Use an explicit PyTorch device such as ``"cpu"`` or ``"cuda:1"``, or omit it for automatic selection.
   
   **Returns:**
   
   * **torch.device** -- The requested device, or the first CUDA device when available and CPU otherwise

Evaluation Metrics
------------------

.. function:: rmse(y, prediction, y_info)
   :noindex:

   Calculate Root Mean Squared Error (RMSE) for regression tasks.
   
   **Parameters:**
   
   * **y** (*np.ndarray*) -- Ground truth values
   * **prediction** (*np.ndarray*) -- Predicted values
   * **y_info** (*dict*) -- Information about the target variable, including normalization policy
   
   **Returns:**
   
   * **float** -- RMSE value, adjusted for normalization if applicable
   
   **Note:**
   
   If y_info['policy'] is 'mean_std', the RMSE is multiplied by the standard deviation
   to denormalize the result.

Configuration Management
------------------------

.. function:: load_config(args, config=None, config_name=None)
   :noindex:

   Load configuration file for model training and save current arguments.
   
   **Parameters:**
   
   * **args** (*argparse.Namespace*) -- Command line arguments
   * **config** (*dict, optional*) -- Pre-loaded configuration dictionary. Defaults to None.
   * **config_name** (*str, optional*) -- Name for the saved config file. Defaults to None.
   
   **Returns:**
   
   * **argparse.Namespace** -- Updated arguments with loaded configuration
   
   **Note:**
   
   Automatically saves the current arguments to a JSON file in the save_path directory.

Hyperparameter Optimization
---------------------------

.. function:: sample_parameters(trial, space, base_config)
   :noindex:

   Sample hyperparameters from the search space using Optuna trial.
   
   **Parameters:**
   
   * **trial** (*optuna.trial.Trial*) -- Optuna trial object for parameter sampling
   * **space** (*dict*) -- Hyperparameter search space definition
   * **base_config** (*dict*) -- Base configuration dictionary
   
   **Returns:**
   
   * **dict** -- Sampled hyperparameters
   
   **Special Distributions:**
   
   * **$mlp_d_layers** -- Special distribution for MLP layer dimensions
   * **$d_token** -- Special distribution for transformer token dimensions
   * **$d_ffn_factor** -- Special distribution for feedforward network factors
   * **?** -- Optional parameters with default values

.. function:: merge_sampled_parameters(config, sampled_parameters)
   :noindex:

   Merge sampled hyperparameters into the base configuration.
   
   **Parameters:**
   
   * **config** (*dict*) -- Base configuration to update
   * **sampled_parameters** (*dict*) -- Sampled parameters to merge
   
   **Note:**
   
   Recursively merges nested dictionaries and overwrites existing parameters.

Argument Parsing
----------------

.. function:: get_classical_args()
   :noindex:

   Parse command line arguments for classical machine learning models.

   **Returns:**
   
   * **tuple** -- (args, default_para, opt_space) where:
     * args: Parsed arguments
     * default_para: Default parameter configurations
     * opt_space: Hyperparameter optimization space
   
   Use ``TALENT.model.method_registry.classical_method_names()`` for the accepted model IDs.

   **Key Parameters:**
   
   * normalization: Data normalization method
   * num_nan_policy: Policy for handling numerical missing values
   * cat_nan_policy: Policy for handling categorical missing values
   * cat_policy: Categorical encoding policy
   * num_policy: Numerical feature processing policy

.. function:: get_deep_args()
   :noindex:

   Parse command line arguments for deep learning models.

   **Returns:**
   
   * **tuple** -- (args, default_para, opt_space) where:
     * args: Parsed arguments
     * default_para: Default parameter configurations
     * opt_space: Hyperparameter optimization space
   
   Use ``TALENT.model.method_registry.deep_method_names()`` for the accepted model IDs.

Results Display
---------------

.. function:: show_results_classical(args, info, metric_name, results_list, time_list)
   :noindex:

   Display results for classical machine learning models.
   
   **Parameters:**
   
   * **args** (*argparse.Namespace*) -- Training arguments
   * **info** (*dict*) -- Dataset information
   * **metric_name** (*list*) -- Names of evaluation metrics
   * **results_list** (*list*) -- List of results from multiple trials
   * **time_list** (*list*) -- Per-seed fit and test prediction times
   
   **Output:**
   
   Prints formatted results including mean, standard deviation, and GPU information.

.. function:: show_results(args, info, metric_name, loss_list, results_list, time_list)
   :noindex:

   Display results for deep learning models.
   
   **Parameters:**
   
   * **args** (*argparse.Namespace*) -- Training arguments
   * **info** (*dict*) -- Dataset information
   * **metric_name** (*list*) -- Names of evaluation metrics
   * **loss_list** (*list*) -- List of training losses
   * **results_list** (*list*) -- List of results from multiple trials
   * **time_list** (*list*) -- Per-seed fit and test prediction times
   
   **Output:**
   
   Prints formatted results including mean loss, metrics, and GPU information.

Hyperparameter Tuning
---------------------

.. function:: tune_hyper_parameters(args, opt_space, train_val_data, info)
   :noindex:

   Perform hyperparameter optimization using Optuna.
   
   **Parameters:**
   
   * **args** (*argparse.Namespace*) -- Training arguments
   * **opt_space** (*dict*) -- Hyperparameter search space
   * **train_val_data** (*tuple*) -- Training and validation data
   * **info** (*dict*) -- Dataset information
   
   **Returns:**
   
   * **argparse.Namespace** -- Updated arguments with optimized hyperparameters
   
   **Features:**
   
   * Uses TPE sampler for efficient optimization
   * Maximizes or minimizes the configured validation metric according to its direction
   * Automatically saves best configuration to JSON file
   * Handles model-specific parameter adjustments

Model Factory
-------------

.. function:: get_method(model)
   :noindex:

   Get the method class for a given model name.
   
   **Parameters:**
   
   * **model** (*str*) -- Model name
   
   **Returns:**
   
   * **class** -- Method class for the specified model
   
   **Raises:**
   
   * **NotImplementedError** -- If the model is not yet implemented
   
   **Supported Models:**
   
   All deep learning and classical models supported by TALENT.

Utility Classes
---------------

.. class:: Averager()
   :noindex:

   A simple averager for tracking running averages.
   
   **Methods:**
   
   .. method:: add(x)
      :noindex:
      
      Add a value to the running average.
      
      **Parameters:**
      
      * **x** (*float*) -- Value to add
   
   .. method:: item()
      :noindex:
      
      Get the current average value.
      
      **Returns:**
      
      * **float** -- Current running average

.. class:: Timer()
   :noindex:

   A timer for measuring elapsed time.
   
   **Methods:**
   
   .. method:: measure(p=1)
      :noindex:
      
      Measure elapsed time since timer creation.
      
      **Parameters:**
      
      * **p** (*int, optional*) -- Period for time formatting. Defaults to 1.
      
      **Returns:**
      
      * **str** -- Formatted time string (e.g., "30s", "2m", "1.5h")

Debugging Utilities
-------------------

.. function:: pprint(x)
   :noindex:

   Pretty print an object using the PrettyPrinter.
   
   **Parameters:**
   
   * **x** (*any*) -- Object to print




