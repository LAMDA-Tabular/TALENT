====================================
How to Use TALENT
====================================

This guide will walk you through how to install, set up and use the TALENT toolbox for benchmarking models on tabular data, running experiments, and adding new methods.

==========================

Installation
==========================

You can install TALENT directly from GitHub:

.. code-block:: bash

     pip install git+https://github.com/LAMDA-Tabular/TALENT.git@main --upgrade

Alternatively, for development purposes you can clone the repository:

.. code-block:: bash

     git clone https://github.com/LAMDA-Tabular/TALENT
     cd TALENT/test

==========================
2. Quick Start
==========================

Here's a basic example of how to run a deep learning model experiment:

.. code-block:: python

     from tqdm import tqdm
     from TALENT.model.utils import get_deep_args, show_results, tune_hyper_parameters, get_method, set_seeds
     from TALENT.model.lib.data import get_dataset
     from TALENT.model.lib.evaluation import evaluate
     from TALENT.model.method_registry import get_method_spec

     if __name__ == '__main__':
         loss_list, results_list, time_list = [], [], []
         args, default_para, opt_space = get_deep_args()
         train_val_data, test_data, info = get_dataset(args.dataset, args.dataset_path)
         if args.tune:
             args = tune_hyper_parameters(args, opt_space, train_val_data, info)

         spec = get_method_spec(args.model_type)
         for seed in tqdm(range(args.seed_num)):
             args.seed = seed    # update seed
             set_seeds(args.seed)
             method = get_method(args.model_type)(args, info['task_type'] == 'regression')
             method.fit(train_val_data, info)
             # evaluate() standardizes predict_proba and (for binary tasks) tunes
             # the decision threshold on the validation split.
             eval_result = evaluate(
                 method, train_val_data, test_data, info,
                 model_name=args.evaluate_option,
                 output_type=spec.output_type,
                 tune_threshold=True,
             )
             loss_list.append(eval_result["loss"])
             results_list.append(eval_result["metrics"])
             metric_name = eval_result["metric_names"]
             time_list.append((method.fit_time, method.predict_time))

         show_results(args, info, metric_name, loss_list, results_list, time_list)


Run the script from command line:

.. code-block:: bash

     python train_model_deep.py --model_type MODEL_NAME

==========================
3. Running Experiments
==========================

TALENT supports running experiments for both deep learning methods and classical machine learning models:

Configure the experiment settings:

Edit the configuration files located in configs/default/[MODEL_NAME].json and configs/opt_space/[MODEL_NAME].json

Run the experiment:

For deep learning methods:

.. code-block:: bash

     python train_model_deep.py --model_type [MODEL_NAME]

For classical machine learning methods:

.. code-block:: bash

     python train_model_classical.py --model_type [MODEL_NAME]

==========================
4. Adding New Methods
==========================

For methods that only need model design:

Add the model class to model/models/

Inherit from model/methods/base.py and override construct_model()

Register the method in the unified registry model/method_registry.py by appending a MethodSpec entry (the CLI choices and get_method() are derived from it)

Add parameter settings in configs/default/[MODEL_NAME].json and configs/opt_space/[MODEL_NAME].json

For methods requiring training process changes, partially override functions based on model/methods/base.py. Refer to existing implementations in model/methods/.

===============================
5. Configuring Hyperparameters
===============================

Hyperparameters can be configured through:

configs/default/: Default parameters for each method

configs/opt_space/: Hyperparameter optimization space

Modify the appropriate .json files to adjust parameters like learning rate, batch size, etc.

==========================
6. Troubleshooting
==========================

If you encounter any issues while using TALENT, try the following steps:

1. **Check the logs**: Review the logs in the `logs/` directory for any error messages.
2. **Verify dependencies**: Ensure that all required dependencies are installed. Refer to the `dependencies.rst` for more information.
3. **Configuration issues**: Double-check your configuration files to ensure the paths, dataset names, and hyperparameters are correct.
4. **Contact**: If you're unable to resolve the issue, feel free to open an issue on GitHub or contact the developers.

==========================
Conclusion
==========================

TALENT provides a flexible and powerful platform for experimenting with both classical and deep learning models on tabular data. By following the steps in this guide, you can quickly set up and run experiments, fine-tune models, and even add your own methods to the toolbox. For any further assistance, refer to the documentation or reach out to the development team.
