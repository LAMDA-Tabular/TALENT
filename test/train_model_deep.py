from tqdm import tqdm
from TALENT.model.utils import (
    get_deep_args,
    show_results,
    tune_hyper_parameters,
    get_method,
    set_seeds,
)
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

    ## Training Stage over different random seeds
    for seed in tqdm(range(args.seed_num)):
        args.seed = seed    # update seed
        set_seeds(args.seed)
        method = get_method(args.model_type)(args, info['task_type'] == 'regression')
        method.fit(train_val_data, info)
        # The evaluator handles predict_proba standardization and, for
        # binary classification, tunes the decision threshold on the
        # validation set (used for Accuracy/F1/Precision/Recall).
        eval_result = evaluate(
            method,
            train_val_data,
            test_data,
            info,
            model_name=args.evaluate_option,
            output_type=spec.output_type,
            tune_threshold=True,
        )
        vl = eval_result["loss"]
        vres = eval_result["metrics"]
        metric_name = eval_result["metric_names"]

        loss_list.append(vl)
        results_list.append(vres)
        time_list.append((method.fit_time, method.predict_time))

    show_results(args, info, metric_name, loss_list, results_list, time_list)
