import abc
import numpy as np
import sklearn.metrics as skm
from sklearn.preprocessing import label_binarize

from TALENT.model.utils import (
    set_seeds,
    get_device,
    check_softmax
)
# from sklearn.externals import joblib
from TALENT.model.lib.data import (
    Dataset,
    data_nan_process,
    data_enc_process,
    num_enc_process,
    data_norm_process,
    data_label_process,
)


class classical_methods(object, metaclass=abc.ABCMeta):
    def __init__(self, args, is_regression):
        self.args = args
        print(args.config)
        self.is_regression = is_regression
        self.D = None
        self.args.device = get_device()
        self.trlog = {}
        assert args.cat_policy != 'indices'

    def data_format(self, is_train = True, N = None, C = None, y = None):
        if is_train:
            self.N, self.C, self.num_new_value, self.imputer, self.cat_new_value = data_nan_process(self.N, self.C, self.args.num_nan_policy, self.args.cat_nan_policy)
            self.y, self.y_info, self.label_encoder = data_label_process(self.y, self.is_regression)
            self.n_bins = self.args.config['fit']['n_bins']
            self.N,self.num_encoder = num_enc_process(self.N,num_policy = self.args.num_policy, n_bins = self.n_bins,y_train=self.y['train'],is_regression=self.is_regression)
            self.N, self.C, self.ord_encoder, self.mode_values, self.cat_encoder = data_enc_process(self.N, self.C, self.args.cat_policy, self.y['train'])
            self.N, self.normalizer = data_norm_process(self.N, self.args.normalization, self.args.seed)
            
            if self.is_regression:
                self.d_out = 1
            else:
                self.d_out = len(np.unique(self.y['train']))
            self.n_num_features = self.N['train'].shape[1] if self.N is not None else 0
            self.n_cat_features = self.C['train'].shape[1] if self.C is not None else 0
            self.d_in = 0 if self.N is None else self.N['train'].shape[1]
        else:
            N_test, C_test, _, _, _ = data_nan_process(N, C, self.args.num_nan_policy, self.args.cat_nan_policy, self.num_new_value, self.imputer, self.cat_new_value)
            y_test, _, _ = data_label_process(y, self.is_regression, self.y_info, self.label_encoder)
            N_test,_ = num_enc_process(N_test,num_policy=self.args.num_policy,n_bins = self.n_bins,y_train=None,encoder = self.num_encoder)
            N_test, C_test, _, _, _ = data_enc_process(N_test, C_test, self.args.cat_policy, None, self.ord_encoder, self.mode_values, self.cat_encoder)
            N_test, _ = data_norm_process(N_test, self.args.normalization, self.args.seed, self.normalizer)
            if N_test is not None and C_test is not None:
                self.N_test,self.C_test = N_test['test'],C_test['test']
            elif N_test is None and C_test is not None:
                self.N_test,self.C_test = None,C_test['test']
            else:
                self.N_test,self.C_test = N_test['test'],None
            self.y_test = y_test['test']
            
    def construct_model(self, model_config = None):
        raise NotImplementedError

    def fit(self, data, info, train = True, config = None):
        N, C, y = data
        # if self.D is None:
        self.D = Dataset(N, C, y, info)
        self.N, self.C, self.y = self.D.N, self.D.C, self.D.y
        self.is_binclass, self.is_multiclass, self.is_regression = self.D.is_binclass, self.D.is_multiclass, self.D.is_regression
          
        if config is not None:
            self.reset_stats_withconfig(config)
        self.data_format(is_train = True)
        self.construct_model()

        # if not train, skip the training process. such as load the checkpoint and directly predict the results
        if not train:
            return
        
    def reset_stats_withconfig(self, config):
        set_seeds(self.args.seed)
        self.config = self.args.config = config

    def _val_proba(self, X):
        """Validation-set class probabilities used for HPO metric selection.

        Defaults to the fitted estimator's ``predict_proba``; subclasses
        without one should override (e.g. distance-based ``NearestCentroid``).
        """
        return self.model.predict_proba(X)

    def _record_best_res(self, val_features):
        """Set ``trlog['best_res']`` for HPO, honouring ``args.tune_metric``.

        With no ``tune_metric`` configured the historical behaviour is kept
        exactly (validation accuracy for classification, std-scaled RMSE for
        regression). Otherwise the configured metric is read from ``metric()``.
        """
        from TALENT.model.lib.tuning_metric import resolve_tune_metric, select_objective
        from sklearn.metrics import accuracy_score, mean_squared_error
        y_val = self.y['val']
        if resolve_tune_metric(self.args) is None:
            y_pred = self.model.predict(val_features)
            if self.is_regression:
                self.trlog['best_res'] = mean_squared_error(y_val, y_pred) ** 0.5 * self.y_info['std']
            else:
                self.trlog['best_res'] = accuracy_score(y_val, y_pred)
            return
        logit = self.model.predict(val_features) if self.is_regression else self._val_proba(val_features)
        vres, names = self.metric(logit, y_val, self.y_info)
        self.trlog['best_res'], _ = select_objective(vres, names, self.args, self.is_regression)

    def metric(self, predictions, labels, y_info, threshold=None):
        """Compute evaluation metrics.

        :param threshold: float or None. If given and the task is binary
            classification, use ``predictions[:, 1] >= threshold`` for the
            hard-prediction metrics (Accuracy, Avg_Recall, Avg_Precision,
            F1). Threshold-independent metrics (LogLoss, AUC, Brier, ECE)
            are not affected. Silently ignored for regression/multiclass.
        """
        if not isinstance(labels, np.ndarray):
            labels = labels.cpu().numpy()
        if not isinstance(predictions, np.ndarray):
            predictions = predictions.cpu().numpy()
        if self.is_regression:
            mae = skm.mean_absolute_error(labels, predictions)
            rmse = skm.mean_squared_error(labels, predictions) ** 0.5
            r2 = skm.r2_score(labels, predictions)
            if y_info['policy'] == 'mean_std':
                mae *= y_info['std']
                rmse *= y_info['std']
            return (mae, r2, rmse), ("MAE", "R2", "RMSE")
        elif self.is_binclass:
            predictions = check_softmax(predictions)
            if threshold is not None:
                hard_preds = (predictions[:, 1] >= threshold).astype(int)
            else:
                hard_preds = predictions.argmax(axis=-1)
            accuracy = skm.accuracy_score(labels, hard_preds)
            avg_recall = skm.balanced_accuracy_score(labels, hard_preds)
            avg_precision = skm.precision_score(labels, hard_preds, average='macro', zero_division=0)
            f1_score = skm.f1_score(labels, hard_preds, average='binary')
            log_loss = skm.log_loss(labels, predictions, labels=y_info['classes'])
            auc = skm.roc_auc_score(labels, predictions[:, 1], labels=y_info['classes']) if len(np.unique(labels)) == 2 else float("nan")
            from TALENT.model.lib.calibration import brier_score, expected_calibration_error
            brier = brier_score(labels, predictions)
            ece = expected_calibration_error(labels, predictions)
            return (
                (accuracy, avg_recall, avg_precision, f1_score, log_loss, auc, brier, ece),
                ("Accuracy", "Avg_Recall", "Avg_Precision", "F1", "LogLoss", "AUC", "Brier", "ECE"),
            )
        elif self.is_multiclass:
            predictions = check_softmax(predictions)
            hard_preds = predictions.argmax(axis=-1)
            accuracy = skm.accuracy_score(labels, hard_preds)
            avg_recall = skm.balanced_accuracy_score(labels, hard_preds)
            avg_precision = skm.precision_score(labels, hard_preds, average='macro', zero_division=0)
            f1_score = skm.f1_score(labels, hard_preds, average='macro')
            log_loss = skm.log_loss(labels, predictions, labels=y_info['classes'])

            present_classes = np.unique(labels)
            if len(present_classes) < 2:
                auc = float("nan")
            else:
                labels_bin = label_binarize(labels, classes=y_info['classes'])
                class_indices = [i for i, c in enumerate(y_info['classes']) if c in present_classes]
                preds_sliced = predictions[:, class_indices]
                labels_bin = labels_bin[:, class_indices]
                auc = skm.roc_auc_score(labels_bin, preds_sliced, labels=present_classes, average='macro', multi_class='ovr')

            from TALENT.model.lib.calibration import brier_score, expected_calibration_error
            brier = brier_score(labels, predictions)
            ece = expected_calibration_error(labels, predictions)

            return (
                (accuracy, avg_recall, avg_precision, f1_score, log_loss, auc, brier, ece),
                ("Accuracy", "Avg_Recall", "Avg_Precision", "F1", "LogLoss", "AUC", "Brier", "ECE"),
            )
        else:
            raise ValueError("Unknown tabular task type")