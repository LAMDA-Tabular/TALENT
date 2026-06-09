from TALENT.model.classical_methods.base import classical_methods
from copy import deepcopy
import os.path as ops
import pickle
import time
from sklearn.metrics import accuracy_score, mean_squared_error

class SvmMethod(classical_methods):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)
        assert(args.cat_policy != 'indices')

    def construct_model(self, model_config = None):
        if model_config is None:
            model_config = self.args.config['model']
        from sklearn.svm import LinearSVC, LinearSVR
        if self.is_regression:
            self.model = LinearSVR(**model_config)
        else:
            from sklearn.calibration import CalibratedClassifierCV
            base_svc = LinearSVC(**model_config, random_state=self.args.seed)
            self.model = CalibratedClassifierCV(
                estimator=base_svc, method='sigmoid', cv=3, n_jobs=1
            )

    def fit(self, data, info, train=True, config=None):
        super().fit(data, info, train, config)
        if not train:
            return
        tic = time.time()
        self.model.fit(self.N['train'], self.y['train'])
        if not self.is_regression:
            y_val_pred = self.model.predict(self.N['val'])
            self.trlog['best_res'] = accuracy_score(self.y['val'], y_val_pred)
        else:
            y_val_pred = self.model.predict(self.N['val'])
            self.trlog['best_res'] = mean_squared_error(self.y['val'], y_val_pred) ** 0.5 * self.y_info['std']
        time_cost = time.time() - tic
        with open(ops.join(self.args.save_path, 'best-val-{}.pkl'.format(self.args.seed)), 'wb') as f:
            pickle.dump(self.model, f)
        return time_cost

    def predict(self, data, info, model_name):
        N, C, y = data
        with open(ops.join(self.args.save_path, 'best-val-{}.pkl'.format(self.args.seed)), 'rb') as f:
            self.model = pickle.load(f)
        self.data_format(False, N, C, y)
        test_label = self.y_test
        if self.is_regression:
            test_logit = self.model.predict(self.N_test)
        else:
            test_logit = self.model.predict_proba(self.N_test)
        vres, metric_name = self.metric(test_logit, test_label, self.y_info)
        # Denormalize regression predictions back to original scale for the returned value
        if self.is_regression and self.y_info.get('policy') == 'mean_std':
            test_logit = test_logit * self.y_info['std'] + self.y_info['mean']
        return vres, metric_name, test_logit