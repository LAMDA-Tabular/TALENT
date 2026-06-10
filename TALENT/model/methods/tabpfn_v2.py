from TALENT.model.methods.base import Method
import torch
import numpy as np
import torch.nn.functional as F

from TALENT.model.lib.data import (
    Dataset,
    data_nan_process,
    data_enc_process,
    data_label_process
)
import time


class TabPFNMethod(Method):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)
        assert(args.normalization == 'none')
        assert(args.cat_policy == 'indices')
        assert(args.num_policy == 'none')
        assert(args.tune != True)


    def data_format(self, is_train = True, N = None, C = None, y = None):
        if is_train:
            self.N, self.C, self.num_new_value, self.imputer, self.cat_new_value = data_nan_process(self.N, self.C, self.args.num_nan_policy, self.args.cat_nan_policy)
            self.y, self.y_info, self.label_encoder = data_label_process(self.y, self.is_regression)
            self.N, self.C, self.ord_encoder, self.mode_values, self.cat_encoder = data_enc_process(self.N, self.C, self.args.cat_policy)
            self.criterion = F.cross_entropy if  not self.is_regression else F.mse_loss
        else:
            N_test, C_test, _, _, _ = data_nan_process(N, C, self.args.num_nan_policy, self.args.cat_nan_policy, self.num_new_value, self.imputer, self.cat_new_value)
            N_test, C_test, _, _, _ = data_enc_process(N_test, C_test, self.args.cat_policy, None, self.ord_encoder, self.mode_values, self.cat_encoder)
            y_test, _, _ = data_label_process(y, self.is_regression, self.y_info, self.label_encoder)
            if N_test is not None and C_test is not None:
                self.N_test,self.C_test = N_test['test'],C_test['test']
            elif N_test is None and C_test is not None:
                self.N_test,self.C_test = None,C_test['test']
            else:
                self.N_test,self.C_test = N_test['test'],None
            self.y_test = y_test['test']


    def construct_model(self, model_config = None,cat_indices=None):
        cat_indices = cat_indices or []
        from TALENT.model.method_registry import resolve_bundled_path
        if self.is_regression:
            from TALENT.model.models.tabpfn_v2 import TabPFNRegressor
            # Use bundled checkpoint if present; otherwise fall back to the
            # TabPFN library's auto-download (model_path="auto").
            model_path = resolve_bundled_path(
                "model/models/models_tabpfn/tabpfn-v2-regressor.ckpt"
            ) or "auto"
            self.model = TabPFNRegressor(
                model_path = model_path,
                device = self.args.device,
                random_state = self.args.seed,
                n_estimators = 8,
                ignore_pretraining_limits = True,
                categorical_features_indices = cat_indices
            )
        else:
            from TALENT.model.models.tabpfn_v2 import TabPFNClassifier
            model_path = resolve_bundled_path(
                "model/models/models_tabpfn/tabpfn-v2-classifier.ckpt"
            ) or "auto"
            self.model = TabPFNClassifier(
                model_path = model_path,
                device = self.args.device,
                random_state = self.args.seed,
                n_estimators = 4,
                ignore_pretraining_limits = True,
                categorical_features_indices = cat_indices
            )


    def fit(self, data, info, train = True, config = None):
        N, C, y = data
        # if self.D is None:
        self.D = Dataset(N, C, y, info)
        self.N, self.C, self.y = self.D.N, self.D.C, self.D.y
        self.is_binclass, self.is_multiclass, self.is_regression = self.D.is_binclass, self.D.is_multiclass, self.D.is_regression
        self.data_format(is_train = True)
        
        sampled_Y = self.y['train']
        cat_indices = []
        if self.N is not None and self.C is not None:
            sampled_X = np.concatenate((self.N['train'],self.C['train']),axis=1)
            cat_indices = [i for i in range(self.N['train'].shape[1],self.N['train'].shape[1]+self.C['train'].shape[1])]
        elif self.N is None and self.C is not None:
            sampled_X = self.C['train']
            cat_indices = [i for i in range(self.C['train'].shape[1])]
        else:
            sampled_X = self.N['train']
        # Row cap: config['general']['sample_size'] override, else the
        # registry's train_row_limit. The TabPFN v2 wrapper subsamples
        # internally, so the cap is forwarded to model.fit().
        sample_size = self.resolve_sample_size()
        if sample_size is None:
            sample_size = sampled_X.shape[0]
        self.sampled_X = sampled_X
        self.sampled_Y = sampled_Y
        self.construct_model(cat_indices=cat_indices)
        self.model.fit(sampled_X,sampled_Y,sample_size)
        self.fit_time = 0  # general model does not require fitting


    def predict(self, data, info, model_name):
        N, C, y = data
        self.data_format(False, N, C, y)
        if self.N_test is not None and self.C_test is not None:
            Test_X = np.concatenate((self.N_test, self.C_test), axis=1)
        elif self.N_test is None and self.C_test is not None:
            Test_X = self.C_test
        else:
            Test_X = self.N_test
        
        tic = time.time()
        if self.is_regression:
            test_logit = self.model.predict(Test_X)
        else:
            test_logit = self.model.predict_proba(Test_X)
        self.predict_time = time.time() - tic
        
        test_label = self.y_test
        vl = self.criterion(torch.tensor(test_logit), torch.tensor(test_label)).item()
        vres, metric_name = self.metric(test_logit, test_label, self.y_info)
        
        # Denormalize regression predictions back to original scale
        if self.is_regression and self.y_info.get('policy') == 'mean_std':
            test_logit = test_logit * self.y_info['std'] + self.y_info['mean']
        print('Test: loss={:.4f}'.format(vl))
        for name, res in zip(metric_name, vres):
            print('[{}]={:.4f}'.format(name, res))
        return vl, vres, metric_name, test_logit
