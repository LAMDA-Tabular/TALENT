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
from TALENT.model.utils import probability_nll
import time

class HyperFastMethod(Method):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)
        assert(is_regression == False)
        assert(args.normalization == 'none')
        assert(args.cat_policy == 'indices')
        assert(args.num_policy == 'none')
        assert(args.tune != True)


    def data_format(self, is_train = True, N = None, C = None, y = None):
        if is_train:
            self.N, self.C, self.num_new_value, self.imputer, self.cat_new_value = data_nan_process(self.N, self.C, self.args.num_nan_policy, self.args.cat_nan_policy)
            self.y, self.y_info, self.label_encoder = data_label_process(self.y, self.is_regression)
            self.N, self.C, self.ord_encoder, self.mode_values, self.cat_encoder = data_enc_process(self.N, self.C, self.args.cat_policy)
            self.criterion = F.cross_entropy
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

    @staticmethod
    def checkpoint_path():
        """Where the HyperFast checkpoint is read from (and downloaded to).

        Bundled copy if present, else a copy left at the historical
        CWD-relative location by an earlier run, else the TALENT cache
        directory (``$TALENT_CACHE_DIR``, default ``~/.cache/talent``).
        """
        import os
        from TALENT.model.method_registry import resolve_bundled_path, talent_cache_dir
        legacy = os.path.join("model", "models", "hyperfast", "hyperfast.ckpt")
        bundled = resolve_bundled_path("model/models/hyperfast/hyperfast.ckpt")
        if bundled is not None:
            return bundled
        if os.path.exists(legacy):
            return legacy
        return os.path.join(talent_cache_dir(), "hyperfast", "hyperfast.ckpt")

    def construct_model(self, model_config = None):
        from TALENT.model.models.hyperfast import HyperFastClassifier, DEFAULT_CONFIG
        from TALENT.model.methods.foundation_base import resolve_n_estimators
        config = dict(DEFAULT_CONFIG, model_path=self.checkpoint_path())
        # The checkpoint is downloaded from figshare on first use.
        self.model = HyperFastClassifier(
            device = self.args.device, seed = self.args.seed,
            n_ensemble = resolve_n_estimators(self.args, 1), config = config,
        )

    def fit(self, data, info, train = True, config = None):
        from TALENT.model.method_registry import weights_download_hint
        N, C, y = data
        self.D = Dataset(N, C, y, info)
        self.N, self.C, self.y = self.D.N, self.D.C, self.D.y
        self.is_binclass, self.is_multiclass, self.is_regression = self.D.is_binclass, self.D.is_multiclass, self.D.is_regression
        self.data_format(is_train = True)
        with weights_download_hint(getattr(self.args, 'model_type', 'hyperfast')):
            self.construct_model()

        sampled_Y = self.y['train']
        if self.N is not None and self.C is not None:
            sampled_X = np.concatenate((self.N['train'], self.C['train']), axis=1)
            n_num = self.N['train'].shape[1]
            cat_features = list(range(n_num, n_num + self.C['train'].shape[1]))
        elif self.N is None and self.C is not None:
            sampled_X = self.C['train']
            cat_features = list(range(self.C['train'].shape[1]))
        else:
            sampled_X = self.N['train']
            cat_features = []
        # Optional row cap via config['general']['sample_size'].
        sampled_X, sampled_Y = self.subsample_train_rows(sampled_X, sampled_Y)
        tic = time.time()
        self.model.fit(sampled_X, sampled_Y, cat_features)
        self.fit_time = time.time() - tic
    
    def predict(self, data, info, model_name):
        N, C, y = data
        self.data_format(False, N, C, y)
        if self.N_test is not None and self.C_test is not None:
            Test_X = np.concatenate((self.N_test,self.C_test),axis=1)
        elif self.N_test is None and self.C_test is not None:
            Test_X = self.C_test
        else:
            Test_X = self.N_test
        
        tic = time.time()
        test_logit = self.model.predict_proba(Test_X)
        self.predict_time = time.time() - tic
        
        test_label = self.y_test
        # predict_proba returns probabilities, so the loss is their NLL.
        vl = probability_nll(test_logit, test_label)
        vres, metric_name = self.metric(test_logit, test_label, self.y_info)
        print('Test: loss={:.4f}'.format(vl))
        for name, res in zip(metric_name, vres):
            print('[{}]={:.4f}'.format(name, res))
        return vl, vres, metric_name, test_logit

