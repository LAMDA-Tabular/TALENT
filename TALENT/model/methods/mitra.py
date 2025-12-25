from TALENT.model.methods.base import Method
import torch
import numpy as np
import torch
import torch.nn.functional as F

from TALENT.model.lib.data import (
    Dataset,
    data_nan_process,
    data_enc_process,
    data_label_process
)
import time


class MitraMethod(Method):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)
        assert args.normalization == 'none'
        assert args.cat_policy == 'indices'
        assert args.num_policy == 'none'
        assert args.tune is not True


    def data_format(self, is_train = True, N = None, C = None, y = None):
        if is_train:
            self.N, self.C, self.num_new_value, self.imputer, self.cat_new_value = data_nan_process(self.N, self.C, self.args.num_nan_policy, self.args.cat_nan_policy)
            self.y, self.y_info, self.label_encoder = data_label_process(self.y, self.is_regression)
            self.N, self.C, self.ord_encoder, self.mode_values, self.cat_encoder = data_enc_process(self.N, self.C, self.args.cat_policy)
            self.criterion = F.cross_entropy if not self.is_regression else F.mse_loss
            self.n_classes = self.y_info['n_classes'] if not self.is_regression else None
        else:
            N_test, C_test, _, _, _ = data_nan_process(N, C, self.args.num_nan_policy, self.args.cat_nan_policy, self.num_new_value, self.imputer, self.cat_new_value)
            N_test, C_test, _, _, _ = data_enc_process(N_test, C_test, self.args.cat_policy, None, self.ord_encoder, self.mode_values, self.cat_encoder)
            y_test, _, _ = data_label_process(y, self.is_regression, self.y_info, self.label_encoder)
            if N_test is not None and C_test is not None:
                self.N_test, self.C_test = N_test['test'], C_test['test']
            elif N_test is None and C_test is not None:
                self.N_test, self.C_test = None, C_test['test']
            else:
                self.N_test, self.C_test = N_test['test'], None
            self.y_test = y_test['test']


    def construct_model(self, model_config = None):
        from TALENT.model.models.mitra import Mitra
        if self.is_regression:
            model_path = "./TALENT/model/models/models_mitra/reg/"
        else:
            model_path = "./TALENT/model/models/models_mitra/cls/"
        self.model = Mitra.from_pretrained(
            path=model_path,
            device="cpu"  # safe tensor initialization on CPU
        ).to(self.args.device)


    def fit(self, data, info, train = True, config = None):
        N, C, y = data
        self.D = Dataset(N, C, y, info)
        self.N, self.C, self.y = self.D.N, self.D.C, self.D.y
        self.is_binclass, self.is_multiclass, self.is_regression = self.D.is_binclass, self.D.is_multiclass, self.D.is_regression
        self.data_format(is_train = True)
        
        y_support = self.y['train']
        
        if self.N is not None and self.C is not None:
            x_support = np.concatenate((self.N['train'], self.C['train']), axis=1)
        elif self.N is None and self.C is not None:
            x_support = self.C['train']
        else:
            x_support = self.N['train']
        
        x_support = x_support.astype(np.float32)
        y_support = y_support.astype(np.float32 if self.is_regression else np.int64)

        self.x_support = torch.from_numpy(x_support).to(self.args.device)  # [n_support, n_feat]
        self.y_support = torch.from_numpy(y_support).to(self.args.device)  # [n_support]

        self.construct_model()
        self.fit_time = 0  # general model does not require fitting


    def predict(self, data, info, model_name):
        start_time = time.time()
        N, C, y = data
        self.data_format(False, N, C, y)
        
        if self.N_test is not None and self.C_test is not None:
            Test_X = np.concatenate((self.N_test, self.C_test), axis=1)
        elif self.N_test is None and self.C_test is not None:
            Test_X = self.C_test
        else:
            Test_X = self.N_test

        x_query = torch.from_numpy(Test_X).float().to(self.args.device)  # [n_query, n_feat]

        n_obs_support = self.x_support.shape[0]
        n_obs_query = x_query.shape[0]
        n_feat = self.x_support.shape[1]
        
        max_samples_support = self.args.config['general']['max_samples_support']
        max_samples_query = self.args.config['general']['max_samples_query']

        if n_obs_support > max_samples_support:
            idx = torch.randperm(n_obs_support)[:max_samples_support] 
            self.x_support = self.x_support[idx, :]
            self.y_support = self.y_support[idx]
            n_obs_support = max_samples_support
        
        results = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, n_obs_query, max_samples_query):
                end = min(start + max_samples_query, n_obs_query)
                x_query_batch = x_query[start:end]   # [batch_query, n_feat]
                batch_size = 1
                batch_n_query = x_query_batch.shape[0]

                x_support_batch = self.x_support.unsqueeze(0)  # [1, n_obs_support, n_feat]
                y_support_batch = self.y_support.unsqueeze(0)  # [1, n_obs_support]
                x_query_batch = x_query_batch.unsqueeze(0)     # [1, batch_n_query, n_feat]

                # no padding
                padding_features = torch.zeros((batch_size, n_feat), dtype=torch.bool, device=self.args.device)
                padding_obs_support = torch.zeros((batch_size, n_obs_support), dtype=torch.bool, device=self.args.device)
                padding_obs_query = torch.zeros((batch_size, batch_n_query), dtype=torch.bool, device=self.args.device)

                test_logit = self.model(
                    x_support=x_support_batch,
                    y_support=y_support_batch,
                    x_query=x_query_batch,
                    padding_features=padding_features,
                    padding_obs_support=padding_obs_support,
                    padding_obs_query__=padding_obs_query,
                ) # [1, batch_n_query, n_classes]

                results.append(test_logit.squeeze(0))

        test_logit = torch.cat(results, dim=0).cpu()
        if not self.is_regression:
            test_logit = test_logit[:, :self.n_classes]
        test_label = self.y_test

        test_label_tensor = torch.tensor(test_label, dtype=torch.float32 if self.is_regression else torch.long)
        
        vl = self.criterion(test_logit, test_label_tensor).item()
        vres, metric_name = self.metric(test_logit, test_label, self.y_info)
        
        # FIX: Denormalize regression predictions
        if self.is_regression and self.y_info.get('policy') == 'mean_std':
            test_logit = test_logit * self.y_info['std'] + self.y_info['mean']

        print('Test: loss={:.4f}'.format(vl))
        for name, res in zip(metric_name, vres):
            print('[{}]={:.4f}'.format(name, res))
        print('Time cost: {:.4f}s'.format(time.time() - start_time))
        return vl, vres, metric_name, test_logit