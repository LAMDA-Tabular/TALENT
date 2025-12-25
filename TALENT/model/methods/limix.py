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


class LimiXMethod(Method):
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
        from TALENT.model.models.limix import LimiXPredictor
        model_path = "./TALENT/model/models/models_limix/LimiX-16M.ckpt"
        config_path = f"./TALENT/model/lib/limix/config/{'reg' if self.is_regression else 'cls'}_default_noretrieval.json"
        self.model = LimiXPredictor(
            device = self.args.device,
            model_path = model_path,
            inference_config = config_path,
            task_type = "Regression" if self.is_regression else "Classification",
            mask_prediction = False,
            seed = self.args.seed
        )


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
        
        self.x_support = x_support.astype(np.float32)
        self.y_support = y_support.astype(np.float32 if self.is_regression else np.int64)
        self.construct_model()
        self.fit_time = 0  # general model does not require fitting


    def predict(self, data, info, model_name):
        N, C, y = data
        self.data_format(False, N, C, y)
        
        if self.N_test is not None and self.C_test is not None:
            x_query = np.concatenate((self.N_test, self.C_test), axis=1)
        elif self.N_test is None and self.C_test is not None:
            x_query = self.C_test
        else:
            x_query = self.N_test
        
        tic = time.time()
        test_logit = self.model.predict(self.x_support, self.y_support, x_query)
        self.predict_time = time.time() - tic
        test_label = self.y_test

        test_logit_tensor = torch.tensor(test_logit, dtype=torch.float32, device=torch.device("cpu"))
        test_label_tensor = torch.tensor(test_label, dtype=torch.float32 if self.is_regression else torch.long)

        vl = self.criterion(test_logit_tensor, test_label_tensor).item()
        vres, metric_name = self.metric(test_logit_tensor, test_label_tensor, self.y_info)

        # ==========================================================
        # FIX: Denormalize regression predictions back to original scale
        # ==========================================================
        if self.is_regression and self.y_info.get('policy') == 'mean_std':
            test_logit_tensor = test_logit_tensor * self.y_info['std'] + self.y_info['mean']

        print('Test: loss={:.4f}'.format(vl))
        for name, res in zip(metric_name, vres):
            print('[{}]={:.4f}'.format(name, res))
        return vl, vres, metric_name, test_logit_tensor