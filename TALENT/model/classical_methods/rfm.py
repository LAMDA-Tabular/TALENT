from TALENT.model.classical_methods.xrfm import XRFMMethod
from TALENT.model.lib.data import (
    Dataset,
)

import os.path as ops
import time
import torch
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, mean_squared_error

from xrfm import RFM


class RFMMethod(XRFMMethod):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)

    def construct_model(self, model_config = None, categorical_info = None):
        if model_config is None:
            model_config = self.args.config['model']

        exponent = model_config.get('exponent', 1.0)
        rfm_params = {
            'model': {
                'kernel': model_config['kernel_type'],
                'bandwidth': model_config['bandwidth'],
                'exponent': exponent,
                'norm_p': exponent + (2 - exponent) * model_config.get('p_interp', 1.0),
                'diag': model_config['diag'],
                'bandwidth_mode': model_config.get('bandwidth_mode', 'constant'),
            }
        }
        self.model = RFM(
            **rfm_params['model'],
            categorical_info = categorical_info,
            device = self.args.device,
            time_limit_s = None,
            tuning_metric = 'mse' if self.is_regression else 'accuracy',
        )
    
    def fit(self, data, info, train=True, config=None, train_on_subset=False):
        N, C, y = data
        # if the method already fit the dataset, skip these steps (such as the hyper-tune process)
        self.D = Dataset(N, C, y, info)
        self.N, self.C, self.y = self.D.N, self.D.C, self.D.y
        self.is_binclass, self.is_multiclass, self.is_regression = self.D.is_binclass, self.D.is_multiclass, self.D.is_regression
        self.n_num_features, self.n_cat_features = self.D.n_num_features, self.D.n_cat_features
        if config:
            self.reset_stats_withconfig(config)

        numerical_indices, categorical_indices, categorical_vectors = self.data_format(is_train = True)
        categorical_info = {
            'categorical_indices': categorical_indices,
            'categorical_vectors': categorical_vectors,
            'numerical_indices': numerical_indices
        }
        self.construct_model(categorical_info=categorical_info)
        
        if self.C is None:
            assert self.N is not None
            X_train = torch.from_numpy(self.N['train'])
            X_val = torch.from_numpy(self.N['val'])
        elif self.N is None:
            assert self.C is not None
            X_train = torch.from_numpy(self.C['train'])
            X_val = torch.from_numpy(self.C['val'])
        else:
            assert self.C is not None and self.N is not None
            X_train = torch.from_numpy(np.concatenate((self.C['train'], self.N['train']), axis=1))
            X_val = torch.from_numpy(np.concatenate((self.C['val'], self.N['val']), axis=1))

        if self.is_regression:
            y_train = self.y['train'].reshape(-1, 1)
            y_val = self.y['val'].reshape(-1, 1)
        else:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            y_train = ohe.fit_transform(self.y['train'].reshape(-1, 1))
            y_val = ohe.transform(self.y['val'].reshape(-1, 1))

        X_train = X_train.to(dtype=torch.float32, device=self.args.device)
        X_val = X_val.to(dtype=torch.float32, device=self.args.device)
        y_train = torch.from_numpy(y_train).to(dtype=torch.float32, device=self.args.device)
        y_val = torch.from_numpy(y_val).to(dtype=torch.float32, device=self.args.device)

        tic = time.time()
        self.model.fit((X_train, y_train), (X_val, y_val), reg = self.args.config['model'].get('reg', 1e-3))
        
        if self.is_regression:
            y_val_pred = self.model.predict(X_val).cpu().numpy()
        else:
            y_val_pred = self.model.predict_proba(X_val)
            y_val_pred = torch.argmax(y_val_pred, dim=1).cpu().numpy()

        if self.is_regression:
            mse = mean_squared_error(self.y['val'], y_val_pred)
            self.trlog['best_res'] = (mse ** 0.5) * self.y_info['std']  # Convert MSE to RMSE and scale
        else:
            self.trlog['best_res'] = accuracy_score(self.y['val'], y_val_pred)

        time_cost = time.time() - tic
        
        checkpoint = {}
        checkpoint['state_dict'] = self.model.state_dict()

        torch.save(checkpoint, ops.join(self.args.save_path, f'best-val-{self.args.seed}.pt'))

        return time_cost


    def predict(self, data, info, model_name):
        N, C, y = data
        self.data_format(False, N, C, y)
        checkpoint = torch.load(ops.join(self.args.save_path, f'best-val-{self.args.seed}.pt'))
        self.model.load_state_dict(checkpoint['state_dict'])

        # Convert test data and labels to tensors
        if self.C_test is None:
            assert self.N_test is not None
            X_test = torch.from_numpy(self.N_test)
        elif self.N_test is None:
            assert self.C_test is not None
            X_test = torch.from_numpy(self.C_test)
        else:
            assert self.C_test is not None and self.N_test is not None
            X_test = torch.from_numpy(np.concatenate((self.C_test, self.N_test), axis=1))

        X_test = X_test.float().to(self.args.device)

        test_label = self.y_test
        if self.is_regression:
            test_output = self.model.predict(X_test).cpu().numpy()
        else:
            test_output = self.model.predict_proba(X_test).cpu().numpy()
        vres, metric_name = self.metric(test_output, test_label, self.y_info)
        # Denormalize regression predictions back to original scale for the returned value
        if self.is_regression and self.y_info.get('policy') == 'mean_std':
            test_output = test_output * self.y_info['std'] + self.y_info['mean']
        return vres, metric_name, test_output