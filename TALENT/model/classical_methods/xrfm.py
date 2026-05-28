from TALENT.model.classical_methods.base import classical_methods
from TALENT.model.lib.data import (
    Dataset,
    data_nan_process,
    data_label_process,
    num_enc_process,
    data_enc_process,
    data_norm_process
)

import os.path as ops
import time
from sklearn.metrics import accuracy_score, mean_squared_error

from xrfm import xRFM

import torch
import numpy as np


class XRFMMethod(classical_methods):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)
        assert(args.cat_policy == 'ohe')
        assert(args.normalization == 'standard')

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
            },
            'fit': {
                'reg': model_config.get('reg', 1e-3),
                'iters': model_config.get('iters', 5),
                'early_stop_rfm': model_config.get('early_stop_rfm', True),
                'early_stop_multiplier':  model_config.get('early_stop_multiplier', 1.1),
                'verbose': True,
            }
        }
        self.model = xRFM(rfm_params, categorical_info=categorical_info, device=self.args.device,
                          min_subset_size=model_config.get('min_subset_size', 60000),
                          classification_mode=model_config.get('classification_mode', 'prevalence'))

    def data_format(self, is_train = True, N = None, C = None, y = None):
        num_numerical_features = self.N['train'].shape[1] if self.N is not None else 0
        if is_train:
            self.N, self.C, self.num_new_value, self.imputer, self.cat_new_value = data_nan_process(self.N, self.C, self.args.num_nan_policy, self.args.cat_nan_policy)
            self.y, self.y_info, self.label_encoder = data_label_process(self.y, self.is_regression)
            self.n_bins = self.args.config['fit']['n_bins']
            self.N,self.num_encoder = num_enc_process(self.N,num_policy = self.args.num_policy, n_bins = self.n_bins,y_train=self.y['train'],is_regression=self.is_regression)
            
            # normalize the numerical features
            self.N, self.normalizer = data_norm_process(self.N, self.args.normalization, self.args.seed)
            
            # do not normalize the categorical features
            self.N, self.C, self.ord_encoder, self.mode_values, self.cat_encoder = data_enc_process(self.N, self.C, self.args.cat_policy, self.y['train'])
            
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
            
            N_test, _ = data_norm_process(N_test, self.args.normalization, self.args.seed, self.normalizer)
            N_test, C_test, _, _, _ = data_enc_process(N_test, C_test, self.args.cat_policy, None, self.ord_encoder, self.mode_values, self.cat_encoder)
            if N_test is not None and C_test is not None:
                self.N_test,self.C_test = N_test['test'],C_test['test']
            elif N_test is None and C_test is not None:
                self.N_test,self.C_test = None,C_test['test']
            else:
                self.N_test,self.C_test = N_test['test'],None
            self.y_test = y_test['test']
            return
        
        numerical_block = torch.arange(num_numerical_features)
        if self.cat_encoder is None:
            return numerical_block, [], []

        categorical_indices = []
        categorical_vectors = []
        numerical_indices_parts = []

        idx = num_numerical_features
        for cats in self.cat_encoder.categories_:
            cat_len = len(cats)
            cat_idxs = torch.arange(idx, idx + cat_len)
            if cat_len > 100:
                categorical_indices.append(cat_idxs)
                categorical_vectors.append(torch.eye(cat_len))
            else:
                numerical_indices_parts.append(cat_idxs)
            idx += cat_len

        if len(numerical_indices_parts) > 0:
            numerical_indices = torch.cat([numerical_block] + numerical_indices_parts)
        else:
            numerical_indices = numerical_block

        return numerical_indices, categorical_indices, categorical_vectors

        

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
        
        X_train = X_train.to(dtype=torch.float32, device=self.args.device)
        X_val = X_val.to(dtype=torch.float32, device=self.args.device)
        y_train = torch.from_numpy(self.y['train'])
        y_val = torch.from_numpy(self.y['val'])

        if len(y_train.shape)==1:
            y_train = y_train.unsqueeze(-1)
            y_val = y_val.unsqueeze(-1)
                
        self.model.tuning_metric = 'mse' if self.is_regression else 'accuracy'

        tic = time.time()
        self.model.fit(X_train, y_train, X_val, y_val)
        y_val_pred = self.model.predict(X_val)

        if self.is_binclass or self.is_multiclass:
            self.trlog['best_res'] = accuracy_score(y_val.numpy(), y_val_pred)
        else:
            mse = mean_squared_error(y_val.numpy(), y_val_pred)
            self.trlog['best_res'] = (mse ** 0.5) * self.y_info['std']  # Convert MSE to RMSE and scale

        time_cost = time.time() - tic
        
        checkpoint = {}
        checkpoint['state_dict'] = self.model.get_state_dict()

        torch.save(checkpoint, ops.join(self.args.save_path, f'best-val-{self.args.seed}.pt'))

        return time_cost

    def predict(self, data, info, model_name):
        N, C, y = data
        self.data_format(False, N, C, y)

        checkpoint = torch.load(ops.join(self.args.save_path, f'best-val-{self.args.seed}.pt'))

        if self.C is None:
            assert self.N is not None
            X_train = torch.from_numpy(self.N['train'])
        elif self.N is None:
            assert self.C is not None
            X_train = torch.from_numpy(self.C['train'])
        else:
            assert self.C is not None and self.N is not None
            X_train = torch.from_numpy(np.concatenate((self.C['train'], self.N['train']), axis=1))


        X_train = X_train.to(dtype=torch.float32, device=self.args.device)
        self.model.load_state_dict(checkpoint['state_dict'], X_train)

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
            test_output = self.model.predict(X_test)
            # Convert to numpy if tensor
            if isinstance(test_output, torch.Tensor):
                test_output = test_output.cpu().numpy()
        else:
            test_output = self.model.predict_proba(X_test)
        vres, metric_name = self.metric(test_output, test_label, self.y_info)
        # Denormalize regression predictions back to original scale for the returned value
        if self.is_regression and self.y_info.get('policy') == 'mean_std':
            test_output = test_output * self.y_info['std'] + self.y_info['mean']
        return vres, metric_name, test_output
    
    def metric(self, predictions, labels, y_info):
        from sklearn import metrics as skm
        if self.is_regression:
            mae = skm.mean_absolute_error(labels, predictions)
            rmse = skm.mean_squared_error(labels, predictions) ** 0.5
            r2 = skm.r2_score(labels, predictions)
            if y_info['policy'] == 'mean_std':
                mae *= y_info['std']
                rmse *= y_info['std']
            return (mae,r2,rmse), ("MAE", "R2", "RMSE")
        else:
            predicted_classes = np.argmax(predictions, axis=1)
            accuracy = skm.accuracy_score(labels, predicted_classes)
            avg_precision = skm.precision_score(labels, predicted_classes, average='binary' if self.is_binclass else 'macro')
            avg_recall = skm.recall_score(labels, predicted_classes, average='binary' if self.is_binclass else 'macro')
            f1_score = skm.f1_score(labels, predicted_classes, average='binary' if self.is_binclass else 'macro')
            return (accuracy, avg_precision, avg_recall, f1_score), ("Accuracy", "Avg_Precision", "Avg_Recall", "F1")