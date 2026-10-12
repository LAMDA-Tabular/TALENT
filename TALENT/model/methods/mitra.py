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
from TALENT.model.method_registry import weights_download_hint
import time


class MitraMethod(Method):
    # Checkpoint locations: bundled under TALENT/model/models/<BUNDLED_DIR>/{cls,reg}/,
    # otherwise downloaded from these Hugging Face repos.
    BUNDLED_DIR = "models_mitra"
    HF_REPOS = {"cls": "autogluon/mitra-classifier", "reg": "autogluon/mitra-regressor"}

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
        import os
        from TALENT.model.models.mitra import Mitra
        from TALENT.model.method_registry import resolve_bundled_path, download_from_hf_hub
        subdir = "reg" if self.is_regression else "cls"
        # Resolve via importlib.resources so the package works from any CWD.
        model_path = resolve_bundled_path(f"model/models/{self.BUNDLED_DIR}/{subdir}/")
        if model_path is None:
            # Fall back to the legacy relative path -- preserves prior
            # behaviour when running from the repository root.
            model_path = f"./TALENT/model/models/{self.BUNDLED_DIR}/{subdir}/"
        if not os.path.exists(os.path.join(model_path, "config.json")):
            # Checkpoint not bundled: fetch the official weights (same
            # config.json + model.safetensors layout) from the Hugging Face Hub.
            model_path = download_from_hf_hub(self.HF_REPOS[subdir])
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
        
        # Row cap: config['general']['sample_size'] override, else the
        # registry's train_row_limit (Mitra attends over the full support set).
        x_support, y_support = self.subsample_train_rows(x_support, y_support)

        x_support = x_support.astype(np.float32)
        y_support = y_support.astype(np.float32 if self.is_regression else np.int64)

        self.x_support = torch.from_numpy(x_support).to(self.args.device)  # [n_support, n_feat]
        self.y_support = torch.from_numpy(y_support).to(self.args.device)  # [n_support]

        with weights_download_hint(getattr(self.args, 'model_type', 'mitra')):
            self.construct_model()
        if not self.is_regression and self.n_classes > self.model.dim_output:
            raise ValueError(
                f"Mitra supports classification with at most {self.model.dim_output} "
                f"classes; got {self.n_classes}."
            )
        self._bins = None
        if self.is_regression and self.model.dim_output > 1:
            self.y_support = self._encode_regression_bins(self.y_support)
        self.fit_time = 0  # in-context model: all work happens in predict()


    def _encode_regression_bins(self, y):
        """Targets -> bin ids for a distributional regression head (Mitra-v2).

        Mirrors AutoGluon's cross-entropy regression path: targets are min-max
        scaled over the support set and bucketized into ``dim_output`` bins on
        ``linspace(-0.5, 1.5, dim_output + 1)``.
        """
        y_min, y_max = y.min(), y.max()
        if not bool(y_max > y_min):
            raise ValueError("Mitra's binned regression head needs at least two distinct target values.")
        n_bins = self.model.dim_output
        self._bins = torch.linspace(-0.5, 1.5, n_bins + 1, device=y.device)
        self._y_range = (float(y_min), float(y_max))
        y_norm = (y - y_min) / (y_max - y_min)
        return (torch.bucketize(y_norm, self._bins) - 1).clamp(0, n_bins - 1).long()


    def _decode_regression_bins(self, logits):
        """Bin logits -> point predictions: softmax-weighted mean of the bin
        centres (the decode used for the released Mitra-v2 regressor), then the
        min-max scaling is undone."""
        bins = self._bins.cpu()
        centers = bins[:-1] + (bins[1] - bins[0]) / 2
        y = (torch.softmax(logits.float(), dim=-1) * centers).sum(dim=-1)
        y_min, y_max = self._y_range
        return y * (y_max - y_min) + y_min


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
        
        general = self.args.config.get('general', {}) or {}
        max_samples_support = general.get('max_samples_support', 8192)
        max_samples_query = general.get('max_samples_query', 1024)

        if n_obs_support > max_samples_support:
            idx = torch.randperm(n_obs_support)[:max_samples_support] 
            self.x_support = self.x_support[idx, :]
            self.y_support = self.y_support[idx]
            n_obs_support = max_samples_support
        
        tic = time.time()
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
        self.predict_time = time.time() - tic

        test_logit = torch.cat(results, dim=0).cpu()
        if not self.is_regression:
            test_logit = test_logit[:, :self.n_classes]
        elif self._bins is not None:
            test_logit = self._decode_regression_bins(test_logit)
        test_label = self.y_test

        test_label_tensor = torch.tensor(test_label, dtype=torch.float32 if self.is_regression else torch.long)
        
        vl = self.criterion(test_logit, test_label_tensor).item()
        vres, metric_name = self.metric(test_logit, test_label, self.y_info)
        
        # Denormalize regression predictions back to original scale
        if self.is_regression and self.y_info.get('policy') == 'mean_std':
            test_logit = test_logit * self.y_info['std'] + self.y_info['mean']

        print('Test: loss={:.4f}'.format(vl))
        for name, res in zip(metric_name, vres):
            print('[{}]={:.4f}'.format(name, res))
        print('Time cost: {:.4f}s'.format(time.time() - start_time))
        return vl, vres, metric_name, test_logit