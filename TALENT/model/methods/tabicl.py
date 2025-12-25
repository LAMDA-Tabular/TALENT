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

def check_softmax(logits):
    """
    Check if the logits are already probabilities, and if not, convert them to probabilities.
    
    :param logits: np.ndarray of shape (N, C) with logits
    :return: np.ndarray of shape (N, C) with probabilities
    """
    # Check if any values are outside the [0, 1] range and Ensure they sum to 1
    if np.any((logits < 0) | (logits > 1)) or (not np.allclose(logits.sum(axis=-1), 1, atol=1e-5)):
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # stabilize by subtracting max
        return exps / np.sum(exps, axis=1, keepdims=True)
    else:
        return logits
    
class TabICLMethod(Method):
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

    def construct_model(self, model_config = None,cat_indices=[]):
            from TALENT.model.lib.tabicl.classifier import TabICLClassifier
            self.model = TabICLClassifier(
                device=self.args.device,
                random_state=self.args.seed,
                checkpoint_version="tabicl-classifier-v1.1-0506.ckpt",
                model_path="./TALENT/model/models/models_tabicl/tabicl-classifier-v1.1-0506.ckpt",
                n_estimators=32,                  # number of ensemble members
                norm_methods=["none", "power"],   # normalization methods to try
                feat_shuffle_method="latin",      # feature permutation strategy
                class_shift=True,                 # whether to apply cyclic shifts to class labels
                outlier_threshold=4.0,            # z-score threshold for outlier detection and clipping
                softmax_temperature=0.9,          # controls prediction confidence
                average_logits=True,              # whether ensemble averaging is done on logits or probabilities
                use_hierarchical=True,            # enable hierarchical classification for datasets with many classes
                batch_size=8,                     # process this many ensemble members together (reduce RAM usage)
                use_amp=True,                     # use automatic mixed precision for faster inference
                allow_auto_download=True,         # whether automatic download to the specified path is allowed
                n_jobs=None,                      # number of threads to use for PyTorch
                verbose=False,                    # print progress messages
                inference_config=None,            # inference configuration for fine-grained control
            )

    def fit(self, data, info, train = True, config = None):
        N, C, y = data
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
        self.sampled_X = sampled_X#[:sample_size]
        self.sampled_Y = sampled_Y# [:sample_size]
        self.construct_model(cat_indices=cat_indices)
        self.model.fit(self.sampled_X,self.sampled_Y)
        self.fit_time = 0  # general model does not require fitting

    def predict(self, data, info, model_name):
        import time
        start_time = time.time()
        N,C,y = data
        self.data_format(False, N, C, y)
        if self.N_test is not None and self.C_test is not None:
            Test_X = np.concatenate((self.N_test,self.C_test),axis=1)
        elif self.N_test is None and self.C_test is not None:
            Test_X = self.C_test
        else:
            Test_X = self.N_test
            
        if self.is_regression:
            test_logit = self.model.predict(Test_X)
        else:
            test_logit = self.model.predict_proba(Test_X)
        test_logit = test_logit.astype(np.float32)
        test_label = self.y_test
        vl = self.criterion(torch.tensor(test_logit),torch.tensor(test_label)).item()
        vres, metric_name = self.metric(test_logit, test_label, self.y_info)

        # FIX: Denormalize regression predictions
        if self.is_regression and self.y_info.get('policy') == 'mean_std':
            test_logit = test_logit * self.y_info['std'] + self.y_info['mean']

        print('Test: loss={:.4f}'.format(vl))
        for name, res in zip(metric_name, vres):
            print('[{}]={:.4f}'.format(name, res))
        print('Time cost: {:.4f}s'.format(time.time() - start_time))
        return vl, vres, metric_name, test_logit
