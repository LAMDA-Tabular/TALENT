from TALENT.model.methods.tabpfn_v2 import TabPFNMethod
from TALENT.model.method_registry import resolve_bundled_path, download_from_hf_hub

# Real-TabPFN: TabPFN v2 classifier with continued pre-training on real data.
REAL_TABPFN_REPO = "Prior-Labs/TabPFN-v2-clf"
REAL_TABPFN_CHECKPOINT = "tabpfn-v2-classifier-finetuned-zk73skhh.ckpt"


class TabPFNRealMethod(TabPFNMethod):
    def construct_model(self, model_config = None,cat_indices=None):
        cat_indices = cat_indices or []
        if self.is_regression:
            raise ValueError("TabPFN-Real only supports classification tasks.")
        else:
            from TALENT.model.models.tabpfn_v2 import TabPFNClassifier
            from TALENT.model.methods.foundation_base import resolve_n_estimators
            # Bundled checkpoint if present, otherwise download the Real-TabPFN
            # checkpoint itself (the TabPFN library's model_path="auto" would
            # load the plain TabPFN v2 classifier instead).
            model_path = resolve_bundled_path(
                f"model/models/models_tabpfn/{REAL_TABPFN_CHECKPOINT}"
            ) or download_from_hf_hub(REAL_TABPFN_REPO, REAL_TABPFN_CHECKPOINT)
            self.model = TabPFNClassifier(
                model_path = model_path,
                device = self.args.device,
                random_state = self.args.seed,
                n_estimators = resolve_n_estimators(self.args, 4),
                ignore_pretraining_limits = True,
                categorical_features_indices = cat_indices
            )
