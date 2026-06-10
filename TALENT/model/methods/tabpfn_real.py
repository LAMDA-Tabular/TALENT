from TALENT.model.methods.tabpfn_v2 import TabPFNMethod
from TALENT.model.method_registry import resolve_bundled_path


class TabPFNRealMethod(TabPFNMethod):
    def construct_model(self, model_config = None,cat_indices=None):
        cat_indices = cat_indices or []
        if self.is_regression:
            raise ValueError("TabPFN-Real only supports classification tasks.")
        else:
            from TALENT.model.models.tabpfn_v2 import TabPFNClassifier
            model_path = resolve_bundled_path(
                "model/models/models_tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt"
            ) or "auto"
            self.model = TabPFNClassifier(
                model_path = model_path,
                device = self.args.device,
                random_state = self.args.seed,
                n_estimators = 4,
                ignore_pretraining_limits = True,
                categorical_features_indices = cat_indices
            )