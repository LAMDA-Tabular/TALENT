
# Contributor Guidelines

## 🤗Contributors 

[![Contributors](https://contrib.rocks/image?repo=LAMDA-Tabular/TALENT)](https://github.com/LAMDA-Tabular/TALENT/graphs/contributors)

## 🌟 How to Contribute

We welcome contributions from the community! There are several ways to contribute:

### New Features

If you'd like to implement a new feature:
1. **First** open a feature request issue to discuss your proposal with committers
2. After approval, implement your feature and submit a pull request

### 🐞 Bug Fixes
For bug fixes:
- Submit a pull request with a clear description of the bug
- Include steps to reproduce if applicable
- If unsure whether something is a bug, open an issue first

### ➕ Adding New Methods
To add new methods to the codebase:

For methods similar to MLP (only model design needed):
1. Add model class in `TALENT/model/models/`
2. Inherit from `TALENT/model/methods/base.py` and override `construct_model()`
3. Register the method in the unified registry at `TALENT/model/method_registry.py`
   by appending a `MethodSpec(...)` entry (declaring `cat_policy`,
   `normalization`, `output_type`, `supports_hpo`, etc.). The CLI argparse
   `choices` lists and `get_method()` are both derived from this registry,
   so no other dispatcher edits are needed.
4. Add parameter settings in:
   - `TALENT/configs/default/[MODEL_NAME].json`
   - `TALENT/configs/opt_space/[MODEL_NAME].json`

For methods requiring training process changes:
- Partially override functions from `TALENT/model/methods/base.py`
- Refer to existing implementations in `TALENT/model/methods/`

For pre-trained foundation models with a scikit-learn style estimator
(`fit` / `predict` / `predict_proba`):
- Inherit from `SklearnFoundationMethod` in `TALENT/model/methods/foundation_base.py`
  and implement only `construct_model(model_config=None, cat_indices=None)`
  (see e.g. `TALENT/model/methods/causilo.py`). Data formatting, the training-row cap,
  fitting, prediction and metrics are shared.
- Register it with `normalization="none"`, `num_policy="none"`,
  `cat_policy=("indices",)` and `supports_hpo=False`, set
  `train_row_limit` if the model has a context-size limit, and describe
  where the pre-trained weights come from in `weights=` (shown to users
  when loading them fails). Weights should download automatically; use
  `download_from_hf_hub` in `TALENT/model/method_registry.py` for Hugging Face
  checkpoints that are not bundled.
- Read the ensemble size from `config['general']['n_estimators']`
  (`resolve_n_estimators` in `foundation_base.py`), so the option has the
  same name for every foundation model.



## Committer

- Si-Yang Liu



## 🏁 Getting Started

### Installation
```bash
git clone https://github.com/LAMDA-Tabular/TALENT.git
cd TALENT
python -m pip install -e .
```

### Running Experiments
1. To override the default settings for a method, edit its configuration files:
   - `TALENT/configs/default/[MODEL_NAME].json`
   - `TALENT/configs/opt_space/[MODEL_NAME].json`

2. Run:
```bash
# For deep methods
python test/train_model_deep.py --model_type mlp --dataset bank --dataset_path example_datasets --seed_num 1 --max_epoch 1

# For classical methods
python test/train_model_classical.py --model_type LogReg --dataset bank --dataset_path example_datasets --seed_num 1
```

### Building the documentation

Read the Docs builds the reStructuredText sources under `docs/`. The checked-in
`docs/_build/` directory is an older snapshot. To preview the current source,
install TALENT and the documentation dependencies, then build into a fresh directory:

```bash
python -m pip install -e . -r docs/requirements.txt
python -m sphinx -b html docs ../talent-docs-html
```

## 📬 Contact

For questions or feature proposals:
- Open an issue on GitHub
- Contact the maintainers:
  - Si-Yang Liu ([liusy@lamda.nju.edu.cn](mailto:liusy@lamda.nju.edu.cn))
  - Hao-Run Cai ([caihr@smail.nju.edu.cn](mailto:caihr@smail.nju.edu.cn))
  - Qile Zhou ([zhouql@lamda.nju.edu.cn](mailto:zhouql@lamda.nju.edu.cn))
  - Jun-Peng Jiang ([jiangjp@lamda.nju.edu.cn](mailto:jiangjp@lamda.nju.edu.cn))
  - Huai-Hong Yin ([yinhh@lamda.nju.edu.cn](mailto:yinhh@lamda.nju.edu.cn))
  - Tao Zhou ([zhout@lamda.nju.edu.cn](mailto:zhout@lamda.nju.edu.cn))
  - Han-Jia Ye ([yehj@lamda.nju.edu.cn](mailto:yehj@lamda.nju.edu.cn))
