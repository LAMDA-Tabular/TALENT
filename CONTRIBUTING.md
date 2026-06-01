
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
1. Add model class in `model/models/`
2. Inherit from `model/methods/base.py` and override `construct_model()`
3. Register the method in the unified registry at `model/method_registry.py`
   by appending a `MethodSpec(...)` entry (declaring `cat_policy`,
   `normalization`, `output_type`, `supports_hpo`, etc.). The CLI argparse
   `choices` lists and `get_method()` are both derived from this registry,
   so no other dispatcher edits are needed.
4. Add parameter settings in:
   - `configs/default/[MODEL_NAME].json`
   - `configs/opt_space/[MODEL_NAME].json`

For methods requiring training process changes:
- Partially override functions from `model/methods/base.py`
- Refer to existing implementations in `model/methods/`



## Committer

- Si-Yang Liu



## 🏁 Getting Started

### Installation
```bash
pip install git+https://github.com/LAMDA-Tabular/TALENT.git@main --upgrade
```

### Running Experiments
1. Edit configuration files:
   - `configs/default/[MODEL_NAME].json`
   - `configs/opt_space/[MODEL_NAME].json`

2. Run:
```bash
# For deep methods
python train_model_deep.py --model_type MODEL_NAME

# For classical methods
python train_model_classical.py --model_type MODEL_NAME
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
