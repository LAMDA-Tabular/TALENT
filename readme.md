<p align="center">
<img src="./resources/TALENT-LOGO.png"  width="1000px">
</p>
<p align="center">
    <a href='https://arxiv.org/abs/2407.04057'><img src='https://img.shields.io/badge/Arxiv-2407.04057-b31b1b.svg?logo=arXiv'></a>
    <a href='https://zhuanlan.zhihu.com/p/708721145'><img src='https://img.shields.io/badge/中文解读-b0.svg?logo=zhihu'></a>
    <a href='https://huggingface.co/datasets/LAMDA-Tabular/TALENT'><img src='https://img.shields.io/badge/%F0%9F%A4%97-TALENT-green'></a>
  <a href="https://github.com/LAMDA-Tabular/TALENT/stargazers"><img src="https://img.shields.io/github/stars/LAMDA-Tabular/TALENT?color=4fb5ee" alt="GitHub stars"></a>
  <a href="https://github.com/LAMDA-Tabular/TALENT/commits/main/"><img src="https://img.shields.io/github/last-commit/LAMDA-Tabular/TALENT?color=blue" alt="Latest commit"></a>
   <br>
    <img src="https://img.shields.io/badge/PYTORCH-2.8%2B-red?style=for-the-badge&logo=pytorch" alt="PyTorch - Version" height="21">
    <img src="https://img.shields.io/badge/PYTHON-3.10-red?style=for-the-badge&logo=python&logoColor=white" alt="Python - Version" height="21">
    <a href='https://lamda-talent.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/lamda-talent/badge/?version=latest' alt='Documentation Status' />
    </a><a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT license"></a>
</p>
<div align="center">
    <p>
        TALENT: A Tabular Analytics and Learning Toolbox
    <p>
    <p>
        <a href="https://arxiv.org/abs/2407.04057">[Paper]</a> <a href="https://zhuanlan.zhihu.com/p/708721145">[中文解读]</a> <a href="https://lamda-talent.readthedocs.io/en/latest">[Docs]</a> 
    <p>
</div>





---

## 🎉 Introduction

Welcome to **TALENT**, a benchmark with a comprehensive machine learning toolbox designed to enhance model performance on tabular data. TALENT integrates advanced deep learning models, classical algorithms, and efficient hyperparameter tuning, offering robust preprocessing capabilities to optimize learning from tabular datasets. The toolbox is user-friendly and adaptable, catering to both novice and expert data scientists.

**TALENT** offers the following advantages:

- **Diverse Methods**: Includes various classical methods, tree-based methods, and the latest popular deep learning methods.
- **Extensive Dataset Collection**: Equipped with 300 datasets, covering a wide range of task types, size distributions, and dataset domains.
- **Customizability**: Easily allows the addition of datasets and methods.
- **Versatile Support**: Supports diverse normalization, encoding, and metrics.

## 📚Citing TALENT

**If you use any content of this repo for your work, please cite the following bib entries:**

```bibtex
@article{ye2024closerlookdeeplearning,
         title={A Closer Look at Deep Learning on Tabular Data}, 
         author={Han-Jia Ye and 
         		 Si-Yang Liu and 
         		 Hao-Run Cai and 
         		 Qi-Le Zhou and 
         		 De-Chuan Zhan},
         journal={arXiv preprint arXiv:2407.00956},
         year={2024}
}

@article{JMLR:v26:25-0512,
  author  = {Si-Yang Liu and
			 Hao-Run Cai and
 			 Qi-Le Zhou and
			 Huai-Hong Yin and
			 Tao Zhou and
			 Jun-Peng Jiang and
			 Han-Jia Ye},
  title   = {Talent: A Tabular Analytics and Learning Toolbox},
  journal = {Journal of Machine Learning Research},
  year    = {2025},
  volume  = {26},
  number  = {226},
  pages   = {1--16},
  url     = {http://jmlr.org/papers/v26/25-0512.html}
}
```




## 📰 What's New

- [2026-11] Added [cross-validation](#cross-validation) with stratified, grouped and temporal folds, inner threshold selection, [fold-averaged Optuna tuning and resumable studies](#tuning-across-folds-and-resuming-a-run). Added training, HPO and inference timing, [replayable splits and matched comparisons](#reproducible-benchmarks), and five foundation model releases (Seldon, Seldon-mini, Inpriora, LimiX-2-1016 and TabDPT 1.3). Fixed classification scoring, group ID handling, target leakage in leave-one-out and ordered categorical encoding, most-frequent imputation for empty categorical columns, HPO metric and trial validation, and training or prediction errors in TabR, GRANDE, PTaRL, TabCaps, XGBoost and TabAutoPNPNet.
  TabICL v1 now validates data across supported scikit-learn versions and honours thread limits during prediction and restores the previous setting afterwards. TabFM can score query rows in chunks. The Python API can return validation predictions and use custom scorers for reporting, tuning and threshold selection. TabPTM classification now supplies its query labels to the threshold evaluator.
  Also fixed ProtoGate vote probabilities, TabPTM training on small datasets and in single precision, regression with column-vector targets in Mitra and TabPFN-v2, and TabPFN-v2 input validation and CPU memory estimation. RFM and xRFM now index expanded features correctly and honour checkpoint-only loading. Resumable runs detect changes to local weights supplied through `checkpoint_path`.

- [2026-10]🌟 Add the September 2026 wave of tabular foundation models: **[TabPFN-3.5](https://github.com/PriorLabs/TabPFN)**, **[LimiX-2](https://github.com/limix-ldm-ai/LimiX)**, **[NVIDIA Kumo Tabular](https://github.com/NVIDIA/structured-data-models)**, **[Mitra-v2](https://arxiv.org/abs/2609.04540)**, **[Causilo](https://github.com/nums-ai/causilo)**, **[Xiaomi-TabLDM](https://github.com/xiaomi-research/xiaomi-tabldm)** and **[EXAONE-Tabular](https://github.com/LGAI-Research/EXAONE-Tabular)**. The new adapters document their weight sources and supported options; `tabpfn_v3` now pins the TabPFN-3 checkpoint (`tabpfn>=9.0` changed the package default to TabPFN-3.5).
- [2026-08]🌟 Add **[TRC](https://arxiv.org/abs/2603.16569)** (TPAMI 2026), a post-hoc representation corrector that trains and freezes an FT-Transformer backbone before learning representation re-estimation and light-space mapping.
- [2026-07]🌟 Add **[TabFM](https://github.com/google-research/tabfm)** (Google Research, zero-shot tabular foundation model) as an optional method via `pip install -U "tabfm[pytorch]"`.
- [2026-06]🌟 Add **[TabPFN v2.5](https://arxiv.org/abs/2511.08667)** (PriorLabs, Nov 2025) and **[TabDPT](https://github.com/layer6ai-labs/TabDPT-inference)** (Layer 6 AI, ICL + retrieval). TALENT now ships the full TabPFN family (v1, v2, v2.5, v3) plus TabDPT alongside the other tabular foundation models.
- [2026-06]🌟 Evaluation improvements: **calibration metrics** (Brier, ECE) are now reported by default for every classifier, **decision thresholds** are tuned on the validation set for binary classification (used for Accuracy/F1/Precision/Recall; threshold-independent metrics like AUC/LogLoss/Brier/ECE are unaffected), and `RunResult` exposes a uniform `predict_proba`/`predict_labels` interface regardless of whether the underlying method natively returns logits or probabilities. Bundled checkpoints are now resolved via `importlib.resources` so methods work from any working directory.
- [2026-05]🌟 Add [TabPFN v3](https://github.com/PriorLabs/TabPFN) (PriorLabs 2026) and [TabICL v2](https://github.com/soda-inria/tabicl) (ICML 2026, regression support added).
- [2026-03]🌟 We have updated the TALENT-extension datasets and results. [Link](https://box.nju.edu.cn/d/b7b23a19ee054aaba7b6/?p=%2F&mode=list)
- [2025-11]🌟 Add [RFM](https://www.science.org/doi/10.1126/science.adi5639) (Science).
- [2025-11]🌟 Add [Real-TabPFN](https://arxiv.org/abs/2507.03971).
- [2025-11]🌟 Add [LimiX](https://arxiv.org/abs/2509.03505).
- [2025-09]🌟 Add [xRFM](https://arxiv.org/abs/2508.10053).
- [2025-08]🌟 Add [Mitra](https://www.amazon.science/blog/mitra-mixed-synthetic-priors-for-enhancing-tabular-foundation-models).
- [2025-06]🌟 Add [TabAutoPNPNet](https://www.mdpi.com/2079-9292/14/6/1165) (Electronics 2025).
- [2025-06]🌟 Add [TabICL](https://arxiv.org/abs/2502.05564) (ICML 2025). The current code is based on TabICL v0.1.2.
- [2025-05]🌟 Check out our three papers **[MMTU](https://github.com/LAMDA-Tabular/MMTU)**, **[Tabular-Temporal-Shift](https://github.com/LAMDA-Tabular/Tabular-Temporal-Shift)**, and [**BETA**](https://github.com/LAMDA-Tabular/BETA) accepted at ICML 2025!
- [2025-04]🌟 Check out our new survey [Representation Learning for Tabular Data: A Comprehensive Survey](https://arxiv.org/abs/2504.16109) ([Repo](https://github.com/LAMDA-Tabular/Tabular-Survey)). We organize existing methods into three main categories according to their generalization capabilities: specialized, transferable, and general models, which provides a comprehensive taxonomy for deep tabular representation methods.🚀🚀🚀
- [2025-02]🌟 Add [T2Gformer](https://arxiv.org/abs/2211.16887) (AAAI 2023).
- [2025-02]🌟 Add [TabPFN v2](https://doi.org/10.1038/s41586-024-08328-6) (Nature).
- [2025-02]🌟 Thanks to [Hengzhe Zhang](https://hengzhe-zhang.github.io/) for providing a [Scikit-Learn compatible wrapper](https://github.com/hengzhe-zhang/scikit-TALENT) for TALENT!
- [2025-01]🌟 Check out our new baseline [ModernNCA](https://openreview.net/pdf?id=JytL2MrlLT) (**ICLR 2025**), inspired by traditional **Neighbor Component Analysis**, which outperforms both tree-based and other deep tabular models, while also reducing training time and model size!🚀🚀🚀
- [2025-01]🌟 Check out our [latest version of the benchmark paper](https://arxiv.org/abs/2407.00956) for updated and expanded results and analysis!
- [2025-01]🌟We have curated and released [new benchmark datasets](https://drive.google.com/drive/folders/1j1zt3zQIo8dO6vkO-K-WE6pSrl71bf0z?usp=drive_link), along with updated [results](https://6sy666.github.io/TALENT-Results/) of the dataset across a broader range of methods. This update focuses on enhancing dataset quality, including removing duplicates, and correcting tasks where bin-class was mistakenly treated as regression. We have also separated the larger datasets and formed the basic benchmark (300 datasets, including 120 bin-class, 80 multi-class, and 100 regression), and the large benchmark (22 datasets).
- [2024-12]🌟 Add [TabM](https://arxiv.org/abs/2410.24210) (ICLR 2025).
- [2024-09]🌟 Add [Trompt](https://arxiv.org/abs/2305.18446) (ICML 2023).
- [2024-09]🌟 Add [AMFormer](https://arxiv.org/abs/2402.02334) (AAAI 2024).
- [2024-08]🌟 Add [GRANDE](https://arxiv.org/abs/2309.17130) (ICLR 2024).
- [2024-08]🌟 Add [Excelformer](https://arxiv.org/abs/2301.02819) (KDD 2024).
- [2024-08]🌟 Add [MLP_PLR](https://arxiv.org/abs/2203.05556) (NeurIPS 2022).
- [2024-07]🌟 Add [RealMLP](https://arxiv.org/abs/2407.04491)(NeurIPS 2024).
- [2024-07]🌟 Add [ProtoGate](https://arxiv.org/abs/2306.12330) (ICML 2024).
- [2024-07]🌟 Add [BiSHop](https://arxiv.org/abs/2404.03830) (ICML 2024).
- [2024-06]🌟 Check out our new baseline [ModernNCA](https://arxiv.org/abs/2407.03257), inspired by traditional **Neighbor Component Analysis**, which outperforms both tree-based and other deep tabular models, while also reducing training time and model size!
- [2024-06]🌟 Check out our [benchmark paper](https://arxiv.org/abs/2407.00956) about tabular data, which provides comprehensive evaluations of classical and deep tabular methods based on our toolbox in a fair manner!

## 🌟 Methods

TALENT integrates an extensive array of 30+ deep learning architectures for tabular data, including but not limited to:

1. **MLP**: A multi-layer neural network, which is implemented according to [RTDL](https://arxiv.org/abs/2106.11959).
2. **ResNet**: A DNN that uses skip connections across many layers, which is implemented according to [RTDL](https://arxiv.org/abs/2106.11959).
3. **[SNN](https://arxiv.org/abs/1706.02515)**: An MLP-like architecture utilizing the SELU activation, which facilitates the training of deeper neural networks.
4. **[DANets](https://arxiv.org/abs/2112.02962)**: A neural network designed to enhance tabular data processing by grouping correlated features and reducing computational complexity.
5. **[TabCaps](https://openreview.net/pdf?id=OgbtSLESnI)**: A capsule network that encapsulates all feature values of a record into vectorial features. Its classifier returns probabilities. Prior versions applied softmax to those probabilities a second time in the public runner.
6. **[DCNv2](https://arxiv.org/abs/2008.13535)**: Consists of an MLP-like module combined with a feature crossing module, which includes both linear layers and multiplications.
7. **[NODE](https://arxiv.org/abs/1909.06312)**: A tree-mimic method that generalizes oblivious decision trees, combining gradient-based optimization with hierarchical representation learning.
8. **[GrowNet](https://arxiv.org/abs/2002.07971)**: A gradient boosting framework that uses shallow neural networks as weak learners. In the historical classification training code, a broadcasting error pairs scores with labels from other rows in the same batch. This affects the interpretation of classification results. Regression uses a separate training objective.
9. **[TabNet](https://arxiv.org/abs/1908.07442)**: A tree-mimic method using sequential attention for feature selection, offering interpretability and self-supervised learning capabilities.
10. **[TabR](https://arxiv.org/abs/2307.14338)**: A deep learning model that integrates a KNN component to enhance tabular data predictions through an efficient attention-like mechanism.
11. **[ModernNCA](https://arxiv.org/abs/2407.03257)**: A deep tabular model inspired by traditional Neighbor Component Analysis, which makes predictions based on the relationships with neighbors in a learned embedding space.
12. **[DNNR](https://arxiv.org/abs/2205.08434)**: Enhances KNN by using local gradients and Taylor approximations for more accurate and interpretable predictions.
13. **[AutoInt](https://arxiv.org/abs/1810.11921)**: A token-based method that uses a multi-head self-attentive neural network to automatically learn high-order feature interactions.
14. **[Saint](https://arxiv.org/abs/2106.01342)**: A token-based method that leverages row and column attention mechanisms for tabular data.
15. **[TabTransformer](https://arxiv.org/abs/2012.06678)**: A token-based method that enhances tabular data modeling by transforming categorical features into contextual embeddings.
16. **[FT-Transformer](https://arxiv.org/abs/2106.11959)**: A token-based method which transforms features to embeddings and applies a series of attention-based transformations to the embeddings.
17. **[TANGOS](https://openreview.net/pdf?id=n6H86gW8u0d)**: A regularization-based method for tabular data that uses gradient attributions to encourage neuron specialization and orthogonalization.
18. **[SwitchTab](https://arxiv.org/abs/2401.02013)**: A self-supervised method tailored for tabular data that improves representation learning through an asymmetric encoder-decoder framework. Following the original paper, our toolkit uses a supervised learning form, optimizing both reconstruction and supervised loss in each epoch.
19. **[PTaRL](https://openreview.net/pdf?id=G32oY4Vnm8)**: A regularization-based framework that enhances prediction by constructing and projecting into a prototype-based space. New Optuna trials enable the configured regularizers. Earlier tuned configurations could disable them by storing their flags as strings.
20. **[TabPFN](https://arxiv.org/abs/2207.01848)**: A general model which involves the use of pre-trained deep neural networks that can be directly applied to any tabular task.
21. **[HyperFast](https://arxiv.org/abs/2402.14335)**: A meta-trained hypernetwork that generates task-specific neural networks for instant classification of tabular data.
22. **[TabPTM](https://arxiv.org/abs/2311.00055)**: A general method for tabular data that standardizes heterogeneous datasets using meta-representations, allowing a pre-trained model to generalize to unseen datasets without additional training.
23. **[BiSHop](https://arxiv.org/abs/2404.03830)**: An end-to-end framework for deep tabular learning which leverages a sparse Hopfield model with adaptable sparsity, enhanced by column-wise and row-wise modules.
24. **[ProtoGate](https://arxiv.org/abs/2306.12330)**: A prototype-based model for feature selection in HDLSS biomedical data that adapts global and local feature selection to enhance prediction accuracy and interpretability, addressing co-adaptation issues through a non-parametric prototype-based mechanism.
25. **[RealMLP](https://arxiv.org/abs/2407.04491)**: An improved multilayer perceptron (MLP).
26. **[MLP_PLR](https://arxiv.org/abs/2203.05556)**: An improved multilayer perceptron (MLP), which utilizes periodic activations.
27. **[Excelformer](https://arxiv.org/abs/2301.02819)**: A deep learning model for tabular data prediction, featuring a semi-permeable attention module to address rotational invariance, tailored data augmentation, and an attentive feedforward network, making it a reliable solution across diverse datasets.
28. **[GRANDE](https://arxiv.org/abs/2309.17130)**: A tree-mimic method for learning hard, axis-aligned decision tree ensembles using end-to-end gradient descent.
29. **[AMFormer](https://arxiv.org/abs/2402.02334)**: A token-based method which improves the transformer architecture for tabular data by incorporating parallel addition and multiplication attention mechanisms, utilizing prompt tokens to constrain feature interactions.
30. **[Trompt](https://arxiv.org/abs/2305.18446)**: A prompt-based deep neural network for tabular data that separates learning into intrinsic column features and sample-specific feature importance.
31. **[TabM](https://arxiv.org/abs/2410.24210)** :  A model based on MLP and variations of BatchEnsemble.
32. **[TabPFN v2](https://doi.org/10.1038/s41586-024-08328-6)**: A general model which involves the use of pre-trained deep neural networks that can be directly applied to any tabular task.
33. **[T2Gformer](https://arxiv.org/abs/2211.16887)**: A Transformer network for tabular learning that processes data guided by relation graphs and uses a Cross-level Readout for global semantics in prediction.
34. **[TabICL](https://arxiv.org/abs/2502.05564)**: A comparable tabular foundation model with performance on par with TabPFN v2.
35. **[TabAutoPNPNet](https://www.mdpi.com/2079-9292/14/6/1165)**: A tabular model based on periodicity, particularly the Fourier transform and Chebyshev polynomials, with performance on par with or superior to FT-Transformer. 
36. **[Mitra](https://www.amazon.science/blog/mitra-mixed-synthetic-priors-for-enhancing-tabular-foundation-models)**: A tabular foundation model learned with mixed synthetic priors.
37. **[LimiX](https://arxiv.org/abs/2509.03505)**: A tabular foundation model that leverages transformers to support a wide range of tasks, from prediction to imputation and causal inference, within a unified architecture.
38. **[Real-TabPFN](https://arxiv.org/abs/2507.03971)**: An enhanced tabular foundation model that extends TabPFNv2 through continued pre-training on real-world datasets for classification tasks. 
39. **[RFM](https://www.science.org/doi/10.1126/science.adi5639)**: A non-deep, backpropagation-free feature learning algorithm, iteratively applies AGOP to a kernel machine to adaptively learn task-specific features. To reload RFM with `train=False`, supply the original training data so the model can reconstruct its kernel centres. Older RFM checkpoint files that contain no fitted weights require a fresh fit.
40. **[xRFM](https://arxiv.org/abs/2508.10053)**: A tabular model that combines RFMs with an adaptive tree structure, enabling it to learn features local to data subsets and scale log-linearly with the number of samples.
41. **[TabPFN v3](https://github.com/PriorLabs/TabPFN)**: TabPFN v3 (PriorLabs 2026), with native context up to ~1M rows × 200 features, 160-class support, and an optional thinking mode. Requires `pip install -U 'tabpfn>=8.0.0'`.
42. **[TabICL v2](https://github.com/soda-inria/tabicl)**: TabICL v2 (ICML 2026), now supporting both classification and regression (via `TabICLRegressor`), with native quantile regression. Requires `pip install -U 'tabicl>=2.0.0'`.
43. **[TabPFN v2.5](https://arxiv.org/abs/2511.08667)**: TabPFN v2.5 (PriorLabs, Nov 2025), the intermediate release between v2 and v3. Scales in-context learning to ~50k rows × 2k features. Requires `pip install -U 'tabpfn>=8.0.0'`.
44. **[TabDPT](https://github.com/layer6ai-labs/TabDPT-inference)**: A tabular foundation model from Layer 6 AI that combines in-context learning with retrieval and self-supervised pre-training on real data, removing fixed context-size limits. Requires `pip install -U tabdpt`.
45. **[TabFM](https://github.com/google-research/tabfm)**: Google's zero-shot tabular foundation model for classification and regression, using in-context learning over training rows as context. Requires `pip install -U "tabfm[pytorch]"`; the upstream model weights use the TabFM Non-Commercial License.
46. **[TRC](https://arxiv.org/abs/2603.16569)**: A post-hoc Tabular Representation Corrector that freezes a trained backbone, learns to remove representation shift, and maps corrected representations into a compact light-embedding space.
47. **[TabPFN-3.5](https://github.com/PriorLabs/TabPFN)**: TabPFN-3.5 (PriorLabs, Sep 2026), the successor of TabPFN-3 with native context up to ~1M rows × 20k features; set `general.fast` for the TabPFN-3.5-Fast checkpoint. Requires `pip install -U 'tabpfn>=9.0.0'`; the weights use a non-commercial license (for headless runs, set `TABPFN_TOKEN` after accepting it on the Prior Labs platform).
48. **[Kumo Tabular](https://github.com/NVIDIA/structured-data-models)**: NVIDIA's tabular foundation model (Sep 2026, small / medium / large via `general.size`) for classification and regression; weights under the commercially usable OpenMDW-1.1 license. Requires Python >= 3.11 and `pip install git+https://github.com/NVIDIA/structured-data-models.git`.
49. **[Mitra-v2](https://arxiv.org/abs/2609.04540)**: The second-generation Mitra (AutoGluon, Sep 2026) for classification and regression (1000-bin distributional head), run zero-shot through TALENT's Mitra code.
50. **[Causilo](https://github.com/nums-ai/causilo)**: A tabular foundation model from Nums AI (Sep 2026) for classification and regression. Requires `pip install -U causilo` (PyTorch >= 2.13); the weights use the non-commercial Causilo License.
51. **[Xiaomi-TabLDM](https://github.com/xiaomi-research/xiaomi-tabldm)**: Xiaomi's tabular foundation model (Sep 2026) for classification and regression. Requires `pip install git+https://github.com/xiaomi-research/xiaomi-tabldm.git`; check the model card for the weights license.
52. **[EXAONE-Tabular](https://github.com/LGAI-Research/EXAONE-Tabular)**: LG AI Research's tabular foundation model (2026) for classification and regression. Requires Python >= 3.11 and `pip install "exaonetabular @ git+https://github.com/LGAI-Research/EXAONE-Tabular.git"`; the weights are for research and education only. The released regressor can fail the publisher’s quantile-grid check on Windows with CPU or CUDA. Classification with the public weights was verified on CPU.
53. **[LimiX-2](https://github.com/limix-ldm-ai/LimiX)**: The 400M-parameter successor of LimiX (StableAI, Sep 2026) for classification (2-10 classes) and regression. Requires Python >= 3.12 and `pip install git+https://github.com/limix-ldm-ai/LimiX.git`; the weights use the StableAI LimiX Non-Commercial License.


🔧 If you want to check the **default hyperparameters and hyperparameter search spaces** of all methods, please visit:  
👉 [https://6sy666.github.io/TALENT-Configs/](https://6sy666.github.io/TALENT-Configs/)

## ☄️ How to Use TALENT

### 🕹️ Quick Start

Install with the newest version through GitHub:

```bash
python -m pip install --upgrade git+https://github.com/LAMDA-Tabular/TALENT.git@main
```

From a source checkout, the CLI scripts live in `test/` and sample data in `example_datasets/`. The deep-learning launcher follows this workflow:

```python
from tqdm import tqdm
from TALENT.benchmark.measurement import elapsed_call
from TALENT.model.utils import get_deep_args, show_results, tune_hyper_parameters, get_method, set_seeds
from TALENT.model.lib.data import get_dataset
from TALENT.model.lib.evaluation import evaluate
from TALENT.model.method_registry import get_method_spec

if __name__ == '__main__':
    loss_list, results_list, time_list = [], [], []
    args, default_para, opt_space = get_deep_args()
    train_val_data, test_data, info = get_dataset(args.dataset, args.dataset_path)
    if args.tune:
        args = tune_hyper_parameters(args, opt_space, train_val_data, info)

    spec = get_method_spec(args.model_type)
    for seed in tqdm(range(args.seed_num)):
        args.seed = seed    # update seed
        set_seeds(args.seed)
        method = get_method(args.model_type)(args, info['task_type'] == 'regression')
        _, fit_time = elapsed_call("fit_seconds", args.device, method.fit, train_val_data, info)
        # evaluate() standardizes predict_proba and (for binary tasks) tunes
        # the decision threshold on the validation split.
        eval_result = evaluate(
            method, train_val_data, test_data, info,
            model_name=args.evaluate_option,
            output_type=spec.output_type,
            tune_threshold=True,
        )
        loss_list.append(eval_result["loss"])
        results_list.append(eval_result["metrics"])
        metric_name = eval_result["metric_names"]
        time_list.append((fit_time, eval_result["predict_time"]))

    show_results(args, info, metric_name, loss_list, results_list, time_list)
```



```bash
python test/train_model_deep.py --model_type mlp --dataset bank --dataset_path example_datasets --seed_num 1 --max_epoch 1
```

For TRC, use integer-encoded categorical features as required by its
FT-Transformer backbone:

```bash
python test/train_model_deep.py --model_type trc --cat_policy indices --dataset bank --dataset_path example_datasets --seed_num 1
```

TRC first trains the backbone for at most `--max_epoch` epochs, freezes it,
and then trains the representation corrector for at most the same number of
epochs. Ablations and paper hyperparameters are available in
`TALENT/configs/default/trc.json`; set `shift_estimator`, `space_mapping`, or
`loss_orth` to `false` to disable the corresponding component.

### 🐍 Python API (library mode)

TALENT also exposes a library-style API alongside the CLI scripts, so you can call methods directly from Python or a Jupyter notebook without manipulating `sys.argv`:

```python
import TALENT
from TALENT.model.lib.data import get_dataset

train_val, test, info = get_dataset("bank", "./example_datasets")

# Single-seed run
result = TALENT.run("LogReg", train_val, test, info)
print(dict(zip(result.metric_names, result.metrics)))
print("Training (s)", result.fit_time, "HPO (s)", result.hpo_time, "Inference (s)", result.predict_time)

# Multi-seed run with hyperparameter tuning
result = TALENT.run("catboost", train_val, test, info, tune=True, n_trials=50, seed_num=3)
print("Mean metrics:", dict(zip(result.metric_names, result.metrics_mean)))
print("Std metrics:",  dict(zip(result.metric_names, result.metrics_std)))
```

#### Reproducible benchmarks

Use stable string row IDs to save the exact training, validation, test and context rows. This example compares logistic regression with the dummy baseline on the same 32 training rows and the same held-out data.

```python
import tempfile
from pathlib import Path

import numpy as np
from TALENT.benchmark import BenchmarkSplit, compare_runs, make_split, result_record

X = np.random.default_rng(7).normal(size=(100, 4))
y = (X[:, 0] > 0).astype(int)
ids = [f"row-{i}" for i in range(len(y))]
info = {"task_type": "binclass", "n_num_features": 4, "n_cat_features": 0}
split = make_split(ids, seed=42).with_context(32)
split.save("split.json")
split = BenchmarkSplit.load("split.json")

methods = ["LogReg", "dummy"]
records = []
with tempfile.TemporaryDirectory() as tmp:
    for method in methods:
        result = split.run(method, ids, X, None, y, info, device="cpu",
                           measure=True, save_path=str(Path(tmp) / method))
        records.append(result_record(result, "demo-v1", "Accuracy"))

report = compare_runs(records, expected={"demo-v1": [split.split_id]},
                      methods=methods, metric="Accuracy", reference="dummy")
print(report["ranking"], report["counts"], report["contrasts"])
```

**Choosing a split.** Pass `groups=` to keep whole groups together or `timestamps=` for chronological training, validation and test periods. Equal timestamps stay together. Validation and test fractions apply to the number of rows, groups or distinct timestamps, respectively. `with_context(k)` selects nested training subsets and preserves both holdouts. Temporal contexts use the most recent training rows. For classification, check class coverage when choosing small contexts. You can also construct `BenchmarkSplit(train, val, test, context)` with your own ordered ID sequences. The manifest stores IDs in plain text and fingerprints their assignments and order. Use non-sensitive IDs and record the dataset version separately. Replaying the manifest accepts reordered source data and checks that the row IDs match.

**Measuring a run.** Every run reports three durations in seconds. `result.fit_time` and `result.predict_time` measure the final fit and test prediction calls, averaged across seeds. Individual times are in `result.per_seed`. `result.hpo_time` measures the Optuna search once across all seeds, including candidate training and validation. It is zero when tuning is disabled or an existing tuned configuration is loaded. Both CLIs report the shared search as `HPO_Time`. CUDA timings synchronise the configured device. Test timing includes preprocessing and metrics inside the adapter's prediction call and excludes the evaluator's separate threshold search. For ICL methods, fit time measures support preparation and any loading performed during fit. Older versions relied on adapter-specific timings, so historical times can differ.

Pass `measure=True` for initialization, shared tuning time (including cache lookup), memory and provenance. Where instrumented, `load_seconds` measures backend initialization with checkpoint loading and overlaps fit time.

The record includes the PyTorch version, CUDA device name, fitted model device, optional backend version and checkpoint identity where observable. Set `hash_checkpoint=True` to hash a resolved local checkpoint file. Memory values are sampled peaks for the current process (RSS with optional `psutil`) and PyTorch allocations on the configured CUDA device. Short-lived allocations can fall between samples. Use one process per run for comparable memory measurements.

Set `capture_support_ids=True` on a measured `split.run` to retain the ordered IDs selected by TALENT's shared support sampler. Its `sample_size` option accepts a row count or a fraction for both classification and regression. Mitra also records the context passed to each test query after its inference cap. External backends can perform further per-query retrieval. Unavailable observations are `None`.

**Comparing methods.** Supply the full planned dataset and split schedule, including runs that produced no result. Record an unsuccessful run with `dataset`, `split_id`, `method`, `status` (`"failed"` or `"skipped"`) and an optional `reason`. The report lists every planned cell, including missing and non-finite results. Rankings use splits completed by every method, average splits within each dataset, then weight datasets equally. Fit, HPO and prediction costs use the same cells and include their measurement counts. Each reference comparison uses its own matched splits and reports per-dataset differences plus a paired Student *t* interval when at least two datasets contribute. These are unadjusted intervals across dataset means. Set `higher_is_better=False` for a loss. Positive differences always favor the method being compared with the reference.

#### Cross-validation

Use the row IDs, feature arrays and task information from the example above. The same runner accepts classical, neural and foundation model IDs.

```python
import TALENT

cv_result = TALENT.cross_validate(
    "LogReg", ids, X, None, y, info, cv=5, split_seed=42, device="cpu",
    threshold_cv=3, threshold_metric="f1",
)
print(cv_result.metrics_mean)
row_ids, probabilities = cv_result.out_of_fold("predict_proba")
```

For a TALENT-format dataset on disk, use `TALENT.cross_validate_from_dataset("LogReg", "bank", "./example_datasets", cv=5)`. It pools the supplied training and validation rows for CV and reserves the original test set. Row IDs such as `train/0` refer to positions in that dataset version. Supply any group or timestamp metadata in training-then-validation order.

- **Outer folds.** Ordinary classification uses stratified folds and regression uses shuffled folds. Pass `groups=group_ids` to keep entire groups together, or `timestamps=dates` for expanding-window evaluation. Equal timestamps stay together. Optional `gap=k` leaves `k` distinct timestamp blocks between training and validation and between validation and test. Initial history and gap rows receive predictions only if they later appear in a test fold.
- **Model and threshold selection.** Each fit reserves `val_fraction` (default 0.2) of its development rows, groups or time blocks for validation. Optuna uses this partition when `tune=True`, for example with `n_trials=20, tune_metric="AUC"`. By default, binary classifiers also select their threshold on validation. Set `threshold_cv=3` to pool out-of-fold probabilities from three inner fits on outer development data and choose one threshold per seed. Inner fits have their own validation data, group or time boundaries, and HPO directories. The final fit uses the outer training and validation partitions and applies that threshold to the outer test predictions. `tune_threshold=False` disables selection.
- **Results and cost.** `results` contains one standard `RunResult` per outer fold. `metrics_mean` averages the seed-averaged metrics across folds with equal weight. `out_of_fold()` returns scored row IDs alongside last-seed predictions. `fold_seconds` includes all inner fits, HPO, threshold selection and the final outer run. Five outer folds with three inner threshold folds require twenty runner calls before counting HPO trials or multiple seeds.
- **Replaying folds.** Inspect `splits` and `threshold_splits` on the result, or call `TALENT.make_folds()` to plan folds before fitting. Save a fold with `split.save(path)`. To replay an inner or temporal manifest against the full source table, call `split.run(..., allow_extra_rows=True)`. To reuse a completed fold's inner-selected thresholds, also pass the same seeds and settings and `decision_threshold={s["seed"]: s["threshold"] for s in fold_result.per_seed}`. Each CV invocation creates a fresh subdirectory under `save_path` for its checkpoints and tuned configurations.

#### Tuning across folds and resuming a run

```python
import TALENT

result = TALENT.cross_validate_from_dataset(
    "LogReg", "bank", "./example_datasets",
    cv=5, tune=True, hpo_cv=3, n_trials=10, tune_metric="AUC",
    resume_dir="./runs/bank-logreg", device="cpu",
)
```

`hpo_cv=3` samples one configuration per Optuna trial, fits it separately on three inner folds, and optimises the unweighted mean of their held-out scores. Each candidate fit has its own validation partition for early stopping. The selected configuration is refitted for the enclosing split. Without an explicit metric, fold-averaged HPO uses accuracy for classification, RMSE for classical regression and MAE for deep regression. With `threshold_cv` enabled, each threshold fit gets its own HPO study restricted to its development rows. Group and time boundaries apply at every level. The example runs 150 candidate fits and five outer refits. Omitting `hpo_cv` retains single-validation HPO.

Set `resume_dir` to an empty local directory, then repeat the same call there after interruption. Completed outer and threshold fits are restored with their predictions, scores and timings. Each tuned fit stores its Optuna history in `study.sqlite3` under its own `fold-N/outer` or `fold-N/inner-M` directory. Histories include sampled parameters, the resolved configuration, trial state and duration. Fold-averaged trials also record the scoring-fold IDs and individual scores. Load a database with `optuna.load_study(study_name="tuning", storage="sqlite:////absolute/path/to/study.sqlite3")` (on Windows use `sqlite:///C:/.../study.sqlite3`).

One process owns each resume directory at a time. TALENT checks input contents, split assignments, configuration, source, runtime and recognised local checkpoint paths before reusing results. Use `resume_key` to identify external model or data revisions that the backend resolves independently. A changed protocol or trial budget uses a new directory. Persisted results contain per-row predictions and exclude live fitted estimators. The normal `save_path` workflow continues to create fresh runs.

Use `hpo_timeout=300` in Python or `--hpo_timeout 300` on the CLI to limit each Optuna search to five minutes. TALENT stops starting new trials at the trial or time limit. Running trials can finish after the budget. Final fitting is outside that budget. Persistent CV subtracts the recorded duration of previous finished trials from the allowance.

Completed trials and recorded candidate failures count toward `n_trials`. An abandoned trial is replaced. Future TPE proposals can differ from an uninterrupted search. Recovered fold time includes the recorded duration of earlier completed trials and fits. The CV result’s `hpo_time` sums the searches for every outer fold and its inner threshold fits. Restoring a completed fold preserves its recorded HPO time. Partial work in a killed trial has no reliable duration. With `measure=True`, `tuning_seconds` records the current tuning call and `recovered_trial_seconds` records prior trial time.

#### Selecting a foundation model release

Use the method ID in `TALENT.run()`, `run_from_dataset()` or the deep-method CLI. Each entry below supports classification and regression. Install only the backend you intend to use.

| Method ID | Optional backend | Weights and configuration |
|---|---|---|
| `limix_v2_1016` | Python 3.12+, [publisher's LimiX package](https://github.com/limix-ldm-ai/LimiX) | Selects [LimiX-2-1016.ckpt](https://huggingface.co/stable-ai/LimiX-2-1016) and the corresponding inference recipe. `limix_v2` selects the original release. |
| `tabdpt_v1_3` | `pip install "tabdpt==1.3.*"` | Uses [Layer 6's 1.3 weights](https://huggingface.co/Layer6/TabDPT). The existing `tabdpt` ID supports other backend versions. |
| `inpriora` | `pip install "scikit-learn>=1.6,<2" "inpriora @ git+https://github.com/ai-for-data/inpriora.git"` | Loads the [classifier or regressor](https://huggingface.co/ai-for-data/inpriora). For offline use, supply the task-specific local directory. |
| `seldon`, `seldon_mini` | `pip install "neuralk-oss>=0.3.0"` | Local SDK inference with [Seldon](https://huggingface.co/Neuralk-AI/seldon) or [Seldon-mini](https://huggingface.co/Neuralk-AI/seldon-mini). Obtain access to the gated weights or supply a local model directory. |

All five entries accept `config["general"]["model_path"]`. LimiX and TabDPT expect a checkpoint file. Seldon and Inpriora expect a directory. TabDPT also accepts its native `model_weight_path` spelling.

The LimiX 2.0 backend imports Triton even during CPU inference. On Windows, the local CPU checks used `triton-windows` for this dependency.

```python
spec = TALENT.get_method_spec("tabdpt_v1_3")
print(spec.preflight(checkpoint_path="/models/tabdpt1_3.safetensors"))

result = TALENT.run(
    "tabdpt_v1_3", train_val, test, info,
    config={"general": {"model_path": "/models/tabdpt1_3.safetensors", "n_estimators": 2}},
)
print(result.provenance)
```

Preflight reads registry and installed-package metadata and checks the existence of an optional local path. A real prediction exercises the backend and weights. `result.provenance` identifies the method, configured device, declared backend distribution, detected version and checkpoint when the adapter can identify it. Unknown values are `None`.

#### Choosing the HPO objective metric

By default the hyperparameter search optimizes TALENT's historical objective
(validation **Accuracy** for classification, **MAE / RMSE** for regression).
With `tune=False`, TALENT fits once per model seed using the method configuration under `TALENT/configs/default/`, or the configuration supplied by the caller. Optuna samples the separate distributions under `TALENT/configs/opt_space/`; the default configuration does not enter the study automatically.

Pass `tune_metric` to optimize any metric that `Method.metric` reports instead;
the optimization direction is inferred automatically.

```python
# Tune on ROC-AUC (classification) or R2 (regression) rather than the default
result = TALENT.run("catboost", train_val, test, info,
                    tune=True, n_trials=50, tune_metric="AUC")

from TALENT.model.lib.tuning_metric import supported_tune_metrics
print(supported_tune_metrics())
# ('Accuracy', 'Avg_Recall', 'Avg_Precision', 'F1', 'AUC',
#  'LogLoss', 'Brier', 'ECE', 'R2', 'MAE', 'RMSE')
```

`tune_metric` is also exposed on the CLI (`--tune_metric AUC`). It defaults to
`None`, which preserves the historical objective. TabNet, PTaRL and TabCaps
use an internal objective for single-validation HPO. With `hpo_cv`, outer
candidate selection can use a reported metric while those methods retain
their own early-stopping criteria.

Tuning checks that the requested metric applies to the task. In native CV, AUC requires every class and R2 requires at least two rows in each validation partition and each held-out HPO scoring fold. Check these conditions when using small groups or short time windows.

#### Custom scores and validation predictions

Use a `TALENT.Scorer` to specify which predictions a metric consumes and whether
larger scores are better. For example, select a model by binary average precision
and choose its decision threshold by validation misclassification cost:

```python
from sklearn.metrics import average_precision_score
from TALENT import Scorer, run

ap = Scorer("AveragePrecision", lambda y, p: average_precision_score(y, p[:, 1]),
            response="proba", greater_is_better=True)

def cost(y, decision):
    return ((5 * ((y == 1) & (decision == 0))) + ((y == 0) & (decision == 1))).mean()

loss = Scorer("DecisionCost", cost, response="labels", greater_is_better=False)
result = run("LogReg", train_val, test, info, tune=True, n_trials=5,
             tune_metric=ap, threshold_metric=loss, scoring=loss,
             return_validation_predictions=True)
val_proba = result.validation_predict_proba
test_cost = dict(zip(result.metric_names, result.metrics))["DecisionCost"]
```

`scoring` adds a test metric. `tune_metric` chooses hyperparameters using
validation data and `threshold_metric` chooses a binary decision threshold.
Classification scorers receive integer labels in sorted training-class order
and either predicted labels or the full probability matrix. Use
`response="prediction"` for regression in original target units.
Custom HPO scores the candidate's selected checkpoint and keeps the adapter's
ordinary early-stopping rule. A standalone search with a custom objective runs
fresh trials. Use resumable CV to reuse those studies. These options also work with `hpo_cv` and
`threshold_cv`. For resumable CV, set `resume_key` to a version for your
scoring functions and change it when their behavior or parameters change.

`return_validation_predictions=True` retains `validation_predictions` and,
for probabilistic classifiers, `validation_predict_proba` in validation-row
order. Top-level arrays refer to the last seed. Each `per_seed` entry has its
own arrays. Binary threshold tuning reuses the same validation pass. Other
runs request one extra prediction pass only when these outputs are requested.

#### TabFM query memory

Set `config={"general": {"predict_chunk_size": 256}}` to score at most 256
query rows per TabFM prediction call with the same fitted training context.
The default starts with the full query split. On CUDA memory exhaustion,
TabFM retries the failed rows in smaller chunks down to one row. The training
context size and the number of ensemble members are separate settings.

Introspect or filter methods via the unified registry:

```python
# What does this method need?
spec = TALENT.get_method_spec("tabicl_v2")
print(spec.cat_policy, spec.normalization, spec.supports_regression, spec.supports_hpo)

# List deep methods registered with the GPU hardware category that support regression
for s in TALENT.list_methods(
    architecture=TALENT.Architecture.DEEP,
    hardware=TALENT.Hardware.GPU,
    supports_regression=True,
):
    print(s.name)
```

Both the CLI scripts and the Python API are backed by the same `MethodSpec` registry, which dispatches to the registered adapter. New adapters also need default and search-space configurations.

The registry is also the single source of truth for foundation-model training-row caps (`train_row_limit`): TabPFN 1k, TabPFN v2 / Real-TabPFN / Mitra / Mitra-v2 10k, TabPFN v2.5 50k, TabICL 500k, TabPFN v3 / TabPFN-3.5 / TabICL v2 1M; TabDPT, TabFM, LimiX, LimiX-2, Kumo Tabular, Causilo, Xiaomi-TabLDM and EXAONE-Tabular have no registry cap. The cap is applied automatically when fitting; setting `config['general']['sample_size']` overrides it for a single run. Likewise, `config['general']['n_estimators']` sets the ensemble size of every foundation model that has one.

**Pre-trained weights.** Foundation models download their weights automatically on first use (from the Hugging Face Hub, or figshare for HyperFast) unless the checkpoint is bundled under `TALENT/model/models/`; `TALENT.get_method_spec(name).weights` states the source of each model. If loading the weights fails, the error names the source and the usual fixes:

- TabPFN v2.5 / v3 / 3.5: accept the model license once; on headless machines, accept it on the [Prior Labs platform](https://docs.priorlabs.ai/models/accessing-model-weights) and set `TABPFN_TOKEN` (and `TABPFN_NO_BROWSER=1`).
- Machines without internet access (e.g. cluster compute nodes): run each model once on a connected machine, then reuse the caches (`HF_HOME` with `HF_HUB_OFFLINE=1`, `TABPFN_MODEL_CACHE_DIR`, and `TALENT_CACHE_DIR` for HyperFast).
- TabPTM: only the regression checkpoint is distributed; the classification checkpoint is not publicly available.
- TabPTM's Bray–Curtis distance can be undefined for opposing standardized values. Check predictions for non-finite values when evaluating such data.



> For researchers:

### 🕹️ Clone

Clone this GitHub repository:

```bash
git clone https://github.com/LAMDA-Tabular/TALENT
cd TALENT
python -m pip install -e .
```

### 🔑 Run experiment

1. From the checkout root, edit `TALENT/configs/default/[MODEL_NAME].json` and `TALENT/configs/opt_space/[MODEL_NAME].json` for global settings and hyperparameters.

2. Run:

    ```bash
    python test/train_model_deep.py --model_type mlp --dataset bank --dataset_path example_datasets --seed_num 1 --max_epoch 1
    ```
    for deep methods, or:
    ```bash
    python test/train_model_classical.py --model_type LogReg --dataset bank --dataset_path example_datasets --seed_num 1
    ```
    for classical methods.	

### 🛠️ How to Add New Methods

For methods like the MLP class that only need to design the model, you only need to:

- Add the model class in `TALENT/model/models`.
- Inherit from `TALENT/model/methods/base.py` and override the `construct_model()` method in the new class.
- Register the method in the unified registry `TALENT/model/method_registry.py` by appending a `MethodSpec(...)` entry. The CLI argparse `choices` and `get_method()` are both derived from this registry, so no other dispatcher edits are needed.
- Add the parameter settings for the new method in `TALENT/configs/default/[MODEL_NAME].json` and `TALENT/configs/opt_space/[MODEL_NAME].json`.

For other methods that require changing the training process, partially override functions based on `TALENT/model/methods/base.py`. For details, refer to the implementation of other methods in `TALENT/model/methods/`.

See our [Contribution Guide](CONTRIBUTING.md) for more details.

### 📦 Dependencies

```bash
   pip install -r requirements.txt
```


**TabR** uses FAISS for nearest-neighbour search. Install a GPU-enabled build for CUDA runs:

```bash
conda install faiss-gpu -c pytorch
```

For CPU runs, install `faiss-cpu` and pass `device="cpu"` to the Python API.

## 🗂️ Benchmark Datasets

Datasets are available at [Google Drive](https://drive.google.com/drive/folders/1j1zt3zQIo8dO6vkO-K-WE6pSrl71bf0z?usp=drive_link).

### 📂 How to Place Datasets

TALENT loads each dataset from `<dataset_path>/<dataset>`. The dataset path can be absolute or relative to your working directory. For example, from the checkout root, `--dataset_path example_datasets --dataset bank` loads `example_datasets/bank`.

Each dataset folder `args.dataset` consists of:

- Numeric features: `N_train/val/test.npy` (can be omitted if there are no numeric features)

- Categorical features: `C_train/val/test.npy` (can be omitted if there are no categorical features)

- Labels: `y_train/val/test.npy`

- `info.json`, which must include the following three contents (task_type can be "regression", "multiclass" or "binclass"):
  
  ```json
  {
    "task_type": "regression", 
    "n_num_features": 10,
    "n_cat_features": 10
  }
  ```

## 📝 Experimental Results

We provide comprehensive evaluations of classical and deep tabular methods based on our toolbox in a fair manner in the Figure. Three tabular prediction tasks, namely, binary classification, multi-class classification, and regression, are considered, and each subfigure represents a different task type.

We use `Accuracy` and `RMSE` as the metrics for classification tasks and regression tasks, respectively. To calibrate the metrics, we choose the average performance rank to compare all methods, where a lower rank indicates better performance, following  [Sheskin (2003)](https://www.taylorfrancis.com/books/mono/10.1201/9781420036268/handbook-parametric-nonparametric-statistical-procedures-david-sheskin). Efficiency is calculated by the average training time in seconds, with lower values denoting better time efficiency. The model size is visually indicated by the radius of the circles, offering a quick glance at the trade-off between model complexity and performance.

<!-- The classical method `SVM` provided in TALENT is a `LinearSVM` to ensure faster training.  We also consider the `Dummy` baseline, which outputs the label of the major class and the average labels for classification and regression tasks, respectively. -->

<div align="center" style="text-align:center;">

  <div style="display:inline-block; width:45%; vertical-align:top; margin-bottom:5px;">
    <img src="./resources/Performance-Efficiency-Size-251005-bin.png" alt="Binary classification" style="width:45%; border-radius:8px;" />
    <div>(a) Binary classification</div>
  </div><div style="display:inline-block; width:45%; vertical-align:top; margin-bottom:5px;">
    <img src="./resources/Performance-Efficiency-Size-251005-mul.png" alt="Multiclass Classification" style="width:45%; border-radius:8px;" />
    <div>(b) Multiclass classification</div>
  </div>

  <div style="display:inline-block; width:45%; vertical-align:top; margin-bottom:5px;">
    <img src="./resources/Performance-Efficiency-Size-251005-reg.png" alt="Regression" style="width:45%; border-radius:8px;" />
    <div>(c) Regression</div>
  </div><div style="display:inline-block; width:45%; vertical-align:top; margin-bottom:5px;">
    <img src="./resources/Performance-Efficiency-Size-251005-all.png" alt="All tasks" style="width:45%; border-radius:8px;" />
    <div>(d) All tasks</div>
  </div>

</div>

<!-- From the comparison, we observe that **CatBoost** achieves the best average rank in most classification and regression tasks. Among all deep tabular methods, **ModernNCA** performs the best in most cases while maintaining an acceptable training cost. These results highlight the effectiveness of CatBoost and ModernNCA in handling various tabular prediction tasks, making them suitable choices for practitioners seeking high performance and efficiency.

These visualizations serve as an effective tool for quickly and fairly assessing the strengths and weaknesses of various tabular methods across different task types, enabling researchers and practitioners to make informed decisions when selecting suitable modeling techniques for their specific needs. -->

## 👨🏫 Acknowledgments

We thank the following repos for providing helpful components/functions in our work:

- [Rtdl-revisiting-models](https://github.com/yandex-research/rtdl-revisiting-models)
- [Rtdl-num-embeddings](https://github.com/yandex-research/rtdl-num-embeddings)
- [Tabular-dl-tabr](https://github.com/yandex-research/tabular-dl-tabr)
- [DANet](https://github.com/WhatAShot/DANet)
- [TabCaps](https://github.com/WhatAShot/TabCaps)
- [DNNR](https://github.com/younader/dnnr)
- [PTaRL](https://github.com/HangtingYe/PTaRL)
- [Saint](https://github.com/somepago/saint)
- [SwitchTab](https://github.com/avivnur/SwitchTab)
- [TabNet](https://github.com/dreamquark-ai/tabnet)
- [TabPFN](https://github.com/automl/TabPFN)
- [Tabtransformer-pytorch](https://github.com/lucidrains/tab-transformer-pytorch)
- [TANGOS](https://github.com/alanjeffares/TANGOS)
- [GrowNet](https://github.com/sbadirli/GrowNet)
- [HyperFast](https://github.com/AI-sandbox/HyperFast)
- [BiSHop](https://github.com/MAGICS-LAB/BiSHop)
- [ProtoGate](https://github.com/SilenceX12138/ProtoGate)
- [Pytabkit](https://github.com/dholzmueller/pytabkit)
- [Excelformer](https://github.com/WhatAShot/ExcelFormer)
- [GRANDE](https://github.com/s-marton/GRANDE)
- [AMFormer](https://github.com/aigc-apps/AMFormer)
- [TabM](https://github.com/yandex-research/tabm)
- [TabICL](https://github.com/soda-inria/tabicl)
- [TabAutoPNPNet](https://github.com/matteo-rizzo/periodic-tabular-dl)
- [LimiX](https://github.com/limix-ldm/LimiX)
- [xRFM](https://github.com/dmbeaglehole/xRFM)
- [TabDPT](https://github.com/layer6ai-labs/TabDPT-inference)
- [TabFM](https://github.com/google-research/tabfm)
- [Structured Data Models (Kumo Tabular)](https://github.com/NVIDIA/structured-data-models)
- [AutoGluon (Mitra)](https://github.com/autogluon/autogluon)
- [Causilo](https://github.com/nums-ai/causilo)
- [Xiaomi-TabLDM](https://github.com/xiaomi-research/xiaomi-tabldm)
- [EXAONE-Tabular](https://github.com/LGAI-Research/EXAONE-Tabular)

## 🤗 Contact

If there are any questions, please feel free to propose new features by opening an issue or contact the author: **Si-Yang Liu** ([liusy@lamda.nju.edu.cn](mailto:liusy@lamda.nju.edu.cn)) and **Hao-Run Cai** ([caihr@lamda.nju.edu.cn](mailto:caihr@lamda.nju.edu.cn)) and **Qile Zhou** ([zhouql@lamda.nju.edu.cn](mailto:zhouql@lamda.nju.edu.cn)) and **Jun-Peng Jiang** ([jiangjp@lamda.nju.edu.cn](mailto:jiangjp@lamda.nju.edu.cn)) and **Huai-Hong Yin** ([yinhh@lamda.nju.edu.cn](mailto:yinhh@lamda.nju.edu.cn)) and **Tao Zhou** ([zhout@lamda.nju.edu.cn]) and **Han-Jia Ye** ([yehj@lamda.nju.edu.cn](mailto:yehj@lamda.nju.edu.cn)). Enjoy the code.

## 🚀 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=LAMDA-Tabular/TALENT&type=Date)](https://star-history.com/#LAMDA-Tabular/TALENT&Date)

> Thanks [LAMDA-PILOT](https://github.com/LAMDA-CL/LAMDA-PILOT) and [LAMDA-ZhiJian](https://github.com/zhangyikaii/LAMDA-ZhiJian) for the template.
