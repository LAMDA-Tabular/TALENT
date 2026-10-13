====================================
Methods in TALENT
====================================

TALENT integrates an extensive array of 30+ deep learning architectures and classical methods for tabular data. Below is a summary of these methods, organized by type.

==========================
Deep Learning Methods
==========================

TALENT offers the following deep learning models, specifically designed to enhance performance on tabular data:

1. `MLP <https://arxiv.org/abs/2106.11959>`_ : A multi-layer neural network implemented according to RTDL.
2. `ResNet <https://arxiv.org/abs/2106.11959>`_ : A DNN that uses skip connections across many layers.
3. `SNN <https://arxiv.org/abs/1706.02515>`_ : A self-normalizing network that uses the SELU activation, enabling deeper network training.
4. `DANets <https://arxiv.org/abs/2112.02962>`_ : Groups correlated features to enhance tabular data processing while reducing computational complexity.
5. `TabCaps <https://openreview.net/pdf?id=OgbtSLESnI>`_ : A capsule network that encapsulates feature values into vectorial representations.
6. `DCNv2 <https://arxiv.org/abs/2008.13535>`_: Combines an MLP module with a feature crossing module using linear layers and multiplications.
7. `NODE <https://arxiv.org/abs/1909.06312>`_: Generalizes oblivious decision trees, blending gradient-based optimization and hierarchical representation learning.
8. `GrowNet <https://arxiv.org/abs/2002.07971>`_ : A gradient boosting method utilizing shallow neural networks as weak learners.
9. `TabNet <https://arxiv.org/abs/1908.07442>`_ : Sequential attention-based method for tabular data, enhancing feature selection and providing interpretability.
10. `TabR <https://arxiv.org/abs/2307.14338>`_ : A model integrating KNN and attention mechanisms to improve prediction accuracy.
11. `ModernNCA <https://arxiv.org/abs/2407.03257>`_ : Inspired by traditional NCA, this model makes predictions using relationships with neighbors in a learned embedding space.
12. `DNNR <https://arxiv.org/abs/2205.08434>`_ : Enhances KNN using local gradients and Taylor approximations for better predictions.
13. `AutoInt <https://arxiv.org/abs/1810.11921>`_ : Uses multi-head self-attention to automatically learn high-order feature interactions.
14. `Saint <https://arxiv.org/abs/2106.01342>`_ : A token-based model that applies row and column attention mechanisms to tabular data.
15. `TabTransformer <https://arxiv.org/abs/2012.06678>`_ : Enhances tabular data modeling by transforming categorical features into contextual embeddings.
16. `FT-Transformer <https://arxiv.org/abs/2106.11959>`_ : A feature transformation-based method using attention mechanisms on tabular data.
17. `TANGOS <https://openreview.net/pdf?id=n6H86gW8u0d>`_ : A regularization-based method encouraging neuron specialization for tabular data.
18. `SwitchTab <https://arxiv.org/abs/2401.02013>`_ : A self-supervised method improving representation learning through an encoder-decoder framework.
19. `PTaRL <https://openreview.net/pdf?id=G32oY4Vnm8>`_ : Enhances prediction by constructing a prototype-based space for regularization.
20. `TabPFN <https://arxiv.org/abs/2207.01848>`_ : A pre-trained model that generalizes across diverse tabular tasks.
21. `HyperFast <https://arxiv.org/abs/2402.14335>`_ : A meta-trained hypernetwork that generates task-specific neural networks for tabular data.
22. `TabPTM <https://arxiv.org/abs/2311.00055>`_ : Standardizes heterogeneous datasets using meta-representations for tabular data.
23. `BiSHop <https://arxiv.org/abs/2404.03830>`_ : A sparse Hopfield model for tabular learning with column-wise and row-wise modules.
24. `ProtoGate <https://arxiv.org/abs/2306.12330>`_ : A prototype-based model for feature selection in HDLSS biomedical data.
25. `RealMLP <https://arxiv.org/abs/2407.04491>`_ : An improved multilayer perceptron (MLP) with better efficiency.
26. `MLP_PLR <https://arxiv.org/abs/2203.05556>`_ : An enhanced MLP that uses periodic activations to improve performance.
27. `Excelformer <https://arxiv.org/abs/2301.02819>`_ : A model featuring semi-permeable attention modules for tabular data, addressing rotational invariance.
28. `GRANDE <https://arxiv.org/abs/2309.17130>`_ : A tree-mimic model using gradient descent for axis-aligned decision trees.
29. `AMFormer <https://arxiv.org/abs/2402.02334>`_ : A transformer-based method for tabular data, with attention mechanisms based on feature interactions.
30. `Trompt <https://arxiv.org/abs/2305.18446>`_ : A prompt-based neural network for separating intrinsic column features and sample-specific feature importance.
31. **`TabM <https://arxiv.org/abs/2410.24210>`_** :  A model based on MLP and variations of BatchEnsemble.  
32. **`TabPFN v2 <https://doi.org/10.1038/s41586-024-08328-6>`_**: A general model which involves the use of pre-trained deep neural networks that can be directly applied to any tabular task.  
33. **`T2Gformer <https://arxiv.org/abs/2211.16887>`_**: A Transformer network for tabular learning that processes data guided by relation graphs and uses a Cross-level Readout for global semantics in prediction.  
34. **`TabICL <https://arxiv.org/abs/2502.05564>`_**: A comparable tabular foundation model with performance on par with TabPFN v2.  
35. **`TabAutoPNPNet <https://www.mdpi.com/2079-9292/14/6/1165>`_**: A tabular model based on periodicity, particularly the Fourier transform and Chebyshev polynomials, with performance on par with or superior to FT-Transformer.
36. **`Mitra <https://www.amazon.science/blog/mitra-mixed-synthetic-priors-for-enhancing-tabular-foundation-models>`_**: A tabular foundation model learned with mixed synthetic priors.
37. **`LimiX <https://arxiv.org/abs/2509.03505>`_**: A tabular foundation model that leverages transformers to support a wide range of tasks, from prediction to imputation and causal inference, within a unified architecture.
38. **`Real-TabPFN <https://arxiv.org/abs/2507.03971>`_**: An enhanced tabular foundation model that extends TabPFNv2 through continued pre-training on real-world datasets for classification tasks.
39. **`TabPFN v3 <https://github.com/PriorLabs/TabPFN>`_**: TabPFN v3 (PriorLabs 2026), with native context up to ~1M rows × 200 features. Requires ``pip install -U 'tabpfn>=8.0.0'``.
40. **`TabICL v2 <https://github.com/soda-inria/tabicl>`_**: TabICL v2 (ICML 2026), supporting both classification and regression. Requires ``pip install -U 'tabicl>=2.0.0'``.
41. **`TabPFN v2.5 <https://arxiv.org/abs/2511.08667>`_**: TabPFN v2.5 (PriorLabs, Nov 2025), the intermediate release between v2 and v3, scaling in-context learning to ~50k rows × 2k features. Requires ``pip install -U 'tabpfn>=8.0.0'``.
42. **`TabDPT <https://github.com/layer6ai-labs/TabDPT-inference>`_**: A tabular foundation model from Layer 6 AI that combines in-context learning with retrieval and self-supervised pre-training on real data. Requires ``pip install -U tabdpt``.
43. **`TabFM <https://github.com/google-research/tabfm>`_**: Google's zero-shot tabular foundation model for classification (up to 10 classes) and regression. Requires ``pip install -U "tabfm[pytorch]"``; the model weights use the TabFM Non-Commercial License.
44. **`TRC <https://arxiv.org/abs/2603.16569>`_**: A post-hoc Tabular Representation Corrector (TPAMI 2026) that freezes a trained FT-Transformer backbone, learns to remove representation shift, and maps corrected representations into a compact light-embedding space.
45. **`TabPFN-3.5 <https://github.com/PriorLabs/TabPFN>`_**: The successor of TabPFN-3 (PriorLabs, Sep 2026) with native context up to ~1M rows × 20k features. Requires ``pip install -U 'tabpfn>=9.0.0'``; the weights use a non-commercial license.
46. **`Kumo Tabular <https://github.com/NVIDIA/structured-data-models>`_**: NVIDIA's tabular foundation model (Sep 2026, small / medium / large) under the commercially usable OpenMDW-1.1 license. Requires Python >= 3.11 and ``pip install git+https://github.com/NVIDIA/structured-data-models.git``.
47. **`Mitra-v2 <https://arxiv.org/abs/2609.04540>`_**: The second-generation Mitra (AutoGluon, Sep 2026) for classification and regression (1000-bin distributional head), run zero-shot.
48. **`Causilo <https://github.com/nums-ai/causilo>`_**: A tabular foundation model from Nums AI (Sep 2026). Requires ``pip install -U causilo`` (PyTorch >= 2.13); the weights use the non-commercial Causilo License.
49. **`Xiaomi-TabLDM <https://github.com/xiaomi-research/xiaomi-tabldm>`_**: Xiaomi's tabular foundation model (Sep 2026). Requires ``pip install git+https://github.com/xiaomi-research/xiaomi-tabldm.git``.
50. **`EXAONE-Tabular <https://github.com/LGAI-Research/EXAONE-Tabular>`_**: LG AI Research's tabular foundation model (2026), weights for research and education only. Requires Python >= 3.11 and ``pip install "exaonetabular @ git+https://github.com/LGAI-Research/EXAONE-Tabular.git"``.
51. **`LimiX-2 <https://github.com/limix-ldm-ai/LimiX>`_**: The 400M-parameter successor of LimiX (StableAI, Sep 2026) for classification (2-10 classes) and regression; weights under the StableAI LimiX Non-Commercial License. Requires Python >= 3.12 and ``pip install git+https://github.com/limix-ldm-ai/LimiX.git``.

All foundation models download their pre-trained weights automatically on first use unless the checkpoint is bundled under ``TALENT/model/models/``; ``TALENT.get_method_spec(name).weights`` states the source of each model.

==========================
Classical Methods
==========================

TALENT integrates the following classical machine learning methods, which serve as strong baselines for tabular data tasks:

1. **CatBoost**: A gradient boosting algorithm that excels at handling categorical features and performing well on tabular datasets.
2. **Dummy Classifier**: A simple baseline method that outputs the most frequent class or mean value, used to benchmark against random or naïve predictions.
3. **K-Nearest Neighbors (KNN)**: A classic instance-based learning algorithm that makes predictions based on the closest training samples.
4. **LightGBM**: A highly efficient gradient boosting framework that uses decision tree algorithms to reduce memory usage and improve speed.
5. **Logistic Regression (LogReg)**: A basic classification method that models the probability of a binary outcome based on input features.
6. **Linear Regression (LR)**: A regression method that models the relationship between a dependent variable and one or more independent variables.
7. **Naive Bayes**: A probabilistic classifier based on Bayes' theorem, particularly useful for categorical data with strong independence assumptions.
8. **Nearest Class Mean (NCM)**: A classifier that assigns a sample to the class whose mean is closest to the sample, based on a distance metric.
9. **Random Forest**: An ensemble learning method that constructs multiple decision trees and merges them to improve the accuracy and robustness of predictions.
10. **Support Vector Machine (SVM)**: A powerful classification method that finds the hyperplane best separating different classes of data.
11. **XGBoost**: An advanced gradient boosting algorithm that is particularly effective for structured/tabular data and provides robust performance in competition settings.
12. `RFM <https://www.science.org/doi/10.1126/science.adi5639>`_: A non-deep, backpropagation-free feature learning algorithm that iteratively applies AGOP to a kernel machine to adaptively learn task-specific features.
13. `xRFM <https://arxiv.org/abs/2508.10053>`_: Combines RFMs with an adaptive tree structure, enabling it to learn features local to data subsets and scale log-linearly with the number of samples.

==========================
Methodology Summary
==========================

TALENT provides a comprehensive toolkit for tabular data analysis, integrating both deep learning and classical machine learning models. These methods offer flexibility, customizability, and ease of integration into various tabular data tasks, making TALENT a powerful resource for researchers and practitioners alike.
