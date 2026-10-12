.. TALENT documentation master file

========================================================
TALENT: A Tabular Analytics and Learning Toolbox
========================================================

.. image:: ../resources/TALENT-LOGO.png
   :width: 1000px
   :align: center

Welcome to **TALENT**, a comprehensive machine learning toolbox designed to enhance model performance on tabular data. 

TALENT integrates advanced deep learning models, classical algorithms, and efficient hyperparameter tuning, offering robust preprocessing capabilities to optimize learning from tabular datasets. The toolbox is user-friendly and adaptable, catering to both novice and expert data scientists.

.. important::
   If you use any content of this repo for your work, please make sure to cite the relevant papers as described in the `Citing TALENT` section below.

==========================
Citing TALENT
==========================

If you use **TALENT** in your research, please consider citing the following works:

.. code-block:: bibtex

    @article{ye2024closerlookdeeplearning,
             title={A Closer Look at Deep Learning on Tabular Data}, 
             author={Han-Jia Ye and Si-Yang Liu and Hao-Run Cai and Qi-Le Zhou and De-Chuan Zhan},
             journal={arXiv preprint arXiv:2407.00956},
             year={2024}
    }

    @article{JMLR:v26:25-0512,
             author  = {Si-Yang Liu and Hao-Run Cai and Qi-Le Zhou and Huai-Hong Yin and Tao Zhou and Jun-Peng Jiang and Han-Jia Ye},
             title   = {Talent: A Tabular Analytics and Learning Toolbox},
             journal = {Journal of Machine Learning Research},
             year    = {2025},
             volume  = {26},
             number  = {226},
             pages   = {1--16},
             url     = {http://jmlr.org/papers/v26/25-0512.html}
    }

==========================
What's New
==========================

Here are the recent updates to **TALENT**:

- [2026-10]🌟 Add the September 2026 wave of tabular foundation models: `TabPFN-3.5 <https://github.com/PriorLabs/TabPFN>`_, `LimiX-2 <https://github.com/limix-ldm-ai/LimiX>`_, `NVIDIA Kumo Tabular <https://github.com/NVIDIA/structured-data-models>`_, `Mitra-v2 <https://arxiv.org/abs/2609.04540>`_, `Causilo <https://github.com/nums-ai/causilo>`_, `Xiaomi-TabLDM <https://github.com/xiaomi-research/xiaomi-tabldm>`_ and `EXAONE-Tabular <https://github.com/LGAI-Research/EXAONE-Tabular>`_. All foundation models now download their weights automatically and share one interface.

- [2026-08]🌟 Add `TRC <https://arxiv.org/abs/2603.16569>`_ (TPAMI 2026), a post-hoc representation corrector that trains and freezes an FT-Transformer backbone before learning representation re-estimation and light-space mapping.

- [2026-07]🌟 Add `TabFM <https://github.com/google-research/tabfm>`_ (Google Research, zero-shot tabular foundation model).

- [2026-06]🌟 Add `TabPFN v2.5 <https://arxiv.org/abs/2511.08667>`_ and `TabDPT <https://github.com/layer6ai-labs/TabDPT-inference>`_.

- [2026-06]🌟 Evaluation improvements: calibration metrics (Brier, ECE) for every classifier, decision thresholds tuned on the validation set for binary classification, and a uniform ``predict_proba`` / ``predict_labels`` interface in ``RunResult``.

- [2026-05]🌟 Add `TabPFN v3 <https://github.com/PriorLabs/TabPFN>`_ and `TabICL v2 <https://github.com/soda-inria/tabicl>`_ (regression support added).

- [2026-03]🌟 Updated the TALENT-extension datasets and results. `Link <https://box.nju.edu.cn/d/b7b23a19ee054aaba7b6/?p=%2F&mode=list>`_

- [2025-11]🌟 Add `RFM <https://www.science.org/doi/10.1126/science.adi5639>`_ (Science), `Real-TabPFN <https://arxiv.org/abs/2507.03971>`_ and `LimiX <https://arxiv.org/abs/2509.03505>`_.

- [2025-09]🌟 Add `xRFM <https://arxiv.org/abs/2508.10053>`_.

- [2025-08]🌟 Add `Mitra <https://www.amazon.science/blog/mitra-mixed-synthetic-priors-for-enhancing-tabular-foundation-models>`_.

- [2025-06]🌟 Add `TabAutoPNPNet <https://www.mdpi.com/2079-9292/14/6/1165>`_ (Electronics 2025)

- [2025-06]🌟 Add `TabICL <https://arxiv.org/abs/2502.05564>`_ (ICML 2025). The current code is based on TabICL v0.1.2.

- [2025-05]🌟 Check out our three papers `MMTU <https://github.com/LAMDA-Tabular/MMTU>`_, `Tabular-Temporal-Shift <https://github.com/LAMDA-Tabular/Tabular-Temporal-Shift>`_, and `BETA <https://github.com/LAMDA-Tabular/BETA>`_ accepted at ICML 2025!

- [2025-04]🌟 Check out our new survey `Representation Learning for Tabular Data: A Comprehensive Survey <https://arxiv.org/abs/2504.16109>`_ (`Repo <https://github.com/LAMDA-Tabular/Tabular-Survey>`_). We organize existing methods into three main categories according to their generalization capabilities: specialized, transferable, and general models, which provides a comprehensive taxonomy for deep tabular representation methods.🚀🚀🚀

- [2025-02]🌟 Add `T2Gformer <https://arxiv.org/abs/2211.16887>`_ (AAAI 2023).

- [2025-02]🌟 Add `TabPFN v2 <https://doi.org/10.1038/s41586-024-08328-6>`_ (Nature).

- [2025-02]🌟 Thanks to `Hengzhe Zhang <https://hengzhe-zhang.github.io/>`_ for providing a `Scikit-Learn compatible wrapper <https://github.com/hengzhe-zhang/scikit-TALENT>`_ for TALENT!

- [2025-01]🌟 Check out our new baseline `ModernNCA <https://openreview.net/pdf?id=JytL2MrlLT>`_ (**ICLR 2025**), inspired by traditional **Neighbor Component Analysis**, which outperforms both tree-based and other deep tabular models, while also reducing training time and model size!🚀🚀🚀

- [2025-01]🌟 Check out our `latest version of the benchmark paper <https://arxiv.org/abs/2407.00956>`_ for updated and expanded results and analysis!

- [2025-01]🌟We have curated and released `new benchmark datasets <https://drive.google.com/drive/folders/1j1zt3zQIo8dO6vkO-K-WE6pSrl71bf0z?usp=drive_link>`_, along with updated `results <https://6sy666.github.io/TALENT-Results/>`_ of the dataset across a broader range of methods. This update focuses on enhancing dataset quality, including removing duplicates, and correcting tasks where bin-class was mistakenly treated as regression. We have also separated the larger datasets and formed the basic benchmark (300 datasets, including 120 bin-class, 80 multi-class, and 100 regression), and the large benchmark (22 datasets).

- [2024-12]🌟 Add `TabM <https://arxiv.org/abs/2410.24210>`_ (ICLR 2025).

- [2024-09]🌟 Add `Trompt <https://arxiv.org/abs/2305.18446>`_ (ICML 2023).

- [2024-09]🌟 Add `AMFormer <https://arxiv.org/abs/2402.02334>`_ (AAAI 2024).

- [2024-08]🌟 Add `GRANDE <https://arxiv.org/abs/2309.17130>`_ (ICLR 2024).

- [2024-08]🌟 Add `Excelformer <https://arxiv.org/abs/2301.02819>`_ (KDD 2024).

- [2024-08]🌟 Add `MLP_PLR <https://arxiv.org/abs/2203.05556>`_ (NeurIPS 2022).

- [2024-07]🌟 Add `RealMLP <https://arxiv.org/abs/2407.04491>`_.

- [2024-07]🌟 Add `ProtoGate <https://arxiv.org/abs/2306.12330>`_ (ICML 2024).

- [2024-07]🌟 Add `BiSHop <https://arxiv.org/abs/2404.03830>`_ (ICML 2024).

- [2024-06]🌟 Check out our new baseline `ModernNCA <https://arxiv.org/abs/2407.03257>`_, inspired by traditional **Neighbor Component Analysis**, which outperforms both tree-based and other deep tabular models, while also reducing training time and model size!

- [2024-06]🌟 Check out our `benchmark paper <https://arxiv.org/abs/2407.00956>`_ about tabular data, which provides comprehensive evaluations of classical and deep tabular methods based on our toolbox in a fair manner!

.. note::
   If you want to view **benchmark results**, please visit:
   `https://6sy666.github.io/TALENT-Results/ <https://6sy666.github.io/TALENT-Results/>`_

   To explore **default hyperparameters and search spaces** of methods in this toolbox, check:
   `https://6sy666.github.io/TALENT-Configs/ <https://6sy666.github.io/TALENT-Configs/>`_

==========================
Contents
==========================

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials

.. toctree::
   :maxdepth: 2
   :caption: Methods

   methods
   
.. toctree::
   :maxdepth: 2
   :caption: Dependencies
   
   dependencies

.. toctree::
   :maxdepth: 2
   :caption: Benchmark_Datasets   

   benchmark_datasets

.. toctree::
   :maxdepth: 2
   :caption: Experimental_Results

   experimental_results

.. toctree::
   :maxdepth: 1
   :caption: API Docs

   api/core
   api/classical_methods
   api/deep_learning
   api/lib

.. toctree::
   :maxdepth: 2
   :caption: Acknowledgements

   acknowledgements

