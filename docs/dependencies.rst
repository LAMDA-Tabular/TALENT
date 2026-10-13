====================================
Dependencies
====================================

TALENT relies on the following dependencies to provide a comprehensive machine learning toolbox for tabular data analysis. Ensure that these packages are installed in your environment before using TALENT:

==========================
Python Libraries
==========================

1. `PyTorch <https://pytorch.org/>`_ : The deep learning framework used for model development and training in TALENT.
2. `scikit-learn <https://scikit-learn.org/stable/>`_ : Provides classical machine learning models and utilities for data preprocessing and evaluation.
3. `pandas <https://pandas.pydata.org/>`_ : A data manipulation and analysis library for handling tabular data.
4. `numpy <https://numpy.org/>`_ : Fundamental package for scientific computing with Python, including support for large, multi-dimensional arrays and matrices.
5. `scipy <https://scipy.org/>`_ : A library for scientific and technical computing, used for optimization, integration, and statistics.
6. `tqdm <https://tqdm.github.io/>`_ : A library for creating progress bars, used to display the progress of model training and data processing.

==========================
Optional Dependencies
==========================

Some methods in TALENT require additional dependencies for specific tasks. If you intend to use the following methods, make sure to install these optional packages:

- `faiss-gpu <https://github.com/facebookresearch/faiss>`_ : Required for **TabR** to efficiently handle nearest neighbor searches. Install via conda:
   
   .. code-block:: bash

      conda install faiss-gpu -c pytorch

- **Tabular foundation models** that are not bundled with TALENT need their upstream package (each method raises an ``ImportError`` with the install command if it is missing):

  - ``tabpfn_v2_5`` / ``tabpfn_v3``: ``pip install -U 'tabpfn>=8.0.0'``; ``tabpfn_v3_5``: ``pip install -U 'tabpfn>=9.0.0'``
  - ``tabicl_v2``: ``pip install -U 'tabicl>=2.0.0'``
  - ``tabdpt``: ``pip install -U tabdpt``
  - ``tabfm``: ``pip install -U "tabfm[pytorch]"``
  - ``causilo``: ``pip install -U causilo`` (PyTorch >= 2.13)
  - ``kumo_tabular``: ``pip install git+https://github.com/NVIDIA/structured-data-models.git`` (Python >= 3.11)
  - ``tabldm``: ``pip install git+https://github.com/xiaomi-research/xiaomi-tabldm.git``
  - ``exaone_tabular``: ``pip install "exaonetabular @ git+https://github.com/LGAI-Research/EXAONE-Tabular.git"`` (Python >= 3.11)
  - ``limix_v2``: ``pip install git+https://github.com/limix-ldm-ai/LimiX.git`` (Python >= 3.12)

  All foundation models download their weights automatically on first use. The bundled ones (TabPFN v2, Real-TabPFN, TabICL, Mitra / Mitra-v2, LimiX / LimiX-2) fetch their checkpoints from the Hugging Face Hub when they are not bundled, which needs ``huggingface_hub`` (listed in ``requirements.txt``). TabPFN v2.5 / v3 / 3.5 require accepting the model license once (set ``TABPFN_TOKEN`` on headless machines). On machines without internet access, run each model once on a connected machine and reuse the caches (``HF_HOME`` with ``HF_HUB_OFFLINE=1``, ``TABPFN_MODEL_CACHE_DIR``, ``TALENT_CACHE_DIR``).


==========================
Installation
==========================

To install the necessary dependencies for TALENT, you can use the following commands:

1. **Using pip**:

   Install the required libraries from the `requirements.txt` file:
   
   .. code-block:: bash
   
      pip install -r requirements.txt
   

2. **Using conda**:

   If you are using `conda`, you can create a new environment and install the dependencies:
   
   .. code-block:: bash
   
      conda create -n talent python=3.10
      conda activate talent
      pip install -r requirements.txt
   

==========================
Additional Notes
==========================

Ensure that your Python version is compatible with the dependencies listed above. TALENT is tested with **Python 3.10** and requires **PyTorch >= 2.8** (see ``pyproject.toml``).