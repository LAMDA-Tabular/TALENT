Library Components
==================


Overview
--------

The library components module provides specialized implementations and utilities for various deep learning models in TALENT. These components include attention mechanisms, feature transformers, model-specific optimizations, and data processing utilities.


Common Features
--------------

All library components in TALENT share the following features:

- PyTorch integration for deep learning models
- Efficient data processing and memory optimization
- Modular design for easy integration
- Support for both numerical and categorical features
- Configurable hyperparameters and architectures
- GPU acceleration support

Available Components
------------------

**Core Utilities:**

- :doc:`Data <lib/data>`: Functions for loading, preprocessing, and preparing tabular data for machine learning tasks, including handling missing values, encoding features, and creating data loaders.
- :doc:`TData <lib/TData>`: Optimized data structure for efficient tabular data handling
- :doc:`num_embeddings <lib/num_embeddings>`: Advanced numerical feature embedding techniques

**Library Components:**

- :doc:`TabNet <lib/tabnet>`: Interpretable deep learning for tabular data
- :doc:`TabPFN <lib/tabpfn>`: Prior-data fitted networks
- :doc:`TabR <lib/tabr>`: Tabular representation learning
- :doc:`TabM <lib/tabm>`: Tabular modeling with transformers
- :doc:`RealMLP <lib/realmlp>`: Real-valued MLP for tabular data
- :doc:`BiSHop <lib/bishop>`: Bidirectional hierarchical attention
- :doc:`NODE <lib/node>`: Neural oblivious decision ensembles
- :doc:`HyperFast <lib/hyperfast>`: Fast hyperparameter optimization
- :doc:`ExcelFormer <lib/excelformer>`: Transformer for tabular data
- :doc:`DANets <lib/danets>`: Deep attention networks
- :doc:`TabCaps <lib/tabcaps>`: Capsule networks for tabular data
- :doc:`TabICL <lib/tabicl>`: In-context learning for tabular data
- :doc:`Periodic Tabular DL <lib/periodic_tab_dl>`: Periodic embeddings for tabular data
- :doc:`TROMPT <lib/trompt>`: Tabular prompting mechanisms
- :doc:`PTARL <lib/ptarl>`: Policy gradient methods for tabular RL
- :doc:`TRC <lib/trc>`: Post-hoc representation re-estimation and light-space mapping
- :doc:`AmFormer <lib/amformer>`: Attention mechanisms for transformers
- :doc:`TabPTM <lib/tabptm>`: Pre-trained models for tabular data
- :doc:`DNNR <lib/dnnr>`: Deep nearest neighbor regression

.. toctree::
   :maxdepth: 2
   :caption: Library Components:

   lib/TData
   lib/data
   lib/num_embeddings
   lib/tabnet
   lib/tabpfn
   lib/tabr
   lib/tabm
   lib/realmlp
   lib/bishop
   lib/node
   lib/hyperfast
   lib/excelformer
   lib/danets
   lib/tabcaps
   lib/tabicl
   lib/periodic_tab_dl
   lib/trompt
   lib/ptarl
   lib/trc
   lib/amformer
   lib/tabptm
   lib/dnnr



