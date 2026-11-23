====================================
Models
====================================

Deep learning models for tabular data, implementing various state-of-the-art architectures.

This section contains all the neural network architectures implemented in TALENT, ranging from simple MLPs to advanced transformer-based models specifically designed for tabular data. Each model implements specific forward pass computations, mathematical operations, and architectural innovations.

Basic Neural Networks
=====================

Multi-Layer Perceptron (MLP)
----------------------------

.. automodule:: TALENT.model.models.mlp
   :members:
   :undoc-members:
   :show-inheritance:

.. class:: MLP
   :noindex:

   Simple feedforward neural network with multiple fully connected layers and ReLU activations.
   
   **Mathematical Formulation:**
   
   For input :math:`x \in \mathbb{R}^{d_{in}}`, the MLP computes:
   
   .. math::
      
      h_0 &= x \\
      h_i &= \text{ReLU}(\text{Linear}(h_{i-1})) = \text{ReLU}(W_i h_{i-1} + b_i) \\
      \text{output} &= W_{\text{head}} h_L + b_{\text{head}}
   
   where :math:`L` is the number of hidden layers.

   .. method:: __init__(d_in, d_out, d_layers, dropout)
      :noindex:
      
      Initialize the MLP architecture.
      
      **Parameters:**
      
      * **d_in** (*int*) -- Input feature dimension
      * **d_out** (*int*) -- Output dimension (number of classes for classification, 1 for regression)
      * **d_layers** (*List[int]*) -- Hidden layer dimensions, e.g., [64, 32] for two hidden layers
      * **dropout** (*float*) -- Dropout probability applied after each hidden layer
      
      **Architecture Construction:**
      
      1. **Hidden Layers:** Creates `nn.Linear` layers with dimensions specified in `d_layers`
      2. **Output Head:** Final linear layer mapping to output dimension
      3. **Dropout Setup:** Configures dropout for regularization during training

   .. method:: forward(x, x_cat=None)
      :noindex:
      
      Forward pass through the MLP network.
      
      **Parameters:**
      
      * **x** (*torch.Tensor*) -- Input numerical features of shape (batch_size, d_in)
      * **x_cat** (*torch.Tensor, optional*) -- Categorical features (not used in MLP, maintained for interface consistency)
      
      **Returns:**
      
      * **torch.Tensor** -- Output predictions of shape (batch_size, d_out) or (batch_size,) for regression
      
      **Forward Pass Implementation:**
      
      .. code-block:: python
         
         for layer in self.layers:
             x = layer(x)  # Linear: x = W @ x + b
             x = F.relu(x)  # ReLU: x = max(0, x)
             if self.dropout:
                 x = F.dropout(x, self.dropout, self.training)
         
         logit = self.head(x)  # Final output layer
         if self.d_out == 1:
             logit = logit.squeeze(-1)  # For regression
      
      **ReLU Activation:**
      
      .. math::
         
         \text{ReLU}(x) = \max(0, x)
      
      **Dropout Regularization:**
      
      During training, randomly sets elements to zero with probability `dropout`:
      
      .. math::
         
         \text{Dropout}(x) = \begin{cases}
         \frac{x}{1-p} & \text{with probability } 1-p \\
         0 & \text{with probability } p
         \end{cases}

Residual Network (ResNet)
-------------------------

.. automodule:: TALENT.model.models.resnet
   :members:
   :undoc-members:
   :show-inheritance:

.. class:: ResNet
   :noindex:

   Deep residual network with skip connections for tabular data, preventing gradient vanishing in deep architectures.
   
   **Mathematical Formulation:**
   
   ResNet uses residual blocks with skip connections:
   
   .. math::
      
      h_{i+1} = h_i + F(h_i, W_i)
   
   where :math:`F(h_i, W_i)` is the residual function.

   .. method:: __init__(d_in, d, d_hidden_factor, n_layers, activation, normalization, hidden_dropout, residual_dropout, d_out)
      :noindex:
      
      Initialize the ResNet architecture with configurable components.
      
      **Parameters:**
      
      * **d_in** (*int*) -- Input feature dimension
      * **d** (*int*) -- Hidden dimension for residual blocks
      * **d_hidden_factor** (*float*) -- Factor to scale hidden layer width within blocks
      * **n_layers** (*int*) -- Number of residual blocks
      * **activation** (*str*) -- Activation function ('relu', 'gelu', 'reglu', 'geglu')
      * **normalization** (*str*) -- Normalization type ('batchnorm', 'layernorm')
      * **hidden_dropout** (*float*) -- Dropout probability within residual blocks
      * **residual_dropout** (*float*) -- Dropout probability for residual connections
      * **d_out** (*int*) -- Output dimension

   .. method:: forward(x, x_cat=None)
      :noindex:
      
      Forward pass through the ResNet architecture.
      
      **Parameters:**
      
      * **x** (*torch.Tensor*) -- Input numerical features
      * **x_cat** (*torch.Tensor, optional*) -- Categorical features (not used)
      
      **Returns:**
      
      * **torch.Tensor** -- Output predictions
      
      **Residual Block Mathematical Implementation:**
      
      For each residual block, the computation follows:
      
      .. math::
         
         \text{residual} &= \text{Norm}(h_i) \\
         \text{residual} &= \text{Linear}(\text{residual}) \\
         \text{residual} &= \text{Activation}(\text{residual}) \\
         \text{residual} &= \text{Dropout}(\text{residual}) \\
         \text{residual} &= \text{Linear}(\text{residual}) \\
         \text{residual} &= \text{Dropout}(\text{residual}) \\
         h_{i+1} &= h_i + \text{residual}
      
      **Activation Functions:**
      
      * **ReLU:** :math:`\text{ReLU}(x) = \max(0, x)`
      * **GELU:** :math:`\text{GELU}(x) = x \cdot \Phi(x)`
      * **ReGLU:** :math:`\text{ReGLU}(x) = a \cdot \text{ReLU}(b)` where :math:`a, b = \text{split}(x)`
      * **GeGLU:** :math:`\text{GeGLU}(x) = a \cdot \text{GELU}(b)` where :math:`a, b = \text{split}(x)`

   .. function:: reglu(x)
      :noindex:
      
      ReGLU activation function for gated linear units.
      
      **Mathematical Definition:**
      
      .. math::
         
         \text{ReGLU}(x) = a \cdot \text{ReLU}(b)
      
      where :math:`a` and :math:`b` are obtained by splitting :math:`x` along the last dimension.

   .. function:: geglu(x)
      :noindex:
      
      GeGLU activation function combining gating with GELU.
      
      **Mathematical Definition:**
      
      .. math::
         
         \text{GeGLU}(x) = a \cdot \text{GELU}(b)
      
      where :math:`a` and :math:`b` are obtained by splitting :math:`x` along the last dimension.

Self-Normalizing Network (SNN)
------------------------------

.. automodule:: TALENT.model.models.snn
   :members:
   :undoc-members:
   :show-inheritance:

.. class:: SNN
   :noindex:

   Lightweight neural network with self-normalizing properties using SELU activation.

   .. method:: __init__(d_in, d_out, d_layers, dropout)
      :noindex:
      
      Initialize SNN with SELU activations for self-normalization.
      
      **Parameters:**
      
      * **d_in** (*int*) -- Input dimension
      * **d_out** (*int*) -- Output dimension  
      * **d_layers** (*List[int]*) -- Hidden layer dimensions
      * **dropout** (*float*) -- Dropout probability

   .. method:: forward(x, x_cat=None)
      :noindex:
      
      Forward pass with SELU activation for self-normalization.
      
      **SELU Activation Mathematical Definition:**
      
      .. math::
         
         \text{SELU}(x) = \lambda \begin{cases}
         x & \text{if } x > 0 \\
         \alpha(e^x - 1) & \text{if } x \leq 0
         \end{cases}
      
      where :math:`\lambda \approx 1.0507` and :math:`\alpha \approx 1.6733`.
      
      **Self-Normalization Property:**
      
      SELU ensures that for normalized inputs, activations maintain:
      - Mean converges to 0
      - Variance converges to 1
      - Enables training of very deep networks without explicit normalization

Transformer-Based Models
========================

Feature Tokenizer Transformer (FT-Transformer)
-----------------------------------------------

.. automodule:: TALENT.model.models.ftt
   :members:
   :undoc-members:
   :show-inheritance:

.. class:: Transformer
   :noindex:

   Advanced transformer architecture specifically designed for tabular data with feature tokenization.
   
   **Mathematical Formulation:**
   
   **Feature Tokenization:**
   
   For numerical features: :math:`t_i^{\text{num}} = W_{\text{num}} x_i + b_{\text{num}}`
   
   For categorical features: :math:`t_i^{\text{cat}} = \text{Embedding}(x_i^{\text{cat}})`

   .. method:: __init__(d_numerical, categories, d_token, n_layers, n_heads, d_ffn_factor, attention_dropout, ffn_dropout, residual_dropout, activation, prenormalization, d_out)
      :noindex:
      
      Initialize the FT-Transformer architecture.
      
      **Parameters:**
      
      * **d_numerical** (*int*) -- Number of numerical features
      * **categories** (*List[int], optional*) -- Cardinalities for categorical features
      * **d_token** (*int*) -- Token embedding dimension
      * **n_layers** (*int*) -- Number of transformer layers
      * **n_heads** (*int*) -- Number of attention heads
      * **d_ffn_factor** (*float*) -- Factor for feed-forward network dimension
      * **attention_dropout** (*float*) -- Dropout for attention weights
      * **ffn_dropout** (*float*) -- Dropout for feed-forward network
      * **residual_dropout** (*float*) -- Dropout for residual connections
      * **activation** (*str*) -- Activation function for FFN
      * **prenormalization** (*bool*) -- Whether to use pre-normalization
      * **d_out** (*int*) -- Output dimension

   .. method:: forward(x_num, x_cat)
      :noindex:
      
      Forward pass through the transformer.
      
      **Parameters:**
      
      * **x_num** (*torch.Tensor, optional*) -- Numerical features of shape (batch_size, d_numerical)
      * **x_cat** (*torch.Tensor, optional*) -- Categorical features of shape (batch_size, n_categorical)
      
      **Returns:**
      
      * **torch.Tensor** -- Output predictions
      
      **Transformer Processing Pipeline:**
      
      1. **Tokenization:** Convert features to tokens using `Tokenizer`
      2. **CLS Token Addition:** Prepend classification token
      3. **Transformer Layers:** Apply multi-head attention and feed-forward networks
      4. **Output Generation:** Use CLS token representation for final prediction
      
      **Transformer Layer Mathematical Implementation:**
      
      For each transformer layer:
      
      .. math::
         
         \text{attn_out} &= \text{MultiHeadAttention}(x, x, x) \\
         x &= \text{LayerNorm}(x + \text{attn_out}) \\
         \text{ffn_out} &= \text{FFN}(x) \\
         x &= \text{LayerNorm}(x + \text{ffn_out})

.. class:: Tokenizer
   :noindex:

   Converts numerical and categorical features into token embeddings for transformer processing.

   .. method:: __init__(d_numerical, categories, d_token, bias)
      :noindex:
      
      Initialize the feature tokenizer.
      
      **Parameters:**
      
      * **d_numerical** (*int*) -- Number of numerical features
      * **categories** (*List[int], optional*) -- Cardinalities of categorical features
      * **d_token** (*int*) -- Token embedding dimension
      * **bias** (*bool*) -- Whether to use bias in tokenization

   .. method:: forward(x_num, x_cat)
      :noindex:
      
      Convert features to token embeddings.
      
      **Tokenization Process:**
      
      **Numerical Features:**
      
      .. math::
         
         \text{tokens}_{\text{num}} = x_{\text{num}} W_{\text{num}} + b_{\text{num}}
      
      **Categorical Features:**
      
      .. math::
         
         \text{tokens}_{\text{cat}} = \text{Embedding}(x_{\text{cat}} + \text{offsets})
      
      **CLS Token:**
      
      .. math::
         
         \text{tokens}_{\text{cls}} = W_{\text{cls}}

   .. property:: n_tokens
      :noindex:
      
      Total number of tokens (numerical + categorical + CLS).
      
      **Returns:**
      
      * **int** -- Total token count

.. class:: MultiheadAttention
   :noindex:

   Multi-head attention mechanism optimized for tabular data.

   .. method:: __init__(d, n_heads, dropout, bias)
      :noindex:
      
      Initialize multi-head attention.
      
      **Parameters:**
      
      * **d** (*int*) -- Input dimension
      * **n_heads** (*int*) -- Number of attention heads
      * **dropout** (*float*) -- Attention dropout probability
      * **bias** (*bool*) -- Whether to use bias in projections

   .. method:: forward(x_q, x_kv, key_compression, value_compression)
      :noindex:
      
      Compute multi-head attention.
      
      **Parameters:**
      
      * **x_q** (*torch.Tensor*) -- Query input
      * **x_kv** (*torch.Tensor*) -- Key and value input
      * **key_compression** (*nn.Linear, optional*) -- Key compression layer
      * **value_compression** (*nn.Linear, optional*) -- Value compression layer
      
      **Returns:**
      
      * **torch.Tensor** -- Attention output
      
      **Multi-Head Attention Mathematical Implementation:**
      
      1. **Linear Projections:**
         
         .. math::
            
            Q = x_q W^Q, \quad K = x_{kv} W^K, \quad V = x_{kv} W^V
      
      2. **Scaled Dot-Product Attention:**
         
         .. math::
            
            \text{attention} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)
      
      3. **Output Computation:**
         
         .. math::
            
            \text{output} = \text{attention} \cdot V
      
      4. **Multi-Head Combination:**
         
         .. math::
            
            \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O

Advanced Tabular Models
=======================

TabNet
------

.. automodule:: TALENT.model.models.tabnet
   :members:
   :undoc-members:
   :show-inheritance:

.. class:: TabNetClassifier
   :noindex:

   Interpretable deep learning model with sequential attention mechanism for classification.
   
   **Mathematical Formulation:**
   
   TabNet uses sequential feature selection through sparsemax attention:
   
   **Feature Selection at Step i:**
   
   .. math::
      
      M^{[i]} = \text{sparsemax}(\text{AttentionTransformer}(f^{[i-1]}))
   
   **Feature Processing:**
   
   .. math::
      
      f^{[i]} = \gamma \odot M^{[i]} \odot h + (1-\gamma) \odot f^{[i-1]}
   
   where :math:`\gamma` is the relaxation parameter.

   .. method:: __init__(n_steps, gamma, n_independent, n_shared, momentum, optimizer_params, scheduler_params, mask_type, lambda_sparse, seed)
      :noindex:
      
      Initialize TabNet classifier.
      
      **Parameters:**
      
      * **n_steps** (*int*) -- Number of decision steps
      * **gamma** (*float*) -- Relaxation parameter for feature selection
      * **n_independent** (*int*) -- Number of independent GLU layers per step
      * **n_shared** (*int*) -- Number of shared GLU layers
      * **momentum** (*float*) -- Momentum for batch normalization
      * **optimizer_params** (*dict*) -- Optimizer configuration
      * **scheduler_params** (*dict*) -- Learning rate scheduler parameters
      * **mask_type** (*str*) -- Type of attention mask ('sparsemax' or 'entmax')
      * **lambda_sparse** (*float*) -- Sparsity regularization coefficient
      * **seed** (*int*) -- Random seed

   .. method:: fit(X_train, y_train, eval_set, eval_name, eval_metric, max_epochs, patience, batch_size, virtual_batch_size, num_workers, drop_last, callbacks)
      :noindex:
      
      Train the TabNet model.
      
      **Training Process:**
      
      1. **Data Preprocessing:** Handle categorical encoding and normalization
      2. **Sequential Training:** Train each decision step sequentially
      3. **Attention Regularization:** Apply sparsity constraints on attention masks
      4. **Early Stopping:** Monitor validation metrics for convergence

   .. method:: predict_proba(X)
      :noindex:
      
      Make probability predictions for classification.
      
      **Parameters:**
      
      * **X** (*torch.Tensor or scipy.sparse matrix*) -- Input features
      
      **Returns:**
      
      * **np.ndarray** -- Class probabilities of shape (n_samples, n_classes)
      
      **Prediction Process:**
      
      1. **Forward Pass:** Process through all decision steps
      2. **Attention Aggregation:** Combine attention from all steps
      3. **Softmax Application:** Convert logits to probabilities
      
      .. math::
         
         P(y=k|x) = \frac{\exp(o_k)}{\sum_{j=1}^K \exp(o_j)}
      
      where :math:`o_k` is the raw output for class :math:`k`.

   .. method:: explain(X, normalize)
      :noindex:
      
      Generate feature importance explanations using attention masks.
      
      **Parameters:**
      
      * **X** (*torch.Tensor*) -- Input features
      * **normalize** (*bool*) -- Whether to normalize importance scores
      
      **Returns:**
      
      * **np.ndarray** -- Feature importance matrix
      
      **Explanation Generation:**
      
      Attention masks from each decision step provide interpretable feature importance:
      
      .. math::
         
         \text{importance}_{ij} = \frac{M^{[i]}_j}{\sum_{k=1}^{n_features} M^{[i]}_k}

.. class:: TabNetRegressor
   :noindex:

   TabNet for regression tasks with mean squared error optimization.

   .. method:: compute_loss(y_pred, y_true)
      :noindex:
      
      Compute mean squared error loss for regression.
      
      **MSE Loss Mathematical Definition:**
      
      .. math::
         
         \mathcal{L}_{\text{MSE}} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2

Tree-Based Neural Models
========================

GRANDE (Gradient-Boosted Neural Decision Ensembles)
---------------------------------------------------

.. automodule:: TALENT.model.models.grande
   :members:
   :undoc-members:
   :show-inheritance:

.. class:: GRANDE
   :noindex:

   Tree-mimic neural network using gradient descent for decision tree simulation.
   
   **Mathematical Formulation:**
   
   GRANDE simulates decision trees using neural operations with entmax for sparse selection.

   .. method:: __init__(batch_size, task_type, depth, n_estimators, dropout)
      :noindex:
      
      Initialize GRANDE model.
      
      **Parameters:**
      
      * **batch_size** (*int*) -- Training batch size
      * **task_type** (*str*) -- 'classification' or 'regression'
      * **depth** (*int*) -- Maximum tree depth
      * **n_estimators** (*int*) -- Number of tree estimators
      * **dropout** (*float*) -- Dropout probability

   .. method:: forward(inputs)
      :noindex:
      
      Forward pass through the GRANDE ensemble.
      
      **Parameters:**
      
      * **inputs** (*torch.Tensor*) -- Input features
      
      **Returns:**
      
      * **torch.Tensor** -- Ensemble predictions
      
      **Tree Simulation Mathematical Implementation:**
      
      1. **Split Decision Computation:**
         
         .. math::
            
            \text{node_result} = \frac{\text{softsign}(s_1 - s_2) + 1}{2}
         
         where :math:`s_1` are learned split thresholds and :math:`s_2` are feature values.
      
      2. **Path Probability Calculation:**
         
         .. math::
            
            p = \prod_{j} ((1-\text{path_id}_j) \cdot \text{node_result}_j + \text{path_id}_j \cdot (1-\text{node_result}_j))
      
      3. **Ensemble Output for Regression:**
         
         .. math::
            
            \text{output} = \sum_{e,l} w_e \cdot p_{e,l} \cdot v_{e,l}
         
         where :math:`w_e` are estimator weights, :math:`p_{e,l}` are leaf probabilities, and :math:`v_{e,l}` are leaf values.
      
      4. **Ensemble Output for Classification:**
         
         .. math::
            
            \text{output} = \sum_{e,l} w_e \cdot p_{e,l} \cdot \text{softmax}(v_{e,l})

   .. method:: get_representation(inputs)
      :noindex:
      
      Extract intermediate tree representations for analysis.
      
      **Returns:**
      
      * **torch.Tensor** -- Tree path representations

Neural Oblivious Decision Ensembles (NODE)
------------------------------------------

.. automodule:: TALENT.model.models.node
   :members:
   :undoc-members:
   :show-inheritance:

.. class:: Node
   :noindex:

   Neural implementation of oblivious decision trees with differentiable splits.

   .. method:: __init__(input_dim, layer_dim, output_dim, num_layers, tree_dim, depth, choice_function, bin_function)
      :noindex:
      
      Initialize NODE architecture.
      
      **Parameters:**
      
      * **input_dim** (*int*) -- Input feature dimension
      * **layer_dim** (*int*) -- Hidden layer dimension
      * **output_dim** (*int*) -- Output dimension
      * **num_layers** (*int*) -- Number of NODE layers
      * **tree_dim** (*int*) -- Number of trees per layer
      * **depth** (*int*) -- Tree depth
      * **choice_function** (*str*) -- Function for feature selection ('entmax15')
      * **bin_function** (*str*) -- Function for threshold selection ('entmoid15')

   .. method:: forward(x)
      :noindex:
      
      Forward pass through oblivious decision trees.
      
      **Decision Tree Mathematical Process:**
      
      1. **Feature Selection:** Use entmax for sparse feature selection
      2. **Threshold Comparison:** Compare features with learned thresholds
      3. **Path Aggregation:** Aggregate predictions along tree paths
      4. **Ensemble Combination:** Combine outputs from multiple trees

GrowNet (Gradient Boosting with Neural Networks)
------------------------------------------------

.. automodule:: TALENT.model.models.grownet
   :members:
   :undoc-members:
   :show-inheritance:

.. class:: GrowNet
   :noindex:

   Gradient boosting framework with neural network weak learners.

   .. method:: __init__(input_dim, output_dim, boost_rate, layers_per_net, layer_dims, dropout)
      :noindex:
      
      Initialize GrowNet with neural weak learners.
      
      **Gradient Boosting Process:**
      
      1. **Weak Learner Training:** Train neural networks on residuals
      2. **Boosting Update:** Add weak learners with adaptive weights
      3. **Gradient Computation:** Compute gradients for next weak learner

   .. method:: forward(x)
      :noindex:
      
      Forward pass through the boosted ensemble.
      
      **Boosting Mathematical Formulation:**
      
      .. math::
         
         F_m(x) = F_{m-1}(x) + \gamma_m h_m(x)
      
      where :math:`h_m` is the m-th weak learner and :math:`\gamma_m` is the boosting rate.

Distance-Based Models
=====================

Modern Neighborhood Component Analysis (ModernNCA)
--------------------------------------------------

.. automodule:: TALENT.model.models.modernNCA
   :members:
   :undoc-members:
   :show-inheritance:

.. class:: ModernNCA
   :noindex:

   Neighborhood Component Analysis-inspired model for embedding-based predictions.
   
   **Mathematical Formulation:**
   
   ModernNCA learns embeddings for distance-based classification.

   .. method:: __init__(d_in, d_out, k, dropout, d_embedding)
      :noindex:
      
      Initialize ModernNCA model.
      
      **Parameters:**
      
      * **d_in** (*int*) -- Input feature dimension
      * **d_out** (*int*) -- Output dimension (number of classes)
      * **k** (*int*) -- Number of nearest neighbors to consider
      * **dropout** (*float*) -- Dropout probability
      * **d_embedding** (*int*) -- Embedding dimension

   .. method:: forward(x, y, candidate_x, candidate_y, is_train)
      :noindex:
      
      Forward pass with neighborhood analysis.
      
      **Parameters:**
      
      * **x** (*torch.Tensor*) -- Query features
      * **y** (*torch.Tensor*) -- Query labels
      * **candidate_x** (*torch.Tensor*) -- Candidate features for nearest neighbor search
      * **candidate_y** (*torch.Tensor*) -- Candidate labels
      * **is_train** (*bool*) -- Training mode flag
      
      **Returns:**
      
      * **torch.Tensor** -- Distance-based predictions
      
      **Distance-Based Prediction Mathematical Implementation:**
      
      1. **Embedding Computation:**
         
         .. math::
            
            e_i = f(x_i), \quad e_j = f(x_j)
         
         where :math:`f` is the learned embedding function.
      
      2. **Distance Computation:**
         
         .. math::
            
            d(x_i, x_j) = ||e_i - e_j||_2
      
      3. **Neighbor Weighting:**
         
         .. math::
            
            p_{ij} = \frac{\exp(-d(x_i, x_j))}{\sum_{k \neq i} \exp(-d(x_i, x_k))}
      
      4. **Final Prediction:**
         
         .. math::
            
            \hat{y}_i = \sum_j p_{ij} y_j

   .. method:: knn_prediction(x, candidate_x, candidate_y, k)
      :noindex:
      
      Make predictions using k-nearest neighbors in embedding space.
      
      **K-NN Process:**
      
      1. **Distance Calculation:** Compute distances in embedding space
      2. **Neighbor Selection:** Find k nearest neighbors
      3. **Prediction Aggregation:** Aggregate neighbor labels with distance weighting

Specialized Architectures
=========================

ExcelFormer (Semi-Permeable Attention)
--------------------------------------

.. automodule:: TALENT.model.models.excelformer
   :members:
   :undoc-members:
   :show-inheritance:

.. class:: ExcelFormer
   :noindex:

   Transformer with semi-permeable attention and mixup training capabilities.

   .. method:: __init__(d_numerical, d_token, n_blocks, attention_dropout, ffn_dropout, residual_dropout, d_out)
      :noindex:
      
      Initialize ExcelFormer architecture.
      
      **Parameters:**
      
      * **d_numerical** (*int*) -- Number of numerical features
      * **d_token** (*int*) -- Token embedding dimension
      * **n_blocks** (*int*) -- Number of transformer blocks
      * **attention_dropout** (*float*) -- Attention dropout probability
      * **ffn_dropout** (*float*) -- Feed-forward dropout probability
      * **residual_dropout** (*float*) -- Residual connection dropout
      * **d_out** (*int*) -- Output dimension

   .. method:: forward(x_num, x_cat, mix_up, beta, mtype)
      :noindex:
      
      Forward pass with optional mixup augmentation.
      
      **Parameters:**
      
      * **x_num** (*torch.Tensor*) -- Numerical features
      * **x_cat** (*torch.Tensor, optional*) -- Categorical features
      * **mix_up** (*bool*) -- Whether to apply mixup
      * **beta** (*float*) -- Mixup parameter (default: 0.5)
      * **mtype** (*str*) -- Mixup type ('feat_mix', 'hidden_mix', 'naive_mix')
      
      **Returns:**
      
      * **tuple** -- (output, feat_masks, shuffled_ids) for mixup training
      
      **Mixup Mathematical Implementation:**
      
      **Feature Mixup:**
      
      .. math::
         
         \tilde{x} = \lambda x_i + (1-\lambda) x_j
      
      **Semi-Permeable Attention:**
      
      .. math::
         
         \text{Attention}_{\text{perm}}(Q, K, V) = \text{mask} \odot \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V

   .. method:: mixup_process(x, beta, mtype)
      :noindex:
      
      Apply mixup augmentation to input features.
      
      **Mixup Types:**
      
      * **feat_mix:** Feature-level mixing with learnable weights
      * **hidden_mix:** Hidden representation mixing
      * **naive_mix:** Simple linear interpolation

ProtoGate (Prototype-Based Gating)
----------------------------------

.. automodule:: TALENT.model.models.protogate
   :members:
   :undoc-members:
   :show-inheritance:

.. class:: ProtoGate
   :noindex:

   Prototype-based model with gating mechanisms for interpretable feature selection.

   .. method:: __init__(input_dim, output_dim, n_prototypes, n_components, dropout)
      :noindex:
      
      Initialize ProtoGate architecture.
      
      **Parameters:**
      
      * **input_dim** (*int*) -- Input feature dimension
      * **output_dim** (*int*) -- Output dimension
      * **n_prototypes** (*int*) -- Number of learned prototypes
      * **n_components** (*int*) -- Number of components per prototype
      * **dropout** (*float*) -- Dropout probability

   .. method:: forward(x)
      :noindex:
      
      Forward pass with prototype-based gating.
      
      **Prototype-Based Processing:**
      
      1. **Prototype Computation:** Learn representative prototypes from data
      2. **Distance Calculation:** Compute distances to prototypes
      3. **Gate Generation:** Use distances to generate feature gates
      4. **Feature Selection:** Apply gates for adaptive feature selection

.. class:: GatingNet
   :noindex:

   Gating network for prototype-based feature selection.

   .. method:: hard_sigmoid(x)
      :noindex:
      
      Hard sigmoid activation for efficient gating.
      
      **Hard Sigmoid Mathematical Definition:**
      
      .. math::
         
         \text{hard_sigmoid}(x) = \max(0, \min(1, \frac{x + 1}{2}))
      
      This provides a piecewise linear approximation to the sigmoid function for computational efficiency.

   .. method:: forward(x)
      :noindex:
      
      Generate gating weights for feature selection.

Retrieval-Based Models
======================

TabR (Tabular Retrieval)
------------------------

.. automodule:: TALENT.model.models.tabr
   :members:
   :undoc-members:
   :show-inheritance:

.. class:: TabR
   :noindex:

   KNN-attention hybrid model with retrieval-based predictions.

   .. method:: __init__(n_num_features, n_cat_features, n_classes, context_size, normalization, num_embeddings, d_main, d_multiplier, encoder_n_blocks, predictor_n_blocks, mixer_normalization, dropout0, dropout1, normalization, activation)
      :noindex:
      
      Initialize TabR architecture.
      
      **Parameters:**
      
      * **n_num_features** (*int*) -- Number of numerical features
      * **n_cat_features** (*int*) -- Number of categorical features  
      * **n_classes** (*int*) -- Number of output classes
      * **context_size** (*int*) -- Maximum context size for retrieval
      * **normalization** (*str*) -- Normalization type
      * **num_embeddings** (*dict*) -- Embedding configurations
      * **d_main** (*int*) -- Main hidden dimension
      * **d_multiplier** (*int*) -- Dimension multiplier
      * **encoder_n_blocks** (*int*) -- Number of encoder blocks
      * **predictor_n_blocks** (*int*) -- Number of predictor blocks
      * **mixer_normalization** (*str*) -- Mixer normalization type
      * **dropout0** (*float*) -- Input dropout
      * **dropout1** (*float*) -- Hidden dropout
      * **activation** (*str*) -- Activation function

   .. method:: forward(x_num, x_cat, candidate_x_num, candidate_x_cat, candidate_y, context_size, is_train)
      :noindex:
      
      Forward pass with retrieval-based attention.
      
      **Retrieval Process:**
      
      1. **Context Selection:** Select relevant examples from training set
      2. **Attention Computation:** Apply attention over retrieved candidates
      3. **Feature Processing:** Process query and candidate features
      4. **Prediction Generation:** Combine retrieval and learned representations

Foundation Models
=================

TabPFN (Tabular Prior-Fitting Networks)
---------------------------------------

.. automodule:: TALENT.model.models.tabpfn
   :members:
   :undoc-members:
   :show-inheritance:

.. class:: TabPFNClassifier
   :noindex:

   Prior-fitting network for zero-shot tabular classification.

   .. method:: __init__(device, base_path)
      :noindex:
      
      Initialize TabPFN with pre-trained weights.
      
      **Foundation Model Features:**
      
      * Pre-trained on diverse tabular datasets
      * No gradient-based training required
      * Immediate deployment capability
      * Context-based learning from examples

   .. method:: fit(X, y)
      :noindex:
      
      Fit the model using in-context learning (no parameter updates).
      
      **In-Context Learning Process:**
      
      1. **Context Setup:** Store training examples as context
      2. **No Weight Updates:** Model weights remain frozen
      3. **Context Encoding:** Encode training data for reference

   .. method:: predict_proba(X)
      :noindex:
      
      Make predictions using in-context learning.
      
      **Zero-Shot Prediction:**
      
      1. **Context Retrieval:** Use stored training context
      2. **Attention Mechanism:** Apply attention over training examples
      3. **Prediction Generation:** Generate predictions without fine-tuning

Regularization Methods
======================

TANGOS Regularization
---------------------

.. automodule:: TALENT.model.models.tangos
   :members:
   :undoc-members:
   :show-inheritance:

.. class:: Tangos
   :noindex:

   MLP with TANGOS regularization for neuron specialization.
   
   **Mathematical Formulation:**
   
   TANGOS applies spatial and spectral regularization to encourage neuron specialization:
   
   .. math::
      
      \mathcal{L}_{\text{TANGOS}} = \mathcal{L}_{\text{task}} + \lambda_1 \mathcal{L}_{\text{spatial}} + \lambda_2 \mathcal{L}_{\text{spectral}}

   .. method:: __init__(d_in, d_out, d_layers, dropout, lambda1, lambda2)
      :noindex:
      
      Initialize TANGOS-regularized MLP.
      
      **Parameters:**
      
      * **d_in** (*int*) -- Input dimension
      * **d_out** (*int*) -- Output dimension
      * **d_layers** (*List[int]*) -- Hidden layer dimensions
      * **dropout** (*float*) -- Dropout probability
      * **lambda1** (*float*) -- Spatial regularization weight
      * **lambda2** (*float*) -- Spectral regularization weight

   .. method:: forward(x, x_cat=None)
      :noindex:
      
      Forward pass with standard MLP architecture.

   .. method:: cal_representation(x)
      :noindex:
      
      Calculate intermediate representations for regularization.
      
      **Parameters:**
      
      * **x** (*torch.Tensor*) -- Input features
      
      **Returns:**
      
      * **torch.Tensor** -- Hidden representations before final layer
      
      **Representation Extraction Process:**
      
      The method extracts intermediate representations by stopping before the final layer:
      
      .. code-block:: python
         
         for i, layer in enumerate(self.layers):
             x = layer(x)
             x = F.relu(x)
             if self.dropout and i != len(self.layers) - 1:
                 x = F.dropout(x, self.dropout, self.training)
         return x  # Return before final head layer
      
      **Regularization Applications:**
      
      * **Spatial Regularization:** Encourages spatial locality in neuron activations
      * **Spectral Regularization:** Promotes spectral diversity in learned representations

Activation Functions Reference
==============================

**Standard Activations:**

.. math::
   
   \text{ReLU}(x) = \max(0, x)

.. math::
   
   \text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]

.. math::
   
   \text{SELU}(x) = \lambda \begin{cases}
   x & \text{if } x > 0 \\
   \alpha(e^x - 1) & \text{if } x \leq 0
   \end{cases}

**Gated Activations:**

.. math::
   
   \text{ReGLU}(x) = a \cdot \text{ReLU}(b) \text{ where } [a, b] = \text{split}(x)

.. math::
   
   \text{GeGLU}(x) = a \cdot \text{GELU}(b) \text{ where } [a, b] = \text{split}(x)

**Probability Functions:**

.. math::
   
   \text{Softmax}(x_i) = \frac{\exp(x_i)}{\sum_{j=1}^K \exp(x_j)}

.. math::
   
   \text{Sparsemax}(z) = \arg\min_{p \in \Delta^{K-1}} ||p - z||_2^2

where :math:`\Delta^{K-1}` is the probability simplex.

Model Usage Examples
====================

**Basic MLP Usage:**

.. code-block:: python
   
   from TALENT.model.models.mlp import MLP
   
   # Initialize MLP
   model = MLP(
       d_in=10,           # Input dimension
       d_out=3,           # Output dimension (3 classes)
       d_layers=[64, 32], # Hidden layer sizes
       dropout=0.1        # Dropout probability
   )
   
   # Forward pass
   x = torch.randn(32, 10)  # Batch of 32 samples, 10 features
   output = model(x)        # Shape: (32, 3)

**ResNet with Advanced Activations:**

.. code-block:: python
   
   from TALENT.model.models.resnet import ResNet
   
   # Initialize ResNet with GeGLU activation
   model = ResNet(
       d_in=15,
       d_out=1,                    # Regression task
       d=128,                      # Hidden dimension
       d_hidden_factor=2.0,        # Hidden expansion factor
       n_layers=4,                 # Number of residual blocks
       activation='geglu',         # GeGLU activation
       normalization='layernorm',  # Layer normalization
       hidden_dropout=0.1,
       residual_dropout=0.1
   )

**FT-Transformer with Mixed Features:**

.. code-block:: python
   
   from TALENT.model.models.ftt import Transformer
   
   # Initialize FT-Transformer
   model = Transformer(
       d_numerical=8,          # 8 numerical features
       categories=[5, 10, 3],  # 3 categorical features with cardinalities
       d_token=64,             # Token dimension
       n_layers=3,             # Number of transformer layers
       n_heads=8,              # Attention heads
       d_ffn_factor=2.0,       # FFN expansion factor
       attention_dropout=0.1,
       ffn_dropout=0.1,
       residual_dropout=0.1,
       activation='reglu',
       prenormalization=True,
       d_out=5                 # 5 classes
   )

**TabNet for Interpretable Classification:**

.. code-block:: python
   
   from TALENT.model.models.tabnet import TabNetClassifier
   
   # Initialize TabNet
   model = TabNetClassifier(
       n_steps=3,              # Decision steps
       gamma=1.3,              # Relaxation parameter
       n_independent=2,        # Independent GLU layers
       n_shared=2,             # Shared GLU layers
       momentum=0.02,          # Batch norm momentum
       lambda_sparse=1e-3      # Sparsity regularization
   )
   
   # Training
   model.fit(X_train, y_train, 
             eval_set=[(X_val, y_val)],
             max_epochs=100)
   
   # Get predictions and explanations
   predictions = model.predict_proba(X_test)
   explanations = model.explain(X_test, normalize=True)

**GRANDE for Tree-like Neural Networks:**

.. code-block:: python
   
   from TALENT.model.models.grande import GRANDE
   
   # Initialize GRANDE
   model = GRANDE(
       batch_size=64,
       task_type='classification',
       depth=4,              # Tree depth
       n_estimators=10,      # Number of trees
       dropout=0.1
   )

**ModernNCA with Distance-Based Learning:**

.. code-block:: python
   
   from TALENT.model.models.modernNCA import ModernNCA
   
   # Initialize ModernNCA
   model = ModernNCA(
       d_in=15,
       d_out=4,              # 4 classes
       k=32,                 # Number of neighbors
       dropout=0.1,
       d_embedding=64        # Embedding dimension
   )
   
   # Training requires candidate examples
   output = model(x, y, candidate_x, candidate_y, is_train=True)

**ExcelFormer with Mixup Training:**

.. code-block:: python
   
   from TALENT.model.models.excelformer import ExcelFormer
   
   # Initialize ExcelFormer
   model = ExcelFormer(
       d_numerical=10,
       d_token=64,
       n_blocks=3,
       attention_dropout=0.1,
       ffn_dropout=0.1,
       d_out=3
   )
   
   # Forward pass with feature mixup
   output, masks, shuffled_ids = model(
       x_num, 
       mix_up=True, 
       beta=0.5, 
       mtype='feat_mix'
   )

**TabPFN for Zero-Shot Learning:**

.. code-block:: python
   
   from TALENT.model.models.tabpfn import TabPFNClassifier
   
   # Initialize pre-trained TabPFN
   model = TabPFNClassifier(device='cuda')
   
   # No training required - just fit context
   model.fit(X_train, y_train)
   
   # Immediate predictions
   predictions = model.predict_proba(X_test)

Model Selection Guidelines
==========================

**For Beginners:**
- **MLP:** Simple, fast, good baseline
- **ResNet:** Better than MLP for deeper networks

**For Best Performance:**
- **FT-Transformer:** State-of-the-art on many datasets
- **TabNet:** Excellent performance with interpretability
- **ModernNCA:** Strong embedding-based performance

**For Interpretability:**
- **TabNet:** Attention-based feature importance
- **GRANDE:** Tree-like decision process
- **ProtoGate:** Prototype-based explanations

**For Speed:**
- **MLP:** Fastest training and inference
- **SNN:** Lightweight with self-normalization
- **TabPFN:** No training required

**For Specific Scenarios:**
- **TabR:** Retrieval-based learning
- **ExcelFormer:** Complex feature interactions with mixup
- **TANGOS:** When regularization is critical 