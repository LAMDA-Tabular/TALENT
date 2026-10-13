Methods
========

Deep learning method implementations that wrap model architectures with training logic.

This section contains method classes that provide a unified interface for training, validation, and prediction across all deep learning models in TALENT. Each method class inherits from the base `Method` class and implements model-specific logic while maintaining consistent APIs.

Base Method Class (method/base.py)
===================================

All method implementations inherit from the base `Method` class which provides the core training, validation, and prediction workflow. Understanding this base class is essential for using any method in TALENT.

.. class:: Method
   :noindex:

   Abstract base class that provides a unified interface for all deep learning methods in TALENT.
   
   **Key Features:**
   
   * Consistent training/validation/prediction workflow across all models
   * Automatic data preprocessing and formatting with multiple encoding options
   * Model construction and optimization setup with configurable parameters
   * Early stopping and checkpoint management for robust training
   * Comprehensive evaluation metrics for regression and classification tasks
   * Flexible handling of numerical and categorical features

Core Base Methods
-----------------

.. method:: __init__(args, is_regression)
   :noindex:
   
   Initialize the method with configuration and task type.
   
   **Parameters:**
   
   * **args** (*argparse.Namespace*) -- Configuration arguments containing model, training, and data processing settings
   * **is_regression** (*bool*) -- Whether the task is regression (True) or classification (False)
   
   **Initialization Process:**
   
   1. **Configuration Setup:** Store arguments and determine task type
   2. **Statistics Reset:** Initialize training counters and timers
   3. **Logging Setup:** Create training log dictionary with best performance tracking
   4. **Device Setup:** Configure GPU/CPU device for training

.. method:: construct_model(model_config=None)
   :noindex:
   
   Abstract method to construct the specific model architecture. Must be implemented by each method subclass.
   
   **Parameters:**
   
   * **model_config** (*dict, optional*) -- Model-specific configuration parameters. If None, uses `args.config['model']`
   
   **Implementation Notes:**
   
   * Each method class overrides this to create its specific model type
   * Model is moved to appropriate device (GPU/CPU) and set to correct precision (float/double)
   * Configuration parameters are model-specific (e.g., hidden dimensions, number of layers, etc.)

.. method:: data_format(is_train=True, N=None, C=None, y=None)
   :noindex:
   
   Format and preprocess data for training or inference. This is the core data processing pipeline.
   
   **Parameters:**
   
   * **is_train** (*bool*) -- Whether formatting for training (True) or inference (False)
   * **N** (*dict, optional*) -- Numerical features dictionary with train/val/test splits
   * **C** (*dict, optional*) -- Categorical features dictionary with train/val/test splits
   * **y** (*dict, optional*) -- Target labels dictionary with train/val/test splits
   
   **Training Mode Processing Pipeline:**
   
   1. **NaN Handling:** 
      .. code-block:: python
         
         self.N, self.C, self.num_new_value, self.imputer, self.cat_new_value = \
             data_nan_process(self.N, self.C, self.args.num_nan_policy, self.args.cat_nan_policy)
   
   2. **Label Processing:**
      .. code-block:: python
         
         self.y, self.y_info, self.label_encoder = \
             data_label_process(self.y, self.is_regression)
   
   3. **Numerical Encoding:** Apply binning, quantile transformation, or other numerical policies
   4. **Categorical Encoding:** Apply ordinal, one-hot, or other categorical encoding strategies
   5. **Normalization:** Apply standardization, min-max scaling, or quantile normalization
   6. **DataLoader Creation:** Create PyTorch DataLoaders for training and validation
   
   **Inference Mode Processing:**
   
   Uses previously fitted transformers (encoders, normalizers) to process test data consistently.

.. method:: fit(data, info, train=True, config=None)
   :noindex:
   
   Main training method that orchestrates the entire training process.
   
   **Parameters:**
   
   * **data** (*tuple*) -- (N, C, y) containing numerical features, categorical features, and labels
   * **info** (*dict*) -- Dataset information including feature names, types, and metadata
   * **train** (*bool*) -- Whether to actually train the model (False for loading checkpoints only)
   * **config** (*dict, optional*) -- Override configuration for hyperparameter tuning
   
   **Returns:**
   
   * **float** -- Total training time in seconds
   
   **Training Process:**
   
   1. **Data Setup:** Create Dataset object and extract feature information
   2. **Data Processing:** Call `data_format()` to preprocess all data
   3. **Model Construction:** Call `construct_model()` to build the neural network
   4. **Optimizer Setup:** Initialize AdamW optimizer with configured learning rate and weight decay
   5. **Training Loop:** For each epoch, call `train_epoch()` and `validate()`
   6. **Early Stopping:** Stop training if validation performance doesn't improve for 20 epochs
   7. **Checkpoint Saving:** Save best and final model weights

.. method:: train_epoch(epoch)
   :noindex:
   
   Train the model for one epoch using the training data.
   
   **Parameters:**
   
   * **epoch** (*int*) -- Current epoch number for logging
   
   **Training Steps per Batch:**
   
   1. **Feature Extraction:** Handle numerical and categorical features appropriately
   2. **Forward Pass:** Compute model predictions
   3. **Loss Computation:** Calculate training loss using appropriate criterion
   4. **Backward Pass:** Compute gradients via backpropagation
   5. **Parameter Update:** Apply optimizer step
   6. **Progress Logging:** Display training progress every 50 batches

.. method:: validate(epoch)
   :noindex:
   
   Validate the model on the validation set and handle early stopping.
   
   **Parameters:**
   
   * **epoch** (*int*) -- Current epoch number
   
   **Validation Process:**
   
   1. **Set Evaluation Mode:** `model.eval()` to disable dropout and batch norm updates
   2. **Inference Loop:** Process validation batches without gradients
   3. **Metric Computation:** Calculate validation metrics using `metric()` method
   4. **Best Model Tracking:** Save model checkpoint if validation performance improved
   5. **Early Stopping Logic:** Increment counter if no improvement; stop after 20 epochs
   6. **Logging:** Record validation results and save training log

.. method:: predict(data, info, model_name)
   :noindex:
   
   Make predictions on test data using a trained model.
   
   **Parameters:**
   
   * **data** (*tuple*) -- (N, C, y) test data
   * **info** (*dict*) -- Dataset information
   * **model_name** (*str*) -- Model checkpoint name ('best-val' or 'epoch-last')
   
   **Returns:**
   
   * **tuple** -- (loss, metrics, metric_names, predictions)
   
   **Prediction Process:**
   
   1. **Model Loading:** Load trained weights from checkpoint
   2. **Data Processing:** Apply fitted transformers to test data
   3. **Inference:** Generate predictions in evaluation mode
   4. **Metric Calculation:** Compute comprehensive evaluation metrics

.. method:: metric(predictions, labels, y_info)
   :noindex:
   
   Compute comprehensive evaluation metrics based on task type.
   
   **Parameters:**
   
   * **predictions** (*np.ndarray*) -- Model predictions
   * **labels** (*np.ndarray*) -- Ground truth labels
   * **y_info** (*dict*) -- Label processing information including classes and normalization details
   
   **Returns:**
   
   * **tuple** -- (metrics_values, metric_names)
   
   **Task-Specific Metrics:**
   
   **Regression Tasks:**
   
   * **MAE:** Mean Absolute Error - :math:`\frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|`
   * **R²:** Coefficient of determination - :math:`1 - \frac{SS_{res}}{SS_{tot}}`
   * **RMSE:** Root Mean Squared Error - :math:`\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}`
   
   **Binary Classification:**
   
   * **Accuracy:** Overall classification accuracy
   * **Balanced Recall:** Balanced accuracy score handling class imbalance
   * **Macro Precision:** Macro-averaged precision across classes
   * **F1 Score:** Binary F1 score - :math:`2 \cdot \frac{precision \cdot recall}{precision + recall}`
   * **Log Loss:** Cross-entropy loss
   * **AUC:** Area under ROC curve for probability predictions
   * **Brier:** Brier score (mean squared error of the predicted probabilities; lower is better)
   * **ECE:** Expected Calibration Error with 15 equal-width bins (lower is better)
   
   **Multi-class Classification:**
   
   * **Accuracy:** Overall classification accuracy
   * **Balanced Recall:** Balanced accuracy score
   * **Macro Precision:** Macro-averaged precision
   * **Macro F1:** Macro-averaged F1 score
   * **Log Loss:** Cross-entropy loss
   * **Macro AUC:** One-vs-Rest AUC score
   * **Brier:** Multi-class Brier score (lower is better)
   * **ECE:** Expected Calibration Error with 15 equal-width bins (lower is better)

Utility Functions
-----------------

.. function:: check_softmax(logits)
   :noindex:

   Ensure logits are properly normalized probabilities for classification tasks.
   
   **Parameters:**
   
   * **logits** (*np.ndarray*) -- Array of shape (N, C) with raw logits or probabilities
   
   **Returns:**
   
   * **np.ndarray** -- Properly normalized probabilities summing to 1
   
   **Mathematical Process:**
   
   1. **Probability Check:** Verify if values are in [0,1] and sum to 1 per sample
   2. **Softmax Application:** If not probabilities, apply numerically stable softmax:
      
      .. math::
         
         p_i = \frac{\exp(x_i - \max(x))}{\sum_{j} \exp(x_j - \max(x))}
   
   3. **Numerical Stability:** Subtract max before exp to prevent overflow

Method Implementations
======================

Methods are organized by complexity. Simple methods primarily use base class functionality, while advanced methods implement specialized training procedures.

Simple Methods (Using Base Class Functions)
-------------------------------------------

These methods only override `construct_model()` and use all base class functionality for training, validation, and prediction.

.. class:: MLPMethod
   :noindex:

   Multi-Layer Perceptron method - the simplest and most widely applicable method.
   
   **Features:** Fast training, good baseline performance, suitable for most tabular tasks
   
   **Requirements:** `cat_policy` cannot be 'indices' (categorical features must be encoded)
   
   **Usage:** Ideal for beginners or when you need fast training with decent performance

.. class:: ResNetMethod
   :noindex:

   Residual Network method with skip connections for deeper networks.
   
   **Features:** Prevents gradient vanishing, supports various activations (ReLU, GELU, ReGLU, GeGLU)
   
   **Usage:** When you need deeper networks than MLP without gradient problems

.. class:: SNNMethod
   :noindex:

   Self-Normalizing Network method using SELU activation.
   
   **Features:** Automatic normalization properties, suitable for deeper networks
   
   **Usage:** Alternative to ResNet when you want self-normalizing properties

.. class:: NodeMethod
   :noindex:

   Neural Oblivious Decision Ensembles method implementing neural decision trees.
   
   **Features:** Tree-like decision making with neural network flexibility
   
   **Usage:** When you want tree-like interpretability with neural network power

.. class:: GrowNetMethod
   :noindex:

   Gradient boosting with neural network weak learners.
   
   **Features:** Combines gradient boosting with neural networks
   
   **Usage:** For ensemble-based approaches with neural components

.. class:: GrandeMethod
   :noindex:

   Gradient-boosted neural decision ensembles for tree-mimic behavior.
   
   **Features:** Neural implementation of decision tree ensembles
   
   **Usage:** When you want tree ensemble performance with neural network flexibility

Transformer-Based Methods (Using Base Class Functions)
------------------------------------------------------

These transformer methods use standard base class training but require specific categorical policies.

.. class:: FTTMethod
   :noindex:

   Feature Tokenizer Transformer - one of the best performing methods.
   
   **Features:** Feature tokenization, multi-head attention, state-of-the-art performance
   
   **Requirements:** `cat_policy` must be 'indices' (uses raw categorical indices)
   
   **Usage:** First choice for best performance on most datasets

.. class:: SaintMethod
   :noindex:

   Self-Attention and Intersample Attention Transformer.
   
   **Features:** Row and column attention, enhanced feature interactions
   
   **Usage:** For complex datasets where feature interactions are important

.. class:: TabTransformerMethod
   :noindex:

   Transformer with column-wise attention for categorical features.
   
   **Features:** Contextual embeddings, strong on categorical-heavy datasets
   
   **Usage:** When your dataset has many important categorical features

Advanced Methods with Custom Data Processing
--------------------------------------------

These methods override `data_format()` and implement specialized data handling.

.. class:: TabNetMethod
   :noindex:

   TabNet with sequential attention and custom data processing pipeline.
   
   **Features:** 
   * Interpretable sequential feature selection
   * Custom TabNet-specific data processing
   * Sparse feature selection with attention visualization
   
   **Requirements:** `cat_policy` cannot be 'indices' (requires encoded categorical features)
   
   **Special Data Processing:**
   
   TabNet bypasses the standard `data_format()` method and implements its own data processing:
   
   .. method:: data_format(is_train=True, N=None, C=None, y=None)
      :noindex:
      
      Custom data processing optimized for TabNet's requirements.
      
      **Key Differences from Base:**
      
      * Direct integration with TabNet's internal categorical handling
      * Specialized preprocessing for TabNet's attention mechanism
      * Custom DataLoader creation for TabNet-compatible data structures
   
   **Usage:** When you need interpretable predictions with attention visualization

Methods with Custom Training Procedures
---------------------------------------

These methods implement specialized training workflows by overriding training-related methods.

.. class:: TabRMethod
   :noindex:

   TabR method implementing KNN-attention hybrid with retrieval-based training.
   
   **Features:**
   * Combines KNN retrieval with attention mechanisms
   * Context-aware predictions using training set as retrieval candidates
   * Custom fit method with retrieval context management
   
   **Requirements:** 
   * `cat_policy` must be 'tabr_ohe' 
   * `num_policy` must be 'none'
   
   **Custom Methods:**
   
   .. method:: fit(data, info, train=True, config=None)
      :noindex:
      
      Enhanced fit method with retrieval-based training setup.
      
      **Special Features:**
      
      * **Context Management:** Maintains context_size=96 for efficient retrieval
      * **Candidate Selection:** Uses full training set as retrieval candidates
      * **Index Tracking:** Manages training indices for neighbor search
      
      **Retrieval Process:**
      
      1. **Setup Retrieval Context:** Initialize training indices and context size limits
      2. **Model Construction:** Build TabR with both attention and KNN components  
      3. **Candidate Management:** Store training data for runtime retrieval
      4. **Training Loop:** Standard epoch-based training with retrieval context
   
   **Usage:** Excellent performance on many datasets, especially with clear patterns

.. class:: ExcelFormerMethod
   :noindex:

   ExcelFormer with semi-permeable attention and mixup training strategies.
   
   **Features:**
   * Multiple mixup strategies (feat_mix, hidden_mix, naive_mix)
   * Mutual information-based feature importance scoring
   * Custom training process with enhanced data augmentation
   
   **Custom Methods:**
   
   .. method:: fit(data, info, train=True, config=None)
      :noindex:
      
      Enhanced fit with mutual information preprocessing.
      
      **Preprocessing Steps:**
      
      1. **MI Score Computation:** Calculate mutual information between features and targets
      2. **Feature Ranking:** Sort features by importance for mixup weighting
      3. **Mixup Configuration:** Setup augmentation parameters based on MI scores
   
   .. method:: train_epoch(epoch)
      :noindex:
      
      Custom training with mixup augmentation strategies.
      
      **Mixup Training Process:**
      
      **Feature Mixup (`mix_type='feat_mix'`):**
      
      .. math::
         
         \lambda = \sum (\text{MI_scores} \odot \text{feat_masks}) \\
         \text{loss} = \lambda \cdot \text{criterion}(\text{pred}, y) + (1-\lambda) \cdot \text{criterion}(\text{pred}, y[\text{shuffled}])
      
      **Hidden Mixup (`mix_type='hidden_mix'`):**
      
      .. math::
         
         \text{loss} = \text{feat_masks} \cdot \text{criterion}(\text{pred}, y) + (1-\text{feat_masks}) \cdot \text{criterion}(\text{pred}, y[\text{shuffled}])
   
   **Usage:** When you need enhanced generalization through sophisticated data augmentation

.. class:: TromptMethod
   :noindex:

   Trompt method with prompt-based learning and multiple prediction cycles.
   
   **Features:**
   * Prompt-based neural architecture separating intrinsic and sample-specific features
   * Multiple learning cycles for improved performance
   * Custom training with repeated targets
   
   **Requirements:** `cat_policy` must be 'indices'
   
   **Custom Training:**
   
   .. method:: train_epoch(epoch)
      :noindex:
      
      Custom training with prompt-based multi-cycle learning.
      
      **Prompt Learning Process:**
      
      1. **Multi-cycle Forward:** Uses `model.forward_for_training()` for multiple prediction cycles
      2. **Target Repetition:** Repeats targets for each prediction cycle
      3. **Cycle Loss:** Computes loss across all cycles for robust learning
      
      **Mathematical Formulation:**
      
      For n_cycles prediction cycles:
      
      .. math::
         
         \text{output} = \text{model.forward_for_training}(X_{num}, X_{cat}) \\
         \text{output} = \text{output.view}(-1, d_{out}) \\
         y_{repeated} = y.\text{repeat_interleave}(n_{cycles})
   
   **Usage:** For complex learning scenarios requiring prompt-based architectures

Methods with Specialized Architectures
--------------------------------------

.. class:: ModernNCAMethod
   :noindex:

   Modern Nearest Class Analysis with embedding-based distance learning.
   
   **Features:**
   * Embedding space optimization for distance-based classification
   * Neighborhood sampling for efficient training
   * Interpretable neighbor relationships
   
   **Usage:** Excellent for datasets with clear class structure and when interpretability through neighbors is desired

.. class:: ProtoGateMethod
   :noindex:

   Prototype-based gating method for interpretable feature selection.
   
   **Features:**
   * Prototype-based learning with gating mechanisms
   * Enhanced interpretability through learned prototypes
   * Suitable for high-dimensional data with sparse relevant features
   
   **Usage:** When you need prototype-based explanations and adaptive feature selection

Foundation Model Methods
------------------------

.. class:: TabPFNMethod
   :noindex:

   Prior-Data Fitted Network (PFN) method using pre-trained foundation models.
   
   **Features:**
   * Pre-trained weights for immediate deployment
   * Zero-shot learning capabilities on new datasets
   * No gradient-based training required
   
   **Usage:** For rapid deployment without training when you have limited data or time

.. class:: TabICLMethod
   :noindex:

   In-context learning method for tabular data using foundation model approaches.
   
   **Features:**
   * Meta-learning from diverse tabular datasets
   * Adaptive to new domains without fine-tuning
   * Foundation model approach for tabular data
   
   **Usage:** When you need fast adaptation to new tabular domains

Method Usage Guidelines
=======================

**For Beginners:**
Start with `MLPMethod` or `ResNetMethod` - they're simple, fast, and provide good baselines.

**For Best Performance:**
Try `FTTMethod`, `TabNetMethod`, or `ModernNCAMethod` - these often achieve state-of-the-art results.

**For Interpretability:**
Use `TabNetMethod` (attention visualization), `ProtoGateMethod` (prototypes), or `ModernNCAMethod` (neighbors).

**For Speed:**
Choose `MLPMethod`, `SNNMethod`, or simple foundation models like `TabPFNMethod`.

**For Complex Features:**
Consider `SaintMethod`, `TabTransformerMethod`, or `ExcelFormerMethod` with mixup.

**Common Usage Pattern:**

.. code-block:: python
   
   from TALENT.model.utils import get_method
   
   # Get method class
   MethodClass = get_method('ftt')  # or 'mlp', 'tabnet', etc.
   
   # Initialize method
   method = MethodClass(args, is_regression=True)
   
   # Train the method  
   time_cost = method.fit(train_data, info)
   
   # Make predictions
   loss, metrics, metric_names, predictions = method.predict(test_data, info, 'best-val') 