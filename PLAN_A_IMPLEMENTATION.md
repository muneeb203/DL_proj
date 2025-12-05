# Plan A Implementation Summary

## Overview
This document summarizes the implementation of Plan A to fix the KENN (Knowledge Enhanced Neural Network) implementation for the CiteSeer dataset replication study.

## Changes Made

### 1. Configuration Updates (`config.py`)
- **Random Seed**: Changed from 42 to 0 to match the replication study
- **Clause Weight Initialization**: Confirmed at 0.5 (constant initial value)
- **Binary Preactivation**: Set to 500 (large value for sharp fuzzy logic threshold)
- **Comments**: Added clarifications about hyperparameter choices

### 2. Complete Experiment Runner (`experiment.py`)
**Created a fully functional ExperimentRunner class that:**
- Loops over specified training set sizes (10%, 25%, 50%, 75%, 90%)
- Uses multiple random seeds for statistical significance (default: 10 runs)
- Creates stratified train/val/test splits preserving class distribution
- Trains both Base MLP and KENN models on identical splits
- Evaluates on validation and test sets
- Records comprehensive results (accuracy, loss, etc.)
- Performs statistical tests (paired t-test, Wilcoxon signed-rank)
- Computes effect sizes (Cohen's d)
- Saves results in JSON and CSV formats

**Key Features:**
- Reproducibility: Sets seeds for NumPy, PyTorch, and CUDA
- Same split used for both Base NN and KENN for fair comparison
- Early stopping with patience=10 and min_delta=0.001
- Full-batch training as per original experiments

### 3. Corrected Hyperparameters
All hyperparameters now match the original KENN experiments:

**Base Neural Network:**
- 3 hidden layers with 50 neurons each
- ReLU activations
- Linear output (no activation before softmax)
- Glorot (Xavier) uniform weight initialization
- Zero bias initialization
- No dropout (dropout_rate = 0.0)

**KENN Configuration:**
- 3 Knowledge Enhancement (KE) layers
- Clause weights initialized to 0.5 (constant)
- Binary preactivation = 500
- Range constraint: [0.0, 500.0]

**Training Configuration:**
- Adam optimizer with β1=0.9, β2=0.99, ε=1e-7
- Learning rate = 0.001
- Full-batch training
- 300 epochs maximum
- Early stopping: patience=10, min_delta=0.001

### 4. Knowledge Enhancement Layer Implementation (`models.py`)

**Implemented correct KENN logic:**

The knowledge clause: **"Papers tend to share topics with the papers they cite"**
- Formally: ∀x∀y [T(x) ∧ Cite(x,y) → T(y)]
- Meaning: If paper x has topic T and cites paper y, then y should also have topic T

**Implementation Details:**
1. **Fuzzy Logic with Łukasiewicz Semantics:**
   - Convert logits to probabilities using sigmoid with binary preactivation (500)
   - This creates sharp, binary-like thresholds

2. **Neighbor Aggregation:**
   - Normalize adjacency matrix by node degree
   - Aggregate neighbor probabilities via matrix multiplication
   - This captures citation-based knowledge

3. **Clause Violation Computation:**
   - Compute: max(0, neighbor_prob - node_prob)
   - This measures how much neighbors disagree with current prediction

4. **Learnable Clause Weights:**
   - One weight per class (initialized to 0.5)
   - Weights are trainable parameters
   - Positive weights enforce the clause, negative can learn exceptions

5. **Probability Enhancement:**
   - Apply weighted corrections to probabilities
   - Clamp to [0, 1] range
   - Convert back to logits (inverse sigmoid)

6. **Stacking:**
   - 3 KE layers applied sequentially after base network
   - Each layer refines predictions using graph structure
   - Final output passed through softmax for classification

**Removed:**
- Simplified KE layer that was causing issues
- Now using only the correct Łukasiewicz-based implementation

### 5. Data Loader Updates (`data_loader.py`)

**Graph Structure:**
- Binary, undirected adjacency matrix
- **Added self-loops**: Each node connected to itself
  - Allows node's own prediction to influence itself in KE layers
  - Standard practice in graph neural networks
- Degree normalization applied in KE layers (not pre-computed)

**Split Handling:**
- Transductive setting: Uses full adjacency matrix
- Stratified splits preserve class distribution
- Reproducible with seed control

### 6. Main Script Updates (`main.py`)
- Default seed changed to 0 (matching replication study)
- Quick test mode for rapid validation
- Comprehensive result reporting
- Hypothesis testing (H1 and H2)

### 7. Test Script (`test_implementation.py`)
Created comprehensive test to verify:
- Dataset loading
- Model instantiation
- Forward pass
- Backward pass and gradient computation
- Clause weight initialization
- Output shapes and ranges

## Architecture Summary

### Base Neural Network
```
Input (3703 features)
  ↓
Linear(3703 → 50) + ReLU
  ↓
Linear(50 → 50) + ReLU
  ↓
Linear(50 → 50) + ReLU
  ↓
Linear(50 → 6) [logits]
  ↓
Softmax → Predictions
```

### KENN Model
```
Input (3703 features)
  ↓
Base NN (3 hidden layers, 50 neurons each)
  ↓ [logits]
KE Layer 1 (uses adjacency matrix)
  ↓ [enhanced logits]
KE Layer 2 (uses adjacency matrix)
  ↓ [enhanced logits]
KE Layer 3 (uses adjacency matrix)
  ↓ [final logits]
Softmax → Predictions
```

## Expected Improvements

Based on Plan A implementation, KENN should now:

1. **Properly encode citation knowledge**: The clause "papers share topics with cited papers" is correctly implemented using Łukasiewicz fuzzy logic

2. **Learn appropriate clause weights**: Starting from 0.5, weights can be learned during training to balance base NN predictions with graph structure

3. **Show performance gains**: Especially on smaller training sets where graph structure provides valuable additional signal

4. **Match original KENN results**: All hyperparameters and architecture now match the original experiments

## Running Experiments

### Quick Test (3 runs, 50 epochs, 2 training sizes)
```bash
python main.py --quick-test
```

### Full Experiment (10 runs, 300 epochs, all training sizes)
```bash
python main.py
```

### Custom Configuration
```bash
python main.py --train-sizes 0.1 0.5 --num-runs 5 --epochs 200 --seed 0
```

## Results Location

Results are saved in `results/` directory:
- `results_YYYYMMDD_HHMMSS.json`: Detailed results with all runs
- `summary_YYYYMMDD_HHMMSS.csv`: Summary statistics and significance tests

## Verification

Run the test script to verify implementation:
```bash
python test_implementation.py
```

Expected output:
- ✓ Dataset loaded successfully
- ✓ Base NN forward/backward pass works
- ✓ KENN forward/backward pass works
- ✓ Clause weights initialized to 0.5
- ✓ All tests passed

## Key Differences from Previous Implementation

1. **Clause weights**: Now correctly initialized to 0.5 (not 0.01 or random)
2. **Fuzzy logic**: Proper Łukasiewicz semantics with binary preactivation=500
3. **Graph structure**: Self-loops added, degree normalization in KE layers
4. **Reproducibility**: Consistent seed usage (seed=0) across all experiments
5. **Complete experiment runner**: Handles multiple runs, splits, and statistical tests
6. **Hyperparameters**: All match original KENN paper exactly

## Next Steps

1. Run quick test to verify everything works
2. Run full experiments with all training sizes
3. Analyze results to verify hypotheses:
   - H1: KENN significantly outperforms Base NN
   - H2: Benefit is larger for smaller training sets
4. Compare with original KENN paper results

## References

- Original KENN paper: Knowledge Enhanced Neural Networks
- Replication study: "Reproduce, Replicate, Reevaluate"
- CiteSeer dataset: Citation network for paper classification
