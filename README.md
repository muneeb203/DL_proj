# KENN Replication Study - CiteSeer Dataset

Implementation of Knowledge Enhanced Neural Networks (KENN) for the CiteSeer citation network classification task.

## Quick Start

### 1. Verify Implementation
```bash
python test_implementation.py
```

### 2. Run Quick Test (Fast validation)
```bash
python main.py --quick-test
```
- 3 runs per configuration
- 50 epochs max
- Tests 10% and 50% training sizes only
- Takes ~5-10 minutes

### 3. Run Full Experiments
```bash
python main.py
```
- 10 runs per configuration
- 300 epochs max with early stopping
- All training sizes: 10%, 25%, 50%, 75%, 90%
- Takes ~1-2 hours (depending on hardware)

## Project Structure

```
New folder/
├── config.py              # Hyperparameters and configuration
├── data_loader.py         # CiteSeer dataset loader
├── models.py              # Base NN and KENN implementations
├── trainer.py             # Training and evaluation logic
├── experiment.py          # Experiment runner with statistical tests
├── main.py                # Main entry point
├── test_implementation.py # Verification tests
├── PLAN_A_IMPLEMENTATION.md  # Detailed implementation notes
└── results/               # Experiment results (created automatically)
```

## Models

### Base Neural Network
- 3 hidden layers (50 neurons each)
- ReLU activations
- Xavier uniform initialization
- No dropout

### KENN (Knowledge Enhanced Neural Network)
- Base NN + 3 Knowledge Enhancement layers
- Encodes citation knowledge: "papers share topics with cited papers"
- Łukasiewicz fuzzy logic with learnable clause weights
- Binary preactivation = 500 for sharp thresholds

## Hyperparameters (Matching Original KENN)

- **Optimizer**: Adam (β1=0.9, β2=0.99, ε=1e-7)
- **Learning Rate**: 0.001
- **Batch Size**: Full-batch
- **Epochs**: 300 (with early stopping)
- **Early Stopping**: patience=10, min_delta=0.001
- **Clause Weights**: Initialized to 0.5
- **Random Seed**: 0 (for reproducibility)

## Command Line Options

```bash
python main.py [OPTIONS]

Options:
  --train-sizes FLOAT [FLOAT ...]  Training set sizes (e.g., 0.1 0.25 0.5)
  --num-runs INT                   Number of runs per configuration (default: 10)
  --device {cuda,cpu}              Device to use (default: cuda)
  --epochs INT                     Maximum epochs (default: 300)
  --seed INT                       Random seed (default: 0)
  --quick-test                     Run quick test mode
```

## Examples

### Test specific training sizes
```bash
python main.py --train-sizes 0.1 0.5 0.9
```

### Run with more iterations
```bash
python main.py --num-runs 20
```

### CPU-only mode
```bash
python main.py --device cpu
```

### Custom seed
```bash
python main.py --seed 42
```

## Results

Results are automatically saved in `results/` directory:

- **JSON file**: Complete results with all runs and metrics
- **CSV file**: Summary statistics and significance tests

### Metrics Reported
- Test accuracy (mean ± std)
- Training/validation accuracy and loss
- Statistical significance (paired t-test, Wilcoxon)
- Effect size (Cohen's d)
- Number of epochs trained

### Hypothesis Testing
- **H1**: KENN significantly outperforms Base NN (p < 0.01)
- **H2**: Benefit is larger for smaller training sets

## Dataset

**CiteSeer**: Citation network dataset
- 3,312 papers (nodes)
- 3,703 features (bag-of-words)
- 6 classes (Agents, AI, DB, IR, ML, HCI)
- 4,715 citation edges (undirected)

Dataset should be located at: `../dataset/CiteSeer/`

## Requirements

```
torch
numpy
pandas
scipy
tqdm
```

Install with:
```bash
pip install torch numpy pandas scipy tqdm
```

## Implementation Details

See `PLAN_A_IMPLEMENTATION.md` for comprehensive documentation of:
- Architecture details
- Knowledge Enhancement Layer implementation
- Fuzzy logic formulation
- Changes from previous versions
- Verification procedures

## Troubleshooting

### Dataset not found
Ensure CiteSeer dataset is at `../dataset/CiteSeer/` relative to this folder.

### CUDA out of memory
Use CPU mode: `python main.py --device cpu`

### Slow training
Try quick test first: `python main.py --quick-test`

## Citation

If you use this implementation, please cite:
- Original KENN paper
- CiteSeer dataset
- Replication study paper

## License

Research and educational use only.
