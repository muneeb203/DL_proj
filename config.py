"""
Configuration file for KENN experiments on CiteSeer dataset.
All hyperparameters are documented based on the reproduction paper.
"""

class Config:
    # Dataset
    DATASET_PATH = "../dataset/CiteSeer"
    NUM_CLASSES = 6
    CLASS_NAMES = ["Agents", "AI", "DB", "IR", "ML", "HCI"]
    
    # Training splits (percentage of labeled nodes for training)
    TRAIN_SIZES = [0.10, 0.25, 0.50, 0.75, 0.90]
    
    # Base Neural Network Architecture
    NN_HIDDEN_LAYERS = 3
    NN_HIDDEN_NEURONS = 50
    NN_HIDDEN_ACTIVATION = "relu"
    NN_OUTPUT_ACTIVATION = "linear"  # Linear before softmax
    NN_WEIGHT_INIT = "glorot_uniform"
    NN_BIAS_INIT = "zeros"
    NN_DROPOUT_RATE = 0.0  # No dropout
    
    # KENN Enhancement Layers
    KENN_NUM_LAYERS = 3
    KENN_CLAUSE_WEIGHT_INIT = 0.5  # Constant initial value as per original KENN
    KENN_BINARY_PREACTIVATIONS = 500  # Large value for sharp fuzzy logic threshold
    KENN_RANGE_CONSTRAINT = (0.0, 500.0)
    
    # Training Configuration
    EPOCHS = 300
    BATCH_SIZE = None  # Full-batch (will be set to dataset size)
    LEARNING_RATE = 0.001
    OPTIMIZER = "adam"
    OPTIMIZER_BETA1 = 0.9
    OPTIMIZER_BETA2 = 0.99
    OPTIMIZER_EPSILON = 1e-7
    LOSS_FUNCTION = "cross_entropy"
    
    # Early Stopping
    EARLY_STOPPING_PATIENCE = 10
    EARLY_STOPPING_MIN_DELTA = 0.001
    
    # Experiment Configuration
    NUM_RUNS = 10  # Number of runs for statistical significance
    RANDOM_SEED = 0  # Use seed=0 as per replication study
    DEVICE = "cuda"  # Will fallback to cpu if cuda not available
    
    # Statistical Testing
    SIGNIFICANCE_LEVEL = 0.01
    
    # Logging
    VERBOSE = True
    SAVE_RESULTS = True
    RESULTS_DIR = "results"
