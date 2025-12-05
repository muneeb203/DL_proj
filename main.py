"""
Main script to run KENN replication experiments on CiteSeer dataset.

This script reproduces the results from:
"Reproduce, Replicate, Reevaluate: The Long but Safe Way to Extend Machine Learning Methods"

The goal is to verify:
H1: KENN significantly outperforms the base neural network
H2: The benefit of knowledge enhancement is larger for smaller training set sizes
"""

import argparse
import torch
import numpy as np
from config import Config
from experiment import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(
        description="KENN Replication on CiteSeer Dataset"
    )
    parser.add_argument(
        '--train-sizes',
        type=float,
        nargs='+',
        default=None,
        help='Training set sizes to evaluate (e.g., 0.1 0.25 0.5 0.75 0.9)'
    )
    parser.add_argument(
        '--num-runs',
        type=int,
        default=10,
        help='Number of runs for each configuration'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for training'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=300,
        help='Maximum number of training epochs'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed (default: 0 as per replication study)'
    )
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Run quick test with fewer runs and epochs'
    )
    
    args = parser.parse_args()
    
    # Update config based on arguments
    config = Config()
    
    if args.train_sizes is not None:
        config.TRAIN_SIZES = args.train_sizes
    
    config.NUM_RUNS = args.num_runs
    config.DEVICE = args.device
    config.EPOCHS = args.epochs
    config.RANDOM_SEED = args.seed
    
    # Quick test mode
    if args.quick_test:
        print("\n" + "="*80)
        print("QUICK TEST MODE")
        print("="*80)
        config.NUM_RUNS = 3
        config.EPOCHS = 50
        config.TRAIN_SIZES = [0.10, 0.50]
        print(f"Reduced to {config.NUM_RUNS} runs and {config.EPOCHS} epochs")
        print(f"Testing only train sizes: {config.TRAIN_SIZES}")
    
    # Print configuration
    print("\n" + "="*80)
    print("EXPERIMENT CONFIGURATION")
    print("="*80)
    print(f"Dataset: CiteSeer")
    print(f"Training sizes: {[f'{int(s*100)}%' for s in config.TRAIN_SIZES]}")
    print(f"Number of runs per configuration: {config.NUM_RUNS}")
    print(f"Maximum epochs: {config.EPOCHS}")
    print(f"Device: {config.DEVICE}")
    print(f"Random seed: {config.RANDOM_SEED}")
    print(f"\nBase NN Architecture:")
    print(f"  - Hidden layers: {config.NN_HIDDEN_LAYERS}")
    print(f"  - Hidden neurons: {config.NN_HIDDEN_NEURONS}")
    print(f"  - Activation: {config.NN_HIDDEN_ACTIVATION}")
    print(f"\nKENN Configuration:")
    print(f"  - KE layers: {config.KENN_NUM_LAYERS}")
    print(f"  - Clause weight init: {config.KENN_CLAUSE_WEIGHT_INIT}")
    print(f"  - Binary preactivations: {config.KENN_BINARY_PREACTIVATIONS}")
    print(f"\nTraining Configuration:")
    print(f"  - Learning rate: {config.LEARNING_RATE}")
    print(f"  - Optimizer: {config.OPTIMIZER}")
    print(f"  - Early stopping patience: {config.EARLY_STOPPING_PATIENCE}")
    print("="*80 + "\n")
    
    # Run experiments
    runner = ExperimentRunner(config)
    results = runner.run_experiments()
    
    # Print final summary
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    print("\nHypothesis Testing:")
    print("-" * 80)
    
    all_significant = True
    deltas = []
    
    for train_size in config.TRAIN_SIZES:
        comp = results['comparison'][train_size]
        base_nn = results['base_nn'][train_size]
        kenn = results['kenn'][train_size]
        
        print(f"\nTraining Size: {int(train_size * 100)}%")
        print(f"  Base NN:  {base_nn['mean']:.4f} ± {base_nn['std']:.4f}")
        print(f"  KENN:     {kenn['mean']:.4f} ± {kenn['std']:.4f}")
        print(f"  Delta:    {comp['delta_mean']:.4f}")
        print(f"  p-value:  {comp['p_value_ttest']:.6f} "
              f"{'✓ Significant' if comp['significant'] else '✗ Not significant'}")
        
        deltas.append(comp['delta_mean'])
        if not comp['significant']:
            all_significant = False
    
    print("\n" + "="*80)
    print("HYPOTHESIS VERIFICATION:")
    print("="*80)
    
    # H1: KENN significantly outperforms Base NN
    print(f"\nH1: KENN significantly outperforms Base NN")
    print(f"    Result: {'✓ SUPPORTED' if all_significant else '✗ NOT SUPPORTED'}")
    print(f"    All training sizes show significant improvement: {all_significant}")
    
    # H2: Benefit is larger for smaller training sets
    print(f"\nH2: Knowledge enhancement benefit is larger for smaller training sets")
    deltas_decreasing = all(deltas[i] >= deltas[i+1] for i in range(len(deltas)-1))
    print(f"    Result: {'✓ SUPPORTED' if deltas_decreasing else '✗ PARTIALLY SUPPORTED'}")
    print(f"    Delta trend: {' > '.join([f'{d:.4f}' for d in deltas])}")
    
    print("\n" + "="*80)
    print("Experiment completed successfully!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
