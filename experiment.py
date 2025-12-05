"""
Experiment runner for KENN replication study.

Orchestrates experiments across multiple training sizes and runs,
collecting statistics and performing significance testing.
"""

import torch
import numpy as np
import pandas as pd
import json
import os
from typing import Dict, List, Tuple
from datetime import datetime
from scipy import stats
from tqdm import tqdm

from config import Config
from data_loader import CiteSeerDataset
from models import BaseNeuralNetwork, KENN, count_parameters
from trainer import Trainer


class ExperimentRunner:
    """
    Orchestrates KENN replication experiments.
    
    For each training size:
    - Creates multiple random splits (stratified by class)
    - Trains Base NN and KENN models
    - Evaluates on test set
    - Collects statistics and performs significance testing
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.dataset = None
        self.results = {
            'base_nn': {},
            'kenn': {},
            'comparison': {},
            'raw_results': []
        }
        
        # Setup device
        self.device = self._setup_device()
        
        # Load dataset
        self._load_dataset()
    
    def _setup_device(self) -> str:
        """Setup computation device (CUDA or CPU)."""
        if self.config.DEVICE == 'cuda' and torch.cuda.is_available():
            device = 'cuda'
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = 'cpu'
            print("Using CPU")
        
        return device
    
    def _load_dataset(self):
        """Load CiteSeer dataset."""
        print("\n" + "="*80)
        print("LOADING DATASET")
        print("="*80)
        
        self.dataset = CiteSeerDataset(self.config.DATASET_PATH)
        
        print(f"Dataset: CiteSeer")
        print(f"Nodes: {self.dataset.num_nodes}")
        print(f"Features: {self.dataset.num_features}")
        print(f"Classes: {self.dataset.num_classes}")
        print(f"Edges: {int(self.dataset.adj_matrix.sum() / 2)}")
        
        # Print class distribution
        print("\nClass distribution:")
        for i, class_name in enumerate(self.dataset.class_names):
            count = np.sum(self.dataset.labels == i)
            print(f"  {class_name}: {count} ({count/self.dataset.num_nodes*100:.1f}%)")
    
    def _create_stratified_split(self, train_size: float, 
                                 seed: int) -> Dict[str, np.ndarray]:
        """
        Create stratified train/val/test split.
        
        Ensures class distribution is preserved across splits.
        Uses the same seed for reproducibility.
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        indices_by_class = []
        for c in range(self.dataset.num_classes):
            class_indices = np.where(self.dataset.labels == c)[0]
            np.random.shuffle(class_indices)
            indices_by_class.append(class_indices)
        
        train_idx = []
        val_idx = []
        test_idx = []
        
        for class_indices in indices_by_class:
            n_total = len(class_indices)
            n_train = max(1, int(n_total * train_size))
            n_val = max(1, int(n_total * 0.1))  # 10% for validation
            
            train_idx.extend(class_indices[:n_train])
            val_idx.extend(class_indices[n_train:n_train + n_val])
            test_idx.extend(class_indices[n_train + n_val:])
        
        return {
            'train': np.array(train_idx),
            'val': np.array(val_idx),
            'test': np.array(test_idx)
        }

    def _train_and_evaluate(self, model_type: str, train_size: float,
                           split: Dict[str, np.ndarray], run: int, seed: int) -> Dict:
        """
        Train and evaluate a single model.
        
        Args:
            model_type: 'base_nn' or 'kenn'
            train_size: Training set size fraction
            split: Dictionary with train/val/test indices
            run: Run number
            seed: Random seed for this run
        
        Returns:
            Dictionary with results
        """
        # Set seeds for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # Convert dataset to torch tensors
        features, labels, adj_matrix = self.dataset.to_torch(self.device)
        
        # Create model
        if model_type == 'base_nn':
            model = BaseNeuralNetwork(
                num_features=self.dataset.num_features,
                num_classes=self.dataset.num_classes,
                hidden_layers=self.config.NN_HIDDEN_LAYERS,
                hidden_neurons=self.config.NN_HIDDEN_NEURONS,
                dropout_rate=self.config.NN_DROPOUT_RATE
            )
            adj_for_training = None
        else:  # kenn
            model = KENN(
                num_features=self.dataset.num_features,
                num_classes=self.dataset.num_classes,
                num_nodes=self.dataset.num_nodes,
                hidden_layers=self.config.NN_HIDDEN_LAYERS,
                hidden_neurons=self.config.NN_HIDDEN_NEURONS,
                num_ke_layers=self.config.KENN_NUM_LAYERS,
                clause_weight_init=self.config.KENN_CLAUSE_WEIGHT_INIT,
                binary_preactivation=self.config.KENN_BINARY_PREACTIVATIONS,
                range_constraint=self.config.KENN_RANGE_CONSTRAINT,
                dropout_rate=self.config.NN_DROPOUT_RATE
            )
            adj_for_training = adj_matrix
        
        # Create trainer with Adam optimizer (β1=0.9, β2=0.99, ε=1e-7)
        trainer = Trainer(
            model=model,
            device=self.device,
            learning_rate=self.config.LEARNING_RATE,
            optimizer_params={
                'betas': (self.config.OPTIMIZER_BETA1, self.config.OPTIMIZER_BETA2),
                'eps': self.config.OPTIMIZER_EPSILON
            }
        )
        
        # Train model
        history = trainer.fit(
            features=features,
            labels=labels,
            train_idx=split['train'],
            val_idx=split['val'],
            adj_matrix=adj_for_training,
            epochs=self.config.EPOCHS,
            early_stopping_patience=self.config.EARLY_STOPPING_PATIENCE,
            early_stopping_min_delta=self.config.EARLY_STOPPING_MIN_DELTA,
            verbose=False
        )
        
        # Evaluate on test set
        test_loss, test_acc = trainer.evaluate(
            features=features,
            labels=labels,
            eval_idx=split['test'],
            adj_matrix=adj_for_training
        )
        
        # Get final training and validation metrics
        final_train_loss = history['train_loss'][-1]
        final_train_acc = history['train_acc'][-1]
        final_val_loss = history['val_loss'][-1]
        final_val_acc = history['val_acc'][-1]
        
        return {
            'model_type': model_type,
            'train_size': train_size,
            'run': run,
            'seed': seed,
            'test_accuracy': test_acc,
            'test_loss': test_loss,
            'train_accuracy': final_train_acc,
            'train_loss': final_train_loss,
            'val_accuracy': final_val_acc,
            'val_loss': final_val_loss,
            'num_epochs': len(history['train_loss']),
            'num_parameters': count_parameters(model)
        }

    def run_experiments(self) -> Dict:
        """
        Run all experiments across training sizes and multiple runs.
        
        Returns:
            Dictionary with aggregated results and statistical tests
        """
        print("\n" + "="*80)
        print("RUNNING EXPERIMENTS")
        print("="*80)
        
        # Create results directory
        os.makedirs(self.config.RESULTS_DIR, exist_ok=True)
        
        # Run experiments for each training size
        for train_size in self.config.TRAIN_SIZES:
            print(f"\n{'='*80}")
            print(f"Training Size: {int(train_size * 100)}%")
            print(f"{'='*80}")
            
            base_nn_results = []
            kenn_results = []
            
            # Run multiple times with different seeds
            for run in tqdm(range(self.config.NUM_RUNS), desc=f"Runs"):
                # Use different seed for each run
                seed = self.config.RANDOM_SEED + run
                
                # Create split for this run
                split = self._create_stratified_split(train_size, seed)
                
                # Train and evaluate Base NN
                base_result = self._train_and_evaluate(
                    model_type='base_nn',
                    train_size=train_size,
                    split=split,
                    run=run,
                    seed=seed
                )
                base_nn_results.append(base_result['test_accuracy'])
                self.results['raw_results'].append(base_result)
                
                # Train and evaluate KENN
                kenn_result = self._train_and_evaluate(
                    model_type='kenn',
                    train_size=train_size,
                    split=split,
                    run=run,
                    seed=seed
                )
                kenn_results.append(kenn_result['test_accuracy'])
                self.results['raw_results'].append(kenn_result)
            
            # Compute statistics
            base_nn_results = np.array(base_nn_results)
            kenn_results = np.array(kenn_results)
            
            # Store aggregated results
            self.results['base_nn'][train_size] = {
                'mean': np.mean(base_nn_results),
                'std': np.std(base_nn_results),
                'min': np.min(base_nn_results),
                'max': np.max(base_nn_results),
                'all_runs': base_nn_results.tolist()
            }
            
            self.results['kenn'][train_size] = {
                'mean': np.mean(kenn_results),
                'std': np.std(kenn_results),
                'min': np.min(kenn_results),
                'max': np.max(kenn_results),
                'all_runs': kenn_results.tolist()
            }
            
            # Perform statistical tests
            # Paired t-test (since same splits are used)
            t_stat, p_value_ttest = stats.ttest_rel(kenn_results, base_nn_results)
            
            # Wilcoxon signed-rank test (non-parametric alternative)
            w_stat, p_value_wilcoxon = stats.wilcoxon(kenn_results, base_nn_results)
            
            # Effect size (Cohen's d)
            delta = kenn_results - base_nn_results
            cohens_d = np.mean(delta) / np.std(delta)
            
            self.results['comparison'][train_size] = {
                'delta_mean': np.mean(delta),
                'delta_std': np.std(delta),
                't_statistic': t_stat,
                'p_value_ttest': p_value_ttest,
                'w_statistic': w_stat,
                'p_value_wilcoxon': p_value_wilcoxon,
                'cohens_d': cohens_d,
                'significant': p_value_ttest < self.config.SIGNIFICANCE_LEVEL
            }
            
            # Print results
            print(f"\nResults for {int(train_size * 100)}% training size:")
            print(f"  Base NN:  {self.results['base_nn'][train_size]['mean']:.4f} ± "
                  f"{self.results['base_nn'][train_size]['std']:.4f}")
            print(f"  KENN:     {self.results['kenn'][train_size]['mean']:.4f} ± "
                  f"{self.results['kenn'][train_size]['std']:.4f}")
            print(f"  Delta:    {self.results['comparison'][train_size]['delta_mean']:.4f}")
            print(f"  p-value:  {p_value_ttest:.6f} "
                  f"({'significant' if p_value_ttest < self.config.SIGNIFICANCE_LEVEL else 'not significant'})")
            print(f"  Cohen's d: {cohens_d:.4f}")
        
        # Save results
        self._save_results()
        
        return self.results
    
    def _save_results(self):
        """Save results to JSON and CSV files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results as JSON
        json_path = os.path.join(self.config.RESULTS_DIR, f"results_{timestamp}.json")
        with open(json_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            results_serializable = {
                'base_nn': {str(k): v for k, v in self.results['base_nn'].items()},
                'kenn': {str(k): v for k, v in self.results['kenn'].items()},
                'comparison': {str(k): v for k, v in self.results['comparison'].items()},
                'raw_results': self.results['raw_results']
            }
            json.dump(results_serializable, f, indent=2)
        
        print(f"\nDetailed results saved to: {json_path}")
        
        # Save summary as CSV
        summary_data = []
        for train_size in self.config.TRAIN_SIZES:
            summary_data.append({
                'train_size': train_size,
                'base_nn_mean': self.results['base_nn'][train_size]['mean'],
                'base_nn_std': self.results['base_nn'][train_size]['std'],
                'kenn_mean': self.results['kenn'][train_size]['mean'],
                'kenn_std': self.results['kenn'][train_size]['std'],
                'delta_mean': self.results['comparison'][train_size]['delta_mean'],
                'p_value': self.results['comparison'][train_size]['p_value_ttest'],
                'cohens_d': self.results['comparison'][train_size]['cohens_d'],
                'significant': self.results['comparison'][train_size]['significant']
            })
        
        df = pd.DataFrame(summary_data)
        csv_path = os.path.join(self.config.RESULTS_DIR, f"summary_{timestamp}.csv")
        df.to_csv(csv_path, index=False)
        
        print(f"Summary saved to: {csv_path}")
