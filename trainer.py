"""
Training and evaluation utilities for KENN experiments.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional, List
from tqdm import tqdm
import time


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = model.state_dict().copy()
        
        return self.early_stop
    
    def load_best_model(self, model: nn.Module):
        """Load the best model state."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


class Trainer:
    """Trainer for neural network models."""
    
    def __init__(self, model: nn.Module, device: str = 'cpu',
                 learning_rate: float = 0.001,
                 optimizer_params: Optional[Dict] = None):
        self.model = model.to(device)
        self.device = device
        
        # Setup optimizer (Adam with specific parameters)
        if optimizer_params is None:
            optimizer_params = {
                'betas': (0.9, 0.99),
                'eps': 1e-7
            }
        
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            **optimizer_params
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(self, features: torch.Tensor, labels: torch.Tensor,
                   train_idx: np.ndarray, adj_matrix: Optional[torch.Tensor] = None) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Convert indices to tensor
        train_idx_tensor = torch.LongTensor(train_idx).to(self.device)
        
        # Forward pass
        if adj_matrix is not None:
            # KENN model
            logits = self.model(features, adj_matrix)
        else:
            # Base NN model
            logits = self.model(features)
        
        # Compute loss only on training nodes
        loss = self.criterion(logits[train_idx_tensor], labels[train_idx_tensor])
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Compute accuracy
        with torch.no_grad():
            preds = torch.argmax(logits[train_idx_tensor], dim=1)
            acc = (preds == labels[train_idx_tensor]).float().mean().item()
        
        return loss.item(), acc
    
    def evaluate(self, features: torch.Tensor, labels: torch.Tensor,
                eval_idx: np.ndarray, adj_matrix: Optional[torch.Tensor] = None) -> Tuple[float, float]:
        """Evaluate model on given indices."""
        self.model.eval()
        
        # Convert indices to tensor
        eval_idx_tensor = torch.LongTensor(eval_idx).to(self.device)
        
        with torch.no_grad():
            # Forward pass
            if adj_matrix is not None:
                logits = self.model(features, adj_matrix)
            else:
                logits = self.model(features)
            
            # Compute loss and accuracy
            loss = self.criterion(logits[eval_idx_tensor], labels[eval_idx_tensor])
            preds = torch.argmax(logits[eval_idx_tensor], dim=1)
            acc = (preds == labels[eval_idx_tensor]).float().mean().item()
        
        return loss.item(), acc
    
    def fit(self, features: torch.Tensor, labels: torch.Tensor,
            train_idx: np.ndarray, val_idx: np.ndarray,
            adj_matrix: Optional[torch.Tensor] = None,
            epochs: int = 300, early_stopping_patience: int = 10,
            early_stopping_min_delta: float = 0.001,
            verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            features: Node features
            labels: Node labels
            train_idx: Training node indices
            val_idx: Validation node indices
            adj_matrix: Adjacency matrix (for KENN)
            epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            early_stopping_min_delta: Minimum delta for early stopping
            verbose: Print progress
        
        Returns:
            Training history
        """
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta
        )
        
        iterator = tqdm(range(epochs), desc="Training") if verbose else range(epochs)
        
        for epoch in iterator:
            # Train
            train_loss, train_acc = self.train_epoch(
                features, labels, train_idx, adj_matrix
            )
            
            # Validate
            if len(val_idx) > 0:
                val_loss, val_acc = self.evaluate(
                    features, labels, val_idx, adj_matrix
                )
            else:
                val_loss, val_acc = train_loss, train_acc
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Update progress bar
            if verbose:
                iterator.set_postfix({
                    'train_loss': f'{train_loss:.4f}',
                    'train_acc': f'{train_acc:.4f}',
                    'val_loss': f'{val_loss:.4f}',
                    'val_acc': f'{val_acc:.4f}'
                })
            
            # Early stopping
            if early_stopping(val_loss, self.model):
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                break
        
        # Load best model
        early_stopping.load_best_model(self.model)
        
        return self.history
    
    def predict(self, features: torch.Tensor, 
                adj_matrix: Optional[torch.Tensor] = None) -> np.ndarray:
        """Make predictions."""
        self.model.eval()
        
        with torch.no_grad():
            if adj_matrix is not None:
                logits = self.model(features, adj_matrix)
            else:
                logits = self.model(features)
            
            preds = torch.argmax(logits, dim=1)
        
        return preds.cpu().numpy()
    
    def get_predictions_and_probabilities(self, features: torch.Tensor,
                                         adj_matrix: Optional[torch.Tensor] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions and class probabilities."""
        self.model.eval()
        
        with torch.no_grad():
            if adj_matrix is not None:
                logits = self.model(features, adj_matrix)
            else:
                logits = self.model(features)
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        
        return preds.cpu().numpy(), probs.cpu().numpy()
