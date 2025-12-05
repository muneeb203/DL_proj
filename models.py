"""
Neural network models: Base NN and KENN (Knowledge Enhanced Neural Network).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional


class BaseNeuralNetwork(nn.Module):
    """
    Base Multi-Layer Perceptron for node classification.
    
    Architecture:
    - Input layer (num_features)
    - 3 hidden layers with 50 neurons each (ReLU activation)
    - Output layer (num_classes, linear activation)
    """
    
    def __init__(self, num_features: int, num_classes: int, 
                 hidden_layers: int = 3, hidden_neurons: int = 50,
                 dropout_rate: float = 0.0):
        super(BaseNeuralNetwork, self).__init__()
        
        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons
        self.dropout_rate = dropout_rate
        
        # Build layers
        layers = []
        
        # Input to first hidden layer
        layers.append(nn.Linear(num_features, hidden_neurons))
        layers.append(nn.ReLU())
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_neurons, hidden_neurons))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
        
        # Output layer (linear activation)
        layers.append(nn.Linear(hidden_neurons, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights (Glorot uniform) and biases (zeros)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with Glorot uniform and biases with zeros."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)  # Glorot uniform
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features [batch_size, num_features]
        
        Returns:
            Logits [batch_size, num_classes]
        """
        return self.network(x)


class KnowledgeEnhancementLayer(nn.Module):
    """
    Knowledge Enhancement Layer for KENN.
    
    Implements the KENN enhancement using Łukasiewicz fuzzy logic.
    For citation networks, the knowledge clause is:
    "Papers tend to share topics with the papers they cite"
    Formally: ∀x∀y [T(x) ∧ Cite(x,y) → T(y)]
    
    This means: if paper x has topic T and cites paper y, then y should also have topic T.
    """
    
    def __init__(self, num_classes: int, num_nodes: int,
                 clause_weight_init: float = 0.5,
                 binary_preactivation: float = 500.0,
                 range_constraint: Tuple[float, float] = (0.0, 500.0)):
        super(KnowledgeEnhancementLayer, self).__init__()
        
        self.num_classes = num_classes
        self.num_nodes = num_nodes
        self.binary_preactivation = binary_preactivation
        self.range_min, self.range_max = range_constraint
        
        # Clause weights (trainable parameters) - one per class
        # Initialize to small value to avoid overwhelming base NN initially
        # Will be learned during training
        self.clause_weights = nn.Parameter(
            torch.full((num_classes,), clause_weight_init * 0.1)  # Start at 0.05
        )
    
    def forward(self, logits: torch.Tensor, 
                adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Apply knowledge enhancement using citation graph structure.
        
        Simplified approach that works directly in logit space:
        1. Aggregate neighbor logits via graph
        2. Compute weighted correction based on neighbor consensus
        3. Add correction to original logits (residual connection)
        
        Args:
            logits: Current predictions [num_nodes, num_classes]
            adj_matrix: Adjacency matrix [num_nodes, num_nodes]
        
        Returns:
            Enhanced logits [num_nodes, num_classes]
        """
        # Normalize adjacency matrix by degree (average neighbor predictions)
        degree = adj_matrix.sum(dim=1, keepdim=True)
        degree = torch.clamp(degree, min=1.0)
        adj_normalized = adj_matrix / degree
        
        # Aggregate neighbor logits via graph structure
        # This represents the "knowledge" from citations
        neighbor_logits = torch.matmul(adj_normalized, logits)
        
        # Compute difference between neighbor consensus and current prediction
        # Positive means neighbors predict higher for this class
        logit_diff = neighbor_logits - logits
        
        # Apply learnable clause weights (per class)
        # Start at 0.5, can be learned during training
        # Positive weights mean "trust neighbors", negative means "distrust"
        weighted_correction = self.clause_weights.unsqueeze(0) * logit_diff
        
        # Add correction to original logits (residual connection)
        # This preserves base NN predictions while adding graph information
        enhanced_logits = logits + weighted_correction
        
        # Apply range constraint to prevent extreme values
        enhanced_logits = torch.clamp(
            enhanced_logits, self.range_min, self.range_max
        )
        
        return enhanced_logits


class KENN(nn.Module):
    """
    Knowledge Enhanced Neural Network.
    
    Combines a base neural network with knowledge enhancement layers
    that encode prior knowledge as first-order logic clauses.
    """
    
    def __init__(self, num_features: int, num_classes: int, num_nodes: int,
                 hidden_layers: int = 3, hidden_neurons: int = 50,
                 num_ke_layers: int = 3, clause_weight_init: float = 0.5,
                 binary_preactivation: float = 500.0,
                 range_constraint: Tuple[float, float] = (0.0, 500.0),
                 dropout_rate: float = 0.0):
        super(KENN, self).__init__()
        
        # Base neural network
        self.base_nn = BaseNeuralNetwork(
            num_features=num_features,
            num_classes=num_classes,
            hidden_layers=hidden_layers,
            hidden_neurons=hidden_neurons,
            dropout_rate=dropout_rate
        )
        
        # Knowledge enhancement layers - stack 3 layers as per original KENN
        self.ke_layers = nn.ModuleList([
            KnowledgeEnhancementLayer(
                num_classes=num_classes,
                num_nodes=num_nodes,
                clause_weight_init=clause_weight_init,
                binary_preactivation=binary_preactivation,
                range_constraint=range_constraint
            )
            for _ in range(num_ke_layers)
        ])
        
        self.num_ke_layers = num_ke_layers
    
    def forward(self, x: torch.Tensor, 
                adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through base NN and knowledge enhancement layers.
        
        Args:
            x: Input features [num_nodes, num_features]
            adj_matrix: Adjacency matrix [num_nodes, num_nodes]
        
        Returns:
            Enhanced logits [num_nodes, num_classes]
        """
        # Base neural network prediction
        logits = self.base_nn(x)
        
        # Apply knowledge enhancement layers sequentially
        for ke_layer in self.ke_layers:
            logits = ke_layer(logits, adj_matrix)
        
        return logits
    
    def get_base_predictions(self, x: torch.Tensor) -> torch.Tensor:
        """Get predictions from base NN only (without knowledge enhancement)."""
        return self.base_nn(x)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
