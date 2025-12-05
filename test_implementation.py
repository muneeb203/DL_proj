"""
Quick test script to verify the KENN implementation works correctly.
"""

import torch
import numpy as np
from config import Config
from data_loader import CiteSeerDataset
from models import BaseNeuralNetwork, KENN, count_parameters

def test_implementation():
    """Test that models can be instantiated and run forward pass."""
    print("="*80)
    print("TESTING KENN IMPLEMENTATION")
    print("="*80)
    
    # Load config
    config = Config()
    
    # Load dataset
    print("\n1. Loading dataset...")
    dataset = CiteSeerDataset(config.DATASET_PATH)
    print(f"   ✓ Dataset loaded: {dataset.num_nodes} nodes, {dataset.num_features} features")
    
    # Convert to torch
    print("\n2. Converting to PyTorch tensors...")
    features, labels, adj_matrix = dataset.to_torch('cpu')
    print(f"   ✓ Features shape: {features.shape}")
    print(f"   ✓ Labels shape: {labels.shape}")
    print(f"   ✓ Adjacency shape: {adj_matrix.shape}")
    
    # Test Base NN
    print("\n3. Testing Base Neural Network...")
    base_nn = BaseNeuralNetwork(
        num_features=dataset.num_features,
        num_classes=dataset.num_classes,
        hidden_layers=config.NN_HIDDEN_LAYERS,
        hidden_neurons=config.NN_HIDDEN_NEURONS,
        dropout_rate=config.NN_DROPOUT_RATE
    )
    print(f"   ✓ Base NN created with {count_parameters(base_nn)} parameters")
    
    # Forward pass
    with torch.no_grad():
        logits = base_nn(features)
    print(f"   ✓ Forward pass successful, output shape: {logits.shape}")
    print(f"   ✓ Output range: [{logits.min():.4f}, {logits.max():.4f}]")
    
    # Test KENN
    print("\n4. Testing KENN Model...")
    kenn = KENN(
        num_features=dataset.num_features,
        num_classes=dataset.num_classes,
        num_nodes=dataset.num_nodes,
        hidden_layers=config.NN_HIDDEN_LAYERS,
        hidden_neurons=config.NN_HIDDEN_NEURONS,
        num_ke_layers=config.KENN_NUM_LAYERS,
        clause_weight_init=config.KENN_CLAUSE_WEIGHT_INIT,
        binary_preactivation=config.KENN_BINARY_PREACTIVATIONS,
        range_constraint=config.KENN_RANGE_CONSTRAINT,
        dropout_rate=config.NN_DROPOUT_RATE
    )
    print(f"   ✓ KENN created with {count_parameters(kenn)} parameters")
    
    # Forward pass
    with torch.no_grad():
        kenn_logits = kenn(features, adj_matrix)
    print(f"   ✓ Forward pass successful, output shape: {kenn_logits.shape}")
    print(f"   ✓ Output range: [{kenn_logits.min():.4f}, {kenn_logits.max():.4f}]")
    
    # Check clause weights
    print("\n5. Checking Knowledge Enhancement Layers...")
    for i, ke_layer in enumerate(kenn.ke_layers):
        weights = ke_layer.clause_weights.data
        print(f"   Layer {i+1} clause weights: {weights.tolist()}")
    
    # Test backward pass
    print("\n6. Testing backward pass...")
    kenn.train()
    optimizer = torch.optim.Adam(kenn.parameters(), lr=0.001)
    
    # Create a small batch
    batch_idx = torch.LongTensor([0, 1, 2, 3, 4])
    logits = kenn(features, adj_matrix)
    loss = torch.nn.functional.cross_entropy(logits[batch_idx], labels[batch_idx])
    
    loss.backward()
    optimizer.step()
    print(f"   ✓ Backward pass successful, loss: {loss.item():.4f}")
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED ✓")
    print("="*80)
    print("\nImplementation is ready for experiments!")
    print("Run: python main.py --quick-test")

if __name__ == "__main__":
    test_implementation()
