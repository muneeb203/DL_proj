"""
Data loading and preprocessing for CiteSeer dataset.
Handles the graph structure, features, labels, and train/val/test splits.
"""

import numpy as np
import torch
from typing import Dict, Tuple, List
import os


class CiteSeerDataset:
    """
    CiteSeer citation network dataset loader.
    
    Dataset structure:
    - citeseer.content: paper_id <word_attributes>+ <class_label>
    - citeseer.cites: cited_paper_id citing_paper_id
    - Preprocessed .npy files for features, labels, and splits
    """
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.features = None
        self.labels = None
        self.adj_matrix = None
        self.num_nodes = None
        self.num_features = None
        self.num_classes = None
        self.paper_ids = None
        self.class_names = ["Agents", "AI", "DB", "IR", "ML", "HCI"]
        
        self._load_data()
    
    def _load_data(self):
        """Load all dataset components."""
        print(f"Loading CiteSeer dataset from {self.data_path}...")
        
        # Load preprocessed features and labels
        self.features = np.load(os.path.join(self.data_path, "features.npy"))
        self.labels = np.load(os.path.join(self.data_path, "labels.npy"))
        
        # Convert one-hot labels to class indices
        if len(self.labels.shape) > 1 and self.labels.shape[1] > 1:
            self.labels = np.argmax(self.labels, axis=1)
        
        # Ensure labels are 1D
        self.labels = self.labels.flatten()
        
        self.num_nodes = self.features.shape[0]
        self.num_features = self.features.shape[1]
        self.num_classes = len(np.unique(self.labels))
        
        # Load graph structure
        self._load_graph()
        
        print(f"Dataset loaded: {self.num_nodes} nodes, "
              f"{self.num_features} features, {self.num_classes} classes")
    
    def _load_graph(self):
        """Load citation graph structure."""
        # Read paper IDs from content file
        content_file = os.path.join(self.data_path, "citeseer.content")
        paper_ids = []
        
        with open(content_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    paper_ids.append(parts[0])
        
        self.paper_ids = paper_ids
        paper_id_to_idx = {pid: idx for idx, pid in enumerate(paper_ids)}
        
        # Build adjacency matrix from citations
        cites_file = os.path.join(self.data_path, "citeseer.cites")
        edges = []
        
        with open(cites_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    cited, citing = parts
                    # Only include edges where both papers are in our dataset
                    if cited in paper_id_to_idx and citing in paper_id_to_idx:
                        cited_idx = paper_id_to_idx[cited]
                        citing_idx = paper_id_to_idx[citing]
                        edges.append((citing_idx, cited_idx))  # citing -> cited
        
        # Create adjacency matrix (binary, undirected)
        self.adj_matrix = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        for src, dst in edges:
            self.adj_matrix[src, dst] = 1.0
            self.adj_matrix[dst, src] = 1.0  # Undirected graph
        
        # Add self-loops (each node is connected to itself)
        # This allows the node's own prediction to influence itself in KE layers
        for i in range(self.num_nodes):
            self.adj_matrix[i, i] = 1.0
        
        print(f"Graph loaded: {len(edges)} edges (undirected with self-loops)")
    
    def get_split_indices(self, train_size: float, 
                          split_type: str = "inductive") -> Dict[str, np.ndarray]:
        """
        Get train/val/test split indices.
        
        Args:
            train_size: Fraction of nodes for training (0.1, 0.25, 0.5, 0.75, 0.9)
            split_type: "inductive" or "transductive"
        
        Returns:
            Dictionary with 'train', 'val', 'test' indices
        """
        # Try to load preprocessed splits
        train_pct = int(train_size * 100)
        
        try:
            if split_type == "inductive":
                train_idx = np.load(os.path.join(
                    self.data_path, "index_x_inductive_training.npy")).flatten()
                val_idx = np.load(os.path.join(
                    self.data_path, "index_x_inductive_validation.npy")).flatten()
                test_idx = np.load(os.path.join(
                    self.data_path, "index_x_inductive_test.npy")).flatten()
            else:
                train_idx = np.load(os.path.join(
                    self.data_path, "index_x_transductive.npy")).flatten()
                val_idx = np.array([])  # Transductive uses all data
                test_idx = np.array([])
            
            # Subsample training set based on train_size
            if train_size < 1.0:
                np.random.shuffle(train_idx)
                n_train = int(len(train_idx) * train_size / 0.9)  # Adjust for base split
                train_idx = train_idx[:n_train]
            
            return {
                'train': train_idx,
                'val': val_idx,
                'test': test_idx
            }
        
        except FileNotFoundError:
            # Create splits manually if preprocessed files don't exist
            return self._create_random_split(train_size)
    
    def _create_random_split(self, train_size: float) -> Dict[str, np.ndarray]:
        """Create random train/val/test split."""
        indices = np.arange(self.num_nodes)
        np.random.shuffle(indices)
        
        n_train = int(self.num_nodes * train_size)
        n_val = int(self.num_nodes * 0.05)  # 5% for validation
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        return {
            'train': train_idx,
            'val': val_idx,
            'test': test_idx
        }
    
    def to_torch(self, device: str = 'cpu') -> Tuple[torch.Tensor, ...]:
        """
        Convert dataset to PyTorch tensors.
        
        Returns:
            features, labels, adj_matrix as torch tensors
        """
        features = torch.FloatTensor(self.features).to(device)
        labels = torch.LongTensor(self.labels).to(device)
        adj_matrix = torch.FloatTensor(self.adj_matrix).to(device)
        
        return features, labels, adj_matrix
    
    def get_class_distribution(self, indices: np.ndarray) -> Dict[str, int]:
        """Get class distribution for given indices."""
        labels_subset = self.labels[indices]
        distribution = {}
        for i, class_name in enumerate(self.class_names):
            distribution[class_name] = np.sum(labels_subset == i)
        return distribution
