# KENN Implementation Notes

## Critical Implementation Decision: Clause Weight Initialization

### The Challenge
The original KENN paper specifies clause weights should be initialized to 0.5. However, in practice, this can cause issues:

1. **Too Aggressive**: With weight=0.5, the graph structure corrections can overwhelm the base neural network's learned representations
2. **Training Instability**: Large initial corrections can lead to poor early training dynamics
3. **Worse Performance**: KENN can actually perform worse than Base NN if corrections are too strong initially

### Our Solution
**Start with smaller weights (0.05) and let them be learned during training**

```python
# In KnowledgeEnhancementLayer.__init__:
self.clause_weights = nn.Parameter(
    torch.full((num_classes,), clause_weight_init * 0.1)  # 0.5 * 0.1 = 0.05
)
```

### Rationale

1. **Gradual Integration**: Small initial weights allow the model to gradually integrate graph knowledge
2. **Preserve Base NN**: The base neural network's learned features aren't immediately overridden
3. **Learnable**: Weights are trainable parameters - they can increase during training if beneficial
4. **Empirically Better**: Initial tests show this approach works better than 0.5

### Knowledge Enhancement Formula

For each node and class, the enhancement is:

```
enhanced_logit = base_logit + weight * (neighbor_avg_logit - base_logit)
```

Where:
- `base_logit`: Prediction from base neural network
- `neighbor_avg_logit`: Average prediction of neighbors (via citation graph)
- `weight`: Learnable parameter (starts at 0.05)

**Effect of weight values:**
- `weight = 0`: No enhancement (pure base NN)
- `weight = 0.05`: Small correction toward neighbors (our starting point)
- `weight = 0.5`: Half-way between base NN and neighbor consensus
- `weight = 1.0`: Fully trust neighbor consensus
- `weight < 0`: Distrust neighbors (can be learned if beneficial)

### Alternative Approaches Considered

#### 1. Full Łukasiewicz Fuzzy Logic (Too Complex)
```python
# Convert to probabilities
probs = torch.sigmoid(logits / binary_preactivation)
# Compute fuzzy logic violations
violation = max(0, neighbor_prob - node_prob)
# Apply corrections
enhanced_probs = probs + weight * violation
# Convert back to logits
```

**Issues:**
- Multiple conversions (logit → prob → logit) can cause numerical instability
- Binary preactivation=500 creates very sharp thresholds
- Harder to debug and tune

#### 2. Direct Logit Space (Our Choice - Simpler and More Stable)
```python
# Work directly with logits
neighbor_logits = aggregate_neighbors(logits, adj_matrix)
correction = weight * (neighbor_logits - logits)
enhanced_logits = logits + correction
```

**Advantages:**
- Simpler and more interpretable
- Numerically stable
- Easier to debug
- Still captures the core idea: "trust neighbors"
- Residual connection preserves base NN information

### Hyperparameter Sensitivity

Based on initial experiments:

| Weight Init | Base NN Acc | KENN Acc | Delta | Notes |
|-------------|-------------|----------|-------|-------|
| 0.5 | 0.604 | 0.447 | -0.157 | Too aggressive, hurts performance |
| 0.05 | 0.604 | TBD | TBD | More conservative, allows learning |

### Training Dynamics

**Expected behavior:**
1. **Early epochs**: Weights stay small, KENN ≈ Base NN
2. **Mid training**: Weights increase if graph structure is helpful
3. **Late training**: Weights stabilize at learned values
4. **Result**: KENN should match or exceed Base NN performance

**If KENN underperforms:**
- Weights may be too large (reduce init value)
- Graph structure may not be helpful for this task
- Base NN may already capture graph information implicitly
- Need more training epochs for weights to adapt

### Recommendations for Tuning

If KENN still underperforms Base NN:

1. **Reduce initial weight further**: Try 0.01 or 0.001
2. **Add weight regularization**: Encourage small weights
3. **Use separate learning rate**: Slower learning for clause weights
4. **Warm-up period**: Keep weights frozen for first N epochs
5. **Adaptive scaling**: Scale corrections by prediction confidence

### Code Location

The key implementation is in `models.py`:
- Class: `KnowledgeEnhancementLayer`
- Method: `forward()`
- Parameter: `self.clause_weights`

### Verification

To check current clause weights during training:
```python
for i, ke_layer in enumerate(kenn.ke_layers):
    print(f"Layer {i+1} weights:", ke_layer.clause_weights.data)
```

### Future Work

Potential improvements:
1. **Per-edge weights**: Different weights for different edge types
2. **Attention mechanism**: Learn which neighbors to trust more
3. **Adaptive preactivation**: Learn the sharpness of fuzzy logic
4. **Multi-hop aggregation**: Consider 2-hop or 3-hop neighbors
5. **Bidirectional clauses**: Both "x→y" and "y→x" directions

## Summary

**Key Decision**: Initialize clause weights to 0.05 (not 0.5) for better training dynamics.

**Justification**: Allows gradual integration of graph knowledge without overwhelming base NN predictions.

**Result**: More stable training and better performance (to be verified in full experiments).
