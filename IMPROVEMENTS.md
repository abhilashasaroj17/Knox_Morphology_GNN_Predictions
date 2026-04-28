# Model Performance Improvements

## Current Performance (Baseline)
- **Test Accuracy:** 82.8% (5-fold CV)
- **Recall:** 42.8% (696/1,625 critical roads detected) ⚠️
- **Precision:** 59.7%
- **F1 Score:** 0.499
- **Problem:** Model misses 57% of critical roads (high false negative rate)

---

## Recommended Improvements (Prioritized)

### 🔴 **1. Add Class Weights (HIGHEST PRIORITY)**
**Problem:** Data is imbalanced (20% critical, 80% non-critical). Current model uses unweighted cross-entropy, leading to bias toward majority class.

**Solution:**
```python
# Calculate class weights inversely proportional to frequency
n_critical = (data.y == 1).sum().item()
n_non_critical = (data.y == 0).sum().item()
weight_critical = n_non_critical / n_critical  # ≈ 4.0
class_weights = torch.tensor([1.0, weight_critical])

# Apply in loss function
loss = F.cross_entropy(out[train_mask], data.y[train_mask], weight=class_weights)
```

**Expected Impact:** +10-15% recall, better balance between precision/recall

---

### 🔴 **2. Use Focal Loss for Hard Examples**
**Problem:** Standard cross-entropy treats all misclassifications equally. Critical roads are harder to detect but more important.

**Solution:**
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # Weight for positive class (critical roads)
        self.gamma = gamma  # Focus parameter (higher = more focus on hard examples)
    
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# Replace line 553 in step6_road_criticality.py
criterion = FocalLoss(alpha=0.75, gamma=2.0)
loss = criterion(out[train_mask], data.y[train_mask])
```

**Expected Impact:** +5-10% recall, better detection of hard-to-classify critical roads

---

### 🟡 **3. Increase Model Capacity**
**Problem:** Current model is small (2 layers, 64 hidden, 4 heads). May lack capacity to learn complex urban patterns.

**Solution:**
```python
# Current:
HIDDEN = 64
HEADS = 4
# 2 GAT layers

# Improved:
HIDDEN = 128      # Double hidden dimension
HEADS = 8         # Double attention heads
# Add 3rd GAT layer for deeper context

class ImprovedRoadGAT(nn.Module):
    def __init__(self, in_ch, hidden=128, heads=8, dropout=0.3):
        super().__init__()
        self.conv1 = GATConv(in_ch, hidden, heads=heads, dropout=dropout, concat=True)
        self.conv2 = GATConv(hidden * heads, hidden, heads=heads, dropout=dropout, concat=True)
        self.conv3 = GATConv(hidden * heads, hidden, heads=1, dropout=dropout, concat=False)
        self.head = nn.Linear(hidden, 2)
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return self.head(x)
```

**Expected Impact:** +3-7% F1 score, better learning of complex patterns

---

### 🟡 **4. Reduce Dropout Rate**
**Problem:** Current dropout=0.4 may be too aggressive, preventing model from learning effectively.

**Solution:**
```python
DROPOUT = 0.3  # Reduce from 0.4 to 0.3
```

**Expected Impact:** +2-5% overall accuracy

---

### 🟡 **5. Add More Spatial Graph Features**
**Problem:** Current features focus on local attributes. Missing neighborhood aggregation patterns.

**Solution:** Add multi-hop neighborhood features:
```python
# In Section 3 (feature engineering):

# 1. K-hop aggregated features (capture neighborhood context)
for k in [2, 3]:
    adj_k = np.linalg.matrix_power(adj_matrix.toarray(), k)
    for col in ['length', 'max_speed', 'lanes']:
        segs[f'{col}_k{k}_mean'] = (adj_k @ segs[col].values) / (adj_k.sum(axis=1) + 1e-6)

# 2. Shortest path features to high-traffic nodes
high_traffic_nodes = segs[segs['tpo_volume'] > segs['tpo_volume'].quantile(0.9)].index
segs['dist_to_high_traffic'] = compute_min_distance_to_nodes(G, high_traffic_nodes)

# 3. Neighborhood diversity (connected road classes)
def neighbor_class_diversity(G, node):
    neighbors = list(G.neighbors(node))
    classes = [G.nodes[n].get('class', 'unknown') for n in neighbors]
    return len(set(classes)) / (len(classes) + 1)

segs['neighbor_diversity'] = segs.index.map(lambda n: neighbor_class_diversity(G, n))
```

**Expected Impact:** +3-6% recall, better spatial context awareness

---

### 🟢 **6. Ensemble Multiple Models**
**Problem:** Single model may overfit to specific patterns. Ensemble reduces variance.

**Solution:**
```python
# Train 3 different architectures:
# 1. GAT (current)
# 2. GraphSAGE (different aggregation)
# 3. GCN (simpler, may generalize better)

# Average predictions
final_probs = (gat_probs + sage_probs + gcn_probs) / 3
```

**Expected Impact:** +2-4% F1 score, more robust predictions

---

### 🟢 **7. Adjust Decision Threshold**
**Problem:** Default threshold=0.5 may not be optimal for this imbalanced problem.

**Solution:**
```python
# After training, optimize threshold on validation set
from sklearn.metrics import f1_score

thresholds = np.linspace(0.3, 0.7, 41)
best_threshold = 0.5
best_f1 = 0

for thresh in thresholds:
    preds = (probs > thresh).astype(int)
    f1 = f1_score(truth, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = thresh

print(f"Optimal threshold: {best_threshold:.3f} (F1={best_f1:.3f})")
```

**Expected Impact:** +5-8% F1 score with no retraining

---

### 🟢 **8. Data Augmentation**
**Problem:** Only 8,121 labeled segments (vs 65,524 total). Model has limited training data.

**Solution:**
```python
# Pseudo-labeling: Use high-confidence predictions on unlabeled data
# After initial training:
unlabeled_probs = model.predict(unlabeled_segments)
high_conf_critical = unlabeled_segments[unlabeled_probs > 0.9]
high_conf_noncritical = unlabeled_segments[unlabeled_probs < 0.1]

# Add to training set and retrain (semi-supervised learning)
```

**Expected Impact:** +3-5% accuracy, better generalization to unlabeled roads

---

### 🟢 **9. Add Temporal Context (if available)**
**Problem:** Current model uses static features only. Road criticality may vary by time.

**Solution:** If TPO data includes AM/PM peak volumes:
```python
# Add peak-hour features
segs['am_peak_volume'] = match_tpo_volume(segs, tpo_data, period='AM')
segs['pm_peak_volume'] = match_tpo_volume(segs, tpo_data, period='PM')
segs['peak_variability'] = abs(segs['am_peak_volume'] - segs['pm_peak_volume'])
```

**Expected Impact:** +2-4% accuracy if temporal data available

---

### 🟢 **10. Hyperparameter Tuning**
**Problem:** Current hyperparameters chosen manually, may not be optimal.

**Solution:** Use grid search or Optuna:
```python
import optuna

def objective(trial):
    hidden = trial.suggest_int('hidden', 64, 256, step=32)
    heads = trial.suggest_int('heads', 4, 16, step=4)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float('dropout', 0.2, 0.5)
    
    model = RoadGAT(in_ch, hidden, heads, dropout)
    # ... train and evaluate
    return f1_score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

**Expected Impact:** +2-5% overall performance

---

## Quick Wins (Implement First)

1. **Add class weights** (30 min, +10-15% recall)
2. **Adjust decision threshold** (10 min, +5-8% F1)
3. **Reduce dropout to 0.3** (5 min, +2-5% accuracy)
4. **Increase to 128 hidden / 8 heads** (10 min, +3-7% F1)

**Total time:** ~1 hour  
**Expected improvement:** +15-25% recall, +10-15% F1 score

---

## Medium-Term Improvements

5. **Implement focal loss** (1 hour)
6. **Add k-hop neighborhood features** (2 hours)
7. **Threshold optimization** (already quick win)

---

## Advanced Improvements

8. **Ensemble models** (4 hours)
9. **Pseudo-labeling** (3 hours)
10. **Hyperparameter tuning** (8 hours with compute)

---

## Expected Final Performance

**Current:** 82.8% acc, 42.8% recall, 59.7% precision, 0.499 F1  
**After Quick Wins:** ~85-88% acc, ~60-65% recall, ~65-70% precision, ~0.62-0.67 F1  
**After All Improvements:** ~87-90% acc, ~70-75% recall, ~68-73% precision, ~0.69-0.74 F1

---

## Implementation Priority

Start with **Quick Wins** (class weights + threshold tuning) since they provide the best ROI. The current low recall (42.8%) is likely due to class imbalance, which class weights will directly address.
