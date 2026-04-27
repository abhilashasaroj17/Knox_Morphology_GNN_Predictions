# Knox County City2Graph — Road Criticality Prediction

Graph Neural Network (GNN) model for predicting critical road segments in Knox County, TN using Overture Maps data and TPO traffic volumes.

## 🗺️ Interactive Maps

**[View Live Map: GNN Completion →](https://abhilashasaroj17.github.io/Knox_Morphology_GNN_Predictions/)**

**[Model Accuracy Analysis →](https://abhilashasaroj17.github.io/Knox_Morphology_GNN_Predictions/model_accuracy.html)**

## Overview

This project uses a Graph Attention Network (GAT) to identify critical road segments based on:
- Road network topology from Overture Maps (65,524 segments)
- TPO traffic volume and V/C ratios (8,121 labeled segments)
- Graph-based learning to extend predictions to unlabeled roads

### Key Results
- **Test Accuracy:** 82.8% (5-fold CV)
- **Recall:** 42.8% (696 of 1,625 critical roads detected)
- **GNN Extension:** 4,556 additional critical roads identified with no TPO label

---

## 📊 Maps

### 1. GNN Completion Map
Shows all 65,524 Overture segments in 6 layers:
- **Group A:** TPO-labeled segments (ground truth from traffic model)
  - A1: Critical (1,625 segments)
  - A2: Non-critical (6,496 segments)
- **Group B:** Unlabeled segments (GNN extension to 57,403 roads)
  - B1: GNN Critical (4,556)
  - B2-B4: GNN Non-critical by probability bands

**Features:**
- Toggle service roads for each layer (LayerControl top-right)
- Hover tooltips with road class, volume, GNN probability
- 3 base map styles (Light, Dark, OSM)

### 2. Model Accuracy Map (Confusion Matrix)
Visualizes GNN performance on the 8,121 TPO-labeled test set:
- 🟢 **TP (696):** Correctly identified critical
- 🟡 **FN (929):** Missed critical roads
- 🟠 **FP (470):** Over-predicted as critical
- ⚪ **TN (6,026):** Correctly identified non-critical

---

## Pipeline Steps

1. **`step1_download_overture.py`** — Download road segments from Overture Maps
2. **`step2_build_graphs.py`** — Build spatial graph with NetworkX
3. **`step3_morphology_features.py`** — Extract urban morphology features
4. **`step4_match_tpo_volumes.py`** — Match TPO traffic volumes to Overture segments
5. **`step5_prepare_training_data.py`** — Convert to PyTorch Geometric format
6. **`step6_road_criticality.py`** — Train GAT model with 5-fold CV
7. **`step7_interactive_maps.py`** — Generate initial prediction maps
8. **`step8_summary_report.py`** — Generate performance report
9. **`step9_comparison_map.py`** — TPO ground truth vs GNN extension map
10. **`step9b_model_accuracy_map.py`** — Confusion matrix visualization

---

## Model Architecture

**Graph Attention Network (GAT)**
- 2 layers, 8 attention heads per layer
- Node features: road class, length, lanes, speed, degree, betweenness
- Edge features: spatial distance, connectivity
- Binary classification: critical (V/C > 0.75) vs non-critical

**Training Details:**
- 5-fold stratified cross-validation
- Class weights to handle imbalance (1:4 critical:non-critical ratio)
- Adam optimizer, BCEWithLogitsLoss
- Early stopping with patience=20

---

## Requirements

```bash
python >= 3.11
torch >= 2.0
torch-geometric
geopandas
networkx
folium
overturemaps
```

Install:
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Usage

Run the full pipeline:
```bash
python step1_download_overture.py
python step2_build_graphs.py
python step3_morphology_features.py
python step4_match_tpo_volumes.py
python step5_prepare_training_data.py
python step6_road_criticality.py
python step7_interactive_maps.py
python step8_summary_report.py
python step9_comparison_map.py
python step9b_model_accuracy_map.py
```

Output HTML maps will be in `outputs/maps/`.

---

## Data Sources

- **Road Network:** [Overture Maps](https://overturemaps.org/) (transportation theme)
- **Traffic Volumes:** Knox TPO 2019 Assignment Model
- **Study Area:** Knox County, TN (36.1° N, -84.0° W)

---

## Citation

If you use this work, please cite:

```bibtex
@misc{knoxcity2graph2026,
  title={Knox County City2Graph: GNN-based Road Criticality Prediction},
  author={Abhilasha Saroj},
  year={2026},
  url={https://github.com/abhilashasaroj17/Knox_Morphology_GNN_Predictions}
}
```

---

## License

MIT License — See LICENSE file for details.
