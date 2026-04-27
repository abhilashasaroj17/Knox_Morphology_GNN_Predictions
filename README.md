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

1. **`step1_download_overture.py`** — Download road segments and buildings from Overture Maps
2. **`step2_build_graphs.py`** — Build spatial graph with NetworkX, compute centrality metrics
3. **`step3_morphology_features.py`** — Extract urban morphology features from building data
   - Building density, coverage, sizes in 500m buffers
   - Street network density and connectivity
   - Land use diversity metrics
4. **`step4_match_tpo_volumes.py`** — Match TPO traffic volumes to Overture segments via spatial join
5. **`step5_prepare_training_data.py`** — Convert to PyTorch Geometric format with all node features
6. **`step6_road_criticality.py`** — Train GAT model with 5-fold stratified CV
   - Uses 15 node features: road attributes + graph metrics + morphology
   - Binary classification: V/C > 0.75 threshold
7. **`step7_interactive_maps.py`** — Generate initial prediction maps
8. **`step8_summary_report.py`** — Generate performance report with figures
9. **`step9_comparison_map.py`** — TPO ground truth vs GNN extension map (6 layers)
10. **`step9b_model_accuracy_map.py`** — Confusion matrix visualization (TP/FP/FN/TN)

---

## Model Architecture

**Graph Attention Network (GAT)**
- 2 layers, 8 attention heads per layer
- Node features: 15 total features per road segment
- Edge features: spatial distance, connectivity
- Binary classification: critical (V/C > 0.75) vs non-critical

**Node Features (15 total):**
1. **Road Class** (categorical → one-hot encoded)
   - `motorway`, `trunk`, `primary`, `secondary`, `tertiary`, `residential`, `service`
2. **Length (m)** — Road segment length in meters
3. **Lanes** — Number of travel lanes
4. **Max Speed (km/h)** — Posted speed limit
5. **Width (m)** — Road width (if available)
6. **Is One-Way** — Binary indicator for one-way streets
7. **Degree Centrality** — Number of connected road segments
8. **Betweenness Centrality** — Measure of how often the segment lies on shortest paths
9. **Closeness Centrality** — Average distance to all other segments
10. **Building Density** — Buildings per km² in 500m buffer
11. **Building Footprint Coverage** — % of area covered by buildings
12. **Street Network Density** — Total km of roads per km² in buffer
13. **Street Connectivity** — Average node degree in buffer
14. **Mean Building Size** — Average building footprint area (m²)
15. **Land Use Mix** — Diversity of building types (residential, commercial, industrial)

**Graph Structure:**
- Nodes: 65,524 road segments (Overture Maps)
- Edges: Spatial connectivity where road segments touch
- Spatial weighting: Edge weights based on Euclidean distance

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

- **Road Network:** [Overture Maps](https://overturemaps.org/) (transportation theme, 65,524 segments)
  - Attributes: road class, lanes, speed limit, geometry
  - Knox County, TN coverage (36.1° N, -84.0° W)
  
- **Traffic Volumes:** Knox TPO 2019 Assignment Model
  - Vehicle volumes on 8,121 major road segments
  - V/C ratios from capacity analysis
  - Used as ground truth labels for GNN training
  
- **Building Data:** [Overture Maps](https://overturemaps.org/) (buildings theme)
  - Building footprints, heights, land use types
  - Used to compute urban morphology features
  - Aggregated in 500m buffers around each road segment
  
- **Urban Morphology Features:** Computed from buildings + roads
  - Building density, coverage, sizes
  - Street network density and connectivity
  - Land use diversity metrics

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
