# Knox County City2Graph — Road Criticality Prediction

Graph Neural Network (GNN) model for predicting critical road segments in Knox County, TN using Overture Maps data and TPO traffic volumes.

## 🗺️ Interactive Maps

**[View Live Map: GNN Completion →](https://abhilashasaroj17.github.io/Knox_Morphology_GNN_Predictions/)**

**[Model Accuracy Analysis →](https://abhilashasaroj17.github.io/Knox_Morphology_GNN_Predictions/model_accuracy.html)**

## Overview

This project uses a Graph Attention Network (GAT) to identify critical road segments based on:
- Road network topology from Overture Maps (65,524 segments)
- TPO traffic volumes and network centrality (8,121 labeled segments)
- Graph-based learning to extend predictions to unlabeled roads

### Key Results
- **Test AUC:** 84.3% (5-fold CV)
- **Test Accuracy:** 82.8%
- **Recall:** 72.5% (1,177 of 1,625 critical roads detected)
- **Precision:** 52.3%
- **F1 Score:** 0.598
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
- 🟢 **TP (1,211):** Correctly identified critical
- 🟡 **FN (414):** Missed critical roads
- 🟠 **FP (985):** Over-predicted as critical
- ⚪ **TN (5,511):** Correctly identified non-critical

---

## Pipeline Steps

1. **`step1_base_datasets.py`** — Load and clean Knox TPO data (TAZ zones, OD matrix, assignment)
2. **`step2_build_graphs.py`** — Download Overture Maps data, build spatial graphs, aggregate morphology to TAZ zones
3. **`step3_regression.py`** — Baseline regression models (OLS, Poisson) for trip generation
4. **`step4_hetero_graph.py`** — Build heterogeneous graph for zone-level GNN
5. **`step5_gnn_train.py`** — Train zone-level GNN for trip production/attraction
6. **`step6_road_criticality.py`** — Train road-level GAT for criticality prediction (main model)
   - Uses 24 node features: road attributes + graph metrics + infrastructure proximity + TAZ morphology
   - Binary classification: criticality score (0.5×volume + 0.5×betweenness), top 20% = critical
7. **`step7_interactive_maps.py`** — Generate initial prediction maps
8. **`step8_summary_report.py`** — Generate performance report with figures
9. **`step9_comparison_map.py`** — TPO ground truth vs GNN extension map (6 layers)
10. **`step9b_model_accuracy_map.py`** — Confusion matrix visualization (TP/FP/FN/TN)

---

## Model Architecture

**Graph Attention Network (GAT)**
- 3 layers, 16 attention heads per layer
- Input: 24 node features per road segment
- Edge features: spatial adjacency (shared endpoints)
- Output: Binary classification probability (critical vs non-critical)

**Input Features (24 per road segment):**

*Road Intrinsic (12 features):*
1. **Length (m)** — Road segment length in meters
2. **Road Class** (encoded) — motorway, trunk, primary, secondary, tertiary, residential, service, etc.
3. **Connector Count** — Number of endpoint connections (degree)
4. **Speed Limit (km/h)** — Posted speed limit
5. **Has Surface** — Surface type information available (binary)
6. **Graph Degree** — Number of adjacent road segments
7. **Betweenness Centrality** — Measure of how often segment lies on shortest paths
8. **Is Bridge** — Bridge flag (binary)
9. **Is Link/Ramp** — Highway link or ramp (binary)
10. **Is Tunnel** — Tunnel flag (binary)
11. **Is Private** — Private road access restriction (binary)
12. **Sinuosity** — Road curvature (actual length / straight-line distance)

*Infrastructure Proximity (6 features):*
13. **Distance to Major Road (m)** — Euclidean distance to nearest highway/arterial
14. **Hops to Major Road** — Network distance (# of segments) to nearest major road
15. **Major Road Density (500m)** — Count of major roads within 500m radius
16. **Betweenness to Major** — Betweenness on paths connecting to major roads
17. **Is Major Road** — Is this segment a highway/arterial? (binary)
18. **Connects to Major** — Directly adjacent to major road (binary)

*TAZ Morphology (6 features from Knox TPO zones):*
19. **Total Employment** — Jobs in TAZ zone containing this segment
20. **Households** — Number of households in TAZ zone
21. **Building Coverage (%)** — Percentage of TAZ covered by buildings
22. **Street Density (km/km²)** — Road network density in TAZ
23. **Building Density (n/km²)** — Buildings per square km in TAZ
24. **Average Footprint (m²)** — Mean building size in TAZ

**Training Labels (Binary Classification):**
- **Source:** Knox TPO 2026 Travel Demand Model
  - Assignment results with traffic volumes
  - Available for 8,121 of 65,524 total segments (12.4% coverage)
  - Covers major roads: motorways, trunks, primary, secondary routes
  
- **Label Definition:**
  - **Criticality Score** = 0.5 × normalized(volume) + 0.5 × normalized(betweenness)
    - Combines traffic demand (TPO volumes) with topological importance (graph centrality)
  - **Critical (1):** Top 20% by criticality score
    - Count: **1,625 segments** (20.0% of labeled set)
    - Roads with high traffic volumes AND/OR high network centrality
    - Critical for traffic flow and network connectivity
  - **Non-critical (0):** Bottom 80% by criticality score
    - Count: **6,496 segments** (80.0% of labeled set)
    - Roads with lower combined importance
    
- **Unlabeled Segments:**
  - **57,403 segments** (87.6% of network) have no TPO volume data
  - Includes: residential streets, service roads, minor collectors
  - GNN extends predictions to these unlabeled roads using:
    - Graph topology (connectivity patterns)
    - Road attributes (class, lanes, speed)
    - Urban morphology (building density, land use)
    - Spatial relationships to labeled roads

**Graph Structure:**
- Nodes: 65,524 road segments (Overture Maps)
- Edges: Spatial connectivity where road segments touch
- Spatial weighting: Edge weights based on Euclidean distance

**Training Details:**
- 5-fold spatial cross-validation (KMeans clustering) on 8,121 labeled segments
- 3-layer GAT with 128 hidden units, 16 attention heads
- Focal Loss (α=0.75, γ=2.0) to handle class imbalance
- Adam optimizer (lr=5e-4, weight_decay=1e-4)
- Early stopping with patience=40, trained for up to 400 epochs
- Threshold optimization per fold to maximize F1 score
- After training, model predicts criticality for all 65,524 segments

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
python step1_base_datasets.py
python step2_build_graphs.py
python step3_regression.py
python step4_hetero_graph.py
python step5_gnn_train.py
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
  
- **Traffic Volumes:** Knox TPO 2026 Assignment Model
  - Vehicle volumes on 8,121 major road segments
  - Combined with betweenness centrality to compute criticality scores
  - Used as ground truth labels for GNN training
  
- **TAZ Zones & Demographics:** Knox TPO Traffic Analysis Zones
  - 508 spatial TAZ polygons covering Knox County
  - Employment, household counts per zone
  - Used for land use context features
  
- **Building Data:** [Overture Maps](https://overturemaps.org/) (buildings theme)
  - Building footprints aggregated to TAZ zones
  - Used to compute urban morphology features
  - Building density, coverage, and average sizes per TAZ
  
- **Urban Morphology Features:** Computed at TAZ zone level
  - Building density, coverage, average footprint sizes
  - Street network density within each TAZ
  - Employment and household density

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
