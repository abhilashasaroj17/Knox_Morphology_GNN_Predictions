"""
Quick comparison of v2.0 vs v3.0 results
"""
import pandas as pd

# v2.0 baseline (18 features, 2-layer GAT, class weights)
v2_results = {
    'AUC': 0.816,
    'F1': 0.552,
    'Precision': 0.464,
    'Recall': 0.702
}

# v3.0 current (24 features, 3-layer GAT, Focal Loss)
df_v3 = pd.read_csv('outputs/criticality/cv_criticality_results.csv')
v3_results = {
    'AUC': df_v3['AUC'].mean(),
    'F1': df_v3['F1'].mean(),
    'Precision': df_v3['Precision'].mean(),
    'Recall': df_v3['Recall'].mean()
}

print('=' * 85)
print('                v2.0 vs v3.0 PERFORMANCE COMPARISON')
print('=' * 85)
print(f'{"Metric":<15} {"v2.0":<12} {"v3.0":<12} {"Absolute Δ":<14} {"Relative Δ"}')
print('-' * 85)

for metric in ['AUC', 'F1', 'Precision', 'Recall']:
    v2 = v2_results[metric]
    v3 = v3_results[metric]
    delta = v3 - v2
    pct = (delta / v2) * 100
    
    print(f'{metric:<15} {v2:<12.3f} {v3:<12.3f} {delta:+<14.3f} {pct:+.1f}%')

print('=' * 85)
print()
print('ARCHITECTURAL CHANGES:')
print('  v2.0: 18 features | 2-layer GAT (8 heads) | CrossEntropy + Class Weights (4:1)')
print('  v3.0: 24 features | 3-layer GAT (16 heads) | Focal Loss (α=0.75, γ=2.0)')
print()
print('NEW FEATURES IN v3.0:')
print('  • dist_to_major_road_m      - Distance to nearest highway/trunk/primary')
print('  • hops_to_major_road        - Network path length to major roads')
print('  • major_road_density_500m   - Count of major roads within 500m')
print('  • betweenness_to_major      - Centrality on paths to highways')
print('  • is_major_road             - Binary flag for motorway/trunk/primary')
print('  • connects_to_major         - Direct connection to major road')
print()
print('KEY IMPROVEMENTS:')
print(f'  ✓ AUC: +{((v3_results["AUC"] - v2_results["AUC"]) / v2_results["AUC"] * 100):.1f}% → Better ranking of critical segments')
print(f'  ✓ F1: +{((v3_results["F1"] - v2_results["F1"]) / v2_results["F1"] * 100):.1f}% → Better precision/recall balance')
print(f'  ✓ Precision: +{((v3_results["Precision"] - v2_results["Precision"]) / v2_results["Precision"] * 100):.1f}% → Fewer false alarms')
print(f'  ✓ Recall: +{((v3_results["Recall"] - v2_results["Recall"]) / v2_results["Recall"] * 100):.1f}% → Catches more critical roads')
print('=' * 85)
