"""
Explain why CV recall (72.5%) differs from HTML map recall (54.6%)
"""
import pandas as pd
import geopandas as gpd

# CV results with optimized thresholds
cv = pd.read_csv('outputs/criticality/cv_criticality_results.csv')
print('=' * 80)
print('CV RESULTS (optimized thresholds per fold):')
print('=' * 80)
print(cv[['fold', 'threshold', 'Recall']].to_string(index=False))
print(f'\nMean CV Recall: {cv["Recall"].mean():.1%} (with optimized thresholds per fold)')

# Final model predictions
segs = gpd.read_file('outputs/criticality/critical_segments.gpkg')
# Filter to TPO-matched segments (those with ground truth labels)
labeled = segs[segs['critical'].notna()].copy()

# Count at threshold 0.5
tp_05 = ((labeled['critical'] == 1) & (labeled['pred_critical'] == 1)).sum()
fn_05 = ((labeled['critical'] == 1) & (labeled['pred_critical'] == 0)).sum()
recall_05 = tp_05 / (tp_05 + fn_05)

print('\n' + '=' * 80)
print('FINAL MODEL (fixed threshold = 0.5):')
print('=' * 80)
print(f'TP: {tp_05}  |  FN: {fn_05}  |  Total Critical: {tp_05 + fn_05}')
print(f'Recall at threshold 0.5: {recall_05:.1%} ← THIS IS WHAT HTML MAP SHOWS')

# What if we use CV average threshold?
avg_threshold = cv['threshold'].mean()
pred_critical_avg = (labeled['pred_prob_critical'] >= avg_threshold).astype(int)
tp_avg = ((labeled['critical'] == 1) & (pred_critical_avg == 1)).sum()
fn_avg = ((labeled['critical'] == 1) & (pred_critical_avg == 0)).sum()
recall_avg = tp_avg / (tp_avg + fn_avg)

print('\n' + '=' * 80)
print(f'FINAL MODEL (threshold = {avg_threshold:.2f}, using CV average):')
print('=' * 80)
print(f'TP: {tp_avg}  |  FN: {fn_avg}  |  Total Critical: {tp_avg + fn_avg}')
print(f'Recall at threshold {avg_threshold:.2f}: {recall_avg:.1%}')

print('\n' + '=' * 80)
print('WHY THE DIFFERENCE:')
print('=' * 80)
print('• CV uses OPTIMIZED thresholds (0.36-0.47) per fold → 72.5% recall')
print('• HTML map uses FIXED threshold (0.50) → 54.6% recall')
print('• Lower threshold = more predictions labeled "critical" = higher recall')
print('')
print('WHICH ONE TO REPORT:')
print('• Report CV recall (72.5%) - shows model quality on unseen data')
print('• Report HTML recall (54.6%) - shows conservative deployment threshold')
print('• Both are valid, just answer different questions!')
print('=' * 80)
