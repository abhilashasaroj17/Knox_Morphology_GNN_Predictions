"""
Explain threshold and the difference between CV vs HTML map evaluation
"""
import pandas as pd
import geopandas as gpd

print('=' * 90)
print('                    WHAT IS A THRESHOLD?')
print('=' * 90)
print("""
The GNN model outputs a PROBABILITY (0.0 to 1.0) for each road being critical.
The THRESHOLD is the cutoff point to make the final yes/no decision:

  • If probability >= threshold → Predict "CRITICAL"
  • If probability < threshold  → Predict "NON-CRITICAL"

EXAMPLE:
  Road A: probability = 0.35
  Road B: probability = 0.55
  Road C: probability = 0.75

  With threshold = 0.5:
    Road A (0.35 < 0.5) → Predict NON-CRITICAL
    Road B (0.55 >= 0.5) → Predict CRITICAL ✓
    Road C (0.75 >= 0.5) → Predict CRITICAL ✓

  With threshold = 0.4:
    Road A (0.35 < 0.4) → Predict NON-CRITICAL
    Road B (0.55 >= 0.4) → Predict CRITICAL ✓
    Road C (0.75 >= 0.4) → Predict CRITICAL ✓

  With threshold = 0.7:
    Road A (0.35 < 0.7) → Predict NON-CRITICAL
    Road B (0.55 < 0.7) → Predict NON-CRITICAL
    Road C (0.75 >= 0.7) → Predict CRITICAL ✓

LOWER threshold = More roads predicted critical = Higher recall, Lower precision
HIGHER threshold = Fewer roads predicted critical = Lower recall, Higher precision
""")

print('=' * 90)
print('           CV RESULTS (What I showed you: 72.5% recall)')
print('=' * 90)

cv = pd.read_csv('outputs/criticality/cv_criticality_results.csv')
print(cv[['fold', 'n_test', 'threshold', 'Recall', 'Precision', 'F1']].to_string(index=False))

print("""
KEY POINTS:
  • Training data: 80% of labeled segments (changes per fold)
  • Testing data: 20% of labeled segments (UNSEEN, changes per fold)
  • Threshold: OPTIMIZED per fold to maximize F1 score (ranges 0.36-0.47)
  • This measures: "How well does the model generalize to new geographic areas?"
  
Example Fold 0:
  - Train on 6,467 segments
  - Test on 1,654 UNSEEN segments
  - Try many thresholds (0.1, 0.2, 0.3, ..., 0.9)
  - Pick best threshold (0.36) that maximizes F1 score
  - Report recall at that optimized threshold: 80.7%
""")

print('=' * 90)
print('           HTML MAP RESULTS (What you see: 54.6% recall)')
print('=' * 90)

segs = gpd.read_file('outputs/criticality/critical_segments.gpkg')
labeled = segs[segs['critical'].notna()].copy()

tp = ((labeled['critical'] == 1) & (labeled['pred_critical'] == 1)).sum()
fn = ((labeled['critical'] == 1) & (labeled['pred_critical'] == 0)).sum()
fp = ((labeled['critical'] == 0) & (labeled['pred_critical'] == 1)).sum()
tn = ((labeled['critical'] == 0) & (labeled['pred_critical'] == 0)).sum()

recall = tp / (tp + fn)
precision = tp / (tp + fp)
f1 = 2 * precision * recall / (precision + recall)

print(f"""
Final model evaluation:
  TP: {tp:,}  |  FN: {fn:,}  |  FP: {fp:,}  |  TN: {tn:,}
  Recall: {recall:.1%}  |  Precision: {precision:.1%}  |  F1: {f1:.3f}

KEY POINTS:
  • Training data: ALL 8,121 labeled segments (100%)
  • Testing data: SAME 8,121 labeled segments (NOT truly unseen!)
  • Threshold: FIXED at 0.50 (not optimized)
  • This measures: "How well does final model perform with conservative threshold?"
  
WARNING: This is NOT a true test because the model was trained on this same data!
It's like asking students questions from the practice exam they studied from.
""")

print('=' * 90)
print('                          WHICH ONE TO TRUST?')
print('=' * 90)
print("""
FOR RESEARCH PAPERS / MODEL QUALITY:
  ✓ Report CV results (72.5% recall)
  ✓ This is the TRUE performance on unseen geographic areas
  ✓ Uses optimized thresholds per fold (0.36-0.47)
  
FOR DEPLOYMENT / OPERATIONS:
  ✓ Use HTML results (54.6% recall) 
  ✓ Conservative threshold (0.5) reduces false alarms
  ✓ Higher precision (70.5%) means fewer "false critical" predictions
  ✓ Operators see fewer unnecessary alerts

BOTH ARE VALID - they answer different questions:
  • CV: "Can the model find critical roads in NEW areas?"  → 72.5% recall ✓
  • HTML: "Using threshold=0.5, how many critical roads caught?" → 54.6% recall
""")

print('=' * 90)
print('              IMPACT OF CHANGING THRESHOLD (on same final model)')
print('=' * 90)

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
results = []

for thresh in thresholds:
    pred = (labeled['pred_prob_critical'] >= thresh).astype(int)
    tp = ((labeled['critical'] == 1) & (pred == 1)).sum()
    fn = ((labeled['critical'] == 1) & (pred == 0)).sum()
    fp = ((labeled['critical'] == 0) & (pred == 1)).sum()
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    results.append({
        'Threshold': thresh,
        'Recall': f'{recall:.1%}',
        'Precision': f'{precision:.1%}',
        'TP': tp,
        'FP': fp
    })

df_thresh = pd.DataFrame(results)
print(df_thresh.to_string(index=False))
print("""
See the trade-off:
  • Lower threshold (0.3) → Catch more critical roads (higher recall) but more false alarms (lower precision)
  • Higher threshold (0.7) → Fewer false alarms (higher precision) but miss more critical roads (lower recall)
  • HTML uses 0.5 as a balanced, conservative choice
""")
print('=' * 90)
