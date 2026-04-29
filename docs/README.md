# Knox County Road Criticality - Interactive Maps & AI Chat

## 🎉 What's New: AI Chat Assistant!

Your GNN road criticality maps now include an **AI-powered chat assistant** that can answer questions about model performance, road characteristics, and statistics.

## 🗺️ Available Maps

### 1. Model Accuracy Map (Confusion Matrix)
**URL:** `model_accuracy.html`

Traditional visualization showing model performance:
- True Positives (1,211 roads)
- False Negatives (414 roads)
- False Positives (985 roads)
- True Negatives (5,511 roads)

**Metrics:** Precision 55.1%, Recall 74.5%, F1 0.634

---

### 2. Model Accuracy Map with AI Chat 🤖 ⭐ NEW!
**URL:** `model_accuracy_chat.html`

Same accuracy map PLUS an integrated AI chatbot!

**Features:**
- Click "🤖 AI Chat" button in top-right
- Ask natural language questions
- Get instant answers about model performance
- Powered by OpenAI GPT-4o-mini

**Example Questions:**
- "What's the model performance?"
- "How many false positives are there?"
- "What's the average length of critical roads?"
- "Show me statistics for motorways"
- "Compare TP and FP counts"

**Requirements:**
- OpenAI API key (get free at https://platform.openai.com/api-keys)
- ~$0.001 per question (less than a penny!)

---

### 3. GNN Completion Map
**URL:** `gnn_completion.html`

Shows GNN predictions across ALL 65,524 road segments:
- TPO ground truth (8,121 labeled)
- GNN predictions (57,403 unlabeled)
- Probability-based coloring
- Network-wide coverage

---

## 🚀 Quick Start

### Option 1: Browse All Maps
Open `maps_index.html` for a beautiful landing page with links to all maps.

### Option 2: Direct Access
- Regular accuracy map: `model_accuracy.html`
- **AI chat version: `model_accuracy_chat.html`** ⭐
- Completion map: `gnn_completion.html`

---

## 🤖 Using the AI Chat

1. **Open the chat map:** `model_accuracy_chat.html`
2. **Click** "🤖 AI Chat" button (top-right)
3. **Enter API key:** Get from https://platform.openai.com/api-keys
4. **Ask questions!**

### Example Questions

**Model Performance:**
```
"What's the model precision and recall?"
"How accurate is the model?"
"What's the F1 score?"
```

**Road Statistics:**
```
"How many roads are true positives?"
"What's the average length of false negatives?"
"How many motorway roads are misclassified?"
```

**Comparisons:**
```
"Compare TP and FP counts"
"What's the difference between recall and precision?"
"How many more FN than FP?"
```

---

## 📊 Model Details

**Architecture:** 3-layer Graph Attention Network (GAT)
- Hidden dimensions: 128
- Attention heads: 16 (layers 1-2), 1 (layer 3)
- Total parameters: ~2.5M

**Training:**
- Loss: Focal Loss (α=0.75, γ=2.0)
- Optimizer: Adam (lr=0.001)
- Cross-validation: 5-fold spatial K-Means
- Threshold: 0.41 (CV-optimized, average of 0.36-0.47)

**Features:** 24 total
- Road geometry (2): length_m, sinuosity
- Road class (3): class_enc, speed_kph, has_surface
- Network topology (3): connector_count, graph_degree, betweenness
- Structural flags (4): is_bridge, is_link, is_tunnel, is_private
- **Highway/Infrastructure (6):** dist_to_major_road_m, hops_to_major_road, major_road_density_500m, betweenness_to_major, is_major_road, connects_to_major
- TAZ land use (6): employment, households, building coverage, street density, building density, avg footprint

**Performance (v3.0):**
- AUC: 0.843 (+3.3% vs v2.0)
- F1: 0.598 (+8.3% vs v2.0)
- Precision: 52.3% (+12.6% vs v2.0)
- Recall: 72.5% (+3.2% vs v2.0)

---

## 🔒 Privacy & Security

**AI Chat:**
- Your API key is **never stored** - only used in your browser
- All queries run client-side (no backend server)
- Road data is public (already in HTML)
- OpenAI processes queries (see their privacy policy)

---

## 📁 File Structure

```
docs/
├── maps_index.html              # Landing page for all maps
├── model_accuracy.html          # Regular accuracy map
├── model_accuracy_chat.html     # AI chat-enabled accuracy map ⭐
├── gnn_completion.html          # GNN completion map
└── figures/
    ├── v3_figS1_training_data_overview.png
    ├── v3_figS2_cv_test_results.png
    ├── v3_figS3_prediction_coverage.png
    ├── v3_figS4_critical_by_class.png
    └── v3_figS5_feature_summary.png
```

---

## 🌐 GitHub Pages Deployment

All files are ready for GitHub Pages!

**Your maps will be available at:**
```
https://[your-username].github.io/[repo-name]/maps_index.html
https://[your-username].github.io/[repo-name]/model_accuracy_chat.html
```

**To deploy:**
1. Push to GitHub: `git add docs/ && git commit -m "Add AI chat maps" && git push`
2. Enable GitHub Pages: Settings → Pages → Source: main branch, /docs folder
3. Wait 1-2 minutes for deployment
4. Visit your URL!

---

## 🎓 Citation

If you use these maps or the AI chat feature in your research, please cite:

```bibtex
@software{knox_road_criticality_2026,
  title={Knox County Road Criticality GNN with AI Assistant},
  author={Your Name},
  year={2026},
  url={https://github.com/your-username/repo-name}
}
```

---

## 💡 Tips

**AI Chat Best Practices:**
- Be specific: "How many motorway false positives?" vs "Tell me about roads"
- Ask for metrics: "What's the recall?" gets exact numbers
- Request comparisons: "Compare TP and FP" shows differences
- One question at a time for best results

**Map Performance:**
- Use "no service" layers to exclude service roads (cleaner view)
- Zoom in for detailed road-level analysis
- Click roads for popup with full attributes
- Use layer control to show/hide categories

---

## 🆘 Troubleshooting

**Chat says "API Error: 401"**
→ Check your API key (should start with `sk-`)

**Chat says "API Error: 429"**
→ Rate limit exceeded. Wait or add credits to OpenAI account.

**Map loads slowly**
→ Normal for 8,000+ segments. Wait for initial load or zoom in.

**Chat gives generic answers**
→ Working with 500-sample dataset. Specific road queries may be limited.

---

**Questions?** Open an issue on GitHub or check `AI_CHAT_USAGE_GUIDE.md` for more details!

**Enjoy your AI-powered road analysis! 🚗💡**
