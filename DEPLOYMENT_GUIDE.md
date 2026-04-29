# 🚀 Deploy AI Chat Maps to GitHub Pages

## Step 1: Add and Commit Files

```powershell
# Add all new AI chat files
git add docs/model_accuracy_chat.html
git add docs/maps_index.html
git add docs/README.md
git add docs/figures/

# Add new utility scripts
git add AI_CHAT_USAGE_GUIDE.md
git add THRESHOLD_CONSISTENCY_UPDATE.md
git add compare_v2_v3.py
git add explain_*.py
git add step9b_model_accuracy_map_CHAT.py

# Add v3.0 updates
git add docs/model_accuracy.html
git add docs/gnn_completion.html
git add step6_road_criticality.py
git add step8_summary_report.py
git add step9_comparison_map.py
git add step9b_model_accuracy_map.py

# Add new v3 figures
git add outputs/figures/v3_*.png

# Commit everything
git commit -m "Add v3.0 with AI chat assistant

- 🤖 NEW: AI-powered chat interface for model queries
- 🎯 Updated to CV-optimized threshold (0.41)
- 📊 Model performance: AUC 84.3%, Recall 74.5%, F1 0.634
- 🏗️ 3-layer GAT with 6 highway infrastructure features
- 📈 Improvements over v2.0: +3.3% AUC, +8.3% F1, +12.6% Precision
- 🗺️ Beautiful landing page with all maps
- 📚 Complete documentation and usage guides

New features:
- Chat assistant powered by OpenAI GPT-4o-mini
- Natural language queries about roads and model performance
- Interactive map with confusion matrix visualization
- CV-consistent threshold across all visualizations
- GitHub Pages-ready deployment"
```

## Step 2: Push to GitHub

```powershell
git push origin main
```

## Step 3: Enable GitHub Pages (if not already enabled)

### Option A: Via GitHub Website
1. Go to your repository on GitHub
2. Click "Settings" tab
3. Scroll to "Pages" section (left sidebar)
4. Under "Source":
   - Branch: **main**
   - Folder: **/docs**
5. Click "Save"
6. Wait 1-2 minutes for deployment

### Option B: Via Command Line
```powershell
# If you have GitHub CLI installed
gh repo edit --enable-pages --pages-branch main --pages-path docs
```

## Step 4: Access Your Live Maps!

Your maps will be available at:
```
https://[your-username].github.io/[repo-name]/maps_index.html
https://[your-username].github.io/[repo-name]/model_accuracy_chat.html
https://[your-username].github.io/[repo-name]/model_accuracy.html
https://[your-username].github.io/[repo-name]/gnn_completion.html
```

**Landing Page (Recommended):**
```
https://[your-username].github.io/[repo-name]/maps_index.html
```

---

## 🎨 What Your Users Will See

### Landing Page (maps_index.html)
- Beautiful gradient background
- 3 interactive cards for each map
- "🆕 AI POWERED" badge on chat map
- Model performance statistics
- Direct links to all visualizations

### AI Chat Map (model_accuracy_chat.html)
- Click "🤖 AI Chat" button (top-right)
- Enter OpenAI API key
- Ask questions:
  - "What's the model performance?"
  - "How many false positives are there?"
  - "What's the average length of critical roads?"
  - "Show me statistics for motorways"

---

## 📝 Update Your Repository Description

On GitHub, update your repo description to:
```
Knox County Road Criticality GNN v3.0 with AI Chat Assistant - 
74.5% recall, 84.3% AUC. Try the live demo with AI-powered queries!
```

Add topics/tags:
```
gnn, graph-neural-network, road-safety, geospatial, ai-assistant, 
openai, transportation, gat, pytorch-geometric, folium
```

---

## 🔗 Share Your Work!

Once deployed, share these links:

**For Interactive Demo:**
```
Check out our AI-powered road criticality maps:
https://[your-username].github.io/[repo-name]/maps_index.html

Try the AI chat assistant to query model performance and road statistics!
```

**For Research Paper:**
```
Interactive supplementary materials available at:
https://[your-username].github.io/[repo-name]

Includes confusion matrix visualization, network-wide predictions,
and AI-powered query interface.
```

---

## ✅ Checklist

Before pushing:
- [ ] Verified chat map works locally (opened map_model_accuracy_CHAT.html)
- [ ] Tested AI chat with OpenAI API key
- [ ] Checked all links work on landing page
- [ ] Reviewed docs/README.md

After pushing:
- [ ] Confirmed GitHub Pages is enabled (Settings → Pages)
- [ ] Waited 1-2 minutes for deployment
- [ ] Tested live URL
- [ ] Verified AI chat works on GitHub Pages
- [ ] Shared link with team/advisor

---

## 🐛 Troubleshooting

**"404 Not Found" after deployment**
→ Wait 2-3 minutes. GitHub Pages takes time to build.
→ Check Settings → Pages shows green checkmark and URL

**Chat doesn't work on GitHub Pages**
→ Check browser console (F12) for errors
→ Verify HTTPS (not HTTP) - OpenAI API requires HTTPS
→ GitHub Pages automatically provides HTTPS ✓

**Large file warning**
→ HTML maps are ~10-65 MB (normal for embedded GeoJSON)
→ GitHub allows up to 100 MB per file ✓
→ If issues, use Git LFS (but shouldn't be necessary)

**Maps load slowly**
→ Normal for 8,000-65,000 road segments
→ Users can zoom in for faster rendering
→ Consider adding loading spinner in future

---

## 🎓 What's Next?

**Phase 2 Enhancements** (if you want):
1. **Full Road Data** (1 hour)
   - Include ALL 8,121 roads in chat (not just 500 sample)
   - Enable more detailed queries

2. **Map Highlighting** (2-3 hours)
   - "Show me all false negatives" → Highlights on map
   - Visual filtering based on queries
   - Zoom to matching roads

3. **Advanced Features** (2-3 days)
   - Vector search for semantic queries
   - Multi-turn conversations with memory
   - Export chat history
   - Custom visualizations from queries

**Let me know if you want any of these!**

---

**Ready to deploy? Run the commands above! 🚀**
