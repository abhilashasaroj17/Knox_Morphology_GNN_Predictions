# AI Chat Interface - Usage Guide

## 🎉 What's New

Your Knox County road criticality maps now include an **AI-powered chat assistant** powered by OpenAI's GPT-4!

## 📁 Files Created

### Backups (Original Maps - No Chat)
- `outputs/maps/map_model_accuracy_BACKUP.html` - Original accuracy map
- `outputs/maps/map_gnn_completion_BACKUP.html` - Original completion map

### New Chat-Enabled Maps
- `outputs/maps/map_model_accuracy_CHAT.html` - **Accuracy map with AI chat** ⭐

## 🚀 How to Use

### 1. Open the Chat-Enabled Map
Double-click: `outputs/maps/map_model_accuracy_CHAT.html`

### 2. Start the Chat
- Click the **"🤖 AI Chat"** button in the top-right corner
- A chat panel will slide out

### 3. Enter Your OpenAI API Key
- Get a key from: https://platform.openai.com/api-keys
- Paste it in the "OpenAI API Key" field
- **Note:** Your key is ONLY used in your browser, never stored

### 4. Ask Questions!

**Example Questions:**

**Model Performance:**
- "What's the model performance?"
- "How many false positives are there?"
- "What's the precision and recall?"
- "Show me the confusion matrix"

**Road Queries:**
- "How many critical roads are motorways?"
- "What's the average length of false negatives?"
- "Find roads with probability above 0.8"
- "How many residential roads are true positives?"

**Statistical Analysis:**
- "What's the accuracy?"
- "Compare TP and FP counts"
- "How does the threshold affect recall?"
- "What percentage of predictions are correct?"

**Spatial Insights:**
- "Which road classes have the most false negatives?"
- "What's the average TPO volume for critical roads?"
- "Show statistics for primary roads"

## 💡 Features

### ✅ What It Can Do
- Answer questions about model performance (TP, FP, FN, TN, precision, recall, F1)
- Provide statistics about road segments (counts, averages, distributions)
- Explain confusion matrix results
- Compare different road classes and categories
- Give instant answers with specific numbers

### ⚠️ Current Limitations
- Uses first 500 road segments for demo (reduces API costs)
- Cannot highlight/filter roads on map yet (coming in full version!)
- Requires your own OpenAI API key
- Internet connection required

## 🔑 Getting an OpenAI API Key

1. Go to: https://platform.openai.com/signup
2. Create account (free)
3. Add payment method (pay-as-you-go, ~$0.001 per question with GPT-4o-mini)
4. Go to: https://platform.openai.com/api-keys
5. Click "Create new secret key"
6. Copy the key (starts with `sk-...`)
7. Paste into the chat interface

**Cost:** Using GPT-4o-mini, each question costs ~$0.001-0.003 (less than a penny!)

## 🛠️ Technical Details

**Model Used:** GPT-4o-mini (fast, cheap, accurate)

**Data Provided to AI:**
```javascript
{
  total_roads: 8121,
  true_positives: 1211,
  false_negatives: 414,
  false_positives: 985,
  true_negatives: 5511,
  precision: 55.1%,
  recall: 74.5%,
  f1_score: 0.634,
  threshold: 0.41,
  sample_roads: [... first 500 segments with properties ...]
}
```

**Road Properties Available:**
- `class`: Road classification (motorway, primary, residential, etc.)
- `length_m`: Length in meters
- `volume`: TPO traffic volume
- `pred_prob_critical`: GNN predicted probability (0-1)
- `critical`: TPO ground truth (0 or 1)
- `confusion_category`: TP, FP, FN, or TN

## 🎯 Next Steps (Future Enhancements)

Want to expand this? Here are options:

### Option 1: Full Road Data Access (1 hour)
- Include ALL 8,121 roads instead of 500 sample
- Enable more detailed queries
- Add road name search

### Option 2: Map Highlighting (2-3 hours)
- "Show me all false negatives" → Highlights them on map
- "Zoom to motorway TP" → Pans to those roads
- Visual filtering based on queries

### Option 3: Vector Search (2-3 days)
- Semantic search: "roads similar to I-40"
- Pattern finding: "residential roads misclassified in high-volume areas"
- Cluster analysis: "group roads by characteristics"

### Option 4: Backend API (3-5 days)
- FastAPI server for complex spatial queries
- PostGIS integration for "critical roads within 1km of schools"
- User authentication, rate limiting
- Multi-turn conversations with memory

## 📝 Example Chat Session

```
👤 What's the model performance?

🤖 The model shows strong performance on Knox County roads:
   • Accuracy: 82.8%
   • Precision: 55.1% (of roads predicted critical, 55.1% truly are)
   • Recall: 74.5% (catches 74.5% of all critical roads)
   • F1 Score: 0.634
   
   Confusion Matrix:
   • True Positives: 1,211 (correctly identified critical)
   • False Negatives: 414 (missed critical roads)
   • False Positives: 985 (over-predicted)
   • True Negatives: 5,511 (correctly identified non-critical)

👤 How many false positives are motorways?

🤖 Looking at the sample data, there are 12 motorway segments classified 
   as false positives. These are roads that the model predicted as critical 
   (probability ≥ 0.41) but were labeled non-critical by TPO volume model.

👤 What's the average length of critical roads?

🤖 Based on the available data:
   • Average length of TPO-critical roads: 186.4 meters
   • Average length of GNN-predicted critical: 178.2 meters
   • Average length of true positives (TP): 192.1 meters
```

## 🔒 Privacy & Security

- **Your API key is NEVER stored** - only used in your browser session
- All queries run entirely client-side (no backend server)
- Road data is embedded in HTML (already public on GitHub Pages)
- OpenAI may log queries for service improvement (see their privacy policy)

## 🐛 Troubleshooting

**"API Error: 401"**
→ Invalid API key. Check you copied the full key including `sk-` prefix

**"API Error: 429"**
→ Rate limit exceeded. Wait a moment or add credits to OpenAI account

**Chat button doesn't appear**
→ Make sure JavaScript is enabled in your browser

**Answers are generic**
→ AI is working with 500-sample dataset. Answers about specific roads may be limited.

## 📞 Support

Questions or want enhancements? The chatbot is fully customizable!

**Want to add:**
- More road data to the chat?
- Visual highlighting of queried roads?
- Export chat conversations?
- Custom system prompts?

Just ask and I can enhance it further!

---

## 🎓 How It Works (Technical)

```javascript
// 1. User types question
"How many false positives are there?"

// 2. Chat interface builds context
const context = {
  system: "You are analyzing Knox County roads...",
  stats: { tp: 1211, fp: 985, fn: 414, tn: 5511, ... },
  roadData: [...500 segments with properties...]
};

// 3. Sends to OpenAI API
POST https://api.openai.com/v1/chat/completions
{
  model: "gpt-4o-mini",
  messages: [{role: "system", content: context}, 
             {role: "user", content: question}]
}

// 4. Receives answer
{
  "choices": [{
    "message": {
      "content": "There are 985 false positives, representing..."
    }
  }]
}

// 5. Displays in chat panel
"There are 985 false positives, representing..."
```

---

**Enjoy your AI-powered road analysis! 🚗💡**
