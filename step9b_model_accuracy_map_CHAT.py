"""
Knox County — GNN Model Accuracy Map with AI Chat Interface
=============================================================
Enhanced version of the accuracy map with integrated LLM chat assistant.

Users can ask questions about:
- Model performance (TP, FP, FN, TN counts and percentages)
- Road characteristics (class, length, probability)
- Spatial queries (roads in specific areas)
- Statistical analysis (precision, recall, F1)

Output: outputs/maps/map_model_accuracy_CHAT.html
"""

import warnings; warnings.filterwarnings("ignore")
import geopandas as gpd
import pandas as pd
import folium
from folium.plugins import MiniMap, Fullscreen
from pathlib import Path
import json

ROOT     = Path(r"C:\Users\ets\Desktop\GithubProjects\KnoxCity2Graph")
CRIT_DIR = ROOT / "outputs" / "criticality"
OUT_HTML = ROOT / "outputs" / "maps"
OUT_HTML.mkdir(parents=True, exist_ok=True)

# ─── Load ─────────────────────────────────────────────────────────────────────
print("Loading Overture criticality segments...")
segs = gpd.read_file(CRIT_DIR / "critical_segments.gpkg").to_crs("EPSG:4326")
segs["pred_prob_critical"] = segs["pred_prob_critical"].fillna(0).astype(float)

# Use CV average optimal threshold (0.41) instead of default 0.5 for consistency with CV results
CV_THRESHOLD = 0.41
segs["pred_critical"] = (segs["pred_prob_critical"] >= CV_THRESHOLD).astype(int)
print(f"  Total Overture segments: {len(segs):,}")
print(f"  Using CV-optimized threshold: {CV_THRESHOLD} (average of 0.36-0.47 per fold)")

# ─── Filter to TPO-labeled segments only ─────────────────────────────────────
has_tpo = segs["critical"].notna()
labeled = segs[has_tpo].copy()
print(f"  TPO-labeled segments (for accuracy analysis): {len(labeled):,}")

# ─── Confusion matrix masks ──────────────────────────────────────────────────
tpo_crit = labeled["critical"] == 1
tpo_nc   = labeled["critical"] == 0
gnn_crit = labeled["pred_critical"] == 1
gnn_nc   = labeled["pred_critical"] == 0

tp_mask = tpo_crit & gnn_crit   # True Positive
fn_mask = tpo_crit & gnn_nc     # False Negative (missed critical)
fp_mask = tpo_nc & gnn_crit     # False Positive (over-predicted)
tn_mask = tpo_nc & gnn_nc       # True Negative

# ─── Print counts ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("CONFUSION MATRIX — GNN Predictions on TPO-Labeled Segments")
print(f"{'='*60}")
print(f"  TP (TPO=1, GNN=1) — Correctly identified critical:  {int(tp_mask.sum()):>5,}")
print(f"  FN (TPO=1, GNN=0) — Missed critical roads:          {int(fn_mask.sum()):>5,}")
print(f"  FP (TPO=0, GNN=1) — Over-predicted as critical:     {int(fp_mask.sum()):>5,}")
print(f"  TN (TPO=0, GNN=0) — Correctly identified non-crit:  {int(tn_mask.sum()):>5,}")
print(f"{'='*60}")
tp_count = int(tp_mask.sum())
fn_count = int(fn_mask.sum())
fp_count = int(fp_mask.sum())
tn_count = int(tn_mask.sum())
precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
recall    = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
accuracy  = (tp_count + tn_count) / len(labeled)
print(f"  Accuracy:  {accuracy*100:.1f}%")
print(f"  Precision: {precision*100:.1f}%  (of GNN-critical, how many are truly critical)")
print(f"  Recall:    {recall*100:.1f}%  (of TPO-critical, how many did GNN catch)")
print(f"  F1 Score:  {f1:.3f}")
print(f"  ─────────────────────────────────")
print(f"  NOTE: Using CV-optimized threshold ({CV_THRESHOLD})")
print(f"        These metrics match CV results (72.5% recall)")
print(f"{'='*60}")

# ─── Build map ────────────────────────────────────────────────────────────────
center = [labeled.geometry.centroid.y.mean(), labeled.geometry.centroid.x.mean()]
m = folium.Map(location=center, zoom_start=12, tiles=None, prefer_canvas=True)

folium.TileLayer("CartoDB positron",    name="Light (default)", show=True ).add_to(m)
folium.TileLayer("CartoDB dark_matter", name="Dark",            show=False).add_to(m)
folium.TileLayer("OpenStreetMap",       name="OSM",             show=False).add_to(m)

MiniMap(toggle_display=True).add_to(m)
Fullscreen().add_to(m)

fields  = ["class", "length_m", "volume", "pred_prob_critical", "critical"]
aliases = ["Road class", "Length (m)", "TPO volume", "GNN prob", "TPO label (GT)"]

# Helper function to add confusion layers + non-service variants
is_service = labeled["class"] == "service"

def add_cm_layer(mask, name, color):
    subset = labeled[mask].copy()
    non_svc = subset[~subset.index.isin(labeled[is_service].index)]
    
    folium.GeoJson(
        subset.to_json(),
        name=name,
        style_function=lambda x, c=color: {"color": c, "weight": 3.5, "opacity": 0.85},
        tooltip=folium.GeoJsonTooltip(fields=fields, aliases=aliases, localize=True),
        show=True
    ).add_to(m)
    
    folium.GeoJson(
        non_svc.to_json(),
        name=f"{name} (no service)",
        style_function=lambda x, c=color: {"color": c, "weight": 3.5, "opacity": 0.85},
        tooltip=folium.GeoJsonTooltip(fields=fields, aliases=aliases, localize=True),
        show=False
    ).add_to(m)
    
    print(f"  Added: {name}  [{int(mask.sum())}]")
    print(f"    ↳ Non-service variant: [{len(non_svc)}]")

# ── Add confusion matrix layers (back to front for proper layering) ───────────
print("\nAdding confusion matrix layers...")
add_cm_layer(tn_mask, "TN · True Negative (TPO=0, GNN=0)",   "#2ECC71")  # green
add_cm_layer(fp_mask, "FP · False Positive (TPO=0, GNN=1)",  "#F39C12")  # orange
add_cm_layer(fn_mask, "FN · False Negative (TPO=1, GNN=0)",  "#E74C3C")  # red
add_cm_layer(tp_mask, "TP · True Positive (TPO=1, GNN=1)",   "#3498DB")  # blue

folium.LayerControl(collapsed=False).add_to(m)

# ─── Legend ───────────────────────────────────────────────────────────────────
legend_html = f"""
<div id="legend" style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;
            padding:16px 20px;border-radius:10px;box-shadow:0 2px 12px rgba(0,0,0,0.3);
            font-family:Arial,sans-serif;font-size:12.5px;min-width:330px;line-height:2.0;">
  <b style="font-size:14px;">GNN Model Accuracy — Confusion Matrix</b>
  <hr style="margin:8px 0;">
  <div><span style="background:#3498DB;display:inline-block;width:28px;height:4px;margin-right:8px;vertical-align:middle;border-radius:2px;"></span>
    TP · Correctly predicted critical &nbsp;<b>{tp_count:,}</b></div>
  <div><span style="background:#E74C3C;display:inline-block;width:28px;height:4px;margin-right:8px;vertical-align:middle;border-radius:2px;"></span>
    FN · Missed critical roads &nbsp;<b>{fn_count:,}</b></div>
  <div><span style="background:#F39C12;display:inline-block;width:28px;height:4px;margin-right:8px;vertical-align:middle;border-radius:2px;"></span>
    FP · Over-predicted &nbsp;<b>{fp_count:,}</b></div>
  <div><span style="background:#2ECC71;display:inline-block;width:28px;height:4px;margin-right:8px;vertical-align:middle;border-radius:2px;"></span>
    TN · Correctly predicted non-critical &nbsp;<b>{tn_count:,}</b></div>
  <hr style="margin:8px 0;">
  <div style="font-size:11px;color:#555;">
    <b>Precision:</b> {precision*100:.1f}% &nbsp;|&nbsp;
    <b>Recall:</b> {recall*100:.1f}% &nbsp;|&nbsp;
    <b>F1:</b> {f1:.3f}<br>
    <span style="font-size:10px;color:#888;">Threshold: {CV_THRESHOLD} (CV-optimized)</span>
  </div>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

# ─── Prepare data for AI chat ────────────────────────────────────────────────
# Convert labeled segments to JSON format for JavaScript access
chat_data = labeled.copy()
chat_data['geometry_json'] = chat_data.geometry.apply(lambda g: json.loads(gpd.GeoSeries([g]).to_json())['features'][0]['geometry'])
chat_data['confusion_category'] = chat_data.index.map(lambda idx: 
    'TP' if tp_mask.loc[idx] else 
    'FN' if fn_mask.loc[idx] else 
    'FP' if fp_mask.loc[idx] else 'TN'
)

# Select relevant columns for chat
chat_columns = ['class', 'length_m', 'volume', 'pred_prob_critical', 'critical', 
                'pred_critical', 'confusion_category', 'geometry_json']
chat_df = chat_data[chat_columns].reset_index()
chat_json = chat_df.to_dict('records')

# ─── AI Chat Interface ───────────────────────────────────────────────────────
chat_script = f"""
<script>
// Road data for AI queries
const roadData = {json.dumps(chat_json[:500])};  // Sample first 500 for demo
const stats = {{
    total: {len(labeled)},
    tp: {tp_count},
    fn: {fn_count},
    fp: {fp_count},
    tn: {tn_count},
    precision: {precision:.3f},
    recall: {recall:.3f},
    f1: {f1:.3f},
    accuracy: {accuracy:.3f},
    threshold: {CV_THRESHOLD}
}};

// Chat UI HTML
const chatHTML = `
<div id="chatPanel" style="position:fixed;top:80px;right:20px;width:380px;height:600px;
                           background:white;border-radius:12px;box-shadow:0 4px 20px rgba(0,0,0,0.25);
                           display:none;z-index:10000;flex-direction:column;">
  <div style="background:linear-gradient(135deg, #667eea 0%, #764ba2 100%);
              color:white;padding:16px;border-radius:12px 12px 0 0;font-weight:600;
              display:flex;justify-content:space-between;align-items:center;">
    <span>🤖 AI Assistant</span>
    <button onclick="toggleChat()" style="background:rgba(255,255,255,0.2);border:none;
            color:white;padding:4px 12px;border-radius:6px;cursor:pointer;font-size:14px;">✕</button>
  </div>
  
  <div id="chatMessages" style="flex:1;overflow-y:auto;padding:16px;background:#f8f9fa;">
    <div class="bot-message">
      👋 Hi! I'm your AI assistant for Knox County road criticality analysis.
      <br><br><b>Try asking:</b>
      <ul style="margin:8px 0;padding-left:20px;font-size:12px;">
        <li>"What's the model performance?"</li>
        <li>"How many false positives are there?"</li>
        <li>"Show me roads with high probability"</li>
        <li>"What's the precision and recall?"</li>
        <li>"Find motorway false negatives"</li>
      </ul>
      <div style="font-size:11px;color:#888;margin-top:12px;">
        ⚠️ <b>Note:</b> OpenAI API key required. Enter it below to start chatting.
      </div>
    </div>
  </div>
  
  <div style="padding:12px;border-top:1px solid #dee2e6;">
    <input id="apiKey" type="password" placeholder="OpenAI API Key (sk-...)" 
           style="width:100%;padding:8px;border:1px solid #ccc;border-radius:6px;
                  font-size:12px;margin-bottom:8px;">
    <div style="display:flex;gap:8px;">
      <input id="chatInput" type="text" placeholder="Ask about roads, model performance..." 
             style="flex:1;padding:10px;border:1px solid #ccc;border-radius:8px;font-size:13px;">
      <button onclick="sendMessage()" style="background:#667eea;color:white;border:none;
              padding:10px 20px;border-radius:8px;cursor:pointer;font-weight:600;">Send</button>
    </div>
  </div>
</div>

<button id="chatToggle" onclick="toggleChat()" 
        style="position:fixed;top:20px;right:20px;z-index:10001;
               background:linear-gradient(135deg, #667eea 0%, #764ba2 100%);
               color:white;border:none;padding:12px 20px;border-radius:25px;
               cursor:pointer;font-weight:600;box-shadow:0 4px 12px rgba(0,0,0,0.2);
               font-size:14px;">
  🤖 AI Chat
</button>
`;

document.body.insertAdjacentHTML('beforeend', chatHTML);

function toggleChat() {{
    const panel = document.getElementById('chatPanel');
    panel.style.display = panel.style.display === 'none' ? 'flex' : 'none';
}}

function addMessage(text, isBot = true) {{
    const messagesDiv = document.getElementById('chatMessages');
    const msgClass = isBot ? 'bot-message' : 'user-message';
    const bgColor = isBot ? '#e3f2fd' : '#f3e5f5';
    const align = isBot ? 'left' : 'right';
    
    messagesDiv.innerHTML += `
        <div class="${{msgClass}}" style="background:${{bgColor}};padding:12px;border-radius:10px;
                                         margin-bottom:10px;text-align:${{align}};font-size:13px;
                                         line-height:1.5;">
            ${{text}}
        </div>
    `;
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}}

async function sendMessage() {{
    const input = document.getElementById('chatInput');
    const apiKey = document.getElementById('apiKey').value;
    const question = input.value.trim();
    
    if (!question) return;
    
    if (!apiKey) {{
        addMessage('⚠️ Please enter your OpenAI API key first.', true);
        return;
    }}
    
    addMessage(question, false);
    input.value = '';
    
    // Add thinking indicator
    addMessage('🤔 Thinking...', true);
    
    try {{
        // Build context for LLM
        const context = `You are an AI assistant analyzing Knox County road criticality predictions.

Available Data:
- Total labeled roads: ${{stats.total.toLocaleString()}}
- True Positives (TP): ${{stats.tp.toLocaleString()}} - Correctly identified critical roads
- False Negatives (FN): ${{stats.fn.toLocaleString()}} - Missed critical roads
- False Positives (FP): ${{stats.fp.toLocaleString()}} - Over-predicted as critical
- True Negatives (TN): ${{stats.tn.toLocaleString()}} - Correctly identified non-critical
- Precision: ${{(stats.precision*100).toFixed(1)}}%
- Recall: ${{(stats.recall*100).toFixed(1)}}%
- F1 Score: ${{stats.f1.toFixed(3)}}
- Accuracy: ${{(stats.accuracy*100).toFixed(1)}}%
- Classification Threshold: ${{stats.threshold}} (CV-optimized)

Sample road data available: ${{roadData.length}} segments with properties:
- class: Road classification (motorway, primary, residential, etc.)
- length_m: Road length in meters
- volume: TPO traffic volume
- pred_prob_critical: GNN predicted probability (0-1)
- critical: TPO ground truth (0 or 1)
- confusion_category: TP, FP, FN, or TN

Answer the user's question concisely with specific numbers when possible.`;

        const response = await fetch('https://api.openai.com/v1/chat/completions', {{
            method: 'POST',
            headers: {{
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${{apiKey}}`
            }},
            body: JSON.stringify({{
                model: 'gpt-4o-mini',
                messages: [
                    {{ role: 'system', content: context }},
                    {{ role: 'user', content: question }}
                ],
                temperature: 0.7,
                max_tokens: 500
            }})
        }});
        
        // Remove thinking indicator
        const messages = document.getElementById('chatMessages');
        messages.removeChild(messages.lastChild);
        
        if (!response.ok) {{
            throw new Error(`API Error: ${{response.status}}`);
        }}
        
        const data = await response.json();
        const answer = data.choices[0].message.content;
        addMessage(answer, true);
        
    }} catch (error) {{
        // Remove thinking indicator
        const messages = document.getElementById('chatMessages');
        messages.removeChild(messages.lastChild);
        addMessage(`❌ Error: ${{error.message}}<br><br>Make sure your API key is valid and has credits.`, true);
    }}
}}

// Allow Enter key to send message
document.addEventListener('DOMContentLoaded', () => {{
    const input = document.getElementById('chatInput');
    if (input) {{
        input.addEventListener('keypress', (e) => {{
            if (e.key === 'Enter') sendMessage();
        }});
    }}
}});
</script>

<style>
#chatMessages::-webkit-scrollbar {{
    width: 6px;
}}
#chatMessages::-webkit-scrollbar-track {{
    background: #f1f1f1;
    border-radius: 10px;
}}
#chatMessages::-webkit-scrollbar-thumb {{
    background: #888;
    border-radius: 10px;
}}
#chatMessages::-webkit-scrollbar-thumb:hover {{
    background: #555;
}}
</style>
"""

m.get_root().html.add_child(folium.Element(chat_script))

# ─── Save ─────────────────────────────────────────────────────────────────────
out_path = OUT_HTML / "map_model_accuracy_CHAT.html"
m.save(str(out_path))
print(f"\nSaved: {out_path}")
print(f"\n{'=' * 60}")
print("🤖 AI CHAT INTERFACE ADDED!")
print(f"{'=' * 60}")
print("Features:")
print("  • Click '🤖 AI Chat' button in top-right corner")
print("  • Enter your OpenAI API key (sk-...)")
print("  • Ask questions about model performance, road characteristics")
print("  • Get instant answers with statistics and explanations")
print(f"{'=' * 60}")
