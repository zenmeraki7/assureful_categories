# """
# 🎯 COMPLETE API SERVER - Matches Cross-Store Training System
# =============================================================
# ✅ Works with cross-store synonyms (washing machine = laundry machine)
# ✅ Uses auto-tags from training
# ✅ Single model (fast predictions)
# ✅ Guaranteed category_id match
# ✅ Real-time classification
# """

# from flask import Flask, request, jsonify, render_template_string
# from sentence_transformers import SentenceTransformer
# import faiss
# import pickle
# import numpy as np
# from pathlib import Path
# import time
# import re

# app = Flask(__name__)

# # ============================================================================
# # GLOBAL VARIABLES
# # ============================================================================

# CACHE_DIR = Path('cache')

# # Model
# encoder = None
# faiss_index = None
# metadata = []
# cross_store_synonyms = {}


# # ============================================================================
# # CROSS-STORE SYNONYM DATABASE (Same as training)
# # ============================================================================

# def build_cross_store_synonyms():
#     """Build cross-store synonym database"""
#     synonyms = {
#         # Appliances
#         'washing machine': {'laundry machine', 'washer', 'clothes washer', 'washing appliance'},
#         'laundry machine': {'washing machine', 'washer', 'clothes washer'},
#         'dryer': {'drying machine', 'clothes dryer', 'tumble dryer'},
#         'refrigerator': {'fridge', 'cooler', 'ice box', 'cooling appliance'},
#         'dishwasher': {'dish washer', 'dish cleaning machine'},
#         'microwave': {'microwave oven', 'micro wave'},
#         'vacuum': {'vacuum cleaner', 'hoover', 'vac'},
        
#         # Electronics
#         'tv': {'television', 'telly', 'smart tv', 'display'},
#         'laptop': {'notebook', 'portable computer', 'laptop computer'},
#         'mobile': {'phone', 'cell phone', 'smartphone', 'cellphone'},
#         'tablet': {'ipad', 'tab', 'tablet computer'},
#         'headphones': {'headset', 'earphones', 'earbuds', 'ear buds'},
#         'speaker': {'audio speaker', 'sound system', 'speakers'},
        
#         # Furniture
#         'sofa': {'couch', 'settee', 'divan'},
#         'wardrobe': {'closet', 'armoire', 'cupboard'},
#         'drawer': {'chest of drawers', 'dresser'},
        
#         # Clothing
#         'pants': {'trousers', 'slacks', 'bottoms'},
#         'sweater': {'jumper', 'pullover', 'sweatshirt'},
#         'sneakers': {'trainers', 'tennis shoes', 'running shoes'},
#         'jacket': {'coat', 'blazer', 'outerwear'},
        
#         # Kitchen
#         'cooker': {'stove', 'range', 'cooking range'},
#         'blender': {'mixer', 'food processor', 'liquidizer'},
#         'kettle': {'electric kettle', 'water boiler'},
        
#         # Baby/Kids
#         'stroller': {'pram', 'pushchair', 'buggy', 'baby carriage'},
#         'diaper': {'nappy', 'nappies'},
#         'pacifier': {'dummy', 'soother'},
        
#         # Tools
#         'wrench': {'spanner', 'adjustable wrench'},
#         'flashlight': {'torch', 'flash light'},
#         'screwdriver': {'screw driver'},
        
#         # Home
#         'tap': {'faucet', 'water tap'},
#         'bin': {'trash can', 'garbage can', 'waste bin'},
#         'curtain': {'drape', 'window covering'},
        
#         # Crafts/Office
#         'guillotine': {'paper cutter', 'paper trimmer', 'blade cutter'},
#         'trimmer': {'cutter', 'cutting tool', 'edge cutter'},
#         'stapler': {'stapling machine', 'staple gun'},
        
#         # Books/Media
#         'magazine': {'periodical', 'journal', 'publication'},
#         'comic': {'comic book', 'graphic novel', 'manga'},
#         'ebook': {'e-book', 'digital book', 'electronic book'},
        
#         # General
#         'kids': {'children', 'child', 'childrens', 'youth', 'junior'},
#         'women': {'womens', 'ladies', 'female', 'lady'},
#         'men': {'mens', 'male', 'gentleman'},
#         'baby': {'infant', 'newborn', 'toddler'},
#     }
    
#     # Build bidirectional mapping
#     expanded = {}
#     for term, syns in synonyms.items():
#         expanded[term] = syns.copy()
#         for syn in syns:
#             if syn not in expanded:
#                 expanded[syn] = set()
#             expanded[syn].add(term)
#             expanded[syn].update(syns - {syn})
    
#     return expanded


# # ============================================================================
# # HELPER FUNCTIONS
# # ============================================================================

# def clean_text(text):
#     """Clean and normalize text"""
#     if not text:
#         return ""
#     text = str(text).lower()
#     text = re.sub(r'[^\w\s-]', ' ', text)
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text


# def extract_cross_store_terms(text):
#     """Extract terms with cross-store variations"""
#     cleaned = clean_text(text)
#     words = cleaned.split()
    
#     all_terms = set()
#     all_terms.add(cleaned)  # Full text
    
#     # Single words
#     for word in words:
#         if len(word) > 2:
#             all_terms.add(word)
#             # Add cross-store synonyms
#             if word in cross_store_synonyms:
#                 all_terms.update(cross_store_synonyms[word])
    
#     # 2-word phrases
#     for i in range(len(words) - 1):
#         if len(words[i]) > 2 and len(words[i+1]) > 2:
#             phrase = f"{words[i]} {words[i+1]}"
#             all_terms.add(phrase)
#             if phrase in cross_store_synonyms:
#                 all_terms.update(cross_store_synonyms[phrase])
    
#     # 3-word phrases
#     if len(words) >= 3:
#         for i in range(len(words) - 2):
#             if all(len(w) > 2 for w in words[i:i+3]):
#                 phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
#                 all_terms.add(phrase)
    
#     return list(all_terms)


# def build_enhanced_query(title, description=""):
#     """Build enhanced query with cross-store intelligence"""
#     # Extract terms with variations
#     all_terms = extract_cross_store_terms(f"{title} {description}")
    
#     # Clean product terms
#     product_terms = [t for t in clean_text(f"{title} {description}").split() if len(t) > 2]
    
#     # Build query
#     # Emphasize original + all variations
#     product_text = ' '.join(product_terms)
#     variations_text = ' '.join(all_terms[:30])  # Top 30 variations
    
#     # Repeat for emphasis
#     emphasized = ' '.join([product_text] * 3)
    
#     query = f"{emphasized} {variations_text} {title} {description}"
    
#     return query, all_terms[:20]


# def encode_query(text):
#     """Encode query using the trained model"""
#     embedding = encoder.encode(
#         text,
#         convert_to_numpy=True,
#         normalize_embeddings=True
#     )
    
#     if embedding.ndim == 1:
#         embedding = embedding.reshape(1, -1)
    
#     return embedding.astype('float32')


# def classify_product(title, description="", top_k=5):
#     """
#     Classify product using trained system
#     Returns: category_id, category_path, confidence, and alternatives
#     """
#     start_time = time.time()
    
#     # Step 1: Build enhanced query with cross-store synonyms
#     query, matched_terms = build_enhanced_query(title, description)
    
#     # Step 2: Encode query
#     query_embedding = encode_query(query)
    
#     # Step 3: Search FAISS index
#     distances, indices = faiss_index.search(query_embedding, top_k)
    
#     # Step 4: Get results
#     results = []
#     for i in range(len(indices[0])):
#         idx = indices[0][i]
#         if idx < len(metadata):
#             meta = metadata[idx]
#             confidence = float(distances[0][i]) * 100
            
#             # Get final product name
#             levels = meta.get('levels', [])
#             final_product = levels[-1] if levels else meta['category_path'].split('/')[-1]
            
#             results.append({
#                 'rank': i + 1,
#                 'category_id': meta['category_id'],
#                 'category_path': meta['category_path'],
#                 'final_product': final_product,
#                 'confidence': round(confidence, 2),
#                 'depth': meta.get('depth', 0)
#             })
    
#     # Best result
#     best = results[0] if results else None
    
#     if not best:
#         return {
#             'error': 'No results found',
#             'product': title
#         }
    
#     # Confidence level
#     conf_pct = best['confidence']
#     if conf_pct >= 90:
#         conf_level = "EXCELLENT"
#     elif conf_pct >= 85:
#         conf_level = "VERY HIGH"
#     elif conf_pct >= 80:
#         conf_level = "HIGH"
#     elif conf_pct >= 75:
#         conf_level = "GOOD"
#     elif conf_pct >= 70:
#         conf_level = "MEDIUM"
#     else:
#         conf_level = "LOW"
    
#     processing_time = (time.time() - start_time) * 1000
    
#     return {
#         'product': title,
#         'category_id': best['category_id'],
#         'category_path': best['category_path'],
#         'final_product': best['final_product'],
#         'confidence': f"{conf_level} ({conf_pct:.2f}%)",
#         'confidence_percent': conf_pct,
#         'depth': best['depth'],
#         'matched_terms': matched_terms,
#         'top_5_results': results,
#         'processing_time_ms': round(processing_time, 2)
#     }


# # ============================================================================
# # SERVER INITIALIZATION
# # ============================================================================

# def load_server():
#     """Load all trained data"""
#     global encoder, faiss_index, metadata, cross_store_synonyms
    
#     print("\n" + "="*80)
#     print("🔄 LOADING TRAINED MODEL")
#     print("="*80 + "\n")
    
#     # Load model
#     print("📥 Loading sentence transformer...")
#     encoder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
#     print("✅ Model loaded\n")
    
#     # Load FAISS index
#     print("📥 Loading FAISS index...")
#     index_path = CACHE_DIR / 'main_index.faiss'
#     if not index_path.exists():
#         raise FileNotFoundError(f"FAISS index not found: {index_path}\nPlease run training first!")
#     faiss_index = faiss.read_index(str(index_path))
#     print(f"✅ Index loaded ({faiss_index.ntotal:,} vectors)\n")
    
#     # Load metadata
#     print("📥 Loading metadata...")
#     meta_path = CACHE_DIR / 'metadata.pkl'
#     if not meta_path.exists():
#         raise FileNotFoundError(f"Metadata not found: {meta_path}\nPlease run training first!")
#     with open(meta_path, 'rb') as f:
#         metadata = pickle.load(f)
#     print(f"✅ Metadata loaded ({len(metadata):,} categories)\n")
    
#     # Load cross-store synonyms
#     print("📥 Loading cross-store synonyms...")
#     syn_path = CACHE_DIR / 'cross_store_synonyms.pkl'
#     if syn_path.exists():
#         with open(syn_path, 'rb') as f:
#             cross_store_synonyms = pickle.load(f)
#         print(f"✅ Cross-store synonyms loaded ({len(cross_store_synonyms)} terms)\n")
#     else:
#         print("⚠️  Cross-store synonyms not found, building default set...")
#         cross_store_synonyms = build_cross_store_synonyms()
#         print(f"✅ Built {len(cross_store_synonyms)} synonym mappings\n")
    
#     print("="*80)
#     print("✅ SERVER READY!")
#     print("="*80 + "\n")


# # ============================================================================
# # HTML INTERFACE
# # ============================================================================

# HTML_TEMPLATE = """
# <!DOCTYPE html>
# <html>
# <head>
#     <title>🎯 Product Category Classifier</title>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <style>
#         * { margin: 0; padding: 0; box-sizing: border-box; }
#         body {
#             font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
#             background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#             min-height: 100vh;
#             padding: 20px;
#         }
#         .container { max-width: 1200px; margin: 0 auto; }
#         .header {
#             text-align: center;
#             color: white;
#             margin-bottom: 30px;
#         }
#         .header h1 { font-size: 2.5em; margin-bottom: 10px; }
#         .badge {
#             background: rgba(255,255,255,0.2);
#             padding: 8px 20px;
#             border-radius: 20px;
#             display: inline-block;
#             margin: 5px;
#             font-size: 0.9em;
#         }
#         .card {
#             background: white;
#             border-radius: 20px;
#             padding: 30px;
#             box-shadow: 0 10px 40px rgba(0,0,0,0.2);
#         }
#         .success-box {
#             background: #d4edda;
#             padding: 15px;
#             border-radius: 8px;
#             margin-bottom: 20px;
#             border-left: 4px solid #28a745;
#             color: #155724;
#         }
#         .form-group { margin-bottom: 20px; }
#         label {
#             display: block;
#             font-weight: 600;
#             margin-bottom: 8px;
#             color: #333;
#         }
#         input, textarea {
#             width: 100%;
#             padding: 12px;
#             border: 2px solid #e0e0e0;
#             border-radius: 8px;
#             font-size: 1em;
#         }
#         input:focus, textarea:focus {
#             outline: none;
#             border-color: #667eea;
#         }
#         textarea { min-height: 80px; resize: vertical; }
#         button {
#             width: 100%;
#             padding: 15px;
#             background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#             color: white;
#             border: none;
#             border-radius: 10px;
#             font-size: 1.1em;
#             cursor: pointer;
#             font-weight: 600;
#             transition: transform 0.2s;
#         }
#         button:hover { transform: translateY(-2px); }
#         .results { display: none; margin-top: 20px; }
#         .results.show { display: block; animation: fadeIn 0.5s; }
#         @keyframes fadeIn {
#             from { opacity: 0; transform: translateY(10px); }
#             to { opacity: 1; transform: translateY(0); }
#         }
#         .section {
#             background: #f8f9fa;
#             padding: 20px;
#             border-radius: 12px;
#             margin-bottom: 15px;
#             border-left: 4px solid #667eea;
#         }
#         .section h3 { color: #667eea; margin-bottom: 12px; }
#         .result-item {
#             background: white;
#             padding: 15px;
#             border-radius: 8px;
#             margin-bottom: 10px;
#             border-left: 3px solid #667eea;
#         }
#         .tag {
#             display: inline-block;
#             background: #667eea;
#             color: white;
#             padding: 6px 12px;
#             border-radius: 15px;
#             margin: 3px;
#             font-size: 0.9em;
#         }
#         .conf-excellent { background: #4caf50; }
#         .conf-very { background: #8bc34a; }
#         .conf-high { background: #cddc39; color: #333; }
#         .conf-good { background: #ff9800; }
#         .conf-medium { background: #ff5722; }
#         .conf-low { background: #9e9e9e; }
#         .loading { display: none; text-align: center; padding: 20px; }
#         .loading.show { display: block; }
#         .spinner {
#             border: 4px solid #f3f3f3;
#             border-top: 4px solid #667eea;
#             border-radius: 50%;
#             width: 40px;
#             height: 40px;
#             animation: spin 1s linear infinite;
#             margin: 0 auto;
#         }
#         @keyframes spin {
#             0% { transform: rotate(0deg); }
#             100% { transform: rotate(360deg); }
#         }
#     </style>
# </head>
# <body>
#     <div class="container">
#         <div class="header">
#             <h1>🎯 Product Category Classifier</h1>
#             <div class="badge">Cross-Store Intelligence</div>
#             <div class="badge">Auto-Tag Support</div>
#             <div class="badge">Real-Time</div>
#         </div>
        
#         <div class="card">
#             <div class="success-box">
#                 <strong>✅ Cross-Store Synonyms Active!</strong><br>
#                 Understands: washing machine = laundry machine | tv = television | kids = children
#             </div>
            
#             <div class="form-group">
#                 <label>Product Title *</label>
#                 <input type="text" id="title" placeholder="e.g., Washing Machine or Laundry Machine" />
#             </div>
            
#             <div class="form-group">
#                 <label>Description (Optional)</label>
#                 <textarea id="desc" placeholder="Additional details..."></textarea>
#             </div>
            
#             <button onclick="classify()">🎯 Classify Product</button>
            
#             <div class="loading" id="loading">
#                 <div class="spinner"></div>
#                 <p style="margin-top: 10px; color: #666;">Analyzing...</p>
#             </div>
            
#             <div class="results" id="results">
#                 <div class="section">
#                     <h3>✅ Best Match</h3>
#                     <div class="result-item">
#                         <div style="margin-bottom: 10px;">
#                             <strong>Product:</strong> <span id="product"></span>
#                         </div>
#                         <div style="margin-bottom: 10px;">
#                             <strong>Category ID:</strong> 
#                             <span id="catId" style="font-size: 1.2em; color: #28a745; font-weight: bold;"></span>
#                         </div>
#                         <div style="margin-bottom: 10px;">
#                             <strong>Final Product:</strong> <span id="finalProd" style="font-weight: 600;"></span>
#                         </div>
#                         <div style="margin-bottom: 10px;">
#                             <strong>Full Path:</strong><br>
#                             <span id="path" style="color: #666; font-size: 0.95em;"></span>
#                         </div>
#                         <div style="margin-bottom: 10px;">
#                             <strong>Confidence:</strong> 
#                             <span id="confidence" class="tag"></span>
#                         </div>
#                         <div style="font-size: 0.9em; color: #666;">
#                             <strong>Depth:</strong> <span id="depth"></span> levels | 
#                             <strong>Time:</strong> <span id="time"></span>ms
#                         </div>
#                     </div>
#                 </div>
                
#                 <div class="section">
#                     <h3>🔗 Matched Terms (Cross-Store Variations)</h3>
#                     <div id="matchedTerms"></div>
#                 </div>
                
#                 <div class="section">
#                     <h3>📋 Top 5 Alternative Matches</h3>
#                     <div id="alternatives"></div>
#                 </div>
#             </div>
#         </div>
#     </div>
    
#     <script>
#         async function classify() {
#             const title = document.getElementById('title').value.trim();
#             const desc = document.getElementById('desc').value.trim();
            
#             if (!title) {
#                 alert('Please enter a product title');
#                 return;
#             }
            
#             document.getElementById('loading').classList.add('show');
#             document.getElementById('results').classList.remove('show');
            
#             try {
#                 const response = await fetch('/classify', {
#                     method: 'POST',
#                     headers: { 'Content-Type': 'application/json' },
#                     body: JSON.stringify({ title, description: desc })
#                 });
                
#                 if (!response.ok) throw new Error('Classification failed');
                
#                 const data = await response.json();
#                 displayResults(data);
#             } catch (error) {
#                 alert('Error: ' + error.message);
#             } finally {
#                 document.getElementById('loading').classList.remove('show');
#             }
#         }
        
#         function displayResults(data) {
#             document.getElementById('results').classList.add('show');
            
#             document.getElementById('product').textContent = data.product;
#             document.getElementById('catId').textContent = data.category_id;
#             document.getElementById('finalProd').textContent = data.final_product;
#             document.getElementById('path').textContent = data.category_path;
#             document.getElementById('depth').textContent = data.depth;
#             document.getElementById('time').textContent = data.processing_time_ms;
            
#             const conf = document.getElementById('confidence');
#             conf.textContent = data.confidence;
#             const confClass = data.confidence.split(' ')[0].toLowerCase().replace('_', '-');
#             conf.className = 'tag conf-' + confClass;
            
#             const matchedHtml = data.matched_terms.map(t => `<span class="tag">${t}</span>`).join('');
#             document.getElementById('matchedTerms').innerHTML = matchedHtml;
            
#             let altHtml = '';
#             data.top_5_results.forEach((item, i) => {
#                 const cls = i === 0 ? 'style="background: #e8f5e9;"' : '';
#                 altHtml += `
#                     <div class="result-item" ${cls}>
#                         <strong>${item.rank}.</strong> ${item.final_product} 
#                         <span class="tag" style="background: #999;">${item.confidence}%</span>
#                         <div style="font-size: 0.85em; color: #666; margin-top: 5px;">
#                             ID: ${item.category_id}
#                         </div>
#                     </div>
#                 `;
#             });
#             document.getElementById('alternatives').innerHTML = altHtml;
#         }
        
#         document.getElementById('title').addEventListener('keypress', function(e) {
#             if (e.key === 'Enter') classify();
#         });
#     </script>
# </body>
# </html>
# """


# # ============================================================================
# # FLASK ROUTES
# # ============================================================================

# @app.route('/')
# def index():
#     """Serve the web interface"""
#     return render_template_string(HTML_TEMPLATE)


# @app.route('/classify', methods=['POST'])
# def classify_route():
#     """API endpoint for classification"""
#     data = request.json
#     title = data.get('title', '').strip()
#     description = data.get('description', '').strip()
    
#     if not title:
#         return jsonify({'error': 'Title required'}), 400
    
#     try:
#         result = classify_product(title, description)
#         return jsonify(result)
#     except Exception as e:
#         print(f"Error: {e}")
#         return jsonify({'error': str(e)}), 500


# @app.route('/health')
# def health():
#     """Health check endpoint"""
#     return jsonify({
#         'status': 'healthy',
#         'categories': len(metadata),
#         'cross_store_synonyms': len(cross_store_synonyms),
#         'model': 'all-mpnet-base-v2'
#     })


# # ============================================================================
# # MAIN
# # ============================================================================

# if __name__ == '__main__':
#     try:
#         load_server()
        
#         print("\n🌐 Server starting...")
#         print("   URL: http://localhost:5000")
#         print("   Press CTRL+C to stop\n")
        
#         app.run(host='0.0.0.0', port=5000, debug=False)
        
#     except FileNotFoundError as e:
#         print(f"\n❌ ERROR: {e}")
#         print("\n💡 Solution: Run training first:")
#         print("   python train.py data/category_id_path_only.csv\n")
#     except Exception as e:
#         print(f"\n❌ UNEXPECTED ERROR: {e}\n")





#!/usr/bin/env python3
"""
API Server for product category classification
Merged UI + classification logic
Model: intfloat/e5-base-v2 (must match training)

Usage:
    python api_server.py

Requirements:
    pip install flask sentence-transformers faiss-cpu numpy pickle5

Files expected in cache/:
    - main_index.faiss
    - metadata.pkl
    - cross_store_synonyms.pkl (optional)

"""

from flask import Flask, request, jsonify, render_template_string
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
from pathlib import Path
import time
import re
import os
from typing import List

# ============================================================================
# CONFIG
# ============================================================================

CACHE_DIR = Path('cache')
MODEL_NAME = 'intfloat/e5-base-v2'  # <-- MUST match the model used during training
FAISS_INDEX_PATH = CACHE_DIR / 'main_index.faiss'
METADATA_PATH = CACHE_DIR / 'metadata.pkl'
SYN_PATH = CACHE_DIR / 'cross_store_synonyms.pkl'

# Server globals
encoder = None
faiss_index = None
metadata = []
cross_store_synonyms = {}

# ============================================================================
# CROSS-STORE SYNONYM FALLBACK
# ============================================================================

def build_cross_store_synonyms():
    """Default cross-store synonyms fallback (bidirectional mapping).
    If you have a trained cross_store_synonyms.pkl produced by training, the
    server will load that file instead. This function only used when no file
    exists in the cache.
    """
    synonyms = {
        'washing machine': {'laundry machine', 'washer', 'clothes washer', 'washing appliance'},
        'laundry machine': {'washing machine', 'washer', 'clothes washer'},
        'dryer': {'drying machine', 'clothes dryer', 'tumble dryer'},
        'refrigerator': {'fridge', 'cooler', 'ice box', 'cooling appliance'},
        'dishwasher': {'dish washer', 'dish cleaning machine'},
        'microwave': {'microwave oven', 'micro wave'},
        'vacuum': {'vacuum cleaner', 'hoover', 'vac'},
        'tv': {'television', 'telly', 'smart tv', 'display'},
        'laptop': {'notebook', 'portable computer', 'laptop computer'},
        'mobile': {'phone', 'cell phone', 'smartphone', 'cellphone'},
        'tablet': {'ipad', 'tab', 'tablet computer'},
        'headphones': {'headset', 'earphones', 'earbuds', 'ear buds'},
        'speaker': {'audio speaker', 'sound system', 'speakers'},
        'sofa': {'couch', 'settee', 'divan'},
        'wardrobe': {'closet', 'armoire', 'cupboard'},
        'drawer': {'chest of drawers', 'dresser'},
        'pants': {'trousers', 'slacks', 'bottoms'},
        'sweater': {'jumper', 'pullover', 'sweatshirt'},
        'sneakers': {'trainers', 'tennis shoes', 'running shoes'},
        'jacket': {'coat', 'blazer', 'outerwear'},
        'cooker': {'stove', 'range', 'cooking range'},
        'blender': {'mixer', 'food processor', 'liquidizer'},
        'kettle': {'electric kettle', 'water boiler'},
        'stroller': {'pram', 'pushchair', 'buggy', 'baby carriage'},
        'diaper': {'nappy', 'nappies'},
        'pacifier': {'dummy', 'soother'},
        'wrench': {'spanner', 'adjustable wrench'},
        'flashlight': {'torch', 'flash light'},
        'screwdriver': {'screw driver'},
        'tap': {'faucet', 'water tap'},
        'bin': {'trash can', 'garbage can', 'waste bin'},
        'curtain': {'drape', 'window covering'},
        'guillotine': {'paper cutter', 'paper trimmer', 'blade cutter'},
        'trimmer': {'cutter', 'cutting tool', 'edge cutter'},
        'stapler': {'stapling machine', 'staple gun'},
        'magazine': {'periodical', 'journal', 'publication'},
        'comic': {'comic book', 'graphic novel', 'manga'},
        'ebook': {'e-book', 'digital book', 'electronic book'},
        'kids': {'children', 'child', 'childrens', 'youth', 'junior'},
        'women': {'womens', 'ladies', 'female', 'lady'},
        'men': {'mens', 'male', 'gentleman'},
        'baby': {'infant', 'newborn', 'toddler'},
    }

    expanded = {}
    for term, syns in synonyms.items():
        expanded[term] = set(syns)
        for syn in syns:
            if syn not in expanded:
                expanded[syn] = set()
            expanded[syn].add(term)
            expanded[syn].update(syns - {syn})
    return expanded

# ============================================================================
# TEXT CLEANING / QUERY BUILDING
# ============================================================================

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = str(text).lower()
    # keep alphanumerics, dashes and spaces
    text = re.sub(r"[^\w\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_cross_store_terms(text: str) -> List[str]:
    cleaned = clean_text(text)
    words = cleaned.split()

    all_terms = set()
    all_terms.add(cleaned)  # full cleaned text

    # single words + synonyms
    for word in words:
        if len(word) > 2:
            all_terms.add(word)
            if word in cross_store_synonyms:
                all_terms.update(cross_store_synonyms[word])

    # 2-word phrases
    for i in range(len(words) - 1):
        if len(words[i]) > 2 and len(words[i + 1]) > 2:
            phrase = f"{words[i]} {words[i+1]}"
            all_terms.add(phrase)
            if phrase in cross_store_synonyms:
                all_terms.update(cross_store_synonyms[phrase])

    # 3-word phrases
    if len(words) >= 3:
        for i in range(len(words) - 2):
            if all(len(w) > 2 for w in words[i:i + 3]):
                phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
                all_terms.add(phrase)

    return list(all_terms)

def build_enhanced_query(title, description="", max_synonyms=10):
    """Build query emphasizing original title and cross-store variations"""
    title_clean = clean_text(title)
    description_clean = clean_text(description)
    
    # Extract cross-store variations
    synonyms_list = extract_cross_store_terms(f"{title_clean} {description_clean}")

    # Emphasize original title 3x, then include top synonyms
    enhanced_query = ' '.join([title_clean] * 3 + synonyms_list[:max_synonyms])
    
    return enhanced_query, synonyms_list[:20]  # return top 20 for matched_terms display

# ============================================================================
# ENCODER / FAISS
# ============================================================================

def encode_query(text: str) -> np.ndarray:
    embedding = encoder.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)
    return embedding.astype('float32')

def classify_product(title, description="", top_k=5):
    """Classify product using e5-base embeddings with cross-store optimization"""
    start_time = time.time()
    
    # Step 1: Build enhanced query
    query_text, matched_terms = build_enhanced_query(title, description)
    
    # Step 2: Encode query
    query_embedding = encoder.encode(
        query_text,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype('float32')
    
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    # Step 3: FAISS search
    distances, indices = faiss_index.search(query_embedding, top_k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx >= len(metadata):
            continue
        meta = metadata[idx]
        # Convert FAISS distance to cosine similarity
        similarity = 1 - distances[0][i]
        confidence_pct = float(similarity) * 100
        
        final_product = meta.get('levels', [])[-1] if meta.get('levels') else meta['category_path'].split('/')[-1]
        
        results.append({
            'rank': i + 1,
            'category_id': meta['category_id'],
            'category_path': meta['category_path'],
            'final_product': final_product,
            'confidence': round(confidence_pct, 2),
            'depth': meta.get('depth', 0)
        })
    
    if not results:
        return {'error': 'No results found', 'product': title}
    
    # Pick best match
    best = results[0]
    conf_pct = best['confidence']
    if conf_pct >= 90:
        conf_level = "EXCELLENT"
    elif conf_pct >= 85:
        conf_level = "VERY HIGH"
    elif conf_pct >= 80:
        conf_level = "HIGH"
    elif conf_pct >= 75:
        conf_level = "GOOD"
    elif conf_pct >= 70:
        conf_level = "MEDIUM"
    else:
        conf_level = "LOW"
    
    processing_time = (time.time() - start_time) * 1000
    
    return {
        'product': title,
        'category_id': best['category_id'],
        'category_path': best['category_path'],
        'final_product': best['final_product'],
        'confidence': f"{conf_level} ({conf_pct:.2f}%)",
        'confidence_percent': conf_pct,
        'depth': best['depth'],
        'matched_terms': matched_terms,
        'top_5_results': results,
        'processing_time_ms': round(processing_time, 2)
    }
    # FAISS returns squared L2 distances or inner product depending on index type.
    # We'll treat lower distance as better. We convert to a 0-100-ish confidence by
    # using a simple heuristic: score = 100 - normalized_distance*100 (clamped).

    # Determine a normalization constant: use mean of top distance if available
    flat_dist = distances[0]
    max_d = float(np.max(flat_dist)) if flat_dist.size else 1.0
    min_d = float(np.min(flat_dist)) if flat_dist.size else 0.0
    range_d = max(1e-6, max_d - min_d)

    for i, idx in enumerate(indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        meta = metadata[idx]
        raw_d = float(distances[0][i])
        # normalize and invert to make higher -> better
        norm = (raw_d - min_d) / range_d
        conf = max(0.0, min(100.0, 100.0 * (1.0 - norm)))

        levels = meta.get('levels') or []
        final_product = levels[-1] if levels else meta.get('category_path', '').split('/')[-1]

        results.append({
            'rank': i + 1,
            'category_id': meta.get('category_id'),
            'category_path': meta.get('category_path'),
            'final_product': final_product,
            'confidence': round(conf, 2),
            'depth': meta.get('depth', 0)
        })

    if not results:
        return {
            'error': 'No results found',
            'product': title
        }

    best = results[0]
    conf_pct = best['confidence']
    if conf_pct >= 90:
        conf_level = "EXCELLENT"
    elif conf_pct >= 85:
        conf_level = "VERY HIGH"
    elif conf_pct >= 80:
        conf_level = "HIGH"
    elif conf_pct >= 75:
        conf_level = "GOOD"
    elif conf_pct >= 70:
        conf_level = "MEDIUM"
    else:
        conf_level = "LOW"

    processing_time = (time.time() - start_time) * 1000.0

    return {
        'product': title,
        'category_id': best['category_id'],
        'category_path': best['category_path'],
        'final_product': best['final_product'],
        'confidence': f"{conf_level} ({conf_pct:.2f}%)",
        'confidence_percent': conf_pct,
        'depth': best['depth'],
        'matched_terms': matched_terms,
        'top_5_results': results,
        'processing_time_ms': round(processing_time, 2)
    }

# ============================================================================
# SERVER LOAD
# ============================================================================

def load_server():
    global encoder, faiss_index, metadata, cross_store_synonyms

    print('\n' + '=' * 80)
    print('🔄 LOADING TRAINED MODEL')
    print('=' * 80 + '\n')

    # Load encoder
    print('📥 Loading sentence transformer...')
    encoder = SentenceTransformer(MODEL_NAME)
    print('✅ Model loaded\n')

    # Load FAISS index
    print('📥 Loading FAISS index...')
    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(f"FAISS index not found: {FAISS_INDEX_PATH}\nPlease run training first!")
    faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
    print(f"✅ Index loaded ({faiss_index.ntotal:,} vectors)\n")

    # Load metadata
    print('📥 Loading metadata...')
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata not found: {METADATA_PATH}\nPlease run training first!")
    with open(METADATA_PATH, 'rb') as f:
        metadata = pickle.load(f)
    print(f"✅ Metadata loaded ({len(metadata):,} categories)\n")

    # Load or build cross-store synonyms
    print('📥 Loading cross-store synonyms...')
    if SYN_PATH.exists():
        with open(SYN_PATH, 'rb') as f:
            cross_store_synonyms = pickle.load(f)
        print(f"✅ Cross-store synonyms loaded ({len(cross_store_synonyms)} terms)\n")
    else:
        print('⚠️  Cross-store synonyms not found, building default set...')
        cross_store_synonyms = build_cross_store_synonyms()
        print(f"✅ Built {len(cross_store_synonyms)} synonym mappings\n")

    print('=' * 80)
    print('✅ SERVER READY!')
    print('=' * 80 + '\n')

# ============================================================================
# HTML TEMPLATE (same as provided)
# ============================================================================

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html>
<head>
    <title>🎯 Product Category Classifier</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .badge {
            background: rgba(255,255,255,0.2);
            padding: 8px 20px;
            border-radius: 20px;
            display: inline-block;
            margin: 5px;
            font-size: 0.9em;
        }
        .card {
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }
        .success-box {
            background: #d4edda;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid #28a745;
            color: #155724;
        }
        .form-group { margin-bottom: 20px; }
        label {
            display: block;
            font-weight: 600;
            margin-bottom: 8px;
            color: #333;
        }
        input, textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 1em;
        }
        input:focus, textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        textarea { min-height: 80px; resize: vertical; }
        button {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1.1em;
            cursor: pointer;
            font-weight: 600;
            transition: transform 0.2s;
        }
        button:hover { transform: translateY(-2px); }
        .results { display: none; margin-top: 20px; }
        .results.show { display: block; animation: fadeIn 0.5s; }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .section {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 15px;
            border-left: 4px solid #667eea;
        }
        .section h3 { color: #667eea; margin-bottom: 12px; }
        .result-item {
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            border-left: 3px solid #667eea;
        }
        .tag {
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 6px 12px;
            border-radius: 15px;
            margin: 3px;
            font-size: 0.9em;
        }
        .conf-excellent { background: #4caf50; }
        .conf-very { background: #8bc34a; }
        .conf-high { background: #cddc39; color: #333; }
        .conf-good { background: #ff9800; }
        .conf-medium { background: #ff5722; }
        .conf-low { background: #9e9e9e; }
        .loading { display: none; text-align: center; padding: 20px; }
        .loading.show { display: block; }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Product Category Classifier</h1>
            <div class="badge">Cross-Store Intelligence</div>
            <div class="badge">Auto-Tag Support</div>
            <div class="badge">Real-Time</div>
        </div>

        <div class="card">
            <div class="success-box">
                <strong>✅ Cross-Store Synonyms Active!</strong><br>
                Understands: washing machine = laundry machine | tv = television | kids = children
            </div>

            <div class="form-group">
                <label>Product Title *</label>
                <input type="text" id="title" placeholder="e.g., Washing Machine or Laundry Machine" />
            </div>

            <div class="form-group">
                <label>Description (Optional)</label>
                <textarea id="desc" placeholder="Additional details..."></textarea>
            </div>

            <button onclick="classify()">🎯 Classify Product</button>

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p style="margin-top: 10px; color: #666;">Analyzing...</p>
            </div>

            <div class="results" id="results">
                <div class="section">
                    <h3>✅ Best Match</h3>
                    <div class="result-item">
                        <div style="margin-bottom: 10px;">
                            <strong>Product:</strong> <span id="product"></span>
                        </div>
                        <div style="margin-bottom: 10px;">
                            <strong>Category ID:</strong> 
                            <span id="catId" style="font-size: 1.2em; color: #28a745; font-weight: bold;"></span>
                        </div>
                        <div style="margin-bottom: 10px;">
                            <strong>Final Product:</strong> <span id="finalProd" style="font-weight: 600;"></span>
                        </div>
                        <div style="margin-bottom: 10px;">
                            <strong>Full Path:</strong><br>
                            <span id="path" style="color: #666; font-size: 0.95em;"></span>
                        </div>
                        <div style="margin-bottom: 10px;">
                            <strong>Confidence:</strong> 
                            <span id="confidence" class="tag"></span>
                        </div>
                        <div style="font-size: 0.9em; color: #666;">
                            <strong>Depth:</strong> <span id="depth"></span> levels | 
                            <strong>Time:</strong> <span id="time"></span>ms
                        </div>
                    </div>
                </div>

                <div class="section">
                    <h3>🔗 Matched Terms (Cross-Store Variations)</h3>
                    <div id="matchedTerms"></div>
                </div>

                <div class="section">
                    <h3>📋 Top 5 Alternative Matches</h3>
                    <div id="alternatives"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        async function classify() {
            const title = document.getElementById('title').value.trim();
            const desc = document.getElementById('desc').value.trim();

            if (!title) {
                alert('Please enter a product title');
                return;
            }

            document.getElementById('loading').classList.add('show');
            document.getElementById('results').classList.remove('show');

            try {
                const response = await fetch('/classify', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ title, description: desc })
                });

                if (!response.ok) throw new Error('Classification failed');

                const data = await response.json();
                displayResults(data);
            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                document.getElementById('loading').classList.remove('show');
            }
        }

        function displayResults(data) {
            document.getElementById('results').classList.add('show');

            document.getElementById('product').textContent = data.product;
            document.getElementById('catId').textContent = data.category_id;
            document.getElementById('finalProd').textContent = data.final_product;
            document.getElementById('path').textContent = data.category_path;
            document.getElementById('depth').textContent = data.depth;
            document.getElementById('time').textContent = data.processing_time_ms;

            const conf = document.getElementById('confidence');
            conf.textContent = data.confidence;
            const confClass = data.confidence.split(' ')[0].toLowerCase().replace('_', '-');
            conf.className = 'tag conf-' + confClass;

            const matchedHtml = data.matched_terms.map(t => `<span class="tag">${t}</span>`).join('');
            document.getElementById('matchedTerms').innerHTML = matchedHtml;

            let altHtml = '';
            data.top_5_results.forEach((item, i) => {
                const cls = i === 0 ? 'style="background: #e8f5e9;"' : '';
                altHtml += `
                    <div class="result-item" ${cls}>
                        <strong>${item.rank}.</strong> ${item.final_product} 
                        <span class="tag" style="background: #999;">${item.confidence}%</span>
                        <div style="font-size: 0.85em; color: #666; margin-top: 5px;">
                            ID: ${item.category_id}
                        </div>
                    </div>
                `;
            });
            document.getElementById('alternatives').innerHTML = altHtml;
        }

        document.getElementById('title').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') classify();
        });
    </script>
</body>
</html>
"""

# ============================================================================
# FLASK APP
# ============================================================================

app = Flask(__name__)


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/classify', methods=['POST'])
def classify_route():
    data = request.get_json(force=True)
    title = data.get('title', '').strip()
    description = data.get('description', '').strip()

    if not title:
        return jsonify({'error': 'Title required'}), 400

    try:
        result = classify_product(title, description)
        return jsonify(result)
    except Exception as e:
        app.logger.exception('Classification error')
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'categories': len(metadata),
        'cross_store_synonyms': len(cross_store_synonyms),
        'model': MODEL_NAME
    })


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    try:
        load_server()
        print('\n🌐 Server starting...')
        print('   URL: http://localhost:5000')
        print('   Press CTRL+C to stop\n')
        # Recommended: run with a production server like gunicorn for production use
        app.run(host='0.0.0.0', port=5000, debug=False)
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print('\n💡 Solution: Run training first to create FAISS index and metadata')
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}\n")

