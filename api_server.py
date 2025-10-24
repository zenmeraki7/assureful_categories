# api_server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')

from classifier import create_classifier
import time

app = Flask(__name__)
CORS(app)

print('\n' + '='*70)
print('STARTING API SERVER')
print('='*70 + '\n')
print('Loading classifier (takes 2-5 minutes first time)...\n')

clf = create_classifier('data/categories_fixed.json')

print('\n' + '='*70)
print('CLASSIFIER READY')
print('='*70 + '\n')

@app.route('/')
def home():
    return '''
    <html>
    <head>
        <title>Insurance Classifier API</title>
        <style>
            body { font-family: Arial; max-width: 900px; margin: 50px auto; padding: 20px; }
            h1 { color: #2c3e50; }
            .stat { background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }
            a { color: #3498db; text-decoration: none; font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>Insurance Category Classifier API</h1>
        <div class="stat">Status: ✅ Running</div>
        <div class="stat">Categories: ''' + str(len(clf.categories_df)) + '''</div>
        <div class="stat">Max Depth: ''' + str(clf.max_depth) + ''' levels</div>
        <h2>Test Interface</h2>
        <p><a href="/test">🧪 Go to Test Page →</a></p>
    </body>
    </html>
    '''

@app.route('/test')
def test_page():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Insurance Classifier</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container {
                max-width: 800px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                padding: 40px;
            }
            h1 { color: #667eea; margin-bottom: 30px; }
            .form-group { margin-bottom: 20px; }
            label {
                display: block;
                margin-bottom: 8px;
                color: #374151;
                font-weight: 600;
            }
            input, textarea {
                width: 100%;
                padding: 12px;
                border: 2px solid #e5e7eb;
                border-radius: 8px;
                font-size: 16px;
                transition: border-color 0.3s;
            }
            input:focus, textarea:focus {
                outline: none;
                border-color: #667eea;
            }
            textarea { resize: vertical; min-height: 80px; }
            .btn-group {
                display: flex;
                gap: 10px;
                margin-bottom: 20px;
            }
            button {
                flex: 1;
                background: #667eea;
                color: white;
                padding: 15px;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s;
            }
            button:hover {
                background: #5568d3;
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
            }
            button:disabled {
                background: #9ca3af;
                cursor: not-allowed;
                transform: none;
            }
            .btn-secondary {
                background: #10b981;
            }
            .btn-secondary:hover {
                background: #059669;
            }
            #result {
                margin-top: 30px;
                padding: 25px;
                background: #f9fafb;
                border-radius: 10px;
                display: none;
            }
            .result-header {
                font-size: 1.5em;
                margin-bottom: 15px;
                color: #10b981;
            }
            .result-item {
                margin: 10px 0;
                padding: 10px;
                background: white;
                border-radius: 5px;
            }
            .result-label {
                font-weight: 600;
                color: #6b7280;
            }
            .result-value {
                color: #1f2937;
                margin-top: 5px;
            }
            .confidence-high { color: #10b981; font-weight: bold; }
            .confidence-medium { color: #f59e0b; font-weight: bold; }
            .confidence-low { color: #ef4444; font-weight: bold; }
            .confidence-very_low { color: #991b1b; font-weight: bold; }
            .insurance-yes {
                background: #d1fae5;
                border-left: 4px solid #10b981;
                padding: 15px;
                margin: 15px 0;
                border-radius: 5px;
            }
            .insurance-no {
                background: #fee2e2;
                border-left: 4px solid #ef4444;
                padding: 15px;
                margin: 15px 0;
                border-radius: 5px;
            }
            .hint {
                font-size: 0.9em;
                color: #6b7280;
                margin-top: 5px;
            }
            .tag-badge {
                display: inline-block;
                background: #e0e7ff;
                color: #4f46e5;
                padding: 4px 12px;
                border-radius: 12px;
                margin: 2px;
                font-size: 0.85em;
            }
            .boost-indicator {
                display: inline-block;
                background: #fef3c7;
                color: #92400e;
                padding: 3px 8px;
                border-radius: 4px;
                font-size: 0.8em;
                font-weight: bold;
                margin-left: 5px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧪 Test Insurance Classifier</h1>
            
            <div class="form-group">
                <label for="title">Product Title *</label>
                <input type="text" id="title" placeholder="e.g., Cricket Bat" onchange="updateTags()" />
            </div>
            
            <div class="form-group">
                <label for="description">Description (optional)</label>
                <textarea id="description" rows="3" placeholder="e.g., Professional cricket bat made from willow" onchange="updateTags()"></textarea>
            </div>
            
            <div class="form-group">
                <label for="vendor">Brand/Vendor (optional)</label>
                <input type="text" id="vendor" placeholder="e.g., Ceat" onchange="updateTags()" />
            </div>
            
            <div class="form-group">
                <label for="tags">
                    Tags <span class="boost-indicator">🔥 3X WEIGHT</span>
                </label>
                <textarea id="tags" rows="2" placeholder="Auto-generated or enter your own..."></textarea>
                <div class="hint">💡 Tags boost confidence by 20-30%! Auto-generated from title & description above.</div>
            </div>
            
            <div class="form-group">
                <label for="price">Price (optional)</label>
                <input type="number" id="price" placeholder="e.g., 999.99" step="0.01" />
            </div>
            
            <div class="btn-group">
                <button onclick="autoGenerateTags()" class="btn-secondary">🏷️ Auto-Generate Tags</button>
                <button onclick="classify()" id="classifyBtn">🔍 Classify Product</button>
            </div>
            
            <div id="result"></div>
        </div>
        
        <script>
        function updateTags() {
            const title = document.getElementById('title').value;
            const tagsField = document.getElementById('tags');
            
            if (!tagsField.value.trim() && title) {
                autoGenerateTags();
            }
        }
        
        function autoGenerateTags() {
            const title = document.getElementById('title').value.toLowerCase();
            const description = document.getElementById('description').value.toLowerCase();
            const vendor = document.getElementById('vendor').value.toLowerCase();
            
            if (!title) {
                alert('Please enter a product title first');
                return;
            }
            
            const allText = title + ' ' + description + ' ' + vendor;
            const words = allText.match(/\\b\\w+\\b/g) || [];
            
            const stopWords = new Set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'made']);
            
            const categoryKeywords = {
                'appliances': ['dishwasher', 'refrigerator', 'oven', 'microwave', 'washer', 'dryer', 'appliance'],
                'electronics': ['phone', 'laptop', 'computer', 'tablet', 'tv', 'television', 'camera', 'headphone', 'speaker'],
                'sports': ['cricket', 'bat', 'football', 'tennis', 'basketball', 'sports', 'fitness', 'exercise'],
                'gaming': ['xbox', 'playstation', 'nintendo', 'gaming', 'console', 'game'],
                'automotive': ['car', 'motorcycle', 'tire', 'battery', 'parts', 'automotive', 'inner', 'tube'],
                'kitchen': ['kitchen', 'cooking', 'chef', 'utensil', 'cookware']
            };
            
            let tags = new Set();
            
            title.split(/\\s+/).forEach(word => {
                word = word.replace(/[^a-z0-9]/g, '');
                if (word.length > 2 && !stopWords.has(word)) {
                    tags.add(word);
                }
            });
            
            if (vendor) {
                tags.add(vendor);
            }
            
            for (const [category, keywords] of Object.entries(categoryKeywords)) {
                for (const keyword of keywords) {
                    if (allText.includes(keyword)) {
                        tags.add(category);
                        tags.add(keyword);
                    }
                }
            }
            
            if (allText.includes('cricket') && allText.includes('bat')) {
                tags.add('cricket bat');
                tags.add('batting equipment');
                tags.add('team sports');
                tags.add('sports equipment');
            }
            if (allText.includes('dishwasher')) {
                tags.add('kitchen appliances');
                tags.add('appliances');
            }
            if (allText.includes('phone') || allText.includes('smartphone')) {
                tags.add('mobile phone');
                tags.add('communication device');
                tags.add('electronics');
            }
            if (allText.includes('inner') && allText.includes('tube')) {
                tags.add('inner tubes');
                tags.add('motorcycle parts');
                tags.add('tire accessories');
            }
            
            const tagArray = Array.from(tags).slice(0, 15);
            document.getElementById('tags').value = tagArray.join(', ');
            
            const tagsField = document.getElementById('tags');
            tagsField.style.borderColor = '#10b981';
            setTimeout(() => {
                tagsField.style.borderColor = '#e5e7eb';
            }, 1000);
        }
        
        async function classify() {
            const title = document.getElementById('title').value;
            const description = document.getElementById('description').value;
            const vendor = document.getElementById('vendor').value;
            const tags = document.getElementById('tags').value;
            const price = parseFloat(document.getElementById('price').value) || 0;
            
            if (!title) {
                alert('Please enter a product title');
                return;
            }
            
            const btn = document.getElementById('classifyBtn');
            const resultDiv = document.getElementById('result');
            
            btn.disabled = true;
            btn.textContent = '⏳ Classifying...';
            resultDiv.style.display = 'block';
            resultDiv.innerHTML = '<div style="text-align:center;padding:20px;color:#667eea;">⏳ Processing...</div>';
            
            try {
                const response = await fetch('/api/classify', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({title, description, vendor, tags, price})
                });
                
                const data = await response.json();
                
                if (data.success) {
                    const r = data.result;
                    let html = '<div class="result-header">✅ Classification Result</div>';
                    
                    html += '<div class="result-item">';
                    html += '<div class="result-label">Product</div>';
                    html += '<div class="result-value">' + title + '</div>';
                    html += '</div>';
                    
                    html += '<div class="result-item">';
                    html += '<div class="result-label">Category ID</div>';
                    html += '<div class="result-value"><strong>' + r.category_id + '</strong></div>';
                    html += '</div>';
                    
                    html += '<div class="result-item">';
                    html += '<div class="result-label">Category</div>';
                    html += '<div class="result-value">' + r.category_path + '</div>';
                    html += '</div>';
                    
                    html += '<div class="result-item">';
                    html += '<div class="result-label">Confidence</div>';
                    html += '<div class="result-value confidence-' + r.confidence + '">';
                    html += r.confidence.toUpperCase() + ' (' + r.similarity.toFixed(4) + ')';
                    html += '</div>';
                    html += '</div>';
                    
                    html += '<div class="result-item">';
                    html += '<div class="result-label">Details</div>';
                    html += '<div class="result-value">';
                    html += 'Depth: ' + r.depth + ' levels | ';
                    html += 'Method: ' + r.prediction_method + ' | ';
                    html += 'Time: ' + r.processing_time_ms + 'ms';
                    html += '</div>';
                    html += '</div>';
                    
                    if (tags) {
                        html += '<div class="result-item">';
                        html += '<div class="result-label">Tags Used (3X Weight! 🔥)</div>';
                        html += '<div class="result-value">';
                        tags.split(',').forEach(tag => {
                            html += '<span class="tag-badge">' + tag.trim() + '</span>';
                        });
                        html += '</div>';
                        html += '</div>';
                    }
                    
                    if (r.insurance.eligible) {
                        html += '<div class="insurance-yes">';
                        html += '<strong>🛡️ Insurance Available</strong><br>';
                        html += 'Monthly Premium: $' + r.insurance.monthly_premium + '<br>';
                        html += 'Coverage Period: ' + r.insurance.coverage_period_months + ' months<br>';
                        html += 'Total Cost: $' + r.insurance.total_coverage;
                        html += '</div>';
                    } else {
                        html += '<div class="insurance-no">';
                        html += '<strong>❌ Insurance Not Available</strong><br>';
                        if (r.confidence === 'low' || r.confidence === 'very_low') {
                            html += 'Reason: Low confidence. Try adding more descriptive tags!';
                        } else if (price <= 50) {
                            html += 'Reason: Product price too low (must be > )';
                        } else {
                            html += 'Reason: Category not eligible for insurance';
                        }
                        html += '</div>';
                    }
                    
                    resultDiv.innerHTML = html;
                } else {
                    resultDiv.innerHTML = '<div class="result-header" style="color: #ef4444;">❌ Error</div>';
                    resultDiv.innerHTML += '<div class="result-item">' + data.error + '</div>';
                }
            } catch (error) {
                resultDiv.innerHTML = '<div class="result-header" style="color: #ef4444;">❌ Error</div>';
                resultDiv.innerHTML += '<div class="result-item">' + error + '</div>';
            } finally {
                btn.disabled = false;
                btn.textContent = '🔍 Classify Product';
            }
        }
        
        document.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && e.ctrlKey) {
                classify();
            }
        });
        </script>
    </body>
    </html>
    '''

@app.route('/api/classify', methods=['POST'])
def classify():
    try:
        data = request.json
        
        start = time.time()
        result = clf.predict(
            title=data.get('title', ''),
            description=data.get('description', ''),
            tags=data.get('tags', ''),
            product_type=data.get('product_type', ''),
            vendor=data.get('vendor', '')
        )
        
        processing_time = (time.time() - start) * 1000
        
        price = data.get('price', 0)
        insurance_info = {
            'eligible': False,
            'monthly_premium': 0,
            'coverage_period_months': 24,
            'total_coverage': 0
        }
        
        if result['confidence'] in ['high', 'medium'] and price > 50:
            insurance_info['eligible'] = True
            insurance_info['monthly_premium'] = round(price * 0.007, 2)
            insurance_info['total_coverage'] = round(price * 0.007 * 24, 2)
        
        return jsonify({
            'success': True,
            'result': {
                'category_id': result['category_id'],
                'category_path': result['category_path'],
                'confidence': result['confidence'],
                'similarity': result['similarity'],
                'depth': result['depth'],
                'prediction_method': result['prediction_method'],
                'levels': result['levels'],
                'insurance': insurance_info,
                'alternatives': result.get('alternatives', []),
                'processing_time_ms': round(processing_time, 2)
            }
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'categories_loaded': len(clf.categories_df),
        'max_depth': clf.max_depth,
        'device': 'cpu'
    })

if __name__ == '__main__':
    print('SERVER RUNNING ON http://localhost:5000')
    print('Test Page: http://localhost:5000/test')
    print('Press Ctrl+C to stop\n')
    
    app.run(host='0.0.0.0', port=5000, debug=False)

# """
# Optimized API Server with Ensemble Models and Context Validation
# Fixed version with proper path handling
# """

# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import os
# import joblib
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import pandas as pd
# import faiss
# from typing import Optional, List, Dict
# import gc

# app = FastAPI(title="Category Prediction API")

# # Add CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Global variables
# fast_model = None
# accurate_model = None
# specialized_model = None
# categories_df = None
# faiss_index = None
# embeddings = None

# # Configuration
# USE_ENSEMBLE = os.getenv("USE_ENSEMBLE", "true").lower() == "true"
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# CACHE_DIR = os.path.join(BASE_DIR, "cache")

# def load_or_cache_model(pkl_path: str, hf_name: str):
#     """Load model from pickle cache if exists, else download and save."""
#     os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
    
#     if os.path.exists(pkl_path):
#         print(f"✅ Loaded model from cache: {pkl_path}")
#         return joblib.load(pkl_path)
#     else:
#         print(f"⬇️ Downloading model: {hf_name}")
#         model = SentenceTransformer(hf_name)
#         joblib.dump(model, pkl_path)
#         print(f"💾 Saved model to: {pkl_path}")
#         return model

# class PredictionRequest(BaseModel):
#     title: str
#     description: Optional[str] = ""
#     tags: Optional[str] = ""
#     product_type: Optional[str] = ""
#     vendor: Optional[str] = ""
#     external_category: Optional[str] = None
#     use_validation: Optional[bool] = True

# class EnsemblePredictor:
#     """Ensemble predictor combining multiple models"""
    
#     def __init__(self, fast_model, accurate_model, specialized_model, 
#                  categories_df, faiss_index, embeddings):
#         self.fast_model = fast_model
#         self.accurate_model = accurate_model
#         self.specialized_model = specialized_model
#         self.categories_df = categories_df
#         self.faiss_index = faiss_index
#         self.embeddings = embeddings
    
#     def enhance_product_text(self, title, description, tags, product_type, vendor):
#         """Enhance product text for better prediction"""
#         parts = [title]
        
#         if description:
#             parts.append(description)
#         if product_type:
#             parts.append(f"Type: {product_type}")
#         if vendor:
#             parts.append(f"Brand: {vendor}")
#         if tags:
#             parts.append(f"Tags: {tags}")
        
#         return " | ".join(parts)
    
#     def predict_with_ensemble(self, product_text: str, top_k: int = 5):
#         """Predict using ensemble of models"""
        
#         # Generate embeddings from all models
#         embeddings_list = []
        
#         # Fast model (lightweight, quick)
#         fast_emb = self.fast_model.encode([product_text], show_progress_bar=False)[0]
#         embeddings_list.append(fast_emb)
        
#         # Accurate model (better quality)
#         accurate_emb = self.accurate_model.encode([product_text], show_progress_bar=False)[0]
#         embeddings_list.append(accurate_emb)
        
#         # Specialized model (domain-specific)
#         specialized_emb = self.specialized_model.encode([product_text], show_progress_bar=False)[0]
#         embeddings_list.append(specialized_emb)
        
#         # Combine embeddings (weighted average)
#         # Weights: fast=0.2, accurate=0.5, specialized=0.3
#         weights = [0.2, 0.5, 0.3]
#         combined_embedding = np.average(embeddings_list, axis=0, weights=weights)
        
#         # Normalize
#         combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)
        
#         # Search in FAISS index
#         distances, indices = self.faiss_index.search(
#             np.array([combined_embedding]).astype('float32'),
#             top_k
#         )
        
#         # Format results
#         results = []
#         for dist, idx in zip(distances[0], indices[0]):
#             if idx < len(self.categories_df):
#                 category = self.categories_df.iloc[idx]
#                 similarity = float(1 / (1 + dist))  # Convert distance to similarity
                
#                 results.append({
#                     'index': int(idx),
#                     'category_id': str(category.get('Category_ID', '')),
#                     'category_path': str(category.get('Category_path', '')),
#                     'similarity': similarity,
#                     'distance': float(dist),
#                     'depth': int(category.get('depth', 0))
#                 })
        
#         return results
    
#     def predict(self, title: str, description: str = "", tags: str = "",
#                 product_type: str = "", vendor: str = "", top_k: int = 5):
#         """Main prediction method"""
        
#         # Enhance text
#         product_text = self.enhance_product_text(title, description, tags, product_type, vendor)
        
#         # Get predictions
#         results = self.predict_with_ensemble(product_text, top_k)
        
#         if not results:
#             raise ValueError("No predictions could be made")
        
#         # Build detailed result for best prediction
#         best = results[0]
#         category = self.categories_df.iloc[best['index']]
        
#         # Extract levels
#         levels = {}
#         category_path = str(category.get('Category_path', ''))
        
#         if category_path:
#             parts = category_path.split('/')
#             for i in range(1, len(parts) + 1):
#                 level_name = category.get(f'level_{i}', '')
#                 if level_name and str(level_name).strip():
#                     levels[f'level_{i}'] = {
#                         'name': str(level_name).strip(),
#                         'path': '/'.join(parts[:i]),
#                         'similarity': round(best['similarity'] * (0.98 ** (i - 1)), 4)
#                     }
        
#         result = {
#             'product_title': title,
#             'category_id': best['category_id'],
#             'category_path': best['category_path'],
#             'similarity': round(best['similarity'], 4),
#             'confidence': self._get_confidence(best['similarity']),
#             'depth': best['depth'],
#             'prediction_method': 'ensemble',
#             'levels': levels,
#             'alternatives': [
#                 {
#                     'category_path': r['category_path'],
#                     'similarity': round(r['similarity'], 4),
#                     'confidence': self._get_confidence(r['similarity'])
#                 }
#                 for r in results[1:top_k]
#             ]
#         }
        
#         return result
    
#     def _get_confidence(self, similarity: float) -> str:
#         """Get confidence level from similarity score"""
#         if similarity >= 0.85:
#             return 'high'
#         elif similarity >= 0.70:
#             return 'medium'
#         elif similarity >= 0.55:
#             return 'low'
#         else:
#             return 'very_low'

# # Global predictor instance
# predictor = None

# def find_categories_file():
#     """Find categories file in multiple possible locations"""
    
#     # Possible file names
#     file_names = [
#         'categories_fixed.json',
#         'categories.json',
#         'categories.csv',
#         'categories.pkl'
#     ]
    
#     # Possible locations
#     search_paths = [
#         BASE_DIR,                                    # Root directory
#         os.path.join(BASE_DIR, 'data'),             # data/ folder
#         os.path.join(BASE_DIR, 'cache'),            # cache/ folder
#         os.path.join(BASE_DIR, '..', 'data'),       # ../data/ folder
#     ]
    
#     # Search for files
#     for path in search_paths:
#         for file_name in file_names:
#             full_path = os.path.join(path, file_name)
#             if os.path.exists(full_path):
#                 return full_path, file_name
    
#     return None, None

# def load_categories():
#     """Load categories from JSON, CSV, or PKL file"""
    
#     file_path, file_name = find_categories_file()
    
#     if file_path is None:
#         raise FileNotFoundError(
#             "❌ Categories file not found!\n"
#             "Searched for: categories_fixed.json, categories.json, categories.csv, categories.pkl\n"
#             "In locations: root, data/, cache/, ../data/\n"
#             "Please ensure one of these files exists in your project."
#         )
    
#     print(f"📊 Loading categories from: {file_path}")
    
#     if file_name.endswith('.json'):
#         df = pd.read_json(file_path, orient="records")
#     elif file_name.endswith('.csv'):
#         df = pd.read_csv(file_path)
#     elif file_name.endswith('.pkl'):
#         df = pd.read_pickle(file_path)
#     else:
#         raise ValueError(f"Unsupported file format: {file_name}")
    
#     print(f"✅ Loaded {len(df)} categories from {file_name}")
#     return df

# @app.on_event("startup")
# async def load_resources():
#     """Load all models and data at startup"""
#     global fast_model, accurate_model, specialized_model
#     global categories_df, faiss_index, embeddings, predictor
    
#     try:
#         print("=" * 60)
#         print("🚀 Loading resources...")
#         print("=" * 60)
#         print(f"Base directory: {BASE_DIR}")
        
#         # Load models
#         print("\n📦 Loading models...")
#         fast_model = load_or_cache_model(
#             os.path.join(CACHE_DIR, "fast_model.pkl"),
#             "sentence-transformers/all-MiniLM-L6-v2"
#         )
        
#         if USE_ENSEMBLE:
#             accurate_model = load_or_cache_model(
#                 os.path.join(CACHE_DIR, "accurate_model.pkl"),
#                 "sentence-transformers/all-mpnet-base-v2"
#             )
#             specialized_model = load_or_cache_model(
#                 os.path.join(CACHE_DIR, "specialized_model.pkl"),
#                 "sentence-transformers/msmarco-distilbert-base-v4"
#             )
#         else:
#             print("⚠️ Ensemble disabled, using fast model only")
#             accurate_model = fast_model
#             specialized_model = fast_model
        
#         # Load categories with smart file detection
#         print("\n📊 Loading categories...")
#         categories_df = load_categories()
        
#         # Load FAISS index
#         print("\n🔍 Loading FAISS index...")
        
#         # Search for FAISS index in multiple locations
#         index_paths = [
#             os.path.join(BASE_DIR, 'categories_index.faiss'),
#             os.path.join(BASE_DIR, 'data', 'categories_index.faiss'),
#             os.path.join(BASE_DIR, 'cache', 'categories_index.faiss'),
#         ]
        
#         faiss_path = None
#         for path in index_paths:
#             if os.path.exists(path):
#                 faiss_path = path
#                 break
        
#         if faiss_path:
#             faiss_index = faiss.read_index(faiss_path)
#             print(f"✅ Loaded FAISS index from {faiss_path} with {faiss_index.ntotal} vectors")
#         else:
#             print("⚠️ No pre-built FAISS index found, creating from embeddings...")
            
#             # Search for embeddings
#             embeddings_paths = [
#                 os.path.join(BASE_DIR, 'embeddings.npy'),
#                 os.path.join(BASE_DIR, 'data', 'embeddings.npy'),
#                 os.path.join(BASE_DIR, 'cache', 'embeddings.npy'),
#             ]
            
#             embeddings_path = None
#             for path in embeddings_paths:
#                 if os.path.exists(path):
#                     embeddings_path = path
#                     break
            
#             if embeddings_path:
#                 embeddings = np.load(embeddings_path)
#                 print(f"✅ Loaded embeddings from {embeddings_path}")
#             else:
#                 # Generate embeddings from categories
#                 print("⚠️ No embeddings found, generating from categories...")
#                 categories_list = categories_df['Category_path'].fillna('').astype(str).tolist()
#                 embeddings = fast_model.encode(
#                     categories_list,
#                     show_progress_bar=True,
#                     batch_size=32
#                 )
#                 # Save for future use
#                 np.save(os.path.join(BASE_DIR, 'embeddings.npy'), embeddings)
#                 print(f"✅ Generated and saved embeddings")
            
#             # Create FAISS index
#             dimension = embeddings.shape[1]
#             faiss_index = faiss.IndexFlatL2(dimension)
#             faiss_index.add(embeddings.astype('float32'))
            
#             # Save for future use
#             faiss.write_index(faiss_index, os.path.join(BASE_DIR, 'categories_index.faiss'))
#             print(f"✅ Created and saved FAISS index with {faiss_index.ntotal} vectors")
        
#         # Verify FAISS index matches categories
#         if faiss_index.ntotal != len(categories_df):
#             print(f"⚠️ WARNING: FAISS index has {faiss_index.ntotal} vectors but categories has {len(categories_df)} rows")
#             print("   Regenerating index to match...")
            
#             categories_list = categories_df['Category_path'].fillna('').astype(str).tolist()
#             embeddings = fast_model.encode(categories_list, show_progress_bar=True, batch_size=32)
            
#             dimension = embeddings.shape[1]
#             faiss_index = faiss.IndexFlatL2(dimension)
#             faiss_index.add(embeddings.astype('float32'))
#             faiss.write_index(faiss_index, os.path.join(BASE_DIR, 'categories_index.faiss'))
#             print(f"✅ Regenerated FAISS index with {faiss_index.ntotal} vectors")
        
#         # Initialize predictor
#         print("\n🎯 Initializing predictor...")
#         predictor = EnsemblePredictor(
#             fast_model, accurate_model, specialized_model,
#             categories_df, faiss_index, embeddings
#         )
        
#         # Memory cleanup
#         gc.collect()
        
#         # Print memory info
#         try:
#             import psutil
#             process = psutil.Process(os.getpid())
#             memory_mb = process.memory_info().rss / (1024 * 1024)
#             print(f"\n💾 Current memory usage: {memory_mb:.2f} MB")
#         except ImportError:
#             print("\n💾 psutil not available, skipping memory check")
        
#         print("\n" + "=" * 60)
#         print("✅ All resources loaded successfully!")
#         print(f"   - Categories: {len(categories_df)}")
#         print(f"   - FAISS vectors: {faiss_index.ntotal}")
#         print(f"   - Ensemble mode: {USE_ENSEMBLE}")
#         print("=" * 60)
        
#     except Exception as e:
#         print(f"\n❌ ERROR loading resources: {e}")
#         import traceback
#         traceback.print_exc()
#         raise

# @app.get("/")
# async def root():
#     """Health check endpoint"""
#     return {
#         "status": "healthy",
#         "message": "Category Prediction API",
#         "version": "2.0",
#         "ensemble_enabled": USE_ENSEMBLE,
#         "categories_loaded": len(categories_df) if categories_df is not None else 0
#     }

# @app.get("/health")
# async def health():
#     """Detailed health check"""
#     return {
#         "status": "healthy",
#         "models_loaded": {
#             "fast": fast_model is not None,
#             "accurate": accurate_model is not None,
#             "specialized": specialized_model is not None
#         },
#         "categories_loaded": len(categories_df) if categories_df is not None else 0,
#         "faiss_index_loaded": faiss_index is not None,
#         "faiss_vectors": faiss_index.ntotal if faiss_index else 0,
#         "predictor_ready": predictor is not None
#     }

# @app.post("/predict")
# async def predict_category(request: PredictionRequest):
#     """
#     Predict category for a product
#     """
#     if predictor is None:
#         raise HTTPException(status_code=503, detail="Service not ready. Models not loaded.")
    
#     if not request.title or request.title.strip() == "":
#         raise HTTPException(status_code=400, detail="Title cannot be empty")
    
#     try:
#         # Get prediction
#         result = predictor.predict(
#             title=request.title,
#             description=request.description or "",
#             tags=request.tags or "",
#             product_type=request.product_type or "",
#             vendor=request.vendor or ""
#         )
        
#         # Apply validation if requested
#         if request.use_validation:
#             try:
#                 from core.context_validator import ContextValidator
                
#                 product_text = predictor.enhance_product_text(
#                     request.title,
#                     request.description or "",
#                     request.tags or "",
#                     request.product_type or "",
#                     request.vendor or ""
#                 )
                
#                 result = ContextValidator.validate_prediction(
#                     result,
#                     product_text=product_text,
#                     title=request.title,
#                     description=request.description or ""
#                 )
                
#                 # Add suggestion if needed
#                 if result.get('needs_review'):
#                     suggested = ContextValidator.suggest_alternative_category(
#                         product_text, request.title
#                     )
#                     if suggested:
#                         result['suggested_category_type'] = suggested
                        
#             except ImportError:
#                 result['validation'] = {'status': 'skipped', 'reason': 'Validator not available'}
        
#         # Apply external category mapping if provided
#         if request.external_category:
#             try:
#                 from core.multi_store_mapper import MultiStoreMapper
                
#                 result = MultiStoreMapper.merge_prediction_with_external(
#                     result,
#                     request.external_category,
#                     request.title,
#                     request.description or ""
#                 )
#             except ImportError:
#                 result['external_mapping'] = {'status': 'skipped', 'reason': 'Mapper not available'}
        
#         return result
        
#     except Exception as e:
#         import traceback
#         error_details = traceback.format_exc()
#         print(f"Prediction error: {error_details}")
#         raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# @app.post("/predict-batch")
# async def predict_batch(products: List[PredictionRequest]):
#     """Predict categories for multiple products"""
#     if predictor is None:
#         raise HTTPException(status_code=503, detail="Service not ready")
    
#     if len(products) > 100:
#         raise HTTPException(status_code=400, detail="Maximum 100 products per batch")
    
#     results = []
#     for product in products:
#         try:
#             result = await predict_category(product)
#             results.append(result)
#         except Exception as e:
#             results.append({
#                 "error": str(e),
#                 "title": product.title
#             })
    
#     return {"results": results, "total": len(products)}

# @app.get("/stats")
# async def get_stats():
#     """Get API statistics"""
#     try:
#         import psutil
#         process = psutil.Process(os.getpid())
#         memory_mb = process.memory_info().rss / (1024 * 1024)
#     except:
#         memory_mb = None
    
#     return {
#         "categories_total": len(categories_df) if categories_df is not None else 0,
#         "memory_usage_mb": round(memory_mb, 2) if memory_mb else "N/A",
#         "ensemble_enabled": USE_ENSEMBLE,
#         "models_loaded": {
#             "fast": fast_model is not None,
#             "accurate": accurate_model is not None,
#             "specialized": specialized_model is not None
#         },
#         "faiss_vectors": faiss_index.ntotal if faiss_index else 0
#     }

# @app.get("/categories")
# async def list_categories(
#     limit: int = 100,
#     offset: int = 0,
#     search: Optional[str] = None
# ):
#     """List categories with optional search"""
#     if categories_df is None:
#         raise HTTPException(status_code=503, detail="Categories not loaded")
    
#     limit = min(limit, 1000)
    
#     df = categories_df
    
#     # Apply search filter
#     if search:
#         mask = df['Category_path'].str.contains(search, case=False, na=False)
#         df = df[mask]
    
#     # Apply pagination
#     total = len(df)
#     df = df.iloc[offset:offset + limit]
    
#     results = []
#     for _, row in df.iterrows():
#         results.append({
#             'category_id': str(row.get('Category_ID', '')),
#             'category_path': str(row.get('Category_path', '')),
#             'depth': int(row.get('depth', 0))
#         })
    
#     return {
#         "categories": results,
#         "total": total,
#         "limit": limit,
#         "offset": offset
#     }

# # Add this at the very end of your api_server.py file, replacing the if __name__ == "__main__": section

# if __name__ == "__main__":
#     import uvicorn
#     port = int(os.environ.get("PORT", 5000))
    
#     # Show all links before starting
#     print("\n" + "=" * 75)
#     print("🚀 Category Prediction API Server - Starting...")
#     print("=" * 75)
#     print(f"📍 Directory: {BASE_DIR}")
#     print(f"🔧 Ensemble:  {USE_ENSEMBLE}")
#     print(f"🌐 Port:      {port}")
#     print("=" * 75)
#     print("\n⏳ Loading models... (wait 30-60 seconds)")
#     print("=" * 75)
    
#     # Start server first
#     import threading
#     import time
    
#     def show_links():
#         """Show links after server starts"""
#         time.sleep(3)  # Wait for server to fully start
        
#         print("\n" + "=" * 75)
#         print("✅ SERVER READY! All models loaded successfully!")
#         print("=" * 75)
        
#         print("\n🌐 MAIN URLS:")
#         print(f"   → http://127.0.0.1:{port}/")
#         print(f"   → http://localhost:{port}/")
        
#         print("\n🎯 QUICK ACCESS - CLICK OR COPY THESE LINKS:")
#         print(f"   • Test Interface:   http://127.0.0.1:{port}/test")
#         print(f"   • API Docs:         http://127.0.0.1:{port}/docs")
#         print(f"   • Health Check:     http://127.0.0.1:{port}/health")
#         print(f"   • Statistics:       http://127.0.0.1:{port}/stats")
        
#         print("\n⚡ INSTANT PREDICTION TESTS:")
#         print(f"   • iPhone Test:      http://127.0.0.1:{port}/test/predict/iPhone 13 Pro")
#         print(f"   • Samsung Test:     http://127.0.0.1:{port}/test/predict/Samsung Galaxy S21")
#         print(f"   • T-Shirt Test:     http://127.0.0.1:{port}/test/predict/Mens Cotton T-Shirt")
#         print(f"   • Sample Cases:     http://127.0.0.1:{port}/test/samples")
        
#         print("\n💡 TIPS:")
#         print("   • Ctrl+Click any link to open in browser (Windows Terminal)")
#         print("   • Or copy and paste link into your browser")
#         print("   • Start with: http://127.0.0.1:{port}/test")
#         print("   • Press Ctrl+C to stop server")
        
#         print("\n" + "=" * 75)
#         print(f"🎉 Ready to test! Open http://127.0.0.1:{port}/test in your browser")
#         print("=" * 75 + "\n")
    
#     # Start link display in background
#     threading.Thread(target=show_links, daemon=True).start()
    
#     # Start server
#     uvicorn.run(
#         app,
#         host="0.0.0.0",
#         port=port,
#         log_level="info"
#     )


