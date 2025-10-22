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

clf = create_classifier('data/categories.json')

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
