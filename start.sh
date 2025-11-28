#!/bin/bash

# Pre-download NLTK assets (safe for CPU)
python3 - << 'EOF'
import nltk
nltk.download('punkt')
nltk.download('stopwords')
EOF

# Start FastAPI using Gunicorn + Uvicorn workers
gunicorn app:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --timeout 600 --bind 0.0.0.0:$PORT
