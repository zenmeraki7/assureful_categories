# core/__init__.py
"""
Core Module - Insurance Category Classifier
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Core ML components for insurance product categorization

MODULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Config
   - Configuration and settings
   - Model definitions
   - Thresholds and parameters
   - Directory setup

2. DataLoader (InsuranceCategoryLoader)
   - Load categories from JSON
   - Parse infinite hierarchy levels
   - Extract category structure
   - Data validation

3. TextEnhancer
   - Text preprocessing
   - Level repetition
   - Brand/model extraction
   - Keyword boosting

4. ModelEnsemble
   - Load 3 transformer models
   - Manage model weights
   - GPU/CPU handling

5. EmbeddingEngine
   - Generate embeddings
   - Combine triple models
   - L2 normalization
   - Cache management

6. SearchBuilder
   - Build FAISS indices
   - Main index (all categories)
   - Level-specific indices
   - Index optimization

7. Predictor
   - Main prediction logic
   - Strategy orchestration
   - Result building
   - Confidence calculation

USAGE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
from core import Config, InsuranceCategoryLoader, ModelEnsemble
from core import EmbeddingEngine, SearchBuilder, Predictor

# Or import all
from core import *
"""

# Import all core components
from .config import Config
from .data_loader import InsuranceCategoryLoader
from .text_enhancer import TextEnhancer
from .model_ensemble import ModelEnsemble
from .embedding_engine import EmbeddingEngine
from .search_builder import SearchBuilder
from .predictor import Predictor


# Define what's available when using "from core import *"
__all__ = [
    # Configuration
    'Config',
    
    # Data Loading
    'InsuranceCategoryLoader',
    
    # Text Processing
    'TextEnhancer',
    
    # Model Management
    'ModelEnsemble',
    
    # Embedding Generation
    'EmbeddingEngine',
    
    # Search Infrastructure
    'SearchBuilder',
    
    # Prediction
    'Predictor',
]


# Module metadata
__version__ = '1.0.0'
__author__ = 'Insurance Classifier Team'
__description__ = 'Core ML components for insurance product categorization'


# Module-level convenience functions
def get_version():
    """Get module version"""
    return __version__


def get_available_components():
    """Get list of available components"""
    return __all__


def print_info():
    """Print module information"""
    print(f"\n{'='*70}")
    print(f"Core Module - Insurance Category Classifier")
    print(f"{'='*70}")
    print(f"Version: {__version__}")
    print(f"Components: {len(__all__)}")
    print(f"\nAvailable components:")
    for i, component in enumerate(__all__, 1):
        print(f"  {i}. {component}")
    print(f"{'='*70}\n")


# Validate imports
def _validate_imports():
    """Validate all imports are successful"""
    missing = []
    
    for component in __all__:
        try:
            globals()[component]
        except KeyError:
            missing.append(component)
    
    if missing:
        raise ImportError(
            f"Failed to import the following components: {', '.join(missing)}"
        )


# Run validation on import
try:
    _validate_imports()
except ImportError as e:
    print(f"⚠️  Warning: {str(e)}")


# Optional: Print info on import (can be disabled)
# Uncomment the line below to show info when core is imported
# print_info()