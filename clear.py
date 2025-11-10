"""
🧹 PROJECT CLEANUP SCRIPT
=========================
Removes unnecessary modules from the category prediction project.
Keeps only essential files for production API server.

Usage:
    python cleanup_project.py --dry-run  # Preview what will be deleted
    python cleanup_project.py            # Actually delete files
"""

import os
import shutil
from pathlib import Path
import argparse

# Files to DELETE (not needed for production API)
FILES_TO_DELETE = [
    # Core modules (only keep what's needed)
    'core/batch_classifier.py',        # Redundant - API has batch endpoint
    'core/model_ensemble.py',          # Complex ensemble not needed
    'core/text_enhancer.py',           # Text enhancement not in use
    
    # Strategy modules (not used by simple API)
    'strategies/__init__.py',
    'strategies/direct.py',
    'strategies/ensemble.py',
    'strategies/progressive.py',
    
    # Data cleaning (only needed once)
    'data_cleaner.py',                 # One-time use
    
    # Synonym management (keep only if regenerating)
    'synonyms.py',                     # Old version
    
    # Documentation
    'Readme.md',                       # Optional - can keep if you want
    
    # Scripts
    'setup.sh',                        # Linux setup (if on Windows only)
    'tags.py',                         # Tag generation (one-time use)
]

# Directories to DELETE
DIRS_TO_DELETE = [
    'strategies',                      # Not used by simple API
]

# KEEP these essential files
ESSENTIAL_FILES = [
    'api_server.py',                   # Main API server ✅
    'core/__init__.py',               # Package init
    'core/embedding_engine.py',        # Core embedding logic
    'core/search_builder.py',          # Search query building
    'data/categories.csv',             # Category data
    'data/category_id_path_only.csv', # Training data
    'cache/main_index.faiss',          # FAISS index
    'cache/metadata.pkl',              # Metadata
    'cache/embeddings.npy',            # Embeddings
    'cache/cross_store_synonyms.pkl',  # Synonyms
    'cache/model_info.json',           # Model info
    'requirements.txt',                # Dependencies
    '.gitignore',                      # Git config
]


def analyze_project():
    """Analyze project structure"""
    print("\n" + "="*80)
    print("📊 PROJECT ANALYSIS")
    print("="*80 + "\n")
    
    total_size = 0
    file_count = 0
    
    for root, dirs, files in os.walk('.'):
        # Skip cache and __pycache__
        if '__pycache__' in root or '.git' in root:
            continue
            
        for file in files:
            filepath = Path(root) / file
            try:
                size = filepath.stat().st_size
                total_size += size
                file_count += 1
            except:
                pass
    
    print(f"📁 Total files: {file_count}")
    print(f"💾 Total size: {total_size / (1024*1024):.2f} MB")
    print()


def preview_cleanup():
    """Preview what will be deleted"""
    print("\n" + "="*80)
    print("🔍 CLEANUP PREVIEW (DRY RUN)")
    print("="*80 + "\n")
    
    delete_count = 0
    delete_size = 0
    
    print("📋 Files to DELETE:\n")
    for file in FILES_TO_DELETE:
        filepath = Path(file)
        if filepath.exists():
            try:
                size = filepath.stat().st_size
                delete_size += size
                delete_count += 1
                print(f"   ❌ {file} ({size / 1024:.1f} KB)")
            except Exception as e:
                print(f"   ⚠️  {file} (error: {e})")
        else:
            print(f"   ⏭️  {file} (not found)")
    
    print(f"\n📂 Directories to DELETE:\n")
    for dir_name in DIRS_TO_DELETE:
        dirpath = Path(dir_name)
        if dirpath.exists() and dirpath.is_dir():
            try:
                # Calculate directory size
                dir_size = sum(f.stat().st_size for f in dirpath.rglob('*') if f.is_file())
                delete_size += dir_size
                file_count = len(list(dirpath.rglob('*')))
                print(f"   ❌ {dir_name}/ ({file_count} files, {dir_size / 1024:.1f} KB)")
            except Exception as e:
                print(f"   ⚠️  {dir_name}/ (error: {e})")
        else:
            print(f"   ⏭️  {dir_name}/ (not found)")
    
    print(f"\n📊 Summary:")
    print(f"   Files to delete: {delete_count}")
    print(f"   Space to free: {delete_size / 1024:.1f} KB")
    print()


def perform_cleanup(dry_run=True):
    """Perform the actual cleanup"""
    if dry_run:
        preview_cleanup()
        print("💡 Run without --dry-run to actually delete files\n")
        return
    
    print("\n" + "="*80)
    print("🗑️  PERFORMING CLEANUP")
    print("="*80 + "\n")
    
    deleted_count = 0
    
    # Delete files
    print("📋 Deleting files...\n")
    for file in FILES_TO_DELETE:
        filepath = Path(file)
        if filepath.exists():
            try:
                filepath.unlink()
                print(f"   ✅ Deleted: {file}")
                deleted_count += 1
            except Exception as e:
                print(f"   ❌ Failed: {file} ({e})")
        else:
            print(f"   ⏭️  Skipped: {file} (not found)")
    
    # Delete directories
    print(f"\n📂 Deleting directories...\n")
    for dir_name in DIRS_TO_DELETE:
        dirpath = Path(dir_name)
        if dirpath.exists() and dirpath.is_dir():
            try:
                shutil.rmtree(dirpath)
                print(f"   ✅ Deleted: {dir_name}/")
                deleted_count += 1
            except Exception as e:
                print(f"   ❌ Failed: {dir_name}/ ({e})")
        else:
            print(f"   ⏭️  Skipped: {dir_name}/ (not found)")
    
    print(f"\n✅ Cleanup complete! Deleted {deleted_count} items\n")


def verify_essentials():
    """Verify essential files are present"""
    print("\n" + "="*80)
    print("🔍 VERIFYING ESSENTIAL FILES")
    print("="*80 + "\n")
    
    missing = []
    present = []
    
    for file in ESSENTIAL_FILES:
        filepath = Path(file)
        if filepath.exists():
            present.append(file)
        else:
            missing.append(file)
    
    if present:
        print(f"✅ Found {len(present)} essential files\n")
    
    if missing:
        print(f"⚠️  Missing {len(missing)} essential files:\n")
        for file in missing:
            print(f"   ❌ {file}")
        print()
    else:
        print("✅ All essential files present!\n")


def main():
    parser = argparse.ArgumentParser(description='Clean up unnecessary project files')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Preview what will be deleted without actually deleting')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze project structure')
    parser.add_argument('--verify', action='store_true',
                       help='Verify essential files are present')
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_project()
        return
    
    if args.verify:
        verify_essentials()
        return
    
    # Default: perform cleanup
    if args.dry_run:
        print("\n⚠️  DRY RUN MODE - No files will be deleted\n")
        analyze_project()
        preview_cleanup()
        verify_essentials()
    else:
        print("\n⚠️  WARNING: This will permanently delete files!")
        response = input("Are you sure you want to continue? (yes/no): ")
        if response.lower() == 'yes':
            perform_cleanup(dry_run=False)
            verify_essentials()
        else:
            print("\n❌ Cleanup cancelled\n")


if __name__ == "__main__":
    main()