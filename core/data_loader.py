# core/data_loader.py
"""
Load and parse YOUR insurance category format
Handles Category_path + Level 1-10 columns
"""

import json
import pandas as pd
from pathlib import Path
from typing import Tuple
from tqdm import tqdm

class InsuranceCategoryLoader:
    """
    Load insurance categories in YOUR format
    
    Expected JSON format:
    [
        {
            "Category_ID": "510192",
            "Category_path": "Appliances/Heating, Cooling & Air Quality/Air Purifiers/HEPA Air Purifiers",
            "Top-Level Category (Level 1)": "Appliances",
            "Sub Category (Level 2)": "Heating, Cooling & Air Quality",
            "Product Category (Level 3)": "Air Purifiers",
            "Product Category (Level 4)": "HEPA Air Purifiers",
            "Product Category (Level 5)": null,
            ...
        }
    ]
    """
    
    @staticmethod
    def load(json_path: Path) -> Tuple[pd.DataFrame, int]:
        """
        Load and parse categories
        
        Returns:
            (dataframe, max_depth)
        """
        print(f"\n{'='*70}")
        print(f"📂 LOADING INSURANCE CATEGORIES")
        print(f"{'='*70}\n")
        print(f"File: {json_path}")
        
        # Load JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        print(f"✅ Loaded {len(df):,} categories")
        
        # Clean and parse
        df = InsuranceCategoryLoader._clean_data(df)
        df, max_depth = InsuranceCategoryLoader._parse_levels(df)
        df = InsuranceCategoryLoader._prepare_text(df, max_depth)
        
        # Statistics
        InsuranceCategoryLoader._print_stats(df, max_depth)
        
        return df, max_depth
    
    @staticmethod
    def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize data"""
        
        print("\n🧹 Cleaning data...")
        
        # Ensure Category_ID is string
        df['Category_ID'] = df['Category_ID'].astype(str)
        
        # Clean Category_path
        df['Category_path'] = df['Category_path'].fillna('').astype(str).str.strip()
        
        # Remove rows with empty Category_path
        df = df[df['Category_path'] != ''].copy()
        
        print(f"✅ Cleaned: {len(df):,} valid categories")
        return df
    
    @staticmethod
    def _parse_levels(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """
        Parse all levels dynamically
        
        TECHNIQUE: DUAL PARSING
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        1. Parse from Category_path (split by /)
        2. Use existing Level columns if available
        3. Merge both for complete coverage
        """
        
        print("\n🔍 Parsing hierarchy levels...")
        
        # Method 1: Parse from Category_path
        max_depth_path = 0
        for path in df['Category_path']:
            depth = len(path.split('/'))
            max_depth_path = max(max_depth_path, depth)
        
        # Method 2: Check existing level columns
        level_cols = [col for col in df.columns if 'Level' in col or 'Category (Level' in col]
        max_depth_cols = len(level_cols)
        
        # Use the maximum
        max_depth = max(max_depth_path, max_depth_cols, 10)  # At least 10
        
        print(f"   Max depth from path: {max_depth_path}")
        print(f"   Max depth from columns: {max_depth_cols}")
        print(f"   Final max depth: {max_depth}")
        
        # Create standardized level columns
        for i in range(1, max_depth + 1):
            df[f'level_{i}'] = None
            df[f'level_{i}_path'] = None
        
        df['depth'] = 0
        
        # Parse each category
        for idx in tqdm(df.index, desc="   Parsing categories"):
            path = df.at[idx, 'Category_path']
            parts = path.split('/')
            
            df.at[idx, 'depth'] = len(parts)
            
            # Populate levels from path
            for i, part in enumerate(parts, 1):
                if i <= max_depth:
                    df.at[idx, f'level_{i}'] = part.strip()
                    df.at[idx, f'level_{i}_path'] = '/'.join(parts[:i])
            
            # Also use existing level columns if available
            for i in range(1, max_depth + 1):
                # Check various column name formats
                possible_names = [
                    f'Top-Level Category (Level {i})',
                    f'Sub Category (Level {i})',
                    f'Product Category (Level {i})',
                    f'Level {i}'
                ]
                
                for col_name in possible_names:
                    if col_name in df.columns:
                        existing_value = df.at[idx, col_name]
                        if pd.notna(existing_value) and str(existing_value).strip():
                            # Use existing value if current is empty
                            if pd.isna(df.at[idx, f'level_{i}']):
                                df.at[idx, f'level_{i}'] = str(existing_value).strip()
        
        print("✅ Hierarchy parsed")
        return df, max_depth
    
    @staticmethod
    def _prepare_text(df: pd.DataFrame, max_depth: int) -> pd.DataFrame:
        """
        Prepare enhanced text for embeddings
        
        TECHNIQUE: LEVEL REPETITION
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        Repeat deeper levels to give them more weight in embeddings
        
        Example:
        Input:  "Appliances/Laundry Appliances/Washers & Dryers"
        Output: "Appliances Laundry Appliances Laundry Appliances 
                 Washers & Dryers Washers & Dryers Washers & Dryers"
        
        Why: Deep levels are more specific and should have higher weight
        """
        
        print("\n📝 Preparing enhanced text...")
        
        text_series = df['Category_path'].copy()
        
        # Add repeated levels
        for i in range(1, max_depth + 1):
            level_col = f'level_{i}'
            if level_col in df.columns:
                # Repetition factor: more for deeper levels
                # Level 1: 1x, Level 2: 2x, Level 3+: 3x
                repetitions = min(i, 3)
                
                level_text = df[level_col].fillna('').apply(
                    lambda x: (' ' + str(x)) * repetitions if x and str(x).strip() else ''
                )
                
                text_series = text_series + level_text
        
        df['enhanced_text'] = text_series
        
        print("✅ Text prepared")
        return df
    
    @staticmethod
    def _print_stats(df: pd.DataFrame, max_depth: int):
        """Print statistics"""
        
        print(f"\n📊 Category Statistics:")
        print(f"   Total categories: {len(df):,}")
        print(f"   Maximum depth: {max_depth} levels")
        print(f"\n   Depth distribution:")
        
        for depth in sorted(df['depth'].unique()):
            if depth > 0:
                count = (df['depth'] == depth).sum()
                pct = count / len(df) * 100
                bar = '█' * int(pct / 2)
                print(f"   {depth:2d} levels: {count:5,} ({pct:5.1f}%) {bar}")