"""
app/core/data_loader.py

A utility for loading, parsing, and preparing the hierarchical category data.
This refactor is fully vectorized for maximum performance.
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from dataclasses import dataclass

# Get a logger for this module
logger = logging.getLogger(__name__)

@dataclass
class CategoryData:
    """
    A simple data container for the loaded category data.
    Makes passing data cleaner than using a tuple.
    """
    dataframe: pd.DataFrame
    max_depth: int


class CategoryLoader:
    """
    Handles loading and parsing of the category JSON file.
    All methods are static as this is a stateless utility.
    """

    @staticmethod
    def load(json_path: Path) -> CategoryData:
        """
        Loads and parses the category file from disk.

        Args:
            json_path: The Path object pointing to 'categories.json'.

        Returns:
            A CategoryData object containing the DataFrame and max_depth.
        
        Raises:
            FileNotFoundError: If the categories.json file is not found.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        logger.info(f"--- 📂 Loading Category Data from {json_path.name} ---")

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            logger.critical(f"FATAL: Categories file not found at: {json_path}")
            raise
        except json.JSONDecodeError as e:
            logger.critical(f"FATAL: Failed to parse {json_path.name}. Invalid JSON. {e}")
            raise

        df = pd.DataFrame(data)
        if df.empty:
            logger.warning("categories.json was loaded but is empty.")
        else:
            logger.info(f"Loaded {len(df):,} raw category entries.")

        # --- Run processing pipeline ---
        df = CategoryLoader._clean_data(df)
        df, max_depth = CategoryLoader._parse_levels(df)
        df = CategoryLoader._prepare_text(df, max_depth)
        CategoryLoader._print_stats(df, max_depth)
        
        logger.info("--- ✅ Category Data Loaded and Parsed ---")
        return CategoryData(dataframe=df, max_depth=max_depth)

    @staticmethod
    def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Cleans and standardizes the raw DataFrame."""
        logger.info("Cleaning and standardizing data...")

        # Ensure Category_ID is string
        if 'Category_ID' in df.columns:
            df['Category_ID'] = df['Category_ID'].astype(str)
        else:
            logger.warning("No 'Category_ID' column found. Using index.")
            df['Category_ID'] = df.index.astype(str)
        
        # Clean Category_path
        if 'Category_path' not in df.columns:
             raise ValueError("FATAL: 'Category_path' column not in categories.json")
             
        df['Category_path'] = df['Category_path'].fillna('').astype(str).str.strip()
        
        # Remove rows with empty Category_path (critical for parsing)
        initial_count = len(df)
        df = df[df['Category_path'] != ''].copy()
        removed_count = initial_count - len(df)
        
        if removed_count > 0:
            logger.warning(f"Removed {removed_count:,} rows with empty 'Category_path'.")
            
        logger.info(f"Cleaned data: {len(df):,} valid categories remaining.")
        return df

    @staticmethod
    def _parse_levels(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """
        Parses all levels dynamically (vectorized).
        """
        logger.info("Parsing hierarchy levels...")

        # 1. Vectorized split and depth calculation
        split_paths = df['Category_path'].str.split('/', expand=False)
        df['depth'] = split_paths.apply(len)
        max_depth_path = df['depth'].max()

        # 2. Check existing level columns (e.g., "Level 10")
        level_cols = [col for col in df.columns if 'Level' in col]
        max_depth_cols = len(level_cols) # This is a proxy
        
        # 3. Determine final max_depth
        max_depth = max(max_depth_path, max_depth_cols, 10) # Ensure at least 10 levels
        
        logger.info(f"Max depth from path: {max_depth_path}, from cols: {max_depth_cols}. Using final: {max_depth}")

        # 4. Extract levels and paths
        for i in range(1, max_depth + 1):
            df[f'level_{i}'] = split_paths.apply(
                lambda parts: parts[i-1].strip() if len(parts) >= i else np.nan
            )
            df[f'level_{i}_path'] = split_paths.apply(
                lambda parts: '/'.join(parts[:i]) if len(parts) >= i else np.nan
            )

        # 5. (Optional) Merge with existing cols
        for i in range(1, max_depth + 1):
            possible_names = [
                f'Top-Level Category (Level {i})',
                f'Sub Category (Level {i})',
                f'Product Category (Level {i})',
                f'Level {i}'
            ]
            for col_name in possible_names:
                if col_name in df.columns:
                    mask = df[f'level_{i}'].isna() & df[col_name].notna()
                    if mask.any():
                        df.loc[mask, f'level_{i}'] = df.loc[mask, col_name].astype(str).str.strip()

        logger.info("Hierarchy parsing complete.")
        return df, max_depth

    @staticmethod
    def _prepare_text(df: pd.DataFrame, max_depth: int) -> pd.DataFrame:
        """
        Prepares enhanced text for embeddings using fully vectorized operations.
        """
        logger.info("Preparing enhanced text (fully vectorized)...")

        # 1. Create the 'A → B → C' hierarchy string
        level_cols = [f'level_{i}' for i in range(1, max_depth + 1)]
        
        # Start with level 1
        hierarchy_str = df['level_1'].fillna('')
        
        # Vectorially concatenate subsequent levels with ' → '
        for i in range(2, max_depth + 1):
            col = f'level_{i}'
            if col in df.columns:
                # Only add ' → ' and the level text if the level is not null
                hierarchy_str += (' → ' + df[col]).where(df[col].notna(), '')

        # 2. Get the leaf category (e.g., 'C' from 'A → B → C')
        # We use numpy's take_along_axis for an advanced vectorized lookup
        
        level_cols_present = [col for col in level_cols if col in df.columns]
        if not level_cols_present:
             logger.warning("No 'level_x' columns found. Using Category_path as enhanced_text.")
             df['enhanced_text'] = df['Category_path']
             return df
             
        level_values = df[level_cols_present].to_numpy(na_value='')
        
        # Get the index for the leaf node (depth - 1), clamped to max available
        max_idx = level_values.shape[1] - 1
        depth_indices = (df['depth'] - 1).clip(0, max_idx).to_numpy().reshape(-1, 1)

        leaf_category = np.take_along_axis(level_values, depth_indices, axis=1).squeeze()

        # 3. Build the final structured prompt
        df['enhanced_text'] = (
            'Insurance Category: ' + hierarchy_str +
            '\nHierarchy Level: ' + df['depth'].astype(str) +
            '\nLeaf Category: ' + leaf_category
        )
        
        df['enhanced_text'] = df['enhanced_text'].str.strip()
        
        logger.info("Structured prompt text prepared.")
        return df

    @staticmethod
    def _print_stats(df: pd.DataFrame, max_depth: int):
        """Logs statistics about the loaded categories."""
        logger.info("--- 📊 Category Statistics ---")
        logger.info(f"Total categories: {len(df):,}")
        logger.info(f"Maximum depth: {max_depth} levels")
        
        depth_counts = df['depth'].value_counts().sort_index()
        stats_lines = ["Depth distribution:"]
        
        for depth in depth_counts.index:
            if depth > 0:
                count = depth_counts[depth]
                pct = count / len(df) * 100
                bar = '█' * int(pct / 2.5) # Scale bar for logs
                stats_lines.append(
                    f"  {depth:2d} levels: {count:6,} ({pct:5.1f}%) {bar}"
                )
        logger.info("\n".join(stats_lines))