"""
Script to create a smaller training dataset from raw data.
Reads from raw dataset, preprocesses, splits, and limits the number of entries.
Useful for faster testing and development.
"""
import os
import sys
from pathlib import Path
import pandas as pd

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.preprocessing import Preprocessor


def create_small_train(raw_file="data/raw/train.csv",
                      output_dir="data/processed/filtered_v1/2k_samples",
                      num_entries=2000,
                      test_size=0.1,
                      val_size=0.1):
    """
    Create a smaller training dataset from raw data.
    
    Args:
        raw_file: Path to raw data.csv file
        output_dir: Directory to save train/val/test splits
        num_entries: Number of entries to take from raw dataset (default: 2000)
        test_size: Proportion of data for test set (default: 0.1)
        val_size: Proportion of data for validation set (default: 0.1)
    """
    # Use absolute paths
    raw_path = project_root / raw_file
    output_path = project_root / output_dir
    
    # Check if raw file exists
    if not raw_path.exists():
        print(f"❌ Raw dataset not found: {raw_path}")
        print(f"   Please ensure the raw dataset is at: {raw_path}")
        return
    
    print("\n" + "="*50)
    print("Creating Smaller Training Dataset")
    print("="*50)
    
    # -------------------------
    # 1. Load raw dataset
    # -------------------------
    print(f"\n📖 Loading raw dataset from {raw_path}...")
    pre = Preprocessor(text_column="clean_text")
    df_raw = pre.load_dataset(str(raw_path))

    print(f"   Original size: {len(df_raw)} entries")
    
    # Limit number of entries
    if num_entries > 0 and num_entries < len(df_raw):
        df_raw = df_raw.head(num_entries).copy()
        print(f"   Limited to: {len(df_raw)} entries")
    elif num_entries > 0:
        print(f"   Requested {num_entries} entries, but only {len(df_raw)} available")
        print(f"   Using all {len(df_raw)} entries")
    
    # -------------------------
    # 2. Clean and preprocess
    # -------------------------
    print(f"\n🔹 Cleaning dataset...")
    df_clean = pre.preprocess_dataframe(df_raw)
    print(f"   After cleaning: {len(df_clean)} entries")
    
    # -------------------------
    # 3. Split dataset
    # -------------------------
    print(f"\n🔹 Splitting dataset (train/val/test)...")
    train_df, val_df, test_df = pre.split_dataset(df_clean, test_size=test_size, val_size=val_size)
    
    print(f"   Train: {len(train_df)} entries ({len(train_df)/len(df_clean)*100:.1f}%)")
    print(f"   Val:   {len(val_df)} entries ({len(val_df)/len(df_clean)*100:.1f}%)")
    print(f"   Test:  {len(test_df)} entries ({len(test_df)/len(df_clean)*100:.1f}%)")
    
    # -------------------------
    # 4. Save splits
    # -------------------------
    print(f"\n💾 Saving splits to {output_path}...")
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(output_path / "train.csv", index=False)
    val_df.to_csv(output_path / "val.csv", index=False)
    test_df.to_csv(output_path / "test.csv", index=False)
    
    print(f"✅ Saved train.csv: {output_path / 'train.csv'}")
    print(f"✅ Saved val.csv:   {output_path / 'val.csv'}")
    print(f"✅ Saved test.csv:   {output_path / 'test.csv'}")
    
    print("\n" + "="*50)
    print("Dataset Creation Complete!")
    print("="*50 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create smaller training dataset from raw data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create dataset with 2000 entries (default)
  python scripts/create_small_train.py
  
  # Create dataset with 5000 entries
  python scripts/create_small_train.py --num 5000
  
  # Create dataset with 1000 entries and custom output directory
  python scripts/create_small_train.py --num 1000 --output data/processed_small
        """
    )
    parser.add_argument("--raw", type=str, default="data/raw/data.csv",
                        help="Path to raw data.csv file (default: data/raw/data.csv)")
    parser.add_argument("--output", type=str, default="data/processed",
                        help="Output directory for train/val/test splits (default: data/processed)")
    parser.add_argument("--num", type=int, default=2000,
                        help="Number of entries to take from raw dataset (default: 2000, use 0 for all)")
    parser.add_argument("--test-size", type=float, default=0.1,
                        help="Proportion of data for test set (default: 0.1)")
    parser.add_argument("--val-size", type=float, default=0.1,
                        help="Proportion of data for validation set (default: 0.1)")
    
    args = parser.parse_args()
    
    create_small_train()

