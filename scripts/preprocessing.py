import os
import re

import pandas as pd
from sklearn.model_selection import train_test_split


class Preprocessor:
    """
    Handles data loading, cleaning, and splitting.
    """

    def __init__(self, text_column="Content", min_length=50):
        self.text_column = text_column
        self.min_length = min_length

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Perform general text cleaning:
        - remove HTML tags
        - remove non-ASCII characters
        - remove multiple spaces
        - strip whitespace
        """
        if not isinstance(text, str):
            return ""

        text = re.sub(r"<.*?>", " ", text)  # HTML
        text = re.sub(r"[^a-zA-Z0-9.,;:'\"!?\\s]", " ", text)  # special chars
        text = re.sub(r"\s+", " ", text)  # multi-spaces
        return text.strip()

    def load_dataset(self, path: str) -> pd.DataFrame:
        """
        Load dataset from CSV file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found at: {path}")

        df = pd.read_csv(path)
        if self.text_column not in df.columns:
            raise ValueError(f"Column {self.text_column} not found in dataset")

        print(f"Loaded dataset with {len(df)} rows.")
        return df

    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the text column and filter extremely short entries.
        """
        df["clean_text"] = df[self.text_column].astype(str).apply(self.clean_text)
        df = df[df["clean_text"].str.len() >= self.min_length].reset_index(drop=True)
        return df

    def split_dataset(self, df: pd.DataFrame, test_size=0.1, val_size=0.1):
        """
        Split into train/validation/test.
        """
        train_df, temp_df = train_test_split(df, test_size=test_size + val_size, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=test_size / (test_size + val_size), random_state=42)

        return train_df, val_df, test_df

    def save_splits(self, train_df, val_df, test_df, out_dir="data/processed"):
        """
        Save split datasets.
        """
        os.makedirs(out_dir, exist_ok=True)

        train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(out_dir, "val.csv"), index=False)
        test_df.to_csv(os.path.join(out_dir, "test.csv"), index=False)

        print(f"Saved processed datasets to: {out_dir}")
