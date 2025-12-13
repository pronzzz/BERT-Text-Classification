"""
Data preparation module for BERT text classification.
Handles dataset download, cleaning, splitting, and exploratory analysis.
"""

import os
import re
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


class DataPreparator:
    """Handles data acquisition, cleaning, and splitting."""
    
    def __init__(self, dataset_name='ag_news', output_dir='data'):
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def download_dataset(self):
        """Download AG News dataset from Kaggle."""
        print(f"Downloading {self.dataset_name} dataset from Kaggle...")
        
        # Download using Kaggle API
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files('amananandrai/ag-news-classification-dataset', path=self.output_dir, unzip=True)
        except Exception as e:
            print(f"Kaggle API failed (likely no credentials): {e}")
            print("Falling back to direct download from mirror...")
            try:
                import requests
                urls = {
                    'train.csv': 'https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv',
                    'test.csv': 'https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv'
                }
                for name, url in urls.items():
                    r = requests.get(url)
                    with open(os.path.join(self.output_dir, name), 'wb') as f:
                        f.write(r.content)
                # Mirror does not have header, load with header=None and assign names
                # Actually checking the curl output, it does NOT have a header row. It starts with data.
                # I need to handle this in read_csv
            except Exception as e2:
                raise RuntimeError(f"Failed to download dataset from Fallback: {e2}") from e

            
        # Read CSVs
        train_path = os.path.join(self.output_dir, 'train.csv')
        test_path = os.path.join(self.output_dir, 'test.csv')
        
        try:
            # Helper to read AG News format (handling potential missing header)
            def read_ag_csv(path):
                # Try reading with header inference
                df = pd.read_csv(path)
                # Check if it looks like the Kaggle format with headers
                if 'Class Index' in df.columns and 'Title' in df.columns:
                    return df
                
                # If not, assume it's the raw format without headers
                # We need to re-read because the first row was consumed as header
                df = pd.read_csv(path, header=None, names=['Class Index', 'Title', 'Description'])
                return df

            train_df = read_ag_csv(train_path)
            test_df = read_ag_csv(test_path)
        except Exception as e:
            # Fallback if filenames are different or download failed silently
            print(f"Error reading CSVs: {e}")
            print(os.listdir(self.output_dir))
            raise

        # Kaggle AG News format: Class Index, Title, Description
        # Create 'text' column by combining Title and Description
        train_df['text'] = train_df['Title'] + " " + train_df['Description']
        test_df['text'] = test_df['Title'] + " " + test_df['Description']
        
        # Rename 'Class Index' to 'label' and adjust to 0-indexed
        train_df['label'] = train_df['Class Index'] - 1
        test_df['label'] = test_df['Class Index'] - 1
        
        # Keep only relevant columns
        train_df = train_df[['text', 'label']]
        test_df = test_df[['text', 'label']]
        
        print(f"Train set: {len(train_df)} samples")
        print(f"Test set: {len(test_df)} samples")
        
        return train_df, test_df
    
    def clean_text(self, text):
        """Clean text: remove HTML, normalize punctuation, lowercase."""
        if pd.isna(text):
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', str(text))
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Lowercase
        text = text.lower().strip()
        
        return text
    
    def clean_dataset(self, df):
        """Clean entire dataset."""
        print("Cleaning dataset...")
        df = df.copy()
        df['text'] = df['text'].apply(self.clean_text)
        
        # Remove empty or very short texts
        df = df[df['text'].str.len() > 10]
        
        # Remove duplicates
        initial_len = len(df)
        df = df.drop_duplicates(subset=['text'])
        removed = initial_len - len(df)
        if removed > 0:
            print(f"Removed {removed} duplicate texts")
        
        return df
    
    def balance_classes(self, df, min_samples_per_class=None):
        """Balance classes by downsampling majority classes."""
        if min_samples_per_class is None:
            # Use minimum class count
            min_samples_per_class = df['label'].value_counts().min()
        
        print(f"Balancing classes to {min_samples_per_class} samples per class...")
        
        balanced_dfs = []
        for label in sorted(df['label'].unique()):
            label_df = df[df['label'] == label]
            if len(label_df) > min_samples_per_class:
                label_df = label_df.sample(n=min_samples_per_class, random_state=42)
            balanced_dfs.append(label_df)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return balanced_df
    
    def split_data(self, train_df, test_size=0.1, val_size=0.1):
        """Split data into train/val/test sets."""
        # First split: train + val vs test
        train_val_df, test_df = train_test_split(
            train_df, 
            test_size=test_size, 
            random_state=42,
            stratify=train_df['label']
        )
        
        # Second split: train vs val
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size / (1 - test_size),  # Adjust for already removed test set
            random_state=42,
            stratify=train_val_df['label']
        )
        
        return train_df, val_df, test_df
    
    def exploratory_analysis(self, train_df, val_df, test_df):
        """Perform exploratory data analysis."""
        print("\n=== Exploratory Data Analysis ===")
        
        # Class distribution
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for idx, (name, df) in enumerate([('Train', train_df), ('Val', val_df), ('Test', test_df)]):
            class_counts = df['label'].value_counts().sort_index()
            axes[idx].bar(class_counts.index, class_counts.values)
            axes[idx].set_title(f'{name} Set Class Distribution')
            axes[idx].set_xlabel('Class')
            axes[idx].set_ylabel('Count')
            axes[idx].set_xticks([0, 1, 2, 3])
            axes[idx].set_xticklabels(['World', 'Sports', 'Business', 'Sci/Tech'])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'class_distribution.png'))
        print("Saved class distribution plot")
        
        # Article length distribution
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for idx, (name, df) in enumerate([('Train', train_df), ('Val', val_df), ('Test', test_df)]):
            lengths = df['text'].str.len()
            axes[idx].hist(lengths, bins=50, edgecolor='black')
            axes[idx].set_title(f'{name} Set Article Length Distribution')
            axes[idx].set_xlabel('Character Count')
            axes[idx].set_ylabel('Frequency')
            axes[idx].axvline(lengths.mean(), color='r', linestyle='--', label=f'Mean: {lengths.mean():.0f}')
            axes[idx].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'length_distribution.png'))
        print("Saved length distribution plot")
        
        # Statistics
        print("\nDataset Statistics:")
        for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            print(f"\n{name} Set:")
            print(f"  Total samples: {len(df)}")
            print(f"  Class distribution: {dict(df['label'].value_counts().sort_index())}")
            print(f"  Avg text length: {df['text'].str.len().mean():.1f} chars")
            print(f"  Min text length: {df['text'].str.len().min()} chars")
            print(f"  Max text length: {df['text'].str.len().max()} chars")
    
    def prepare(self, balance=True, min_samples=30000):
        """Main preparation pipeline."""
        # Download
        train_df, test_df_original = self.download_dataset()
        
        # Clean
        train_df = self.clean_dataset(train_df)
        test_df_original = self.clean_dataset(test_df_original)
        
        # Balance if requested
        if balance:
            train_df = self.balance_classes(train_df, min_samples_per_class=min_samples)
        
        # Split
        train_df, val_df, test_df = self.split_data(train_df)
        
        # Use original test set if provided, otherwise use split test
        if len(test_df_original) > 0:
            test_df = test_df_original
        
        # Save
        train_df.to_csv(os.path.join(self.output_dir, 'train.csv'), index=False)
        val_df.to_csv(os.path.join(self.output_dir, 'val.csv'), index=False)
        test_df.to_csv(os.path.join(self.output_dir, 'test.csv'), index=False)
        
        print(f"\nSaved datasets to {self.output_dir}/")
        print(f"Train: {len(train_df)} samples")
        print(f"Val: {len(val_df)} samples")
        print(f"Test: {len(test_df)} samples")
        
        # Exploratory analysis
        self.exploratory_analysis(train_df, val_df, test_df)
        
        return train_df, val_df, test_df


if __name__ == '__main__':
    preparator = DataPreparator()
    preparator.prepare(balance=True, min_samples=30000)


