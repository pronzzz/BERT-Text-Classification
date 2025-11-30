"""
Custom Dataset class for BERT text classification.
"""

import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast


class NewsDataset(Dataset):
    """Dataset class for news article classification."""
    
    def __init__(self, texts, labels, tokenizer, max_len=256, multi_label=False):
        """
        Args:
            texts: List of text strings
            labels: List of labels (single int for single-label, list of ints for multi-label)
            tokenizer: BERT tokenizer
            max_len: Maximum sequence length
            multi_label: Whether this is multi-label classification
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.multi_label = multi_label
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        
        # Handle labels
        if self.multi_label:
            # Multi-label: convert list to tensor
            if isinstance(self.labels[idx], list):
                label_tensor = torch.FloatTensor(self.labels[idx])
            else:
                # Convert single label to one-hot
                label_tensor = torch.zeros(4)
                label_tensor[self.labels[idx]] = 1.0
        else:
            # Single-label: just the class index
            label_tensor = torch.LongTensor([self.labels[idx]]).squeeze()
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label_tensor
        }


def collate_fn(batch):
    """Custom collate function for dynamic padding (optional optimization)."""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


