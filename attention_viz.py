"""
Attention visualization utilities for BERT models.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification


class AttentionVisualizer:
    """Visualize attention weights from BERT models."""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    def get_attention_weights(self, text, layer_idx=-1):
        """
        Extract attention weights from a specific layer.
        
        Args:
            text: Input text string
            layer_idx: Which layer to extract from (-1 for last layer)
        
        Returns:
            attention_weights: numpy array of shape (num_heads, seq_len, seq_len)
            tokens: List of token strings
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=256,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Forward pass with output_attentions=True
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )
        
        # Get attention from specified layer
        # DistilBERT has 6 layers, each with 12 attention heads
        attentions = outputs.attentions[layer_idx]  # Shape: (batch, heads, seq_len, seq_len)
        
        # Remove batch dimension
        attentions = attentions.squeeze(0)  # Shape: (heads, seq_len, seq_len)
        
        # Convert to numpy
        attention_weights = attentions.cpu().numpy()
        
        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        return attention_weights, tokens
    
    def visualize_attention_head(self, attention_weights, tokens, head_idx=0, save_path=None):
        """
        Visualize attention weights for a specific head.
        
        Args:
            attention_weights: numpy array (num_heads, seq_len, seq_len)
            tokens: List of token strings
            head_idx: Which attention head to visualize
            save_path: Path to save the figure
        """
        # Get attention for specific head
        attn = attention_weights[head_idx]
        
        # Limit to actual tokens (remove padding)
        num_tokens = len([t for t in tokens if t != '[PAD]'])
        attn = attn[:num_tokens, :num_tokens]
        tokens_display = tokens[:num_tokens]
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            attn,
            xticklabels=tokens_display,
            yticklabels=tokens_display,
            cmap='Blues',
            cbar=True,
            square=True,
            linewidths=0.1
        )
        plt.title(f'Attention Weights - Head {head_idx}')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        plt.xticks(rotation=90, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved attention visualization to {save_path}")
        else:
            plt.show()
        plt.close()
    
    def visualize_average_attention(self, attention_weights, tokens, save_path=None):
        """
        Visualize average attention across all heads.
        
        Args:
            attention_weights: numpy array (num_heads, seq_len, seq_len)
            tokens: List of token strings
            save_path: Path to save the figure
        """
        # Average across heads
        avg_attn = np.mean(attention_weights, axis=0)
        
        # Limit to actual tokens
        num_tokens = len([t for t in tokens if t != '[PAD]'])
        avg_attn = avg_attn[:num_tokens, :num_tokens]
        tokens_display = tokens[:num_tokens]
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            avg_attn,
            xticklabels=tokens_display,
            yticklabels=tokens_display,
            cmap='Blues',
            cbar=True,
            square=True,
            linewidths=0.1
        )
        plt.title('Average Attention Weights (All Heads)')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        plt.xticks(rotation=90, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved average attention visualization to {save_path}")
        else:
            plt.show()
        plt.close()
    
    def get_attention_to_cls(self, attention_weights, tokens):
        """
        Get attention weights to the [CLS] token (first token).
        Useful for understanding what the model focuses on for classification.
        
        Args:
            attention_weights: numpy array (num_heads, seq_len, seq_len)
            tokens: List of token strings
        
        Returns:
            cls_attention: numpy array (num_heads, seq_len) - attention from each position to [CLS]
        """
        # Attention to [CLS] is the first column
        cls_attention = attention_weights[:, :, 0]  # Shape: (heads, seq_len)
        
        # Average across heads
        avg_cls_attention = np.mean(cls_attention, axis=0)
        
        return avg_cls_attention
    
    def visualize_cls_attention(self, text, save_path=None):
        """
        Visualize which tokens attend most to [CLS] token.
        This shows what the model considers important for classification.
        """
        attention_weights, tokens = self.get_attention_weights(text)
        cls_attention = self.get_attention_to_cls(attention_weights, tokens)
        
        # Limit to actual tokens
        num_tokens = len([t for t in tokens if t != '[PAD]'])
        cls_attention = cls_attention[:num_tokens]
        tokens_display = tokens[:num_tokens]
        
        # Create bar plot
        plt.figure(figsize=(14, 6))
        indices = np.argsort(cls_attention)[-20:]  # Top 20 tokens
        plt.barh(range(len(indices)), cls_attention[indices])
        plt.yticks(range(len(indices)), [tokens_display[i] for i in indices])
        plt.xlabel('Attention Weight to [CLS]')
        plt.title('Top 20 Tokens by Attention to [CLS] Token')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved CLS attention visualization to {save_path}")
        else:
            plt.show()
        plt.close()
    
    def analyze_example(self, text, output_dir='attention_viz'):
        """
        Comprehensive attention analysis for a single example.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Analyzing attention for text: {text[:100]}...")
        
        attention_weights, tokens = self.get_attention_weights(text)
        
        # Visualize average attention
        self.visualize_average_attention(
            attention_weights,
            tokens,
            save_path=os.path.join(output_dir, 'average_attention.png')
        )
        
        # Visualize CLS attention
        self.visualize_cls_attention(
            text,
            save_path=os.path.join(output_dir, 'cls_attention.png')
        )
        
        # Visualize a few individual heads
        for head_idx in [0, 3, 6, 11]:  # Sample different heads
            self.visualize_attention_head(
                attention_weights,
                tokens,
                head_idx=head_idx,
                save_path=os.path.join(output_dir, f'head_{head_idx}_attention.png')
            )


