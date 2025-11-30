"""
Main training script for BERT text classification.
"""

import os
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast
from dataset import NewsDataset, collate_fn
from trainer import Trainer
from attention_viz import AttentionVisualizer


def load_data(data_dir):
    """Load train, validation, and test datasets."""
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    val_df = pd.read_csv(os.path.join(data_dir, 'val.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    
    train_texts = train_df['text'].tolist()
    train_labels = train_df['label'].tolist()
    
    val_texts = val_df['text'].tolist()
    val_labels = val_df['label'].tolist()
    
    test_texts = test_df['text'].tolist()
    test_labels = test_df['label'].tolist()
    
    return (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels)


def main():
    parser = argparse.ArgumentParser(description='BERT Text Classification')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing train.csv, val.csv, test.csv')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='distilbert-base-uncased',
                        help='Pretrained model name')
    parser.add_argument('--num_labels', type=int, default=4,
                        help='Number of classification labels')
    parser.add_argument('--multi_label', action='store_true',
                        help='Use multi-label classification')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=100,
                        help='Number of warmup steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum gradient norm for clipping')
    parser.add_argument('--max_len', type=int, default=256,
                        help='Maximum sequence length')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save outputs')
    
    # Attention visualization
    parser.add_argument('--viz_attention', action='store_true',
                        help='Generate attention visualizations after training')
    parser.add_argument('--viz_examples', type=int, default=3,
                        help='Number of examples to visualize attention for')
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels) = load_data(args.data_dir)
    
    print(f"Train samples: {len(train_texts)}")
    print(f"Val samples: {len(val_texts)}")
    print(f"Test samples: {len(test_texts)}")
    
    # Initialize tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_name)
    
    # Create datasets
    train_dataset = NewsDataset(
        train_texts, train_labels, tokenizer, args.max_len, args.multi_label
    )
    val_dataset = NewsDataset(
        val_texts, val_labels, tokenizer, args.max_len, args.multi_label
    )
    test_dataset = NewsDataset(
        test_texts, test_labels, tokenizer, args.max_len, args.multi_label
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Initialize trainer
    trainer = Trainer(args, train_loader, val_loader, test_loader)
    
    # Train
    trainer.train()
    
    # Load best model for attention visualization
    if args.viz_attention and trainer.best_model_path:
        print("\n" + "="*50)
        print("Generating Attention Visualizations")
        print("="*50)
        
        # Load best model
        trainer.load_checkpoint(trainer.best_model_path)
        
        # Initialize visualizer
        visualizer = AttentionVisualizer(
            trainer.model,
            tokenizer,
            trainer.device
        )
        
        # Visualize attention for a few examples
        viz_dir = os.path.join(args.output_dir, 'attention_visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Sample examples from test set
        import random
        random.seed(42)
        sample_indices = random.sample(range(len(test_texts)), min(args.viz_examples, len(test_texts)))
        
        for idx, example_idx in enumerate(sample_indices):
            example_text = test_texts[example_idx]
            example_label = test_labels[example_idx]
            
            class_names = ['World', 'Sports', 'Business', 'Sci/Tech']
            print(f"\nExample {idx + 1}:")
            print(f"  Text: {example_text[:100]}...")
            print(f"  True Label: {class_names[example_label]}")
            
            example_dir = os.path.join(viz_dir, f'example_{idx + 1}')
            visualizer.analyze_example(example_text, output_dir=example_dir)
        
        print(f"\nAttention visualizations saved to {viz_dir}")


if __name__ == '__main__':
    main()


