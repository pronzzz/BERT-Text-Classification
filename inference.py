"""
Inference script for making predictions with a trained BERT model.
"""

import os
import argparse
import torch
import pandas as pd
from transformers import DistilBertTokenizerFast
from dataset import NewsDataset
from trainer import Trainer
from torch.utils.data import DataLoader


def load_model(checkpoint_path, device, args):
    """Load a trained model from checkpoint."""
    # Initialize model
    from transformers import DistilBertForSequenceClassification
    
    model = DistilBertForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=args.num_labels,
        problem_type='multi_label_classification' if args.multi_label else 'single_label_classification'
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"Validation accuracy: {checkpoint.get('val_acc', 'N/A'):.4f}")
    
    return model


def predict_text(model, tokenizer, text, device, max_len=256, multi_label=False):
    """Predict class for a single text."""
    # Tokenize
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=max_len,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
    
    if multi_label:
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        predictions = (probs >= 0.5).astype(int)
        return predictions, probs
    else:
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        prediction = torch.argmax(logits, dim=-1).cpu().numpy()[0]
        return prediction, probs


def main():
    parser = argparse.ArgumentParser(description='BERT Text Classification Inference')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model_name', type=str, default='distilbert-base-uncased',
                        help='Pretrained model name')
    parser.add_argument('--num_labels', type=int, default=4,
                        help='Number of classification labels')
    parser.add_argument('--multi_label', action='store_true',
                        help='Use multi-label classification')
    parser.add_argument('--max_len', type=int, default=256,
                        help='Maximum sequence length')
    
    # Input options
    parser.add_argument('--text', type=str, default=None,
                        help='Single text to classify')
    parser.add_argument('--file', type=str, default=None,
                        help='CSV file with texts to classify (must have "text" column)')
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help='Output file for predictions')
    
    args = parser.parse_args()
    
    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.checkpoint, device, args)
    
    # Load tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_name)
    
    # Class names
    class_names = ['World', 'Sports', 'Business', 'Sci/Tech']
    
    # Single text prediction
    if args.text:
        print(f"\nClassifying text: {args.text[:100]}...")
        prediction, probs = predict_text(model, tokenizer, args.text, device, args.max_len, args.multi_label)
        
        print("\nPrediction Results:")
        if args.multi_label:
            print("Predicted classes:")
            for i, (name, pred) in enumerate(zip(class_names, prediction)):
                if pred == 1:
                    print(f"  - {name}: {probs[i]:.4f}")
        else:
            print(f"Predicted class: {class_names[prediction]}")
            print("\nClass probabilities:")
            for name, prob in zip(class_names, probs):
                print(f"  {name}: {prob:.4f}")
    
    # File prediction
    elif args.file:
        print(f"\nLoading texts from {args.file}...")
        df = pd.read_csv(args.file)
        
        if 'text' not in df.columns:
            raise ValueError("CSV file must have a 'text' column")
        
        predictions = []
        probabilities = []
        
        for idx, text in enumerate(df['text']):
            if idx % 100 == 0:
                print(f"Processing {idx}/{len(df)}...")
            
            prediction, probs = predict_text(model, tokenizer, text, device, args.max_len, args.multi_label)
            
            if args.multi_label:
                predictions.append(prediction)
                probabilities.append(probs)
            else:
                predictions.append(prediction)
                probabilities.append(probs)
        
        # Add predictions to dataframe
        if args.multi_label:
            for i, name in enumerate(class_names):
                df[f'pred_{name}'] = [p[i] for p in predictions]
                df[f'prob_{name}'] = [p[i] for p in probabilities]
        else:
            df['predicted_label'] = predictions
            df['predicted_class'] = [class_names[p] for p in predictions]
            for i, name in enumerate(class_names):
                df[f'prob_{name}'] = [p[i] for p in probabilities]
        
        # Save results
        df.to_csv(args.output, index=False)
        print(f"\nPredictions saved to {args.output}")
        
        # Print summary
        if not args.multi_label:
            print("\nPrediction Summary:")
            print(df['predicted_class'].value_counts())
    
    else:
        print("Please provide either --text or --file argument")
        parser.print_help()


if __name__ == '__main__':
    main()


