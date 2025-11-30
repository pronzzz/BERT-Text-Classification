"""
Trainer class for fine-tuning BERT models.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    DistilBertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class Trainer:
    """Trainer for BERT fine-tuning."""
    
    def __init__(self, args, train_loader, val_loader, test_loader=None):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Model setup
        self.model = DistilBertForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=args.num_labels,
            problem_type='multi_label_classification' if args.multi_label else 'single_label_classification'
        ).to(self.device)
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # Scheduler
        total_steps = len(train_loader) * args.epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_acc = 0.0
        self.best_model_path = None
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        losses = []
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.args.epochs}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.max_grad_norm)
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            losses.append(loss.item())
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = np.mean(losses)
        self.train_losses.append(avg_loss)
        print(f"Epoch {epoch} - Train Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def evaluate(self, loader, split_name='Validation'):
        """Evaluate model on a dataset."""
        self.model.eval()
        all_preds = []
        all_labels = []
        losses = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Evaluating {split_name}"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                losses.append(loss.item())
                
                logits = outputs.logits
                
                if self.args.multi_label:
                    # Multi-label: apply sigmoid and threshold
                    probs = torch.sigmoid(logits).cpu().numpy()
                    preds = (probs >= 0.5).astype(int)
                    labels_np = labels.cpu().numpy()
                else:
                    # Single-label: take argmax
                    preds = torch.argmax(logits, dim=-1).cpu().numpy()
                    labels_np = labels.cpu().numpy()
                
                all_preds.append(preds)
                all_labels.append(labels_np)
        
        # Concatenate all predictions and labels
        if self.args.multi_label:
            all_preds = np.vstack(all_preds)
            all_labels = np.vstack(all_labels)
        else:
            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)
        
        avg_loss = np.mean(losses)
        
        # Calculate metrics
        if self.args.multi_label:
            accuracy = accuracy_score(all_labels, all_preds)
            f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
        else:
            accuracy = accuracy_score(all_labels, all_preds)
            f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
        
        print(f"\n{split_name} Results:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 (Macro): {f1_macro:.4f}")
        print(f"  F1 (Micro): {f1_micro:.4f}")
        
        # Classification report
        class_names = ['World', 'Sports', 'Business', 'Sci/Tech']
        print(f"\nClassification Report:")
        if self.args.multi_label:
            # For multi-label, flatten for report
            print(classification_report(
                all_labels.flatten(),
                all_preds.flatten(),
                target_names=class_names,
                zero_division=0
            ))
        else:
            print(classification_report(
                all_labels,
                all_preds,
                target_names=class_names,
                zero_division=0
            ))
        
        # Confusion matrix
        if not self.args.multi_label:
            cm = confusion_matrix(all_labels, all_preds)
            self.plot_confusion_matrix(cm, class_names, split_name)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'predictions': all_preds,
            'labels': all_labels
        }
    
    def plot_confusion_matrix(self, cm, class_names, split_name):
        """Plot and save confusion matrix."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title(f'Confusion Matrix - {split_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        filename = os.path.join(self.args.output_dir, f'confusion_matrix_{split_name.lower()}.png')
        plt.savefig(filename)
        print(f"Saved confusion matrix to {filename}")
        plt.close()
    
    def train(self):
        """Main training loop."""
        print(f"\nStarting training for {self.args.epochs} epochs...")
        print(f"Model: {self.args.model_name}")
        print(f"Learning rate: {self.args.learning_rate}")
        print(f"Batch size: {self.args.batch_size}")
        print(f"Max sequence length: {self.args.max_len}")
        print(f"Multi-label: {self.args.multi_label}\n")
        
        for epoch in range(1, self.args.epochs + 1):
            # Train
            self.train_epoch(epoch)
            
            # Validate
            val_results = self.evaluate(self.val_loader, 'Validation')
            self.val_losses.append(val_results['loss'])
            self.val_accuracies.append(val_results['accuracy'])
            
            # Save best model
            if val_results['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_results['accuracy']
                self.best_model_path = os.path.join(
                    self.args.output_dir,
                    f'best_model_epoch_{epoch}.pt'
                )
                self.save_checkpoint(self.best_model_path, epoch)
                print(f"Saved best model (val_acc: {self.best_val_acc:.4f})")
            
            # Save checkpoint every epoch
            checkpoint_path = os.path.join(
                self.args.output_dir,
                f'checkpoint_epoch_{epoch}.pt'
            )
            self.save_checkpoint(checkpoint_path, epoch)
        
        # Plot training history
        self.plot_training_history()
        
        # Final evaluation on test set
        if self.test_loader is not None:
            print("\n" + "="*50)
            print("Final Test Evaluation")
            print("="*50)
            test_results = self.evaluate(self.test_loader, 'Test')
        
        print(f"\nTraining completed! Best validation accuracy: {self.best_val_acc:.4f}")
        if self.best_model_path:
            print(f"Best model saved at: {self.best_model_path}")
    
    def save_checkpoint(self, path, epoch):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': self.val_accuracies[-1] if self.val_accuracies else 0.0,
            'args': self.args
        }, path)
    
    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch']
    
    def plot_training_history(self):
        """Plot training history."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(self.train_losses, label='Train Loss')
        axes[0].plot(self.val_losses, label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training History - Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy plot
        axes[1].plot(self.val_accuracies, label='Val Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training History - Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        filename = os.path.join(self.args.output_dir, 'training_history.png')
        plt.savefig(filename)
        print(f"Saved training history to {filename}")
        plt.close()


