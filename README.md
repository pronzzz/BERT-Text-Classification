# BERT Text Classification

A comprehensive project for fine-tuning BERT (specifically DistilBERT) for multi-class news article classification. This project implements the full pipeline from data preparation to model evaluation and attention visualization.

## 📋 Project Overview

This project fine-tunes a pretrained BERT model (DistilBERT) for classifying news articles into four categories:
- **World**: International news and world events
- **Sports**: Sports news and updates
- **Business**: Business and financial news
- **Sci/Tech**: Science and technology news

## 🚀 Features

- **Data Preparation**: Automated download, cleaning, and splitting of AG News dataset
- **Model Fine-tuning**: Fine-tune DistilBERT with configurable hyperparameters
- **Comprehensive Evaluation**: Accuracy, F1-score, confusion matrix, and classification reports
- **Attention Visualization**: Visualize attention weights to understand model behavior
- **Checkpointing**: Save and load model checkpoints
- **Training History**: Track and visualize training progress

## 📁 Project Structure

```
BERT-Text-Classification/
├── data_preparation.py    # Data download, cleaning, and preprocessing
├── dataset.py             # Custom Dataset class for PyTorch
├── trainer.py             # Training loop and evaluation
├── attention_viz.py        # Attention visualization utilities
├── main.py                # Main training script
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd BERT-Text-Classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Data Preparation

First, prepare the dataset:

```bash
python data_preparation.py
```

This will:
- Download the AG News dataset from Hugging Face
- Clean the text (remove HTML, normalize, lowercase)
- Balance classes (30K samples per class by default)
- Split into train/val/test sets (80%/10%/10%)
- Generate exploratory data analysis plots

The prepared data will be saved in the `data/` directory:
- `train.csv`: Training set
- `val.csv`: Validation set
- `test.csv`: Test set

## 🎯 Training

Train the model with default settings:

```bash
python main.py --data_dir data --output_dir outputs
```

### Custom Training Options

```bash
python main.py \
    --data_dir data \
    --output_dir outputs \
    --epochs 5 \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --max_len 256 \
    --viz_attention \
    --viz_examples 5
```

### Arguments

- `--data_dir`: Directory containing train.csv, val.csv, test.csv (default: `data`)
- `--model_name`: Pretrained model name (default: `distilbert-base-uncased`)
- `--epochs`: Number of training epochs (default: 3)
- `--batch_size`: Batch size (default: 16)
- `--learning_rate`: Learning rate (default: 2e-5)
- `--max_len`: Maximum sequence length (default: 256)
- `--output_dir`: Directory to save outputs (default: `outputs`)
- `--viz_attention`: Generate attention visualizations after training
- `--viz_examples`: Number of examples to visualize (default: 3)
- `--multi_label`: Use multi-label classification (default: single-label)

## 📈 Evaluation

The training script automatically evaluates on validation and test sets. Metrics include:

- **Accuracy**: Overall classification accuracy
- **F1-Score**: Macro and micro F1-scores
- **Classification Report**: Per-class precision, recall, and F1
- **Confusion Matrix**: Visual confusion matrix for error analysis

Results are saved in the output directory:
- `confusion_matrix_validation.png`
- `confusion_matrix_test.png`
- `training_history.png`

## 🔍 Attention Visualization

To visualize attention weights, use the `--viz_attention` flag:

```bash
python main.py --data_dir data --output_dir outputs --viz_attention --viz_examples 5
```

This generates:
- Average attention across all heads
- Attention to [CLS] token (shows important words for classification)
- Individual head visualizations

Visualizations are saved in `outputs/attention_visualizations/`.

## 📝 Model Checkpoints

Model checkpoints are saved automatically:
- `checkpoint_epoch_{N}.pt`: Checkpoint after each epoch
- `best_model_epoch_{N}.pt`: Best model based on validation accuracy

To load a checkpoint:

```python
from trainer import Trainer
import torch

# Initialize trainer (requires args and data loaders)
trainer = Trainer(args, train_loader, val_loader, test_loader)
trainer.load_checkpoint('outputs/best_model_epoch_3.pt')
```

## 🎓 Understanding the Model

### DistilBERT

This project uses **DistilBERT**, a distilled version of BERT that:
- Has 40% fewer parameters than BERT
- Runs 60% faster while retaining 95% of BERT's performance
- Uses 6 transformer layers (vs 12 in BERT)
- Uses 12 attention heads per layer

### Attention Mechanism

BERT uses self-attention to understand relationships between words:
- Each word attends to all other words in the sequence
- Multiple attention heads capture different types of relationships
- The [CLS] token aggregates information for classification

## 🔧 Extensions & Improvements

### 1. Model Variants
- Try `bert-base-uncased` for higher accuracy (slower)
- Try `roberta-base` for potentially better performance
- Try `albert-base-v2` for even smaller model size

### 2. Handling Long Articles
- Implement hierarchical encoding for articles > 256 tokens
- Use Longformer or Big Bird for native long-sequence support

### 3. Data Augmentation
- Back-translation
- Synonym replacement
- Paraphrasing

### 4. Multi-Label Classification
Use the `--multi_label` flag to enable multi-label classification where articles can belong to multiple categories.

### 5. Hyperparameter Tuning
- Use Optuna or Ray Tune for automated hyperparameter search
- Experiment with different learning rates, batch sizes, and dropout rates

## 📊 Expected Results

With default settings on AG News:
- **Accuracy**: ~90-92%
- **F1-Score (Macro)**: ~90-92%
- **Training Time**: ~30-60 minutes on GPU (depending on hardware)

## 🐛 Troubleshooting

### Out of Memory
- Reduce `--batch_size` (e.g., 8 or 4)
- Reduce `--max_len` (e.g., 128)
- Use gradient accumulation

### Slow Training
- Use GPU if available (automatically detected)
- Increase `--batch_size` if memory allows
- Use smaller model (already using DistilBERT)

### Poor Performance
- Train for more epochs (`--epochs 5`)
- Adjust learning rate
- Check data quality and balance
- Try different model variants

## 📚 References

- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [DistilBERT: a distilled version of BERT](https://arxiv.org/abs/1910.01108)
- [Hugging Face Transformers](https://huggingface.co/transformers/)

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


