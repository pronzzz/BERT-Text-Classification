# Quick Start Guide

This guide will help you get started with the BERT Text Classification project quickly.

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Prepare Data

Download and prepare the AG News dataset:

```bash
python data_preparation.py
```

This will:
- Download the AG News dataset (~120K samples)
- Clean and preprocess the text
- Balance classes to 30K samples each
- Split into train/val/test sets
- Generate exploratory analysis plots

The prepared data will be saved in the `data/` directory.

## Step 3: Train the Model

Train with default settings (recommended for first run):

```bash
python main.py --data_dir data --output_dir outputs
```

For a more comprehensive run with attention visualization:

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

**Expected Training Time:**
- CPU: ~2-4 hours
- GPU (CUDA): ~30-60 minutes

## Step 4: Make Predictions

### Single Text Prediction

```bash
python inference.py \
    --checkpoint outputs/best_model_epoch_3.pt \
    --text "Apple announces new iPhone with advanced AI features"
```

### Batch Prediction from CSV

Create a CSV file with a `text` column:

```csv
text
"Breaking: Major tech company announces breakthrough in quantum computing"
"Championship game ends in dramatic overtime victory"
"Stock market reaches all-time high amid economic recovery"
```

Then run:

```bash
python inference.py \
    --checkpoint outputs/best_model_epoch_3.pt \
    --file input_texts.csv \
    --output predictions.csv
```

## Step 5: View Results

After training, check the `outputs/` directory for:

- `training_history.png` - Training loss and validation accuracy curves
- `confusion_matrix_validation.png` - Validation set confusion matrix
- `confusion_matrix_test.png` - Test set confusion matrix
- `best_model_epoch_N.pt` - Best model checkpoint
- `attention_visualizations/` - Attention weight visualizations (if `--viz_attention` was used)

## Example Workflow

```bash
# 1. Setup
pip install -r requirements.txt

# 2. Prepare data
python data_preparation.py

# 3. Train model
python main.py --data_dir data --output_dir outputs --epochs 3 --viz_attention

# 4. Make predictions
python inference.py --checkpoint outputs/best_model_epoch_3.pt --text "Your news article text here"
```

## Troubleshooting

### Out of Memory Error

Reduce batch size:
```bash
python main.py --batch_size 8 --data_dir data --output_dir outputs
```

### Slow Training

- Ensure you're using GPU if available (automatically detected)
- Increase batch size if memory allows
- Reduce `--max_len` (e.g., 128 instead of 256)

### Poor Performance

- Train for more epochs: `--epochs 5`
- Try different learning rates: `--learning_rate 3e-5` or `--learning_rate 1e-5`
- Check data quality in `data/` directory
- Review confusion matrix to identify problematic classes

## Next Steps

- Explore attention visualizations to understand model behavior
- Experiment with different model variants (see README.md)
- Try multi-label classification with `--multi_label` flag
- Implement data augmentation for better performance


