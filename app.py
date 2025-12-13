from flask import Flask, request, jsonify, render_template_string
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import os

app = Flask(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None
if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
class_names = ['World', 'Sports', 'Business', 'Sci/Tech']

def load_model():
    global model, tokenizer
    output_dir = 'outputs'
    # Find the best model
    best_model_path = None
    if os.path.exists(output_dir):
        files = os.listdir(output_dir)
        # Look for best_model_epoch_*.pt
        best_models = [f for f in files if f.startswith('best_model_epoch_') and f.endswith('.pt')]
        if best_models:
            # Pick the last one (assuming higher epoch or just any valid one)
            best_model_path = os.path.join(output_dir, sorted(best_models)[-1])
            print(f"Loading model from {best_model_path}")
    
    if not best_model_path:
        print("No best model found. Please train the model first.")
        # Fallback for testing without trained model if needed, but better to warn
        return

    model_name = 'distilbert-base-uncased'
    
    try:
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        
        # Load model structure
        model = DistilBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=4
        )
        
        # Load state dict
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.route('/')
def home():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>News Classifier</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 800px; margin: 0 auto; padding: 40px; background-color: #f5f7fa; color: #333; }
            .container { background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; margin-bottom: 20px; text-align: center; }
            p.subtitle { text-align: center; color: #666; margin-bottom: 30px; }
            textarea { width: 100%; height: 150px; padding: 15px; margin-bottom: 20px; border: 2px solid #e1e8ed; border-radius: 8px; resize: vertical; font-size: 16px; transition: border-color 0.3s; box-sizing: border-box; }
            textarea:focus { border-color: #3498db; outline: none; }
            button { display: block; width: 100%; background-color: #3498db; color: white; padding: 12px; border: none; border-radius: 8px; cursor: pointer; font-size: 18px; font-weight: 600; transition: background-color 0.3s; }
            button:hover { background-color: #2980b9; }
            #result { margin-top: 30px; padding: 20px; border-radius: 8px; display: none; text-align: center; }
            .result-success { background-color: #dbfadd; border: 1px solid #c3e6cb; color: #155724; }
            .result-error { background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
            .label { font-weight: 800; font-size: 1.4em; display: block; margin-bottom: 5px; }
            .confidence { font-size: 0.9em; opacity: 0.8; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>News Classifier</h1>
            <p class="subtitle">Enter a news headline or article to classify it into: World, Sports, Business, or Sci/Tech</p>
            <textarea id="textInput" placeholder="E.g., NASA announces new mission to Mars..."></textarea>
            <button onclick="predict()">Classify News</button>
            <div id="result"></div>
        </div>

        <script>
            async function predict() {
                const text = document.getElementById('textInput').value;
                const resultDiv = document.getElementById('result');
                
                if (!text.trim()) {
                    resultDiv.className = 'result-error';
                    resultDiv.innerHTML = 'Please enter some text.';
                    resultDiv.style.display = 'block';
                    return;
                }
                
                resultDiv.style.display = 'none';
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ text: text })
                    });
                    const data = await response.json();
                    
                    if (data.error) {
                        resultDiv.className = 'result-error';
                        resultDiv.innerHTML = `Error: ${data.error}`;
                    } else {
                        resultDiv.className = 'result-success';
                        resultDiv.innerHTML = `
                            <span class="label">${data.class}</span>
                            <span class="confidence">Confidence: ${(data.confidence * 100).toFixed(1)}%</span>
                        `;
                    }
                    resultDiv.style.display = 'block';
                } catch (e) {
                    console.error(e);
                    resultDiv.className = 'result-error';
                    resultDiv.innerHTML = "Network error. Please try again.";
                    resultDiv.style.display = 'block';
                }
            }
        </script>
    </body>
    </html>
    ''')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded. Please ensure the model is trained and available.'}), 500
    
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=256, padding='max_length').to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence, predicted_class = torch.max(probs, dim=-1)
        
    return jsonify({
        'class': class_names[predicted_class.item()],
        'confidence': confidence.item()
    })

if __name__ == '__main__':
    load_model()
    app.run(debug=True, port=5001)
