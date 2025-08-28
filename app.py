
from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import pickle
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import logging

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Model classes (same as in training)
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, num_layers=2, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout, bidirectional=False)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        output = self.dropout(hidden[-1])
        output = self.fc(output)
        return torch.sigmoid(output)

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, num_layers=2, dropout=0.3):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                             batch_first=True, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.bilstm(embedded)
        output = torch.cat((hidden[-2], hidden[-1]), dim=1)
        output = self.dropout(output)
        output = self.fc(output)
        return torch.sigmoid(output)

class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, num_filters=100, filter_sizes=[3, 4, 5], dropout=0.3):
        super(CNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, 1)

    def forward(self, x):
        embedded = self.embedding(x).transpose(1, 2)
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(embedded))
            pooled = torch.max_pool1d(conv_out, kernel_size=conv_out.size(2))
            conv_outputs.append(pooled.squeeze(2))

        output = torch.cat(conv_outputs, dim=1)
        output = self.dropout(output)
        output = self.fc(output)
        return torch.sigmoid(output)

# Global variables for model and preprocessing
model = None
vocab = None
config = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    global model, vocab, config

    try:
        # Load model checkpoint
        checkpoint = torch.load('models/best_toxicity_model.pth', map_location=device)

        # Load vocabulary
        with open('models/vocabulary.pkl', 'rb') as f:
            vocab = pickle.load(f)

        # Load preprocessing config
        with open('models/preprocessing_config.pkl', 'rb') as f:
            config = pickle.load(f)

        # Initialize model based on saved class name
        model_class = checkpoint['model_class']
        if model_class == 'LSTMClassifier':
            model = LSTMClassifier(checkpoint['vocab_size'], **checkpoint['model_config'])
        elif model_class == 'BiLSTMClassifier':
            model = BiLSTMClassifier(checkpoint['vocab_size'], **checkpoint['model_config'])
        elif model_class == 'CNNClassifier':
            model = CNNClassifier(checkpoint['vocab_size'], **checkpoint['model_config'])

        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        app.logger.info("Model loaded successfully!")
        return True

    except Exception as e:
        app.logger.error(f"Error loading model: {str(e)}")
        return False

def clean_text(text):
    if not text:
        return ""

    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def advanced_preprocess(text):
    text = clean_text(text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = [token for token in tokens if len(token) > 1]

    return ' '.join(tokens)

def tokenize_text(text, vocab, max_length=128):
    processed_text = advanced_preprocess(text)
    words = processed_text.split()
    sequence = [vocab.get(word, 1) for word in words]  # 1 is <UNK>

    if len(sequence) > max_length:
        sequence = sequence[:max_length]
    else:
        sequence = sequence + [0] * (max_length - len(sequence))  # 0 is <PAD>

    return sequence

def predict_toxicity(text):
    try:
        sequence = tokenize_text(text, vocab, config['max_length'])
        input_tensor = torch.tensor([sequence]).to(device)

        with torch.no_grad():
            output = model(input_tensor).item()

        is_toxic = output > 0.5
        confidence = output if is_toxic else 1 - output

        return {
            'text': text,
            'is_toxic': bool(is_toxic),
            'confidence': float(confidence),
            'toxicity_score': float(output),
            'prediction': 'Toxic' if is_toxic else 'Non-Toxic'
        }

    except Exception as e:
        app.logger.error(f"Error in prediction: {str(e)}")
        return {
            'error': f'Prediction failed: {str(e)}'
        }

@app.route('/')
def home():
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Toxicity Shield - Ultra Modern</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        :root {
            --primary-color: #2e8b57;
            --secondary-color: #0e4b99;
            --accent-color: #0f3460;
            --success-color: #20b2aa;
            --danger-color: #dc3545;
            --warning-color: #ffa500;
            --dark-bg: #0f172a;
            --card-bg: rgba(15, 52, 96, 0.15);
            --text-primary: #f0f8ff;
            --text-secondary: #e0f2e7;
            --text-muted: #87ceeb;
            --glass-bg: rgba(46, 139, 87, 0.15);
            --glass-border: rgba(46, 139, 87, 0.3);
        }

        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #0f3460 0%, #0e4b99 35%, #2e8b57 100%);
            min-height: 100vh;
            color: var(--text-primary);
            overflow-x: hidden;
            position: relative;
        }

        body::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: 
                radial-gradient(circle at 20% 20%, rgba(46, 139, 87, 0.4) 0%, transparent 50%),
                radial-gradient(circle at 80% 80%, rgba(14, 75, 153, 0.4) 0%, transparent 50%),
                radial-gradient(circle at 40% 40%, rgba(15, 52, 96, 0.3) 0%, transparent 50%);
            z-index: -2;
        }

        .animated-bg {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
            overflow: hidden;
        }

        .floating-shapes {
            position: absolute;
            width: 100%;
            height: 100%;
        }

        .shape {
            position: absolute;
            border-radius: 50%;
            background: linear-gradient(45deg, rgba(46, 139, 87, 0.15), rgba(14, 75, 153, 0.15));
            animation: float 6s ease-in-out infinite;
        }

        .shape:nth-child(1) {
            width: 80px;
            height: 80px;
            top: 20%;
            left: 10%;
            animation-delay: 0s;
        }

        .shape:nth-child(2) {
            width: 120px;
            height: 120px;
            top: 60%;
            right: 10%;
            animation-delay: 2s;
        }

        .shape:nth-child(3) {
            width: 60px;
            height: 60px;
            bottom: 20%;
            left: 20%;
            animation-delay: 4s;
        }

        @keyframes float {
            0%, 100% { transform: translateY(0px) rotate(0deg); }
            50% { transform: translateY(-20px) rotate(180deg); }
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
            position: relative;
            z-index: 1;
        }

        .header {
            text-align: center;
            margin-bottom: 3rem;
            animation: slideDown 1s ease-out;
        }

        .logo {
            display: inline-flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1rem;
        }

        .logo-icon {
            width: 60px;
            height: 60px;
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            border-radius: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            animation: pulse 2s infinite;
            box-shadow: 0 0 30px rgba(99, 102, 241, 0.5);
        }

        .title {
            font-size: 3.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #fff, #cbd5e1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.5rem;
            animation: glow 3s ease-in-out infinite alternate;
        }

        .subtitle {
            font-size: 1.2rem;
            color: var(--text-secondary);
            font-weight: 400;
        }

        .main-card {
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            border-radius: 24px;
            padding: 3rem;
            box-shadow: 
                0 25px 50px -12px rgba(0, 0, 0, 0.5),
                0 0 0 1px rgba(255, 255, 255, 0.1);
            animation: slideUp 1s ease-out 0.3s both;
            position: relative;
            overflow: hidden;
        }

        .main-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
            animation: shimmer 3s infinite;
        }

        .form-group {
            margin-bottom: 2rem;
        }

        .form-label {
            display: block;
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 0.75rem;
        }

        .textarea-container {
            position: relative;
        }

        .form-textarea {
            width: 100%;
            min-height: 150px;
            padding: 1.5rem;
            background: rgba(30, 41, 59, 0.5);
            border: 2px solid transparent;
            border-radius: 16px;
            color: var(--text-primary);
            font-size: 1rem;
            font-family: inherit;
            resize: vertical;
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
        }

        .form-textarea:focus {
            outline: none;
            border-color: var(--primary-color);
            box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.1), 0 0 20px rgba(99, 102, 241, 0.3);
            transform: translateY(-2px);
        }

        .form-textarea::placeholder {
            color: var(--text-muted);
        }

        .char-counter {
            position: absolute;
            bottom: 12px;
            right: 16px;
            font-size: 0.875rem;
            color: var(--text-muted);
            font-family: 'JetBrains Mono', monospace;
        }

        .analyze-btn {
            width: 100%;
            padding: 1.25rem 2rem;
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            border: none;
            border-radius: 16px;
            color: white;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
            box-shadow: 0 10px 25px rgba(46, 139, 87, 0.3);
        }

        .analyze-btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 15px 35px rgba(46, 139, 87, 0.4);
        }

        .analyze-btn:active {
            transform: translateY(-1px);
        }

        .analyze-btn.loading {
            pointer-events: none;
        }

        .btn-content {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            transition: opacity 0.3s ease;
        }

        .loading-spinner {
            width: 20px;
            height: 20px;
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-top: 2px solid white;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            display: none;
        }

        .result-container {
            margin-top: 2rem;
            opacity: 0;
            transform: translateY(20px);
            transition: all 0.5s ease;
        }

        .result-container.show {
            opacity: 1;
            transform: translateY(0);
        }

        .result-card {
            padding: 2rem;
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            position: relative;
            overflow: hidden;
        }

        .result-card.safe {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(6, 182, 212, 0.2));
            border-color: rgba(16, 185, 129, 0.3);
        }

        .result-card.toxic {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(245, 158, 11, 0.2));
            border-color: rgba(239, 68, 68, 0.3);
        }

        .result-header {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1.5rem;
        }

        .result-icon {
            width: 48px;
            height: 48px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
        }

        .result-icon.safe {
            background: linear-gradient(135deg, var(--success-color), var(--accent-color));
        }

        .result-icon.toxic {
            background: linear-gradient(135deg, var(--danger-color), var(--warning-color));
        }

        .result-title {
            font-size: 1.5rem;
            font-weight: 700;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }

        .metric-card {
            background: rgba(255, 255, 255, 0.05);
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .metric-label {
            font-size: 0.875rem;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
        }

        .metric-value {
            font-size: 1.75rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }

        .confidence-bar {
            width: 100%;
            height: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 0.5rem;
        }

        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
            transition: width 1s ease;
            border-radius: 4px;
        }

        .footer {
            text-align: center;
            margin-top: 3rem;
            padding-top: 2rem;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            color: var(--text-muted);
        }

        .tech-stack {
            display: flex;
            justify-content: center;
            gap: 1rem;
            margin-top: 1rem;
            flex-wrap: wrap;
        }

        .tech-badge {
            background: rgba(255, 255, 255, 0.1);
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.875rem;
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: all 0.3s ease;
        }

        .tech-badge:hover {
            background: rgba(255, 255, 255, 0.2);
            transform: translateY(-2px);
        }

        @keyframes slideDown {
            from { opacity: 0; transform: translateY(-30px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }

        @keyframes glow {
            from { text-shadow: 0 0 20px rgba(255, 255, 255, 0.5); }
            to { text-shadow: 0 0 30px rgba(255, 255, 255, 0.8), 0 0 40px rgba(99, 102, 241, 0.5); }
        }

        @keyframes shimmer {
            0% { left: -100%; }
            100% { left: 100%; }
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        @media (max-width: 768px) {
            .container { padding: 1rem; }
            .title { font-size: 2.5rem; }
            .main-card { padding: 2rem; }
            .metrics-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="animated-bg">
        <div class="floating-shapes">
            <div class="shape"></div>
            <div class="shape"></div>
            <div class="shape"></div>
        </div>
    </div>

    <div class="container">
        <header class="header">
            <div class="logo">
                <div class="logo-icon">
                    <i class="fas fa-shield-alt"></i>
                </div>
            </div>
            <h1 class="title">AI Toxicity Shield</h1>
            <p class="subtitle">Advanced Neural Network-Powered Content Analysis</p>
        </header>

        <main class="main-card">
            <form id="predictionForm">
                <div class="form-group">
                    <label for="commentText" class="form-label">
                        <i class="fas fa-comment-dots"></i>
                        Enter your comment for analysis
                    </label>
                    <div class="textarea-container">
                        <textarea 
                            id="commentText" 
                            class="form-textarea"
                            placeholder="Type or paste your comment here to analyze its toxicity level using our advanced AI model..."
                            maxlength="1000"
                            required
                        ></textarea>
                        <div class="char-counter">
                            <span id="charCount">0</span>/1000
                        </div>
                    </div>
                </div>

                <button type="submit" class="analyze-btn" id="analyzeBtn">
                    <div class="btn-content">
                        <i class="fas fa-brain"></i>
                        <span>Analyze with AI</span>
                    </div>
                    <div class="loading-spinner"></div>
                </button>
            </form>

            <div id="resultContainer" class="result-container">
                <!-- Results will be populated here -->
            </div>
        </main>

        <footer class="footer">
            <p>&copy; 2025 AI Toxicity Shield - Powered by Advanced Deep Learning</p>
            <div class="tech-stack">
                <span class="tech-badge">PyTorch</span>
                <span class="tech-badge">BiLSTM</span>
                <span class="tech-badge">NLP</span>
                <span class="tech-badge">Flask</span>
                <span class="tech-badge">Deep Learning</span>
            </div>
        </footer>
    </div>

    <script>
        // Character counter
        const textarea = document.getElementById('commentText');
        const charCount = document.getElementById('charCount');

        textarea.addEventListener('input', function() {
            charCount.textContent = this.value.length;
        });

        // Form submission
        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();

            const text = textarea.value.trim();
            const resultContainer = document.getElementById('resultContainer');
            const analyzeBtn = document.getElementById('analyzeBtn');
            const btnContent = analyzeBtn.querySelector('.btn-content');
            const spinner = analyzeBtn.querySelector('.loading-spinner');

            if (!text) {
                showError('Please enter some text to analyze!');
                return;
            }

            // Show loading state
            analyzeBtn.classList.add('loading');
            btnContent.style.display = 'none';
            spinner.style.display = 'block';

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({text: text})
                });

                const data = await response.json();

                if (data.error) {
                    showError(data.error);
                } else {
                    showResult(data);
                }
            } catch (error) {
                showError('Network error: ' + error.message);
            } finally {
                // Hide loading state
                analyzeBtn.classList.remove('loading');
                btnContent.style.display = 'flex';
                spinner.style.display = 'none';
            }
        });

        function showResult(data) {
            const resultContainer = document.getElementById('resultContainer');
            const isToxic = data.is_toxic;
            const confidence = (data.confidence * 100).toFixed(1);
            const toxicityScore = (data.toxicity_score * 100).toFixed(1);

            const resultClass = isToxic ? 'toxic' : 'safe';
            const resultIcon = isToxic ? 'fa-exclamation-triangle' : 'fa-check-circle';
            const resultTitle = data.prediction;

            resultContainer.innerHTML = `
                <div class="result-card ${resultClass}">
                    <div class="result-header">
                        <div class="result-icon ${resultClass}">
                            <i class="fas ${resultIcon}"></i>
                        </div>
                        <h3 class="result-title">${resultTitle}</h3>
                    </div>

                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-label">Confidence Level</div>
                            <div class="metric-value">${confidence}%</div>
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: ${confidence}%"></div>
                            </div>
                        </div>

                        <div class="metric-card">
                            <div class="metric-label">Toxicity Score</div>
                            <div class="metric-value">${toxicityScore}%</div>
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: ${toxicityScore}%"></div>
                            </div>
                        </div>
                    </div>
                </div>
            `;

            resultContainer.classList.add('show');
        }

        function showError(message) {
            const resultContainer = document.getElementById('resultContainer');

            resultContainer.innerHTML = `
                <div class="result-card toxic">
                    <div class="result-header">
                        <div class="result-icon toxic">
                            <i class="fas fa-exclamation-circle"></i>
                        </div>
                        <h3 class="result-title">Error</h3>
                    </div>
                    <p style="color: var(--text-secondary); margin-top: 1rem;">${message}</p>
                </div>
            `;

            resultContainer.classList.add('show');
        }

        // Add some interactive effects
        document.addEventListener('mousemove', function(e) {
            const shapes = document.querySelectorAll('.shape');
            const x = e.clientX / window.innerWidth;
            const y = e.clientY / window.innerHeight;

            shapes.forEach((shape, index) => {
                const speed = (index + 1) * 0.5;
                shape.style.transform = `translate(${x * speed * 20}px, ${y * speed * 20}px)`;
            });
        });
    </script>
</body>
</html>'''

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        text = data['text']

        if not text.strip():
            return jsonify({'error': 'Empty text provided'}), 400

        result = predict_toxicity(text)
        return jsonify(result)

    except Exception as e:
        app.logger.error(f"Error in /predict endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device)
    })

@app.route('/model_info')
def model_info():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        with open('models/model_metrics.pkl', 'rb') as f:
            metrics = pickle.load(f)

        return jsonify({
            'model_type': metrics['best_model'],
            'validation_accuracy': metrics['validation_accuracy'],
            'validation_roc_auc': metrics['validation_roc_auc'],
            'vocabulary_size': len(vocab) if vocab else 0,
            'max_sequence_length': config['max_length'] if config else 0
        })
    except Exception as e:
        return jsonify({'error': f'Could not load model info: {str(e)}'}), 500

if __name__ == '__main__':
    if load_model():
        print("Starting Flask app with loaded model...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model. Please check if model files exist in 'models/' directory.")
