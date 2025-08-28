# Comment Toxicity Classification Flask App

## Setup Instructions

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Make sure you have the trained model files in the 'models/' directory:
   - models/best_toxicity_model.pth
   - models/vocabulary.pkl
   - models/preprocessing_config.pkl
   - models/model_metrics.pkl

3. Run the Flask app:
   ```
   python app.py
   ```

4. Open your browser and go to: http://localhost:5000

## API Endpoints

- `/` - Web interface for testing
- `/predict` - POST endpoint for predictions (JSON: {"text": "your comment"})
- `/health` - Health check
- `/model_info` - Model information

## Example API Usage

```python
import requests

# Make a prediction
response = requests.post('http://localhost:5000/predict', 
                        json={'text': 'This is a test comment'})
result = response.json()
print(result)
```

## Model Features

- Deep Learning with PyTorch
- Multiple model architectures (LSTM, BiLSTM, CNN)
- Advanced NLP preprocessing
- Real-time toxicity detection
- Confidence scoring
- Web interface for easy testing
