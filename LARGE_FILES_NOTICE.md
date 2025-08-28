# Large Files Notice

This repository contains the complete Comment Toxicity Classification project, but some large files have been excluded from version control due to GitHub's file size limitations:

## Excluded Files (Available Locally)
- `train.csv` (~65 MB) - Training dataset
- `test.csv` (~57 MB) - Test dataset  
- `models/best_toxicity_model.pth` - Trained PyTorch model
- `models/model_metrics.pkl` - Model performance metrics
- `models/preprocessing_config.pkl` - Preprocessing configuration
- `models/vocabulary.pkl` - Vocabulary mapping

## To Run This Project Locally
1. Download the original datasets from Kaggle or your data source
2. Place the CSV files in the root directory
3. Train the models using the provided notebook, or
4. Use pre-trained model files if available

## Dataset Information
- Total samples: 159,571 comments
- Toxic comments: 15,294 (9.6%)
- Non-toxic comments: 144,277 (90.4%)
- Features: comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

## Model Performance
- Best Model: BiLSTM
- Validation Accuracy: 84.75%
- Architecture: Bidirectional LSTM with embedding layer
- Comprehensive hypothesis testing included

For questions about obtaining the dataset or model files, please refer to the documentation in the notebook.
