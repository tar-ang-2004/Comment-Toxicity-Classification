
import torch
import torch.nn as nn
import pickle
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

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

def load_model(model_path, vocab_path, config_path):
    """Load the trained model and preprocessing components"""
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')

    # Load vocabulary
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Load preprocessing config
    with open(config_path, 'rb') as f:
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
    model.eval()

    return model, vocab, config
