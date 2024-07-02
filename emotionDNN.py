import os
import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Define emotions
emotions = ['sad', 'notsad']
max_length = 100

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_features(file_path, max_length=100):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        if mfccs.shape[1] < max_length:
            mfccs = np.pad(mfccs, ((0, 0), (0, max_length - mfccs.shape[1])), mode='constant')
        elif mfccs.shape[1] > max_length:
            mfccs = mfccs[:, :max_length]
        return mfccs.T  # Transpose to shape [max_length, 13]
    except Exception as e:
        print(f"Error extracting features from {file_path}: {str(e)}")
        return None

class EmotionDNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(EmotionDNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Load the pre-trained model
input_size = 13 * max_length  # 13 MFCC features times the maximum length
hidden_size = 256
num_classes = len(emotions)

model = EmotionDNN(input_size, hidden_size, num_classes).to(device)
model_path = 'DNN_exec14_model.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Inference on new audio file
def predict_emotion(audio_file_path, model, max_length=100):
    features = extract_features(audio_file_path, max_length)
    if features is not None:
        features = features.flatten()
        features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            predicted_emotion = emotions[predicted.item()]
            return predicted_emotion
    else:
        return None

# Example usage for inference
audio_file_path = '1001_DFA_SAD_XX.wav'  # Replace with your audio file path
predicted_emotion = predict_emotion(audio_file_path, model, max_length)
if predicted_emotion:
    print(f'Predicted Emotion: {predicted_emotion}')
else:
    print('Failed to extract features from the audio file.')
