import os
import numpy as np
import librosa
import torch
from torch import nn

# Define emotions
emotions = ['sad', 'notsad']
max_length = 100

# Function to extract features from audio file
def extract_features(file_path, max_length=100):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        if mfccs.shape[1] < max_length:
            mfccs = np.pad(mfccs, ((0, 0), (0, max_length - mfccs.shape[1])), mode='constant')
        elif mfccs.shape[1] > max_length:
            mfccs = mfccs[:, :max_length]
        mfccs = np.expand_dims(mfccs, axis=0)  # Add channel dimension
        return mfccs
    except Exception as e:
        print(f"Error extracting features from {file_path}: {str(e)}")
        return None

# Define the model class
class EmotionClassifier(nn.Module):
    def __init__(self, num_classes, max_length):
        super(EmotionClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * (max_length // 8) * (13 // 8), 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the model
num_classes = len(emotions)
model = EmotionClassifier(num_classes, max_length)
model.load_state_dict(torch.load('C:/Users/green/Desktop/my code AI/CNNmodel11.pt'))
model.eval()

# Example usage for prediction
file_path = 'mhappy.wav'
features = extract_features(file_path, max_length)
if features is not None:
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    # Predict emotion
    with torch.no_grad():
        logits = model(features)
        pred = torch.argmax(logits, dim=-1)
        print(f'Predicted emotion: {emotions[pred.item()]}')
